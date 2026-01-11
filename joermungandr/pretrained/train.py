import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel

from .config import RerankerConfig, RerankerTrainConfig
from .data import RerankerDataset, collate_reranker_batch, load_jsonl
from .download import load_model, load_tokenizer


@torch.compile
def compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, token_yes_id: int, token_no_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute binary cross-entropy loss on yes/no logits.

    Args:
        logits: Model output logits [B, S, V].
        labels: Binary labels [B].
        token_yes_id: Token ID for "yes".
        token_no_id: Token ID for "no".

    Returns:
        Tuple of (loss, predictions).
    """
    # Get logits for the last token
    last_logits = logits[:, -1, :]

    # Extract yes/no logits
    yes_logits = last_logits[:, token_yes_id]
    no_logits = last_logits[:, token_no_id]

    # Binary classification loss
    scores = torch.stack([no_logits, yes_logits], dim=1)
    log_probs = F.log_softmax(scores, dim=1)

    loss = F.nll_loss(log_probs, labels)

    # Predictions for accuracy
    preds = scores.argmax(dim=1)

    return loss, preds


@torch.compile
def forward_and_loss(
    model: PreTrainedModel,
    batch: dict[str, torch.Tensor],
    token_yes_id: int,
    token_no_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass and compute loss.

    Args:
        model: The language model.
        batch: Batch with input_ids, attention_mask, labels.
        token_yes_id: Token ID for "yes".
        token_no_id: Token ID for "no".

    Returns:
        Tuple of (loss, predictions).
    """
    # Model forward returns CausalLMOutput with .logits attribute
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits
    return compute_loss(logits, batch["labels"], token_yes_id, token_no_id)


def create_optimizer_and_scheduler(
    model: PreTrainedModel,
    lr: float,
    weight_decay: float,
    num_training_steps: int,
    warmup_ratio: float,
) -> tuple[AdamW, SequentialLR]:
    """Create AdamW optimizer with cosine schedule and linear warmup.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        num_training_steps: Total number of training steps.
        warmup_ratio: Fraction of steps for warmup.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_warmup_steps = int(num_training_steps * warmup_ratio)
    num_decay_steps = num_training_steps - num_warmup_steps

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1.0, total_iters=num_warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_decay_steps, eta_min=0.0)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps]
    )

    return optimizer, scheduler


def evaluate(
    model: PreTrainedModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    token_yes_id: int,
    token_no_id: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a dataset.

    Args:
        model: The language model.
        dataloader: Evaluation data loader.
        token_yes_id: Token ID for "yes".
        token_no_id: Token ID for "no".
        device: Device to run on.

    Returns:
        Dict with loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, preds = forward_and_loss(model, batch, token_yes_id, token_no_id)

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == batch["labels"]).sum().item()
            total_samples += batch_size

    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


def train_reranker(model_config: RerankerConfig, train_config: RerankerTrainConfig) -> None:
    """Fine-tune reranker on (start, goal, guide) triples.

    Args:
        model_config: Model configuration.
        train_config: Training configuration.
    """
    if train_config.train_data_path is None:
        raise ValueError("train_data_path must be specified")

    # Set random seeds for reproducibility
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = None
    if train_config.logging_dir is not None:
        log_dir = Path(train_config.logging_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))

    print(f"Loading model: {model_config.model_id}")
    model = load_model(model_config.model_id, model_config)
    tokenizer = load_tokenizer(model_config.model_id)
    model = model.to(device)  # pyright: ignore[reportArgumentType]

    token_yes_id = tokenizer.convert_tokens_to_ids("yes")
    token_no_id = tokenizer.convert_tokens_to_ids("no")
    assert isinstance(token_yes_id, int), "Expected single token ID for 'yes'"
    assert isinstance(token_no_id, int), "Expected single token ID for 'no'"

    print(f"Loading training data: {train_config.train_data_path}")
    train_samples = list(load_jsonl(train_config.train_data_path))
    train_dataset = RerankerDataset(train_samples, tokenizer, model_config.max_length)
    print(f"Training examples: {len(train_dataset)}")

    eval_dataset = None
    if train_config.eval_data_path:
        print(f"Loading eval data: {train_config.eval_data_path}")
        eval_samples = list(load_jsonl(train_config.eval_data_path))
        eval_dataset = RerankerDataset(eval_samples, tokenizer, model_config.max_length)
        print(f"Eval examples: {len(eval_dataset)}")

    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int), "Expected int pad_token_id"
    collate_fn = functools.partial(collate_reranker_batch, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
        )

    num_training_steps = (
        len(train_loader) // train_config.gradient_accumulation_steps
    ) * train_config.num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        num_training_steps=num_training_steps,
        warmup_ratio=train_config.warmup_ratio,
    )

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if writer:
        writer.add_text("config/model", str(model_config))
        writer.add_text("config/train", str(train_config))

    # Training loop
    print("\nStarting training...")
    print(f"{'Step':<8} | {'Loss':<12} | {'LR':<12} | {'Acc':<8}")
    print("-" * 50)

    global_step = 0
    accum_steps = train_config.gradient_accumulation_steps
    model.train()

    for epoch in range(train_config.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, preds = forward_and_loss(model, batch, token_yes_id, token_no_id)
            loss = loss / accum_steps  # Scale loss for accumulation
            loss.backward()

            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                # Use unscaled loss for logging (multiply back by accum_steps)
                unscaled_loss = loss.item() * accum_steps
                acc = (preds == batch["labels"]).float().mean().item()

                if writer:
                    writer.add_scalar("train/loss", unscaled_loss, global_step)
                    writer.add_scalar("train/accuracy", acc, global_step)
                    writer.add_scalar("train/lr", current_lr, global_step)
                    writer.add_scalar("train/epoch", epoch, global_step)

                if global_step % train_config.log_interval == 0:
                    print(
                        f"{global_step:<8} | {unscaled_loss:<12.4f} | {current_lr:<12.6f} | {acc:<8.4f}"
                    )

                if eval_loader and global_step % train_config.eval_interval == 0:
                    metrics = evaluate(model, eval_loader, token_yes_id, token_no_id, device)
                    print(
                        f"  [Eval] Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                    )

                    if writer:
                        writer.add_scalar("eval/loss", metrics["loss"], global_step)
                        writer.add_scalar("eval/accuracy", metrics["accuracy"], global_step)

                    model.train()

                if global_step % train_config.save_interval == 0:
                    ckpt_path = output_dir / f"step_{global_step}"
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"  [Saved] {ckpt_path}")

    # Save final model
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved final model to {final_path}")

    # Final evaluation
    if eval_loader:
        metrics = evaluate(model, eval_loader, token_yes_id, token_no_id, device)
        print(f"Final eval - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        if writer:
            writer.add_scalar("eval/final_loss", metrics["loss"], global_step)
            writer.add_scalar("eval/final_accuracy", metrics["accuracy"], global_step)

    if writer:
        writer.close()

    print("\nTraining complete.")
