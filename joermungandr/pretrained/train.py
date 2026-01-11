from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import RerankerConfig, RerankerTrainConfig
from .data import RerankerDataset, collate_reranker_batch, load_reranker_dataset
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
    logits: torch.Tensor = outputs.logits
    return compute_loss(logits, batch["labels"], token_yes_id, token_no_id)


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

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup TensorBoard logging
    writer = None
    if train_config.logging_dir is not None:
        log_dir = Path(train_config.logging_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))

    # Load model and tokenizer
    print(f"Loading model: {model_config.model_id}")
    model = load_model(model_config.model_id, model_config)
    tokenizer = load_tokenizer(model_config.model_id)
    model = model.to(device)  # pyright: ignore[reportArgumentType]

    # Get yes/no token IDs
    token_yes_id = tokenizer.convert_tokens_to_ids("yes")
    token_no_id = tokenizer.convert_tokens_to_ids("no")
    assert isinstance(token_yes_id, int), "Expected single token ID for 'yes'"
    assert isinstance(token_no_id, int), "Expected single token ID for 'no'"

    # Load datasets
    print(f"Loading training data: {train_config.train_data_path}")
    train_dataset = load_reranker_dataset(
        train_config.train_data_path, tokenizer, model_config.max_length
    )
    print(f"Training examples: {len(train_dataset)}")

    eval_dataset: RerankerDataset | None = None
    if train_config.eval_data_path:
        print(f"Loading eval data: {train_config.eval_data_path}")
        eval_dataset = load_reranker_dataset(
            train_config.eval_data_path, tokenizer, model_config.max_length
        )
        print(f"Eval examples: {len(eval_dataset)}")

    # Create data loaders
    # pad_token_id is int for standard tokenizers
    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int), "Expected int pad_token_id"
    collate_fn = partial(collate_reranker_batch, pad_token_id=pad_token_id)

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    eval_loader: DataLoader[dict[str, torch.Tensor]] | None = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
        )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    # Setup scheduler - use get_cosine_schedule_with_warmup for clearer typing
    num_training_steps = len(train_loader) * train_config.num_epochs
    num_warmup_steps = int(num_training_steps * train_config.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        import math

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Setup output directory
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log hyperparameters
    if writer:
        writer.add_text("config/model", str(model_config))
        writer.add_text("config/train", str(train_config))

    # Training loop
    print("\nStarting training...")
    print(f"{'Step':<8} | {'Loss':<12} | {'LR':<12} | {'Acc':<8}")
    print("-" * 50)

    global_step = 0
    model.train()

    for epoch in range(train_config.num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            loss, preds = forward_and_loss(model, batch, token_yes_id, token_no_id)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            # Optimizer step
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            acc = (preds == batch["labels"]).float().mean().item()

            # TensorBoard logging
            if writer:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/accuracy", acc, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/epoch", epoch, global_step)

            # Console logging
            if global_step % train_config.log_interval == 0:
                print(
                    f"{global_step:<8} | {loss.item():<12.4f} | {current_lr:<12.6f} | {acc:<8.4f}"
                )

            # Evaluation
            if eval_loader and global_step % train_config.eval_interval == 0:
                metrics = evaluate(model, eval_loader, token_yes_id, token_no_id, device)
                print(f"  [Eval] Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

                if writer:
                    writer.add_scalar("eval/loss", metrics["loss"], global_step)
                    writer.add_scalar("eval/accuracy", metrics["accuracy"], global_step)

                model.train()

            # Save checkpoint
            if global_step % train_config.save_interval == 0:
                ckpt_path = output_dir / f"step_{global_step}"
                save_checkpoint(model, tokenizer, ckpt_path)
                print(f"  [Saved] {ckpt_path}")

    # Save final model
    final_path = output_dir / "final"
    save_checkpoint(model, tokenizer, final_path)
    print(f"\nSaved final model to {final_path}")

    # Final evaluation
    if eval_loader:
        metrics = evaluate(model, eval_loader, token_yes_id, token_no_id, device)
        print(f"Final eval - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        if writer:
            writer.add_scalar("eval/final_loss", metrics["loss"], global_step)
            writer.add_scalar("eval/final_accuracy", metrics["accuracy"], global_step)

    # Close TensorBoard writer
    if writer:
        writer.close()

    print("\nTraining complete.")


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    path: Path,
) -> None:
    """Save model and tokenizer checkpoint.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        path: Directory to save to.
    """
    # HF save_pretrained has incomplete type stubs
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
