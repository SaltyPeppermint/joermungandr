from datetime import datetime

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tensorboardX import SummaryWriter

from checkpoint import save_checkpoint
from config import ModelConfig, TrainConfig
from model import Encoder


def create_scheduler(config: TrainConfig) -> optax.Schedule:
    """Create a learning rate schedule with linear warmup and cosine decay."""
    warmup_steps = int(config.total_steps * config.warmup_ratio)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=warmup_steps,
        decay_steps=config.total_steps - warmup_steps,
        end_value=0.0,
    )


def create_train_step(mesh: Mesh):
    """Create a JIT-compiled training step with data-parallel sharding."""
    data_sharding = NamedSharding(mesh, P("data"))

    @jax.jit
    def train_step(model, optimizer, batch):
        """
        Execute a single training step with data parallelism.

        Args:
            batch: Tuple of (input_ids, seg_ids, mlm_targets, mlm_mask, nsp_labels)
                - input_ids: [B, L] token IDs
                - seg_ids: [B, L] segment IDs
                - mlm_targets: [B, L] target token IDs for MLM
                - mlm_mask: [B, L] boolean mask for MLM positions
                - nsp_labels: [B] binary labels for NSP

        Returns:
            Tuple of (total_loss, mlm_loss, nsp_loss), each a scalar.
        """
        input_ids, seg_ids, mlm_targets, mlm_mask, nsp_labels = batch

        def loss_fn(model):
            mlm_logits, nsp_logits = model(input_ids, seg_ids)

            mlm_losses = optax.softmax_cross_entropy_with_integer_labels(mlm_logits, mlm_targets)
            mlm_loss = jnp.sum(mlm_losses * mlm_mask) / (jnp.sum(mlm_mask) + 1e-9)

            nsp_loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(nsp_logits, nsp_labels)
            )
            return mlm_loss + nsp_loss, (mlm_loss, nsp_loss)

        (loss, (mlm_loss, nsp_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, mlm_loss, nsp_loss

    def sharded_step(model, optimizer, batch):
        batch = jax.device_put(batch, data_sharding)
        return train_step(model, optimizer, batch)

    return sharded_step


def train_loop(model_config: ModelConfig, train_config: TrainConfig):
    """Run the training loop with TensorBoard logging and multi-GPU data parallelism."""

    # Set up checkpoint directory
    ckpt_path = train_config.checkpoint_path
    if ckpt_path:
        ckpt_path.mkdir(parents=True, exist_ok=True)

    # Set up device mesh for data parallelism
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(devices, axis_names=("data",))
    global_batch_size = train_config.batch_size_per_device * num_devices

    print(f"Running on {num_devices} device(s), global batch size: {global_batch_size}")

    replicated = NamedSharding(mesh, P())

    # Initialize model and optimizer
    rngs = nnx.Rngs(train_config.seed)
    model = Encoder(model_config, rngs=rngs)

    scheduler = create_scheduler(train_config)
    tx = optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adamw(learning_rate=scheduler, weight_decay=train_config.weight_decay),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Replicate model and optimizer state across devices
    nnx.update(model, jax.device_put(nnx.state(model), replicated))
    nnx.update(optimizer, jax.device_put(nnx.state(optimizer), replicated))

    train_step = create_train_step(mesh)

    log_dir = train_config.log_dir or f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    writer.add_text("model_config", str(model_config))
    writer.add_text("train_config", str(train_config))
    writer.add_text("devices", f"{num_devices} device(s)")

    print(f"Logging to {log_dir}")
    print("Starting training...")
    print(f"{'Step':<6} | {'Total Loss':<12} | {'MLM Loss':<10} | {'NSP Loss':<10} | {'LR':<10}")
    print("-" * 60)

    for step in range(train_config.total_steps):
        # Generate dummy batch
        inp = jax.random.randint(
            rngs.step(), (global_batch_size, train_config.seq_len), 0, model_config.vocab_size
        )
        seg = jax.random.randint(
            rngs.step(), (global_batch_size, train_config.seq_len), 0, model_config.num_segments
        )
        mlm_mask = jax.random.uniform(rngs.step(), (global_batch_size, train_config.seq_len)) < 0.15
        mlm_targets = jax.random.randint(
            rngs.step(), (global_batch_size, train_config.seq_len), 0, model_config.vocab_size
        )
        nsp_labels = jax.random.randint(rngs.step(), (global_batch_size,), 0, 2)

        loss, mlm, nsp = train_step(model, optimizer, (inp, seg, mlm_targets, mlm_mask, nsp_labels))
        current_lr = scheduler(optimizer.step.value)

        # Logged losses are from one shard only (noisier but unbiased estimate of global loss)
        writer.add_scalar("loss/total", float(loss), step)
        writer.add_scalar("loss/mlm", float(mlm), step)
        writer.add_scalar("loss/nsp", float(nsp), step)
        writer.add_scalar("lr", float(current_lr), step)

        if step % train_config.log_interval == 0:
            print(
                f"{step:<6} | {loss:.4f}       | {mlm:.4f}     | {nsp:.4f}     | {current_lr:.6f}"
            )

        if ckpt_path and (step + 1) % train_config.save_interval == 0:
            save_checkpoint(model, ckpt_path, step + 1)
            print(f"Saved checkpoint at step {step + 1}")

    # Save final checkpoint
    if ckpt_path:
        save_checkpoint(model, ckpt_path, train_config.total_steps)
        print(f"Saved final checkpoint at step {train_config.total_steps}")

    writer.close()
    print("\nTraining complete.")
