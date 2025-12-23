from datetime import datetime
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tensorboardX import SummaryWriter

from .checkpoint import save_checkpoint
from .config import ModelConfig, TrainConfig
from .data import Batch, dummy_data_generator
from .model import Encoder

TrainStep = Callable[
    [Encoder, nnx.Optimizer, Batch],
    Tuple[Array, Array, Array, Encoder, nnx.Optimizer],
]


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


def create_train_step(mesh: Mesh) -> TrainStep:
    """Create a JIT-compiled training step with data-parallel sharding."""
    data_sharding = NamedSharding(mesh, P("data"))

    @jax.jit
    def train_step(
        model: Encoder,
        optimizer: nnx.Optimizer,
        batch: Batch,
    ) -> Tuple[Array, Array, Array, Encoder, nnx.Optimizer]:
        """
        Execute a single training step with data parallelism.

        Args:
            batch: A `Batch` object containing training data.

        Returns:
            Tuple of (total_loss, mlm_loss, nsp_loss), each a scalar, plus the
            updated model and optimizer.
        """
        batch = jax.device_put(batch, data_sharding)

        def loss_fn(model: Encoder) -> Tuple[Array, Tuple[Array, Array]]:
            mlm_logits, nsp_logits = model(
                batch.input_ids, batch.seg_ids, mask=batch.attention_mask
            )

            mlm_losses = optax.softmax_cross_entropy_with_integer_labels(
                mlm_logits, batch.mlm_targets
            )
            mlm_loss = jnp.sum(mlm_losses * batch.mlm_mask) / (jnp.sum(batch.mlm_mask) + 1e-9)

            nsp_loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(nsp_logits, batch.nsp_labels)
            )

            return mlm_loss + nsp_loss, (mlm_loss, nsp_loss)

        (loss, (mlm_loss, nsp_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, mlm_loss, nsp_loss, model, optimizer

    return train_step


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

    data_generator = dummy_data_generator(model_config, train_config, global_batch_size, rngs)

    print(f"Logging to {log_dir}")
    print("Starting training...")
    print(f"{'Step':<6} | {'Total Loss':<12} | {'MLM Loss':<10} | {'NSP Loss':<10} | {'LR':<10}")
    print("-" * 60)

    for step in range(train_config.total_steps):
        batch = next(data_generator)
        loss, mlm, nsp, model, optimizer = train_step(model, optimizer, batch)
        current_lr = scheduler(optimizer.step.value)

        # Logged losses are from one shard only (noisier but unbiased estimate of global loss)
        writer.add_scalar("loss/total", float(loss), step)
        writer.add_scalar("loss/mlm", float(mlm), step)
        writer.add_scalar("loss/nsp", float(nsp), step)
        writer.add_scalar("lr", float(current_lr), step)

        if step % train_config.log_interval == 0:
            print(
                f"{step:<6} | {float(loss):.4f}      | {float(mlm):.4f}    | {float(nsp):.4f}     | {current_lr:.6f}"
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
