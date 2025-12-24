import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from jax.sharding import NamedSharding

from .. import checkpoint as ckpt
from .. import train_utils
from ..config import ModelConfig, TrainConfig
from .data import Seq2SeqBatch, dummy_seq2seq_generator
from .model import Seq2Seq


@nnx.jit(donate_argnames=("model", "optimizer"), static_argnames=("data_sharding",))
def train_step(
    model: Seq2Seq,
    optimizer: nnx.Optimizer,
    batch: Seq2SeqBatch,
    data_sharding: NamedSharding,
) -> tuple[Array, Seq2Seq, nnx.Optimizer]:
    """Execute a single training step with data parallelism.

    Returns:
        (loss, model, optimizer)
    """
    batch = jax.device_put(batch, data_sharding)

    def loss_fn(model: Seq2Seq) -> Array:
        logits = model(
            batch.encoder_ids,
            batch.encoder_seg_ids,
            batch.decoder_ids,
            encoder_mask=batch.encoder_mask,
            decoder_mask=batch.decoder_mask,
        )

        losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
        loss = jnp.sum(losses * batch.label_mask) / (jnp.sum(batch.label_mask) + 1e-9)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss, model, optimizer


def train_loop(model_config: ModelConfig, train_config: TrainConfig) -> None:
    """Run the seq2seq training loop with TensorBoard logging and multi-GPU data parallelism."""

    ckpt_path = ckpt.setup_dir(train_config)

    # Set up device mesh for data parallelism
    num_devices, replicated, data_sharding = train_utils.setup_device_mesh()
    global_batch_size = train_config.batch_size_per_device * num_devices

    print(f"Running on {num_devices} device(s), global batch size: {global_batch_size}")

    # Initialize model and optimizer
    rngs = nnx.Rngs(train_config.seed)
    model = Seq2Seq(model_config, rngs=rngs)
    optimizer, scheduler = train_utils.create_optimizer(model, train_config)

    # Replicate model and optimizer state across devices
    train_utils.replicate_on_devices(model, optimizer, replicated)

    writer = train_utils.setup_logging(train_config, model_config, num_devices)

    # Data generator
    data_generator = dummy_seq2seq_generator(model_config, train_config, global_batch_size, rngs)

    print("Starting training...")
    print(f"{'Step':<6} | {'Loss':<12} | {'LR':<10}")
    print("-" * 40)

    for step in range(train_config.total_steps):
        batch = next(data_generator)
        loss, model, optimizer = train_step(model, optimizer, batch, data_sharding)
        current_lr = scheduler(optimizer.step.value)

        writer.add_scalar("loss/total", float(loss), step)
        writer.add_scalar("lr", float(current_lr), step)

        if step % train_config.log_interval == 0:
            print(f"{step:<6} | {float(loss):.4f}       | {current_lr:.6f}")

        ckpt.maybe_save(model, ckpt_path, step, train_config.save_interval)

    # Save final checkpoint
    if ckpt_path:
        ckpt.save(model, ckpt_path, train_config.total_steps)
        print(f"Saved final checkpoint at step {train_config.total_steps}")

    writer.close()
    print("\nTraining complete.")
