import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from jax.sharding import NamedSharding

from .. import checkpoint as ckpt
from .. import train_utils
from ..config import ModelConfig, TrainConfig
from .data import EncoderBatch, dummy_encoder_generator
from .model import Encoder


@nnx.jit(donate_argnames=("model", "optimizer"), static_argnames=("data_sharding",))
def train_step(
    model: Encoder, optimizer: nnx.Optimizer, batch: EncoderBatch, data_sharding: NamedSharding
) -> tuple[Array, Array, Array, Encoder, nnx.Optimizer]:
    """Execute a single training step with data parallelism.

    Returns:
        (total_loss, mlm_loss, nsp_loss, model, optimizer)
    """
    batch = jax.device_put(batch, data_sharding)

    def loss_fn(model: Encoder) -> tuple[Array, tuple[Array, Array]]:
        mlm_logits, nsp_logits = model(batch.input_ids, batch.seg_ids, mask=batch.padding_mask)

        mlm_losses = optax.softmax_cross_entropy_with_integer_labels(mlm_logits, batch.mlm_targets)
        mlm_loss = jnp.sum(mlm_losses * batch.mlm_mask) / (jnp.sum(batch.mlm_mask) + 1e-9)

        nsp_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(nsp_logits, batch.nsp_labels)
        )

        return mlm_loss + nsp_loss, (mlm_loss, nsp_loss)

    (loss, (mlm_loss, nsp_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, mlm_loss, nsp_loss, model, optimizer


def train_loop(model_config: ModelConfig, train_config: TrainConfig) -> None:
    """Run the encoder training loop with TensorBoard logging and multi-GPU data parallelism."""

    ckpt_path = ckpt.setup_dir(train_config)

    num_devices, replicated, data_sharding = train_utils.setup_device_mesh()
    global_batch_size = train_config.batch_size_per_device * num_devices

    print(f"Running on {num_devices} device(s), global batch size: {global_batch_size}")

    rngs = nnx.Rngs(train_config.seed)
    model = Encoder(model_config, rngs=rngs)
    optimizer, scheduler = train_utils.create_optimizer(model, train_config)

    train_utils.replicate_on_devices(model, optimizer, replicated)

    writer = train_utils.setup_logging(train_config, model_config, num_devices)

    data_generator = dummy_encoder_generator(model_config, train_config, global_batch_size, rngs)

    print("Starting training...")
    print(f"{'Step':<6} | {'Total Loss':<12} | {'MLM Loss':<10} | {'NSP Loss':<10} | {'LR':<10}")
    print("-" * 60)

    for step in range(train_config.total_steps):
        batch = next(data_generator)
        loss, mlm, nsp, model, optimizer = train_step(model, optimizer, batch, data_sharding)
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

        ckpt.maybe_save(model, ckpt_path, step, train_config.save_interval)

    if ckpt_path:
        ckpt.save(model, ckpt_path, train_config.total_steps)
        print(f"Saved final checkpoint at step {train_config.total_steps}")

    writer.close()
    print("\nTraining complete.")
