from datetime import datetime
from pathlib import Path

import jax
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tensorboardX import SummaryWriter

from .checkpoint import save_checkpoint
from .config import ModelConfig, TrainConfig


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


def create_optimizer(
    model: nnx.Module, train_config: TrainConfig
) -> tuple[nnx.Optimizer, optax.Schedule]:
    """Create optimizer with gradient clipping and AdamW."""
    scheduler = create_scheduler(train_config)
    tx = optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adamw(learning_rate=scheduler, weight_decay=train_config.weight_decay),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer, scheduler


def setup_device_mesh() -> tuple[int, NamedSharding, NamedSharding]:
    """Set up device mesh for data parallelism.

    Returns:
        (num_devices, replicated_sharding, data_sharding)
    """
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(devices, axis_names=("data",))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data"))
    return num_devices, replicated, data_sharding


def replicate_on_devices(model: nnx.Module, optimizer: nnx.Optimizer, sharding: NamedSharding):
    """Replicate model and optimizer state across devices."""
    nnx.update(model, jax.device_put(nnx.state(model), sharding))
    nnx.update(optimizer, jax.device_put(nnx.state(optimizer), sharding))


def setup_logging(
    train_config: TrainConfig, model_config: ModelConfig, num_devices: int
) -> SummaryWriter:
    """Set up TensorBoard logging."""
    log_dir = train_config.log_dir or f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    writer.add_text("model_config", str(model_config))
    writer.add_text("train_config", str(train_config))
    writer.add_text("devices", f"{num_devices} device(s)")
    print(f"Logging to {log_dir}")
    return writer


def setup_checkpoint_dir(train_config: TrainConfig) -> Path | None:
    """Set up checkpoint directory."""
    ckpt_path = train_config.checkpoint_path
    if ckpt_path:
        ckpt_path.mkdir(parents=True, exist_ok=True)
    return ckpt_path


def maybe_save_checkpoint(model: nnx.Module, ckpt_path: Path | None, step: int, save_interval: int):
    """Save checkpoint if at save interval."""
    if ckpt_path and (step + 1) % save_interval == 0:
        save_checkpoint(model, ckpt_path, step + 1)
        print(f"Saved checkpoint at step {step + 1}")
