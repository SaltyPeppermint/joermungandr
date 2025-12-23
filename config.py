from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the transformer encoder."""

    vocab_size: int = 32000
    dim: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    num_layers: int = 6
    num_segments: int = 3
    intermediate_dim: int = 2048
    max_len: int = 512


@dataclass
class TrainConfig:
    """Configuration for training hyperparameters and paths."""

    # Optimization
    lr: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training
    total_steps: int = 100
    batch_size_per_device: int = 8
    seq_len: int = 32
    seed: int = 42

    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    tokenizer_path: str = "tok.save"
    log_dir: str | None = None
    checkpoint_dir: str | None = None

    @property
    def checkpoint_path(self) -> Path | None:
        return Path(self.checkpoint_dir) if self.checkpoint_dir else None
