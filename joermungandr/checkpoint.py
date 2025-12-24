from pathlib import Path

import orbax.checkpoint as ocp
import sentencepiece as spm
from flax import nnx

from .config import TrainConfig


def load_tokenizer(path: str) -> spm.SentencePieceProcessor:
    """Load a SentencePiece tokenizer from disk."""
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def setup_dir(train_config: TrainConfig) -> Path | None:
    """Set up checkpoint directory."""
    ckpt_path = train_config.checkpoint_path
    if ckpt_path:
        ckpt_path.mkdir(parents=True, exist_ok=True)
    return ckpt_path


def maybe_save(model: nnx.Module, ckpt_path: Path | None, step: int, save_interval: int):
    """Save checkpoint if at save interval."""
    if ckpt_path and (step + 1) % save_interval == 0:
        save(model, ckpt_path, step + 1)
        print(f"Saved checkpoint at step {step + 1}")


def save(model: nnx.Module, path: Path, step: int) -> None:
    """Save model checkpoint to disk."""
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path / f"step_{step}", nnx.state(model))


def load(model: nnx.Module, path: Path) -> None:
    """Load model checkpoint from disk."""
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(path, nnx.state(model))
    nnx.update(model, state)
