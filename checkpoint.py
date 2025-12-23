from pathlib import Path

import orbax.checkpoint as ocp
import sentencepiece as spm
from flax import nnx

from model import Encoder


def load_tokenizer(path: str) -> spm.SentencePieceProcessor:
    """Load a SentencePiece tokenizer from disk."""
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def save_checkpoint(model: Encoder, path: Path, step: int) -> None:
    """Save model checkpoint to disk."""
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path / f"step_{step}", nnx.state(model))


def load_checkpoint(model: Encoder, path: Path) -> None:
    """Load model checkpoint from disk."""
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(path, nnx.state(model))
    nnx.update(model, state)
