from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .config import RerankerConfig


def download_model(
    model_id: str = "Qwen/Qwen3-Reranker-4B",
    cache_dir: str | None = None,
    force: bool = False,
) -> Path:
    """Download model and tokenizer from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Optional custom cache directory.
        force: If True, re-download even if already cached.

    Returns:
        Path to the downloaded model directory.
    """
    path: str = snapshot_download(model_id, cache_dir=cache_dir, force_download=force)
    return Path(path)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Must be one of {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def load_tokenizer(model_id_or_path: str) -> PreTrainedTokenizer:
    """Load tokenizer with correct padding configuration for reranking.

    Args:
        model_id_or_path: HuggingFace model ID or local path.

    Returns:
        Configured AutoTokenizer.
    """
    # AutoTokenizer.from_pretrained has incomplete type stubs
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id_or_path: str, config: RerankerConfig) -> PreTrainedModel:
    """Load model with optional LoRA for fine-tuning.

    Args:
        model_id_or_path: HuggingFace model ID or local path.
        config: Reranker configuration.
        for_training: If True, prepare model for training with LoRA.

    Returns:
        Loaded model, optionally wrapped with LoRA.
    """
    torch_dtype = get_torch_dtype(config.torch_dtype)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    return AutoModelForCausalLM.from_pretrained(model_id_or_path, **model_kwargs)
