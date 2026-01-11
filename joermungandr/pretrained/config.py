from dataclasses import dataclass


@dataclass
class RerankerConfig:
    """Configuration for the Qwen3 Reranker model."""

    model_id: str = "Qwen/Qwen3-Reranker-4B"
    max_length: int = 8192
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = "flash_attention_2"

    # # LoRA configuration for future use?
    # use_lora: bool = False
    # lora_r: int = 16
    # lora_alpha: int = 32
    # lora_dropout: float = 0.05
    # lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class RerankerTrainConfig:
    """Configuration for reranker fine-tuning."""

    # Optimization
    lr: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Paths
    output_dir: str = "checkpoints/reranker"
    logging_dir: str = "runs/reranker"

    # Data
    train_data_path: str | None = None
    eval_data_path: str | None = None

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 100

    # Misc
    seed: int = 42
