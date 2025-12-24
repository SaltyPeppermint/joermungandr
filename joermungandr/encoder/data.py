from typing import Generator

import flax.struct
import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from ..config import ModelConfig, TrainConfig


@flax.struct.dataclass
class EncoderBatch:
    """A batch of data for training.

    Attributes:
        input_ids: [B, L]
        seg_ids: [B, L]
        padding_mask: [B, L] boolean mask for attention (True for real, False for padding).
        mlm_targets: [B, L] target token IDs for MLM.
        mlm_mask: [B, L] boolean mask for MLM positions.
        nsp_labels: [B] binary labels for NSP.
    """

    input_ids: Array
    seg_ids: Array
    padding_mask: Array
    mlm_targets: Array
    mlm_mask: Array
    nsp_labels: Array


def dummy_encoder_generator(
    model_config: ModelConfig,
    train_config: TrainConfig,
    global_batch_size: int,
    rngs: nnx.Rngs,
) -> Generator[EncoderBatch]:
    """Yield batches of dummy data for encoder training with simulated padding.

    Generates sequences in BERT format: [CLS] segment_A [SEP] segment_B [SEP] [PAD]...
    """
    # Special token IDs
    pad_id = 0
    cls_id = 2
    sep_id = 3
    first_content_id = 5  # First non-special token ID

    seq_len = train_config.seq_len

    while True:
        # Determine segment lengths for each example
        # We need: [CLS] + seg_A + [SEP] + seg_B + [SEP] <= seq_len
        # Minimum 1 token per segment, so min total = 5 tokens
        max_content_len = seq_len - 3  # Reserve space for [CLS], [SEP], [SEP]
        seg_a_lens = jax.random.randint(rngs.step(), (global_batch_size,), 1, max_content_len)
        remaining = max_content_len - seg_a_lens
        seg_b_lens = jax.random.randint(
            rngs.step(), (global_batch_size,), 1, jnp.maximum(remaining, 1) + 1
        )

        total_lens = 1 + seg_a_lens + 1 + seg_b_lens + 1

        content_tokens = jax.random.randint(
            rngs.step(), (global_batch_size, seq_len), first_content_id, model_config.vocab_size
        )

        # Build input_ids: [CLS] seg_A [SEP] seg_B [SEP] [PAD]...
        positions = jnp.arange(seq_len)
        seg_a_end = 1 + seg_a_lens  # Position after seg_A (where first [SEP] goes)
        seg_b_end = seg_a_end + 1 + seg_b_lens  # Position after seg_B (where second [SEP] goes)

        # Construct token IDs
        is_cls = positions == 0
        is_first_sep = positions == seg_a_end[:, None]
        is_second_sep = positions == seg_b_end[:, None]
        is_pad = positions >= total_lens[:, None]

        input_ids = jnp.where(is_cls, cls_id, content_tokens)
        input_ids = jnp.where(is_first_sep, sep_id, input_ids)
        input_ids = jnp.where(is_second_sep, sep_id, input_ids)
        input_ids = jnp.where(is_pad, pad_id, input_ids)

        # Padding mask: True for real tokens, False for padding
        padding_mask = ~is_pad

        # Segment IDs: 0 for [CLS] + seg_A + [SEP], 1 for seg_B + [SEP], 0 for padding
        seg_ids = jnp.where(positions <= seg_a_end[:, None], 0, 1)
        seg_ids = jnp.where(is_pad, 0, seg_ids)

        # MLM mask: randomly mask content tokens (not special tokens)
        is_special = is_cls | is_first_sep | is_second_sep | is_pad
        mlm_candidates = ~is_special
        mlm_random = jax.random.uniform(rngs.step(), (global_batch_size, seq_len))
        mlm_mask = mlm_candidates & (mlm_random < train_config.mlm_mask_prob)

        # MLM targets (what the masked tokens should be)
        mlm_targets = jax.random.randint(
            rngs.step(), (global_batch_size, seq_len), first_content_id, model_config.vocab_size
        )

        nsp_labels = jax.random.randint(rngs.step(), (global_batch_size,), 0, 2)

        yield EncoderBatch(
            input_ids=input_ids,
            seg_ids=seg_ids,
            padding_mask=padding_mask,
            mlm_targets=mlm_targets,
            mlm_mask=mlm_mask,
            nsp_labels=nsp_labels,
        )
