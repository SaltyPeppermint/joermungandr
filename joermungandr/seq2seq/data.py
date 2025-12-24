from typing import Generator

import flax.struct
import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from ..config import ModelConfig, TrainConfig


@flax.struct.dataclass
class Seq2SeqBatch:
    """A batch of data for seq2seq training.

    Attributes:
        encoder_ids: [B, S]
        encoder_seg_ids: [B, S]
        encoder_mask: [B, S]
        decoder_ids: [B, T]
        decoder_mask: [B, T]
        labels: [B, T] target token IDs for next-token prediction.
        label_mask: [B, T] boolean mask for loss computation.
    """

    encoder_ids: Array
    encoder_seg_ids: Array
    encoder_mask: Array
    decoder_ids: Array
    decoder_mask: Array
    labels: Array
    label_mask: Array


def dummy_seq2seq_generator(
    model_config: ModelConfig,
    train_config: TrainConfig,
    global_batch_size: int,
    rngs: nnx.Rngs,
) -> Generator[Seq2SeqBatch]:
    """Yield batches of dummy data for seq2seq training with simulated padding.

    Encoder format: [CLS] segment_A [SEP] segment_B [SEP] [PAD]...
    Decoder input:  [CLS] token_1 token_2 ... token_n [PAD]...
    Labels:         token_1 token_2 ... token_n [SEP] [PAD]...
    """
    # Special token IDs
    pad_id = 0
    cls_id = 2  # Also used as decoder BOS. Corresponds to BOS in tokenizer
    sep_id = 3  # Also used as decoder EOS. Corresponds to EOS in tokenizer
    first_content_id = 5  # First non-special token ID

    seq_len = train_config.seq_len

    while True:
        # === Encoder inputs ===
        # Structure: [CLS] segment_A [SEP] segment_B [SEP] [PAD]...
        max_content_len = seq_len - 3  # Reserve space for [CLS], [SEP], [SEP]
        seg_a_lens = jax.random.randint(rngs.step(), (global_batch_size,), 1, max_content_len)
        remaining = max_content_len - seg_a_lens
        seg_b_lens = jax.random.randint(
            rngs.step(), (global_batch_size,), 1, jnp.maximum(remaining, 1) + 1
        )

        encoder_total_lens = 1 + seg_a_lens + 1 + seg_b_lens + 1

        encoder_content = jax.random.randint(
            rngs.step(), (global_batch_size, seq_len), first_content_id, model_config.vocab_size
        )

        # Build encoder_ids
        positions = jnp.arange(seq_len)
        seg_a_end = 1 + seg_a_lens
        seg_b_end = seg_a_end + 1 + seg_b_lens

        is_cls = positions == 0
        is_first_sep = positions == seg_a_end[:, None]
        is_second_sep = positions == seg_b_end[:, None]
        is_encoder_pad = positions >= encoder_total_lens[:, None]

        encoder_ids = jnp.where(is_cls, cls_id, encoder_content)
        encoder_ids = jnp.where(is_first_sep, sep_id, encoder_ids)
        encoder_ids = jnp.where(is_second_sep, sep_id, encoder_ids)
        encoder_ids = jnp.where(is_encoder_pad, pad_id, encoder_ids)

        # Segment IDs: 0 for [CLS] + seg_A + [SEP], 1 for seg_B + [SEP], 0 for padding
        encoder_seg_ids = jnp.where(positions <= seg_a_end[:, None], 0, 1)
        encoder_seg_ids = jnp.where(is_encoder_pad, 0, encoder_seg_ids)

        encoder_mask = ~is_encoder_pad

        # === Decoder inputs and labels ===
        # Decoder input: [CLS] content_tokens [PAD]...
        # Labels: content_tokens [SEP] [PAD]...
        # Variable decoder content lengths (excluding BOS/EOS)
        max_decoder_content = seq_len - 1  # Reserve 1 for [CLS] in input or [SEP] in labels
        decoder_content_lens = jax.random.randint(
            rngs.step(), (global_batch_size,), 1, max_decoder_content + 1
        )

        decoder_content = jax.random.randint(
            rngs.step(), (global_batch_size, seq_len), first_content_id, model_config.vocab_size
        )

        # Decoder input: [CLS] + content[:-1], length = decoder_content_lens
        # We shift right: position 0 is [CLS], positions 1..n are content[0..n-1]
        decoder_ids = jnp.concatenate(
            [jnp.full((global_batch_size, 1), cls_id), decoder_content[:, :-1]], axis=1
        )
        # Mask out padding positions
        decoder_total_lens = 1 + decoder_content_lens  # [CLS] + content
        is_decoder_pad = positions >= decoder_total_lens[:, None]
        decoder_ids = jnp.where(is_decoder_pad, pad_id, decoder_ids)
        decoder_mask = ~is_decoder_pad

        # Labels: content + [SEP], then padding
        # Position i in labels corresponds to what decoder should predict at position i
        labels = decoder_content.at[jnp.arange(global_batch_size), decoder_content_lens].set(sep_id)
        label_total_lens = decoder_content_lens + 1  # content + [SEP]
        is_label_pad = positions >= label_total_lens[:, None]
        labels = jnp.where(is_label_pad, pad_id, labels)
        label_mask = ~is_label_pad

        yield Seq2SeqBatch(
            encoder_ids=encoder_ids,
            encoder_seg_ids=encoder_seg_ids,
            encoder_mask=encoder_mask,
            decoder_ids=decoder_ids,
            decoder_mask=decoder_mask,
            labels=labels,
            label_mask=label_mask,
        )
