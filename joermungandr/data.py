from typing import Generator

import flax.struct
import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .config import ModelConfig, TrainConfig


@flax.struct.dataclass
class Seq2SeqBatch:
    """A batch of data for seq2seq training.

    Attributes:
        encoder_ids: [B, S] encoder input token IDs.
        encoder_seg_ids: [B, S] segment IDs for encoder.
        encoder_mask: [B, S] boolean mask for encoder attention.
        decoder_ids: [B, T] decoder input token IDs (shifted right).
        decoder_mask: [B, T] boolean mask for decoder attention.
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


@flax.struct.dataclass
class EncoderBatch:
    """A batch of data for training.

    Attributes:
        input_ids: [B, L] token IDs.
        seg_ids: [B, L] segment IDs.
        attention_mask: [B, L] boolean mask for attention (True for real, False for padding).
        mlm_targets: [B, L] target token IDs for MLM.
        mlm_mask: [B, L] boolean mask for MLM positions.
        nsp_labels: [B] binary labels for NSP.
    """

    input_ids: Array
    seg_ids: Array
    attention_mask: Array
    mlm_targets: Array
    mlm_mask: Array
    nsp_labels: Array


def dummy_encoder_generator(
    model_config: ModelConfig,
    train_config: TrainConfig,
    global_batch_size: int,
    rngs: nnx.Rngs,
) -> Generator[EncoderBatch]:
    """Yield batches of dummy data for encoder training with simulated padding."""
    pad_id = 0  # Assuming [PAD] token is ID 0

    while True:
        # Generate base data
        # Start from 1 to avoid generating pad_id tokens as real data
        inp = jax.random.randint(
            rngs.step(), (global_batch_size, train_config.seq_len), 1, model_config.vocab_size
        )
        seg = jax.random.randint(
            rngs.step(), (global_batch_size, train_config.seq_len), 0, model_config.num_segments
        )
        mlm_mask = (
            jax.random.uniform(rngs.step(), (global_batch_size, train_config.seq_len))
            < train_config.mlm_mask_prob
        )

        mlm_targets = jax.random.randint(
            rngs.step(), (global_batch_size, train_config.seq_len), 0, model_config.vocab_size
        )
        nsp_labels = jax.random.randint(rngs.step(), (global_batch_size,), 0, 2)

        # Simulate variable sequence lengths and create attention mask
        # Sequences can be between 50% and 100% of max length
        seq_lens = jax.random.randint(
            rngs.step(), (global_batch_size,), train_config.seq_len // 2, train_config.seq_len + 1
        )
        attention_mask = jnp.arange(train_config.seq_len) < seq_lens[:, None]

        # Apply padding and update masks
        # Set padded tokens in input to pad_id
        inp = jnp.where(attention_mask, inp, pad_id)

        # Ensure MLM mask does not include padded tokens
        mlm_mask = mlm_mask & attention_mask

        yield EncoderBatch(
            input_ids=inp,
            seg_ids=seg,
            attention_mask=attention_mask,
            mlm_targets=mlm_targets,
            mlm_mask=mlm_mask,
            nsp_labels=nsp_labels,
        )


def dummy_seq2seq_generator(
    model_config: ModelConfig,
    train_config: TrainConfig,
    global_batch_size: int,
    rngs: nnx.Rngs,
) -> Generator[Seq2SeqBatch]:
    """Yield batches of dummy data for seq2seq training with simulated padding."""
    pad_id = 0
    bos_id = 1

    while True:
        # Encoder inputs (two segments separated by SEP)
        encoder_ids = jax.random.randint(
            rngs.step(), (global_batch_size, train_config.seq_len), 2, model_config.vocab_size
        )
        # Segment IDs: first half segment 0, second half segment 1
        half_len = train_config.seq_len // 2
        encoder_seg_ids = jnp.concatenate(
            [
                jnp.zeros((global_batch_size, half_len), dtype=jnp.int32),
                jnp.ones((global_batch_size, train_config.seq_len - half_len), dtype=jnp.int32),
            ],
            axis=1,
        )

        # Variable encoder lengths
        encoder_lens = jax.random.randint(
            rngs.step(), (global_batch_size,), train_config.seq_len // 2, train_config.seq_len + 1
        )
        encoder_mask = jnp.arange(train_config.seq_len) < encoder_lens[:, None]
        encoder_ids = jnp.where(encoder_mask, encoder_ids, pad_id)

        # Decoder inputs and targets
        decoder_len = train_config.seq_len
        # Generate target sequence (what decoder should predict)
        target_seq = jax.random.randint(
            rngs.step(), (global_batch_size, decoder_len), 2, model_config.vocab_size
        )
        # Decoder input is BOS + target[:-1] (teacher forcing)
        decoder_ids = jnp.concatenate(
            [jnp.full((global_batch_size, 1), bos_id), target_seq[:, :-1]], axis=1
        )
        # Labels are the target sequence
        labels = target_seq

        # Variable decoder lengths
        decoder_lens = jax.random.randint(
            rngs.step(), (global_batch_size,), decoder_len // 2, decoder_len + 1
        )
        decoder_mask = jnp.arange(decoder_len) < decoder_lens[:, None]
        decoder_ids = jnp.where(decoder_mask, decoder_ids, pad_id)
        label_mask = decoder_mask

        yield Seq2SeqBatch(
            encoder_ids=encoder_ids,
            encoder_seg_ids=encoder_seg_ids,
            encoder_mask=encoder_mask,
            decoder_ids=decoder_ids,
            decoder_mask=decoder_mask,
            labels=labels,
            label_mask=label_mask,
        )
