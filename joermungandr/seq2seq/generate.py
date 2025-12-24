import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .model import Seq2Seq


def generate(
    model: Seq2Seq,
    encoder_ids: Array,
    encoder_seg_ids: Array,
    *,
    encoder_mask: Array | None = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0,
    rngs: nnx.Rngs | None = None,
) -> Array:
    """Generate sequences autoregressively from encoder inputs.

    Args:
        model: The Seq2Seq model to use for generation.
        encoder_ids: Encoder input token IDs of shape [B, S].
        encoder_seg_ids: Segment IDs for encoder of shape [B, S].
        encoder_mask: Optional encoder padding mask of shape [B, S].
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: If set, only sample from top-k tokens.
        top_p: If set, use nucleus sampling with cumulative probability p.
        bos_id: Beginning-of-sequence token ID.
        eos_id: End-of-sequence token ID.
        pad_id: Padding token ID.
        rngs: Random number generator for sampling. Required if temperature > 0.

    Returns:
        Generated token IDs of shape [B, max_new_tokens].
    """
    batch_size = encoder_ids.shape[0]

    # Encode the input sequence once
    encoder_out = model.encode(encoder_ids, encoder_seg_ids, mask=encoder_mask)

    # Initialize decoder input with BOS token
    decoder_ids = jnp.full((batch_size, 1), bos_id, dtype=jnp.int32)

    # Track which sequences have finished (encountered EOS)
    finished = jnp.zeros(batch_size, dtype=jnp.bool_)

    for _ in range(max_new_tokens):
        # Get logits for the current sequence
        logits = model.decode(
            decoder_ids,
            encoder_out,
            encoder_mask=encoder_mask,
        )

        # Get logits for the last token
        next_token_logits = logits[:, -1, :]  # [B, vocab_size]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Sample or take argmax
        if temperature == 0.0:
            # Greedy decoding
            next_token = jnp.argmax(next_token_logits, axis=-1)
        else:
            # Sampling
            if rngs is None:
                raise ValueError("rngs must be provided when temperature > 0")

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
                # Set all non-top-k logits to -inf
                next_token_logits = jnp.full_like(next_token_logits, -jnp.inf)
                next_token_logits = next_token_logits.at[
                    jnp.arange(batch_size)[:, None], top_k_indices
                ].set(top_k_logits)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_indices = jnp.argsort(next_token_logits, axis=-1)[:, ::-1]
                sorted_logits = jnp.take_along_axis(next_token_logits, sorted_indices, axis=-1)
                sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least one token
                sorted_indices_to_remove = jnp.concatenate(
                    [
                        jnp.zeros((batch_size, 1), dtype=jnp.bool_),
                        sorted_indices_to_remove[:, :-1],
                    ],
                    axis=-1,
                )

                # Scatter back to original indexing
                indices_to_remove = jnp.zeros_like(next_token_logits, dtype=jnp.bool_)
                indices_to_remove = indices_to_remove.at[
                    jnp.arange(batch_size)[:, None], sorted_indices
                ].set(sorted_indices_to_remove)

                next_token_logits = jnp.where(indices_to_remove, -jnp.inf, next_token_logits)

            # Sample from the filtered distribution
            next_token = jax.random.categorical(rngs.step(), next_token_logits, axis=-1)

        # For finished sequences, use pad token
        next_token = jnp.where(finished, pad_id, next_token)

        # Append to decoder_ids
        decoder_ids = jnp.concatenate([decoder_ids, next_token[:, None]], axis=1)

        # Update finished status
        finished = finished | (next_token == eos_id)

        # Early stopping if all sequences are finished
        if jnp.all(finished):
            break

    # Return only the generated tokens (exclude BOS)
    return decoder_ids[:, 1:]


def generate_batch(
    model: Seq2Seq,
    encoder_ids: Array,
    encoder_seg_ids: Array,
    *,
    encoder_mask: Array | None = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    num_return_sequences: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0,
    rngs: nnx.Rngs | None = None,
) -> Array:
    """Generate multiple sequences per input using sampling.

    This function replicates each input num_return_sequences times and generates
    different sequences for each replica.

    Args:
        model: The Seq2Seq model to use for generation.
        encoder_ids: Encoder input token IDs of shape [B, S].
        encoder_seg_ids: Segment IDs for encoder of shape [B, S].
        encoder_mask: Optional encoder padding mask of shape [B, S].
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: If set, only sample from top-k tokens.
        top_p: If set, use nucleus sampling with cumulative probability p.
        num_return_sequences: Number of sequences to generate per input.
        bos_id: Beginning-of-sequence token ID.
        eos_id: End-of-sequence token ID.
        pad_id: Padding token ID.
        rngs: Random number generator for sampling. Required if temperature > 0.

    Returns:
        Generated token IDs of shape [B * num_return_sequences, max_new_tokens].
    """
    if num_return_sequences == 1:
        return generate(
            model,
            encoder_ids,
            encoder_seg_ids,
            encoder_mask=encoder_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            rngs=rngs,
        )

    # Replicate inputs
    encoder_ids = jnp.repeat(encoder_ids, num_return_sequences, axis=0)
    encoder_seg_ids = jnp.repeat(encoder_seg_ids, num_return_sequences, axis=0)
    if encoder_mask is not None:
        encoder_mask = jnp.repeat(encoder_mask, num_return_sequences, axis=0)

    return generate(
        model,
        encoder_ids,
        encoder_seg_ids,
        encoder_mask=encoder_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        rngs=rngs,
    )
