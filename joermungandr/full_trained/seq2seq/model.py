import jax.numpy as jnp
from flax import nnx
from jax import Array

from ..config import ModelConfig
from ..layers import DecoderBlock, EncoderBlock


class Seq2Seq(nnx.Module):
    """seq2seq transformer for sequence-to-sequence tasks.

    The encoder takes two sentences separated by a SEP token and produces
    contextualized representations. The decoder autoregressively generates
    the output sequence.
    """

    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        module_kwargs = {"rngs": rngs, "dtype": dtype, "param_dtype": param_dtype}
        linear_kwargs = {**module_kwargs, "use_bias": False}

        # Shared embeddings for encoder and decoder
        self.tok_emb = nnx.Embed(config.vocab_size, config.dim, **module_kwargs)
        self.seg_emb = nnx.Embed(config.num_segments, config.dim, **module_kwargs)

        # Encoder
        self.encoder_blocks = nnx.List(
            [EncoderBlock(config, **module_kwargs) for _ in range(config.num_layers)]
        )
        self.encoder_norm = nnx.RMSNorm(config.dim, **module_kwargs)

        # Decoder
        self.decoder_blocks = nnx.List(
            [DecoderBlock(config, **module_kwargs) for _ in range(config.num_layers)]
        )
        self.decoder_norm = nnx.RMSNorm(config.dim, **module_kwargs)

        # Output projection (tied with embeddings via transpose)
        self.lm_head = nnx.Linear(config.dim, config.vocab_size, **linear_kwargs)

    def encode(
        self,
        input_ids: Array,
        seg_ids: Array,
        *,
        mask: Array | None = None,
    ) -> Array:
        """
        Encode the input sequence (two sentences separated by SEP).

        Args:
            input_ids: Token IDs of shape [B, S].
            seg_ids: Segment IDs of shape [B, S] (0 for first sentence, 1 for second).
            mask: Optional padding mask of shape [B, S].

        Returns:
            Encoder output of shape [B, S, D].
        """
        x = self.tok_emb(input_ids) + self.seg_emb(seg_ids)

        for block in self.encoder_blocks:
            x = block(x, mask)

        return self.encoder_norm(x)

    def decode(
        self,
        decoder_ids: Array,
        encoder_out: Array,
        *,
        decoder_mask: Array | None = None,
        encoder_mask: Array | None = None,
    ) -> Array:
        """
        Decode autoregressively given encoder output.

        Args:
            decoder_ids: Decoder input token IDs of shape [B, T].
            encoder_out: Encoder output of shape [B, S, D].
            decoder_mask: Optional decoder padding mask of shape [B, T].
            encoder_mask: Optional encoder padding mask of shape [B, S].

        Returns:
            Logits of shape [B, T, vocab_size].
        """
        x = self.tok_emb(decoder_ids)

        for block in self.decoder_blocks:
            x = block(x, encoder_out, decoder_mask=decoder_mask, encoder_mask=encoder_mask)

        x = self.decoder_norm(x)
        return self.lm_head(x)

    def __call__(
        self,
        encoder_ids: Array,
        encoder_seg_ids: Array,
        decoder_ids: Array,
        *,
        encoder_mask: Array | None = None,
        decoder_mask: Array | None = None,
    ) -> Array:
        """
        Full forward pass for training.

        Args:
            encoder_ids: Encoder input tokens [B, S] (sent1 + SEP + sent2).
            encoder_seg_ids: Segment IDs [B, S] (0 for sent1, 1 for sent2).
            decoder_ids: Decoder input tokens [B, T] (target sequence with BOS).
            encoder_mask: Optional encoder padding mask [B, S].
            decoder_mask: Optional decoder padding mask [B, T].

        Returns:
            Logits of shape [B, T, vocab_size].
        """
        encoder_out = self.encode(encoder_ids, encoder_seg_ids, mask=encoder_mask)
        return self.decode(
            decoder_ids,
            encoder_out,
            decoder_mask=decoder_mask,
            encoder_mask=encoder_mask,
        )
