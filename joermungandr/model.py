import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .config import ModelConfig


class SwiGLU(nnx.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        linear_kwargs = {
            "use_bias": False,
            "rngs": rngs,
            "dtype": dtype,
            "param_dtype": param_dtype,
        }
        self.w1 = nnx.Linear(config.dim, config.intermediate_dim, **linear_kwargs)
        self.w2 = nnx.Linear(config.dim, config.intermediate_dim, **linear_kwargs)
        self.w3 = nnx.Linear(config.intermediate_dim, config.dim, **linear_kwargs)

    def __call__(self, x: Array) -> Array:
        return self.w3(nnx.silu(self.w1(x)) * self.w2(x))


def precompute_freqs_cis(dim: int, end: int, theta: float) -> Array:
    """
    Precompute complex exponential frequencies for rotary position embeddings.

    Args:
        dim: Head dimension (D).
        end: Maximum sequence length.
        theta: Base frequency for RoPE.

    Returns:
        Complex array of shape [end, D//2] containing e^(i * freq * pos).
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)


def apply_rope(x: Array, freqs_cis: Array) -> Array:
    """
    Apply rotary position embeddings.

    Args:
        x: Input tensor of shape [B, L, H, D].
        freqs_cis: Precomputed frequencies of shape [max_len, D//2].

    Returns:
        Tensor of shape [B, L, H, D] with rotary embeddings applied.
    """
    B, L, H, D = x.shape
    # Cast freqs to the input dtype for mixed-precision training.
    freqs_cos = jnp.real(freqs_cis).astype(x.dtype)
    freqs_sin = jnp.imag(freqs_cis).astype(x.dtype)

    # Broadcast freqs: [L, D//2] -> [1, L, 1, D//2]
    freqs_cos = freqs_cos[None, :L, None, :]
    freqs_sin = freqs_sin[None, :L, None, :]

    # Split input into real and imaginary parts.
    x_real = x[..., 0::2]
    x_imag = x[..., 1::2]

    # Apply rotation.
    x_out_real = x_real * freqs_cos - x_imag * freqs_sin
    x_out_imag = x_real * freqs_sin + x_imag * freqs_cos

    # Reconstruct the output tensor by interleaving the parts.
    return jnp.stack([x_out_real, x_out_imag], axis=-1).reshape(B, L, H, D)


class GQA(nnx.Module):
    """Grouped-query attention with rotary position (RoPE) embeddings.

    Supports both self-attention and cross-attention modes.
    """

    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs,
        *,
        is_cross_attention: bool = False,
        is_causal: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.num_heads = config.num_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.dim // config.num_heads
        self.is_cross_attention = is_cross_attention
        self.is_causal = is_causal

        linear_kwargs = {
            "use_bias": False,
            "rngs": rngs,
            "dtype": dtype,
            "param_dtype": param_dtype,
        }

        self.q_proj = nnx.Linear(config.dim, config.num_heads * self.head_dim, **linear_kwargs)
        self.k_proj = nnx.Linear(config.dim, config.num_kv_heads * self.head_dim, **linear_kwargs)
        self.v_proj = nnx.Linear(config.dim, config.num_kv_heads * self.head_dim, **linear_kwargs)
        self.o_proj = nnx.Linear(config.num_heads * self.head_dim, config.dim, **linear_kwargs)

        self.freqs_cis = nnx.Cache(
            precompute_freqs_cis(self.head_dim, config.max_len, config.rope_theta)
        )

    def __call__(
        self,
        x: Array,
        *,
        kv: Array | None = None,
        mask: Array | None = None,
    ) -> Array:
        """
        Apply grouped-query attention.

        Args:
            x: Input tensor of shape [B, L, D] (queries).
            kv: Optional key-value tensor of shape [B, S, D] for cross-attention.
                If None, self-attention is performed.
            mask: Optional padding mask of shape [B, S] for key/value positions.

        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, _ = x.shape
        kv_input = kv if kv is not None else x
        S = kv_input.shape[1]

        # Project and reshape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(kv_input).reshape(B, S, self.num_kv, self.head_dim)
        v = self.v_proj(kv_input).reshape(B, S, self.num_kv, self.head_dim)

        # Apply RoPE (only for self-attention, not cross-attention)
        if not self.is_cross_attention:
            q = apply_rope(q, self.freqs_cis.value)
            k = apply_rope(k, self.freqs_cis.value)

        # Build attention mask [B, 1, L, S]
        attn_mask = None
        if self.is_causal:
            attn_mask = jnp.tril(jnp.ones((L, S), dtype=jnp.bool_))
        if mask is not None:
            pad_mask = mask[:, None, None, :]  # [B, 1, 1, S]
            attn_mask = pad_mask if attn_mask is None else (attn_mask & pad_mask)

        out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)

        return self.o_proj(out.reshape(B, L, -1))


class EncoderBlock(nnx.Module):
    """Pre-norm transformer block with GQA attention and SwiGLU MLP."""

    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        module_kwargs = {"rngs": rngs, "dtype": dtype, "param_dtype": param_dtype}
        self.norm1 = nnx.RMSNorm(config.dim, **module_kwargs)
        self.attn = GQA(config, **module_kwargs)
        self.norm2 = nnx.RMSNorm(config.dim, **module_kwargs)
        self.mlp = SwiGLU(config, **module_kwargs)

    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nnx.Module):
    """Pre-norm decoder block with self-attention, cross-attention, and SwiGLU MLP."""

    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        module_kwargs = {"rngs": rngs, "dtype": dtype, "param_dtype": param_dtype}
        self.norm1 = nnx.RMSNorm(config.dim, **module_kwargs)
        self.self_attn = GQA(config, is_causal=True, **module_kwargs)
        self.norm2 = nnx.RMSNorm(config.dim, **module_kwargs)
        self.cross_attn = GQA(config, is_cross_attention=True, **module_kwargs)
        self.norm3 = nnx.RMSNorm(config.dim, **module_kwargs)
        self.mlp = SwiGLU(config, **module_kwargs)

    def __call__(
        self,
        x: Array,
        encoder_out: Array,
        *,
        decoder_mask: Array | None = None,
        encoder_mask: Array | None = None,
    ) -> Array:
        """
        Args:
            x: Decoder input of shape [B, T, D].
            encoder_out: Encoder output of shape [B, S, D].
            decoder_mask: Padding mask for decoder [B, T].
            encoder_mask: Padding mask for encoder output [B, S].
        """
        x = x + self.self_attn(self.norm1(x), mask=decoder_mask)
        x = x + self.cross_attn(self.norm2(x), kv=encoder_out, mask=encoder_mask)
        x = x + self.mlp(self.norm3(x))
        return x


class Encoder(nnx.Module):
    """BERT-style encoder with MLM and NSP heads."""

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

        self.tok_emb = nnx.Embed(config.vocab_size, config.dim, **module_kwargs)
        self.seg_emb = nnx.Embed(config.num_segments, config.dim, **module_kwargs)
        self.blocks = nnx.List(
            [EncoderBlock(config, **module_kwargs) for _ in range(config.num_layers)]
        )
        self.norm_final = nnx.RMSNorm(config.dim, **module_kwargs)
        self.mlm_head = nnx.Linear(config.dim, config.vocab_size, **linear_kwargs)
        self.nsp_head = nnx.Linear(config.dim, 2, **linear_kwargs)

    def __call__(
        self, input_ids: Array, seg_ids: Array, *, mask: Array | None = None
    ) -> tuple[Array, Array]:
        """
        Forward pass through the encoder.

        Args:
            input_ids: Token IDs of shape [B, L].
            seg_ids: Segment IDs of shape [B, L].
            mask: Optional attention mask of shape [B, L] (padding mask).

        Returns:
            Tuple of (mlm_logits, nsp_logits):
                - mlm_logits: [B, L, vocab_size] logits for masked language modeling.
                - nsp_logits: [B, 2] logits for next sentence prediction.
        """
        x = self.tok_emb(input_ids) + self.seg_emb(seg_ids)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm_final(x)
        mlm_logits = self.mlm_head(x)
        # Use first token (CLS) for sentence-level classification
        nsp_logits = self.nsp_head(x[:, 0, :])
        return mlm_logits, nsp_logits


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
