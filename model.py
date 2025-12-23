import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from config import ModelConfig


class SwiGLU(nnx.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.w1 = nnx.Linear(config.dim, config.intermediate_dim, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(config.dim, config.intermediate_dim, use_bias=False, rngs=rngs)
        self.w3 = nnx.Linear(config.intermediate_dim, config.dim, use_bias=False, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.w3(nnx.silu(self.w1(x)) * self.w2(x))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Array:
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
        x: Input tensor of shape [B, H, L, D].
        freqs_cis: Precomputed frequencies of shape [max_len, D//2].

    Returns:
        Tensor of shape [B, H, L, D] with rotary embeddings applied.
    """
    B, H, L, D = x.shape
    # Pair adjacent elements as complex: [B, H, L, D] -> [B, H, L, D//2] complex
    x_complex = jax.lax.complex(x[..., 0::2], x[..., 1::2])
    # Broadcast freqs: [L, D//2] -> [1, 1, L, D//2]
    freqs_cis = freqs_cis[None, None, :L, :]
    x_out = x_complex * freqs_cis
    # Unpack complex back to real: [B, H, L, D//2] complex -> [B, H, L, D]
    return jnp.stack([jnp.real(x_out), jnp.imag(x_out)], axis=-1).reshape(B, H, L, D)


class GQA(nnx.Module):
    """Grouped-query attention with rotary position (RoPE) embeddings."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.num_heads = config.num_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.dim // config.num_heads

        self.q_proj = nnx.Linear(
            config.dim, config.num_heads * self.head_dim, use_bias=False, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            config.dim, config.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            config.dim, config.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            config.num_heads * self.head_dim, config.dim, use_bias=False, rngs=rngs
        )

        self.freqs_cis = nnx.Cache(precompute_freqs_cis(self.head_dim, config.max_len))

    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        """
        Apply grouped-query attention.

        Args:
            x: Input tensor of shape [B, L, D].
            mask: Optional attention mask of shape [B, 1, L, L] or broadcastable.

        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, _ = x.shape

        # Project and reshape: [B, L, D] -> [B, L, H, head_dim]
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv, self.head_dim)

        # Transpose for attention: [B, L, H, head_dim] -> [B, H, L, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        q = apply_rope(q, self.freqs_cis.value)
        k = apply_rope(k, self.freqs_cis.value)

        # Attention: [B, H, L, head_dim] -> [B, H, L, head_dim]
        out = jax.nn.dot_product_attention(q, k, v, mask=mask)

        # Transpose back: [B, H, L, head_dim] -> [B, L, H, head_dim] -> [B, L, D]
        out = jnp.transpose(out, (0, 2, 1, 3))
        return self.o_proj(out.reshape(B, L, -1))


class TransformerBlock(nnx.Module):
    """Pre-norm transformer block with GQA attention and SwiGLU MLP."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.norm1 = nnx.RMSNorm(config.dim, rngs=rngs)
        self.attn = GQA(config, rngs=rngs)
        self.norm2 = nnx.RMSNorm(config.dim, rngs=rngs)
        self.mlp = SwiGLU(config, rngs=rngs)

    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nnx.Module):
    """BERT-style encoder with MLM and NSP heads."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.tok_emb = nnx.Embed(config.vocab_size, config.dim, rngs=rngs)
        self.seg_emb = nnx.Embed(config.num_segments, config.dim, rngs=rngs)
        self.blocks = nnx.List(
            [TransformerBlock(config, rngs=rngs) for _ in range(config.num_layers)]
        )
        self.norm_final = nnx.RMSNorm(config.dim, rngs=rngs)
        self.mlm_head = nnx.Linear(config.dim, config.vocab_size, use_bias=False, rngs=rngs)
        self.nsp_head = nnx.Linear(config.dim, 2, use_bias=False, rngs=rngs)

    def __call__(
        self, input_ids: Array, seg_ids: Array, mask: Array | None = None
    ) -> tuple[Array, Array]:
        """
        Forward pass through the encoder.

        Args:
            input_ids: Token IDs of shape [B, L].
            seg_ids: Segment IDs of shape [B, L].
            mask: Optional attention mask of shape [B, 1, L, L] or broadcastable.

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
