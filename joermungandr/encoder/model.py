import jax.numpy as jnp
from flax import nnx
from jax import Array

from ..config import ModelConfig
from ..layers import EncoderBlock


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
