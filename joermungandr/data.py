from dataclasses import dataclass
from typing import Generator

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .config import ModelConfig, TrainConfig


@jax.tree_util.register_pytree_node_class
@dataclass
class Batch:
    """A batch of data for training, registered as a JAX PyTree.

    Attributes:
        input_ids: [B, L] token IDs.
        seg_ids: [B, L] segment IDs.
        attention_mask: [B, L] boolean mask for attention (1 for real, 0 for padding).
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

    def tree_flatten(self):
        """Flatten the PyTree."""
        children = (
            self.input_ids,
            self.seg_ids,
            self.attention_mask,
            self.mlm_targets,
            self.mlm_mask,
            self.nsp_labels,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the PyTree."""
        return cls(*children)


def dummy_data_generator(
    model_config: ModelConfig,
    train_config: TrainConfig,
    global_batch_size: int,
    rngs: nnx.Rngs,
) -> Generator[Batch]:
    """
    Yields batches of dummy data as Batch objects, including padding simulation.
    """
    pad_id = 0  # Assuming [PAD] token is ID 0

    while True:
        # 1. Generate base data
        inp = jax.random.randint(
            rngs.step(),
            (global_batch_size, train_config.seq_len),
            1,  # Start from 1 to avoid generating pad_id tokens as real data
            model_config.vocab_size,
        )
        seg = jax.random.randint(
            rngs.step(),
            (global_batch_size, train_config.seq_len),
            0,
            model_config.num_segments,
        )
        mlm_mask = (
            jax.random.uniform(rngs.step(), (global_batch_size, train_config.seq_len))
            < train_config.mlm_mask_prob
        )
        mlm_targets = jax.random.randint(
            rngs.step(),
            (global_batch_size, train_config.seq_len),
            0,
            model_config.vocab_size,
        )
        nsp_labels = jax.random.randint(rngs.step(), (global_batch_size,), 0, 2)

        # 2. Simulate variable sequence lengths and create attention mask
        # Sequences can be between 50% and 100% of max length
        seq_lens = jax.random.randint(
            rngs.step(),
            (global_batch_size,),
            train_config.seq_len // 2,
            train_config.seq_len + 1,
        )
        attention_mask = jnp.arange(train_config.seq_len) < seq_lens[:, None]

        # 3. Apply padding and update masks
        # Set padded tokens in input to pad_id
        inp = jnp.where(attention_mask, inp, pad_id)

        # Ensure MLM mask does not include padded tokens
        mlm_mask = mlm_mask & attention_mask

        yield Batch(
            input_ids=inp,
            seg_ids=seg,
            attention_mask=attention_mask,
            mlm_targets=mlm_targets,
            mlm_mask=mlm_mask,
            nsp_labels=nsp_labels,
        )
