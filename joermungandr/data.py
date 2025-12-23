from dataclasses import dataclass
from typing import Generator

import jax
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
        mlm_targets: [B, L] target token IDs for MLM.
        mlm_mask: [B, L] boolean mask for MLM positions.
        nsp_labels: [B] binary labels for NSP.
    """

    input_ids: Array
    seg_ids: Array
    mlm_targets: Array
    mlm_mask: Array
    nsp_labels: Array

    def tree_flatten(self):
        """Flatten the PyTree."""
        children = (self.input_ids, self.seg_ids, self.mlm_targets, self.mlm_mask, self.nsp_labels)
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
    """Yields batches of dummy data as Batch objects."""
    while True:
        inp = jax.random.randint(
            rngs.step(),
            (global_batch_size, train_config.seq_len),
            0,
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

        yield Batch(
            input_ids=inp,
            seg_ids=seg,
            mlm_targets=mlm_targets,
            mlm_mask=mlm_mask,
            nsp_labels=nsp_labels,
        )
