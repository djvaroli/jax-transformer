from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from wheeljax.utils.mask import create_lookahead_mask


@dataclass
class TestConfig:
    batch_size: int = 2
    seq_len: int = 4
    model_dim: int = 8
    n_layers: int = 2
    n_heads: int = 2
    random_seed = 42

    def generate_random_2d_padding_mask(self, n_samples: int, length: int) -> jax.Array:
        """Creates an additive Array with random positions marked as -inf. to be used as a padding mask.

        Args:
            n_samples (int): number of samples in the batch.
            length (int): length of the sequence.

        Returns:
            Array: additive Array with random positions marked as -inf. to be used as a padding mask.
                Array shape (n_samples, length)
        """
        key = jax.random.PRNGKey(self.random_seed)
        return jax.numpy.log(jax.random.uniform(key, (n_samples, length)) > 0.1)

    def create_attention_mask(
        self,
        dimensions: tuple[int, int],
        mask_type: Literal["look_ahead", "padding", "all"] | None,
    ) -> Array:
        """Craetes an attention mask

        Args:
            dimensions (tuple[int, int]): mask dimensions in the form (n_samples, length).
            mask_type (str): the type of mask to create.

        Returns:
            Array: 4D attention mask.
        """
        n_samples, length = dimensions

        if mask_type == "lookahead":
            attention_mask = create_lookahead_mask(length)[None, None, :, :]
        elif mask_type == "padding":
            attention_mask = self.generate_random_2d_padding_mask(n_samples, length)[
                :, None, None, :
            ]
        elif mask_type == "all":
            lookahead_mask = create_lookahead_mask(length)[None, None, :, :]
            padding_mask = self.generate_random_2d_padding_mask(n_samples, length)[
                :, None, None, :
            ]
            attention_mask = lookahead_mask + padding_mask
        else:
            attention_mask = jnp.zeros((n_samples, 1, length, length))

        return attention_mask


# a global testing config class that can be accessed by all tests in the session
@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    return TestConfig()
