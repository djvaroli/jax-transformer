import pytest
from conftest import TestConfig
from jax import numpy as jnp

from optimusjx.utils.mask import create_lookahead_mask, create_padding_mask

test_config = TestConfig()


@pytest.mark.parametrize("length", (1, 2, 3, 4))
def test_create_lookahead_mask(length: int):
    # mask will be of the same dimension as inputs
    mask = create_lookahead_mask(length)

    # should be (length, length)
    assert mask.shape == (length, length)

    # expect -inf on the upper triangular part and 0 on the lower triangular part
    expected_mask = jnp.triu(jnp.full(mask.shape, -jnp.inf), k=1)
    assert jnp.allclose(mask, expected_mask), f"{mask} != {expected_mask}"


@pytest.mark.parametrize("n_samples", (1, 2, 3, 4))
@pytest.mark.parametrize("length", (1, 2, 3, 4))
def test_create_padding_mask(n_samples: int, length: int):
    dummy_mask = test_config.generate_random_2d_padding_mask(n_samples, length)
    # set all -inf to 0, then treat the zeros as the padding token
    dummy_mask = dummy_mask.at[dummy_mask == -1 * jnp.inf].set(0)

    mask = create_padding_mask(dummy_mask, 0)

    # should be (n_samples, length)
    assert mask.shape == (n_samples, length)

    # check that zeros are replaced with -infs
    assert jnp.allclose(
        mask[dummy_mask == 0], -jnp.inf
    ), "Expected masked positions to have -inf"
