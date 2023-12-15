import pytest
from jax import numpy as jnp

from optimusjx.utils.mask import lookahead_mask, paddig_mask


@pytest.mark.parametrize(
    "input_shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]
)
def test_lookahead_mask(
    input_shape: tuple[int, ...],
):
    inputs = jnp.ones(input_shape)
    if len(input_shape) < 2 or len(input_shape) > 4:
        with pytest.raises(ValueError):
            mask = lookahead_mask(inputs)
        return

    # mask will be of the same dimension as inputs
    mask = lookahead_mask(inputs)

    seq_len = inputs.shape[-1] if len(input_shape) == 2 else inputs.shape[-2]

    # should be (seq_len, seq_len)
    assert mask.shape[-2:] == (seq_len, seq_len)

    # expect -inf on the upper triangular part and 0 on the lower triangular part
    expected_mask = jnp.triu(jnp.full(mask.shape, -jnp.inf), k=1)
    assert jnp.allclose(mask, expected_mask), f"{mask} != {expected_mask}"
