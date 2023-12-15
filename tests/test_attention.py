import jax
import jax.numpy as jnp
import pytest
from conftest import TestConfig

from optimusjx.model import MultiHeadedAttention, scaled_dot_product_attn
from optimusjx.utils.mask import lookahead_mask

test_config = TestConfig()


@pytest.mark.parametrize(
    "input_shape",
    [
        (
            test_config.batch_size,
            test_config.n_heads,
            test_config.seq_len,
            test_config.model_dim,
        )
    ],
)
@pytest.mark.parametrize("mask_attention", [False, True])
def test_scaled_dot_product_attention(
    input_shape: tuple[int, ...], mask_attention: bool
):
    inputs = 2 * jnp.ones(input_shape)
    q = k = v = inputs

    if mask_attention:
        mask = lookahead_mask(inputs)
    else:
        mask = None

    # test that the attention weights sum to 1
    _, attn_weights = scaled_dot_product_attn(q, k, v, mask=mask)

    # should sum to 1 along the last axis
    assert jnp.allclose(attn_weights.sum(-1), 1.0)


@pytest.mark.parametrize("mask_attention", [False, True])
@pytest.mark.parametrize("train", [True, False])
def test_multi_head_attention(mask_attention: bool, train: bool):
    input_shape = (test_config.batch_size, test_config.seq_len, test_config.model_dim)
    mha = MultiHeadedAttention(
        n_heads=test_config.n_heads, model_dim=test_config.model_dim
    )
    rng = jax.random.PRNGKey(test_config.random_seed)

    inputs = 2 * jnp.ones(input_shape)
    mask = lookahead_mask(inputs) if mask_attention else None

    mha_params = mha.init({"params": rng}, inputs, train=True, attention_mask=mask)[
        "params"
    ]

    # test that the attention weights sum to 1
    _, attn_weights = mha.apply(
        {"params": mha_params}, inputs, train=train, attention_mask=mask
    )

    # should sum to 1 along the last axis
    assert jnp.allclose(attn_weights.sum(-1), 1.0)
