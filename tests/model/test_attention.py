import jax
import jax.numpy as jnp
import pytest
from conftest import TestConfig

from wheeljax.model import MultiHeadedAttention, scaled_dot_product_attn

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
@pytest.mark.parametrize("mask_type", ["lookahead", "padding", "all", None])
def test_scaled_dot_product_attention(
    input_shape: tuple[int, ...], mask_type: str | None
):
    inputs = 2 * jnp.ones(input_shape)
    q = k = v = inputs
    n_samples, _, seq_len, _ = inputs.shape

    # returns a 4D mask. If type is None then creates a mask of all zeros
    attention_mask = test_config.create_attention_mask((n_samples, seq_len), mask_type)

    # test that the attention weights sum to 1
    _, attn_weights = scaled_dot_product_attn(q, k, v, mask=attention_mask)

    # should sum to 1 along the last axis
    assert jnp.allclose(attn_weights.sum(-1), 1.0)

    # broadcast the mask to shape of attention weights
    attention_mask = jnp.broadcast_to(attention_mask, attn_weights.shape)
    masked_positions = attention_mask == -jnp.inf

    # check that attention weights at masked positions are 0
    assert jnp.allclose(
        attn_weights[masked_positions], 0.0
    ), "attention weights at masked positions should be 0"


@pytest.mark.parametrize("mask_type", ["lookahead", "padding", "all", None])
@pytest.mark.parametrize("train", [True, False])
def test_multi_head_attention(mask_type: str | None, train: bool):
    input_shape = (test_config.batch_size, test_config.seq_len, test_config.model_dim)
    mha = MultiHeadedAttention(
        n_heads=test_config.n_heads, model_dim=test_config.model_dim
    )
    rng = jax.random.PRNGKey(test_config.random_seed)
    inputs = 2 * jnp.ones(input_shape)
    n_samples, seq_len, _ = inputs.shape

    attention_mask = test_config.create_attention_mask(
        (n_samples, seq_len), mask_type=mask_type
    )

    mha_params = mha.init(
        {"params": rng}, inputs, train=True, attention_mask=attention_mask
    )["params"]

    # test that the attention weights sum to 1
    _, attn_weights = mha.apply(
        {"params": mha_params}, inputs, train=train, attention_mask=attention_mask
    )

    # should sum to 1 along the last axis
    assert jnp.allclose(attn_weights.sum(-1), 1.0)

    # broadcast the mask to shape of attention weights
    attention_mask = jnp.broadcast_to(attention_mask, attn_weights.shape)
    masked_positions = attention_mask == -jnp.inf

    # check that attention weights at masked positions are 0
    assert jnp.allclose(
        attn_weights[masked_positions], 0.0
    ), "attention weights at masked positions should be 0"
