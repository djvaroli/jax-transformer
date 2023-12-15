import jax
import pytest
from conftest import TestConfig

from optimusjx.model import EncoderBlock
from optimusjx.utils import lookahead_mask

test_config = TestConfig()


@pytest.mark.parametrize("mask_attention", [False, True])
def test_encoder_block(mask_attention: bool):
    enc_block = EncoderBlock(
        test_config.model_dim,
        test_config.n_heads,
        test_config.model_dim,
        dropout_rate=0.1,
    )
    rng = jax.random.PRNGKey(test_config.random_seed)

    inputs = jax.random.normal(
        rng, (test_config.batch_size, test_config.seq_len, test_config.model_dim)
    )

    mask = lookahead_mask(inputs) if mask_attention else None
    params = enc_block.init(
        {"params": rng, "dropout": rng}, inputs, train=True, attention_mask=mask
    )["params"]

    outputs = enc_block.apply(
        {"params": params},
        inputs,
        train=True,
        attention_mask=mask,
        rngs={"dropout": rng},
    )
