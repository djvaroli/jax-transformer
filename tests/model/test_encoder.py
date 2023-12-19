import jax
import pytest
from conftest import TestConfig

from bumblejax.model import EncoderBlock, TransformerEncoder

test_config = TestConfig()


@pytest.mark.parametrize("mask_type", ["lookahead", "padding", "all", None])
@pytest.mark.parametrize("train", [False, True])
def test_encoder_block(mask_type: str | None, train: bool):
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

    n_samples, seq_len, _ = inputs.shape
    attention_mask = test_config.create_attention_mask((n_samples, seq_len), mask_type)

    params = enc_block.init(
        {"params": rng, "dropout": rng},
        inputs,
        train=train,
        attention_mask=attention_mask,
    )["params"]

    outputs = enc_block.apply(
        {"params": params},
        inputs,
        train=train,
        attention_mask=attention_mask,
        rngs={"dropout": rng},
    )


@pytest.mark.parametrize("mask_type", ["lookahead", "padding", "all", None])
@pytest.mark.parametrize("train", [False, True])
def test_transformer_encoder(mask_type: str | None, train: bool):
    t_encoder = TransformerEncoder(
        test_config.model_dim,
        test_config.n_layers,
        test_config.n_heads,
        test_config.model_dim,
        dropout_rate=0.1,
    )

    rng = jax.random.PRNGKey(test_config.random_seed)
    inputs = jax.random.normal(
        rng, (test_config.batch_size, test_config.seq_len, test_config.model_dim)
    )

    n_samples, seq_len, _ = inputs.shape
    attention_mask = test_config.create_attention_mask((n_samples, seq_len), mask_type)

    params = t_encoder.init(
        {"params": rng, "dropout": rng},
        inputs,
        train=train,
        attention_mask=attention_mask,
    )["params"]

    outputs = t_encoder.apply(
        {"params": params},
        inputs,
        train=train,
        attention_mask=attention_mask,
        rngs={"dropout": rng},
    )
