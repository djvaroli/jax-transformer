import jax
import pytest
from conftest import TestConfig

from wheeljax.model import TransformerLM

test_config = TestConfig()


@pytest.mark.parametrize("mask_type", ["lookahead", "padding", "all", None])
@pytest.mark.parametrize("train", [False, True])
def test_transformer(mask_type: str | None, train: bool):
    """
    Test the transformer model
    """
    # TODO: only tests basic forward pass
    rng = jax.random.PRNGKey(test_config.random_seed)
    transformer = TransformerLM(
        vocab_size=10,
        model_dim=test_config.model_dim,
        dim_feedforward=2,
        num_heads=2,
        num_encoder_layers=1,
    )

    mask_dims = (test_config.batch_size, test_config.seq_len)

    # create two identical masks to test that passing both works
    padding_mask = test_config.create_attention_mask(mask_dims, mask_type)
    lookahead_mask = test_config.create_attention_mask(mask_dims, mask_type)

    inputs = jax.random.randint(
        rng, (test_config.batch_size, test_config.seq_len), 0, 10
    )
    params = transformer.init(
        {"params": rng, "dropout": rng},
        inputs,
        lookahead_mask=lookahead_mask,
        padding_mask=padding_mask,
        train=train,
    )["params"]

    logits = transformer.apply(
        {"params": params},
        inputs,
        lookahead_mask=lookahead_mask,
        padding_mask=padding_mask,
        train=train,
        rngs={"dropout": rng},
    )
