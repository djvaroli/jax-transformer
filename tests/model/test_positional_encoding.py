import jax
from conftest import TestConfig

from wheeljax.model import PositionalEncoding

test_config = TestConfig()


def test_positional_encoding():
    # TODO: only tests basic forward pass, check correct sin/cos functions
    pe = PositionalEncoding(test_config.model_dim)
    inputs = jax.numpy.zeros(
        (test_config.batch_size, test_config.seq_len, test_config.model_dim)
    )

    rng = jax.random.PRNGKey(test_config.random_seed)
    params = pe.init({"params": rng}, inputs)

    outputs = pe.apply({"params": params}, inputs)
