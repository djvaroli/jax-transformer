from dataclasses import dataclass

import pytest


@dataclass
class TestConfig:
    batch_size: int = 2
    seq_len: int = 4
    model_dim: int = 8
    n_layers: int = 2
    n_heads: int = 2
    random_seed = 42


# a global testing config class that can be accessed by all tests in the session
@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    return TestConfig()
