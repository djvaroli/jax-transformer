from typing import Any

import jax
import numpy as np
from flax import linen as nn
from jax import Array


class PositionalEncoding(nn.Module):
    model_dim: int
    max_len: int = 5000

    def setup(self) -> Array:
        # 3D to be broadcastable with 3D inputs to transformers
        self.pe = np.zeros((1, self.max_len, self.model_dim))

        # value depends on position and on the index within the position vector
        # all even rows have sin(f(position)) and all odd rows have cos(f(position))
        positions = np.arange(0, self.max_len, dtype=np.float32)

        # the div_term only depends on the index within a given vector (is invariant to position in the sequence)
        # only need array with model_dim // 2 entries since same value in every pair of entries.
        # I.e. div_term[0] == div_term[1], div_term[2] == div_term[3]
        # raising to a negative exponent is equivalent to taking the reciprocal
        div_term = np.exp(
            -1 * np.arange(0, self.model_dim, 2) * (np.log(10000.0) / self.model_dim)
        )

        # sin at even positions, cos at odd positions
        self.pe[:, :, 0::2] = np.sin(positions * div_term)
        self.pe[:, :, 1::2] = np.cos(positions * div_term)

        # make it a Jax device array
        self.pe = jax.device_put(self.pe)

    def __call__(self, inputs: Array) -> Array:
        """Add positional encoding to inputs

        Args:
            inputs (Array): an array of shape (batch_size, seq_len, model_dim)

        Returns:
            Array: inputs with positional encoding added. Array of shape (batch_size, seq_len, model_dim)
        """
        inputs = inputs + self.pe[:, : inputs.shape[1], :]
        return inputs
