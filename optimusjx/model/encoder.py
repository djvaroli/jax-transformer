from typing import Any

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from .attention import MultiHeadedAttention


def is_dropout(layer: nn.Module) -> bool:
    return isinstance(layer, nn.Dropout)


class EncoderBlock(nn.Module):
    model_dim: int
    n_heads: int
    dim_feedforward: int
    dropout_rate: float = 0.0

    def setup(self) -> None:
        self.mha = MultiHeadedAttention(self.model_dim, self.n_heads)
        self.mlp = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_rate),
            nn.relu,
            nn.Dense(self.model_dim),
        ]
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(
        self, inputs: Array, train: bool = True, attention_mask: Array | None = None
    ):
        """Forward pass through the Transformer encoder block.

        Args:
            inputs (Array): array of shape (batch_size, seq_len, model_dim)
            train (bool, optional): whether to run in train mode. Defaults to `True`.
            attention_mask (Array | None, optional): mask to be applied to attention logits.
                Defaults to None.
        """

        # out will have shape (b, seq_len, model_dim)
        out, _ = self.mha(inputs, train=train, attention_mask=attention_mask)

        # maintains shape
        out = inputs + self.dropout(out, deterministic=not train)
        out = self.layernorm1(out)

        # pass through MLP, maintains shape
        # keep pre-mlp input since we have another residual connection
        mlp_out = out
        for layer in self.mlp:
            mlp_out = (
                layer(mlp_out)
                if not is_dropout(layer)
                else layer(mlp_out, deterministic=not train)
            )

        out = inputs + self.dropout(mlp_out, deterministic=not train)
        return self.layernorm2(out)
