from typing import Any, Callable

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from .attention import MultiHeadedAttention, sqrt_model_dim_scaling


def is_dropout(layer: nn.Module) -> bool:
    return isinstance(layer, nn.Dropout)


class EncoderBlock(nn.Module):
    model_dim: int
    n_heads: int
    dim_feedforward: int
    dropout_rate: float = 0.0
    scaling_function: Callable[[Array], Array] = sqrt_model_dim_scaling

    def setup(self) -> None:
        self.mha = MultiHeadedAttention(
            self.model_dim, self.n_heads, self.scaling_function
        )
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
    ) -> Array:
        """Forward pass through the Transformer encoder block.

        Args:
            inputs (Array): array of shape (batch_size, seq_len, model_dim)
            train (bool, optional): whether to run in train mode. Defaults to `True`.
            attention_mask (Array | None, optional): an additive mask to be applied to attention logits.
                Defaults to None. Positions to be masked are set to -inf, and positions to be kept are set to 0.

        Returns:
            Array: output array of shape (batch_size, seq_len, model_dim)
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


class TransformerEncoder(nn.Module):
    model_dim: int
    n_layers: int
    n_heads: int
    dim_feedforward: int
    dropout_rate: float = 0.0
    scaling_function: Callable[[Array], Array] = sqrt_model_dim_scaling

    def setup(self) -> None:
        self.encoder_stack = [
            EncoderBlock(
                self.model_dim,
                self.n_heads,
                self.dim_feedforward,
                self.dropout_rate,
                self.scaling_function,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(
        self, inputs: Array, train: bool = True, attention_mask: Array | None = None
    ) -> Array:
        """Forward pass through the Transformer encoder.

        Args:
            inputs (Array): array of shape (batch_size, seq_len, model_dim)
            train (bool, optional): whether to run in train mode. Defaults to `True`.
            attention_mask (Array | None, optional): an additive mask to be applied to the
                attention logits. Positions to be masked are set to -inf, and positions to be kept
                are set to 0. Defaults to None.

        Returns:
            Array: output array of shape (batch_size, seq_len, model_dim)
        """
        out = inputs
        for block in self.encoder_stack:
            out = block(out, train=train, attention_mask=attention_mask)

        return out

    def get_attention_maps(
        self, inputs: Array, train: bool = True, attention_mask: Array | None = None
    ) -> list[Array]:
        """Returns the encoder attention maps for each encoder block.

        Args:
            inputs (Array): input array of shape (batch_size, seq_len, model_dim)
            train (bool, optional): wether to run model in train model.
            attention_mask (Array | None, optional): an additive mask to apply to attention logits.
                Positions to be masked are set to -inf, and positions to be kept are set to 0. Defaults to None.

        Returns:
            list[Array]: a list of length (n_heads) containing the output maps.
        """

        attention_maps = []
        out = inputs
        for block in self.encoder_stack:
            _, attn_weights = block.mha(out, train=train, attention_mask=attention_mask)
            attention_maps.append(attn_weights)
            out = block(out, train=train, attention_mask=attention_mask)

        return attention_maps
