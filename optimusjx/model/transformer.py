import jax
from flax import linen as nn
from jax import Array

from .encoder import TransformerEncoder
from .positional_encoding import PositionalEncoding


class TransformerLM(nn.Module):
    vocab_size: int
    model_dim: int = 512
    dim_feedforward: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 8
    dropout_rate: float = 0.1
    input_dropout_rate: float = 0.1
    max_len: int = 5000
    name: str = "transformer-language-model"

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        train: bool = True,
        lookahead_mask: Array | None = None,
        padding_mask: Array | None = None,
    ) -> Array:
        """Forward pass of TransformerLM

        Args:
            inputs (Array): an array of shape (batch_size, seq_len)
            train (bool, optional): Whether to run in train mode. Defaults to True.
            lookahead_mask (Array | None, optional): an array of shape (seq_len, seq_len) or
                (..., ..., seq_len, seq_len). Defaults to None. If specified and is 2D,
                will be expanded to 4D tensor, otherwise passed to MHA as is.
            padding_mask (Array | None, optional): an array of shape (batch_size, seq_len) or
                (batch_size, ..., ..., seq_len). Defaults to None. If specified and is 2D,
                will be expanded to 4D tensor, otherwise passed to MHA as is.

        Returns:
            Array: an array of shape (batch_size, seq_len, vocab_size)
        """
        if padding_mask.ndim not in [2, 4]:
            raise ValueError(
                f"Padding mask must be a 2D or 4D tensor. Got {padding_mask.ndim}D"
            )

        if lookahead_mask.ndim not in [2, 4]:
            raise ValueError(
                f"Lookahead mask must be a 2D or 4D tensor. Got {lookahead_mask.ndim}D"
            )

        # create the combined attention mask
        attention_mask = None

        if padding_mask is None:
            padding_mask = jax.numpy.zeros_like(inputs)

        if lookahead_mask is None:
            seq_len = inputs.shape[1]
            lookahead_mask = jax.numpy.zeros((seq_len, seq_len))

        # expand them to 4D arrays if necessary
        if padding_mask.ndim < 4:
            padding_mask = padding_mask[:, None, None, :]

        if lookahead_mask.ndim < 4:
            lookahead_mask = lookahead_mask[None, None, :, :]

        # combine into a single attention mask
        attention_mask = padding_mask + lookahead_mask

        # embed inputs
        out = nn.Embed(self.vocab_size, self.model_dim)(inputs)
        out = PositionalEncoding(self.model_dim, self.max_len)(out)
        out = nn.Dense(self.model_dim)(out)
        out = nn.Dropout(rate=self.input_dropout_rate)(out, deterministic=not train)

        # pass through Transformer encoder
        out = TransformerEncoder(
            self.model_dim,
            self.num_heads,
            self.dim_feedforward,
            self.num_encoder_layers,
            self.dropout_rate,
        )(out, train=train, attention_mask=attention_mask)

        # pass through final dense layer
        out = nn.Dense(self.vocab_size)(out)

        return out
