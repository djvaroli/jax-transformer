from warnings import warn

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
    _debug: bool = False

    def toggle_debug(self) -> bool:
        self._debug = not self._debug
        return self._debug

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        train: bool = True,
        lookahead_mask: Array | None = None,
        padding_mask: Array | None = None,
        **kwargs,
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

        Kwargs:
            Any key-word arguments not excplicitly defined will be ignored.

        Returns:
            Array: an array of shape (batch_size, seq_len, vocab_size)
        """
        if len(kwargs):
            for key in kwargs.keys():
                warn(f"Transformer recieved unknown keyword argument {key} - ignoring")

        if padding_mask is not None and padding_mask.ndim not in [2, 4]:
            raise ValueError(
                f"Padding mask must be a 2D or 4D tensor. Got {padding_mask.ndim}D"
            )

        if lookahead_mask is not None and lookahead_mask.ndim not in [2, 4]:
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
        embedding = nn.Embed(self.vocab_size, self.model_dim)
        out = embedding(inputs)
        if self._debug:
            print(out, "embedding\n")

        out = PositionalEncoding(self.model_dim, self.max_len)(out)
        if self._debug:
            print(out, "positional encoding\n")
        out = nn.Dense(self.model_dim)(out)
        if self._debug:
            print(out, "dense-1\n")
        out = nn.Dropout(rate=self.input_dropout_rate)(out, deterministic=not train)
        if self._debug:
            print(out, "dropout-1\n")

        # pass through Transformer encoder
        out = TransformerEncoder(
            self.model_dim,
            self.num_heads,
            self.dim_feedforward,
            self.num_encoder_layers,
            self.dropout_rate,
        )(out, train=train, attention_mask=attention_mask)
        if self._debug:
            print(out, "transformer\n")

        # more efficient to re-use the embedding matrix as final dense layer
        out = jax.numpy.matmul(out, jax.numpy.transpose(embedding.embedding, (1, 0)))
        # out = nn.Dense(self.vocab_size)(out)
        if self._debug:
            print(out, "lm-head\n")

        return out
