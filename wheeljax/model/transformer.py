import time
from warnings import warn

import jax
import tqdm
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

        if self._debug:
            print(out, "lm-head\n")

        return out

    def generate(
        self,
        params,
        input_tokens: Array,
        max_new_tokens: int = 20,
        stop_token_id: int | None = None,
        temperature: float = 1.0,
        rng_key: jax.random.KeyArray | None = None,
    ) -> Array:
        """Generate an output sequence given an input sequence

        Args:
            params: model parameters.
            input_tokens (Array): an array of shape (1, ...) corresponding to an initial sequence.
            max_new_tokens (int): maximum number of new tokens to generate.
            stop_token_id (int, optional): token id to stop generation at. Defaults to None.
            temperature (float, optional): controls the randomness when sampling tokens. Defaults to 1.0.
                Higher values will increase randomness, lower values will decrease randomness.
            rng_key (jax.random.KeyArray, optional): random number generator key. Defaults to None.
                If None, then a new key will be generated, using the current time as a seed.

        Returns:
            Array: an array of shape (1, ...) corresponding to the generated sequence.
        """
        tokens = input_tokens
        if tokens.shape[0] != 1:
            raise ValueError("Method implemented for a single input sequence only.")

        rng_key = (
            rng_key if rng_key is not None else jax.random.PRNGKey(int(time.time()))
        )

        with tqdm.tqdm(total=max_new_tokens, desc="Generating") as pbar:
            for _ in range(max_new_tokens):
                if stop_token_id is not None and tokens[0, -1] == stop_token_id:
                    break

                next_token_logits = self.apply(
                    {"params": params},
                    tokens,
                    padding_mask=None,
                    lookahead_mask=None,
                    train=False,
                )[:, -1:, :]

                next_token_logits = next_token_logits / temperature
                next_token = jax.random.categorical(rng_key, next_token_logits)

                tokens = jax.numpy.concatenate([tokens, next_token], axis=-1)
                rng_key, _ = jax.random.split(rng_key, 2)
                pbar.update(1)

                # clean up memory
                del next_token_logits

        return tokens
