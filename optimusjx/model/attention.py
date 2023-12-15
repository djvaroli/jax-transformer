from typing import Optional, Tuple

import jax
from flax import linen as nn
from jax import Array
from jax import numpy as jnp

# Here's what GPT4 said about the implementation below:
# In conclusion, this version of the scaled dot-product attention mechanism is well-implemented.
# The code is clean, efficient, and follows good coding practices.
# It seems ready for use or integration into a larger system or model. Well done!
# Yay!


@jax.vmap
def _scaled_dot_product_attention_with_logits(
    q: Array,
    k: Array,
    v: Array,
    mask: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """Performs scaled dot product attention for a single attention head.

    Note:
        * function is vectorized over the batch dimension using jax.vmap.

    Args:
        q (Array): query matrix with shape (seq_len, Dk).
        k (Array): keys matrix with shape (seq_len, Dk).
        v (Array): values matrix with shape (seq_len, Dv).
        mask (Optional[Array], optional): padding mask with shape (..., seq_len, seq_len). Defaults to None.
            If specified, expected to be an additive mask, i.e. positions to be masked are set to -inf.
            and positions to be attended to are set to 0.

    Returns:
        Tuple[Array, Array]:
            Array: scaled dot product attention output with shape (seq_len, Dv).
            Array: attention weights with shape (seq_len, seq_len).
            Array: attention logits with shape (seq_len, seq_len).
    """
    seq_len, vector_dim = q.shape[-2:]

    if mask is None:
        # mask will be broadcasted appropriately
        mask = jnp.zeros((seq_len, seq_len))

    # will ensure that variance of dot product remains sigma^4 ~= 1 since we init with sigma = 1
    # see dotprod-step-by-step.ipynb for more details
    variance_scale_factor = 1 / jnp.sqrt(vector_dim)

    # compute QK^T, transpose K to get (Dk, seq_len)
    # output shape (..., seq_len, seq_len)
    attention_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * variance_scale_factor

    # apply additive mask
    # values to be masked are set to -inf, values to be attended to are set to 0
    attention_logits = attention_logits + mask

    # apply softmax element-wise along the rows
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)

    # weighted average of value vectors (weighted retrieval)
    result = jnp.matmul(attention_weights, v)

    return result, attention_weights, attention_logits


def scaled_dot_product_attn(
    q: Array,
    k: Array,
    v: Array,
    mask: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Performs scaled dot product attention for a single attention head.

    Args:
        q (Array): query matrix with shape (b, seq_len, Dk).
        k (Array): keys matrix with shape (b, seq_len, Dk).
        v (Array): values matrix with shape (seq_len, Dv).
        mask (Optional[Array], optional): padding mask with shape (..., seq_len, seq_len). Defaults to None.
            If specified, expected to be an additive mask, i.e. positions to be masked are set to -inf.
            and positions to be attended to are set to 0.

    Returns:
        Tuple[Array, Array]:
            Array: scaled dot product attention output with shape (b, ..., seq_len, Dv).
            Array: attention weights with shape (b, ..., seq_len, seq_len).
    """
    result, attention_weights, _ = _scaled_dot_product_attention_with_logits(
        q, k, v, mask
    )
    return result, attention_weights


class MultiHeadedAttention(nn.Module):
    """Multi-head attention layer.

    Input args:
        model_dim (int): the model representation dimension. Will be the size of
            the last dimension of the output array.
        n_heads (int): the number of attention heads. `model_dim` must be evenly divisible
            by `n_heads`.
    """

    model_dim: int
    n_heads: int

    def setup(self) -> None:
        if self.model_dim % self.n_heads != 0:
            raise ValueError(
                "Model dimension must be evenly divisible by the number of attention heads."
            )

        self.qkv_dense = nn.Dense(3 * self.model_dim)
        self.out_dense = nn.Dense(self.model_dim)

    def _expand_mask(self, mask: Array) -> Array:
        """Ensures that mask is 4D by expanding dimensions until that is the case."""
        if mask.ndim < 2:
            raise ValueError("Mask must be at least 2D")

        # expected shape (b, seq_len, seq_len), expand to (b, 1, seq_len, seq_len)
        if mask.ndim == 3:
            mask = jnp.expand_dims(mask, 1)

        # expected_shape (seq_len, seq_len), expand to (1, 1, seq_len, seq_len)
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)

        return mask

    def __call__(
        self,
        inputs: Array,
        train: bool = True,
        attention_mask: Array | None = None
        # TODO: add padding mask
    ) -> tuple[Array, Array]:
        """Perform attention over an input array.

        Args:
            inputs (Array): input of array of shape (batch_size, seq_len, model_dim)
            train (bool, optional): wether to operate in train mode. Defaults to True.
            attention_mask (Array | None, optional): additive attention mask. Defaults to None.

        Returns:
            tuple[Array, Array]: attention array and attention weights.
                attention array will have shape (batch_size, seq_len, model_dim)
                attention weights will have shape (batch_size, n_heads, seq_len, seq_len)
        """
        # inputs shape (b, seq_len, model_dim)
        batch_size, seq_len, dim = inputs.shape

        # qkv shape (b, seq_len, 3 * model_dim)
        qkv = self.qkv_dense(inputs)

        # qkv shape (b, seq_len, n_heads, 3 * model_dim // n_heads)
        qkv = jnp.reshape(qkv, (batch_size, seq_len, self.n_heads, -1))

        # qkv shape (b, n_heads, seq_len, 3 * model_dim // n_heads)
        qkv = jnp.transpose(qkv, (0, 2, 1, 3))

        # q,k,v shape (b, n_heads, seq_len, 3 * model_dim // n_heads // 3)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # attn shape (b, n_heads, seq_len, 3 * model_dim // n_heads // 3)
        # TODO: pass in mask
        if attention_mask is not None:
            attention_mask = self._expand_mask(attention_mask)

        attn, attn_weights = scaled_dot_product_attn(q, k, v, attention_mask)

        # attn shape (b, seq_len, n_heads, 3 * model_dim // n_heads // 3)
        attn = jnp.transpose(attn, (0, 2, 1, 3))

        # attn shape (b, seq_len, 3 * model_dim // 3) -> (b, seq_len, model_dim)
        attn = jnp.reshape(attn, (batch_size, seq_len, dim))

        return self.out_dense(attn), attn_weights
