from typing import Optional, Tuple

import jax
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
        mask (Optional[Array], optional): padding mask with shape (seq_len, seq_len). Defaults to None.
            If specified must be a binary matrix with 1s indicating valid positions and 0s indicating masked positions.
            Expected shape is (seq_len, seq_len) where seq_len is the length of the sequence.
        
    Returns:
        Tuple[Array, Array]: 
            Array: scaled dot product attention output with shape (seq_len, Dv).
            Array: attention weights with shape (seq_len, seq_len).
            Array: attention logits with shape (seq_len, seq_len).
    """
    seq_len, vector_dim = q.shape
    if mask is None:
        mask = jnp.ones((seq_len, seq_len))
    
    # will ensure that variance of dot product remains sigma^4 ~= 1 since we init with sigma = 1
    # see dotprod-step-by-step.ipynb for more details
    variance_scale_factor = 1 / jnp.sqrt(vector_dim)

    # compute QK^T, transpose K to get (Dk, seq_len)
    # output shape (seq_len, seq_len)
    attention_logits = jnp.matmul(q, jnp.swapaxes(k, 0, 1)) * variance_scale_factor

    # apply mask if provided
    # positions at 1 remain unchanged, positions at 0 are set to -inf (-1e10)
    # 1 - mask = 1 for masked positions, 0 for valid positions
    attention_logits = attention_logits - 1e10 * (1 - mask)

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
        mask (Optional[Array], optional): padding mask with shape (seq_len, seq_len). Defaults to None.
            If specified must be a binary matrix with 1s indicating valid positions and 0s indicating masked positions.
            Expected shape is (seq_len, seq_len) where seq_len is the length of the sequence.
        
    Returns:
        Tuple[Array, Array]: 
            Array: scaled dot product attention output with shape (b, seq_len, Dv).
            Array: attention weights with shape (b, seq_len, seq_len).
    """ 
    result, attention_weights, _ = _scaled_dot_product_attention_with_logits(q, k, v, mask)
    return result, attention_weights 

    