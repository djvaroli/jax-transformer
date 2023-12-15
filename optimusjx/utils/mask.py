from jax import Array
from jax import numpy as jnp


def lookahead_mask(inputs: Array, dtype: str | jnp.dtype = "float32") -> Array:
    """Creates an additive lookahead attention mask based on the shape of the inputs.
    Positions that should not be attended to will contain -inf. Unmasked positions will contain 0.

    Args:
        inputs (Array): a 2D array of shape (batch_size, seq_len),
            a 3D array of shape (batch_size, seq_len, model_dim), or
            a 4D array of shape (batch_size, n_heads, seq_len, model_dim).
        dtype (str | jnp.dtype): dtype of the mask array. Defaults to 'float32'.

    Returns:
        Array: a 2D array of shape (seq_len, seq_len) or 3D array of shape (batch_size, seq_len, seq_len),
        or a 4D array of shape (batch_size, n_heads, seq_len, seq_len) with -inf on the upper triangular part
        and 0 on the lower triangular part.

    Example:
        >>> import jax.numpy as np
        >>> inputs = np.ones((2, 3))
        >>> mask = lookahead_mask(inputs)
        >>> print(mask)
        >>>  Array([[  0., -inf, -inf],
        >>> [  0.,   0., -inf]], dtype=float32)
    """

    if inputs.ndim < 2 or inputs.ndim > 4:
        raise ValueError("Expected inputs to be a 2D, 3D or 4D tensor.")

    if inputs.ndim == 2:
        seq_len = inputs.shape[-1]
        mask_shape = (seq_len, seq_len)

    elif inputs.ndim == 3:
        seq_len = inputs.shape[-2]
        mask_shape = (inputs.shape[0], seq_len, seq_len)

    else:
        seq_len = inputs.shape[-2]
        mask_shape = (inputs.shape[0], inputs.shape[1], seq_len, seq_len)

    # initialize binary mask with 0s on the upper triangular part
    binary_mask = jnp.tril(jnp.ones(mask_shape, dtype=dtype))

    # Using log on the lower triangular part of the mask to set upper triangular to -inf
    return jnp.log(binary_mask)


def paddig_mask(
    inputs: Array, padding_token_index: int, dtype: str | jnp.dtype = "float32"
) -> Array:
    """Creates and returns an additive mask for padding tokens.

    Positions with padding token index set to -inf are not attended to.
    Positions with padding token index set to 0 are attended to.

    Args:
        inputs (Array): a 2D array of shape (batch_size, seq_len).
        padding_token_index (int, optional): index of the padding token. Defaults to 0.
        dtype (str | jnp.dtype, optional): dtype of the mask array. Defaults to 'float32'.

    Returns:
        Array: a 2D array of shape (batch_size, seq_len) with -inf on the padding tokens.
    """

    if inputs.ndim != 2:
        raise ValueError("Expected inputs to be a 2D tensor.")

    # will have zeros at positions with padding token
    inverse_mask = inputs != padding_token_index

    # will have -inf at positions with padding token
    return jnp.log(inverse_mask).astype(dtype)
