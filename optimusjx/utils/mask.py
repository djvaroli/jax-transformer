from jax import Array
from jax import numpy as jnp


def create_lookahead_mask(length: int, dtype: str | jnp.dtype = "float32") -> Array:
    """Creates an additive lookahead attention mask based on the shape of the inputs.
    Positions that should not be attended to will contain -inf. Unmasked positions will contain 0.

    Args:
        length (int): the length of the side of the the square lookahead mask.
        dtype (str | jnp.dtype): dtype of the mask array. Defaults to 'float32'.

    Returns:
        Array: a 2D array of shape (length, length).

    Example:
        >>> import jax.numpy as np
        >>> mask = create_lookahead_mask(3)
        >>> print(mask)
        >>> Array([[  0., -inf, -inf],
        >>>        [  0.,   0., -inf],
        >>>        [  0.,   0.,   0.]], dtype=float32)
    """
    mask_shape = (length, length)

    # initialize binary mask with 0s on the upper triangular part
    binary_mask = jnp.tril(jnp.ones(mask_shape, dtype=dtype))

    # Using log on the lower triangular part of the mask to set upper triangular to -inf
    return jnp.log(binary_mask)


def create_padding_mask(
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
    inverse_mask = (inputs != padding_token_index).astype(dtype)

    # will have -inf at positions with padding token
    return jnp.log(inverse_mask).astype(dtype)
