from collections import UserDict
from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as np

from ..utils.mask import create_lookahead_mask, create_padding_mask


class JaxBatch(UserDict):
    def device_put(self, device: str) -> "JaxBatch":
        for key, value in self.items():
            if isinstance(value, jax.Array):
                self[key] = jax.device_put(value, device)


def collate_for_clm(examples: list[dict[str, list]], key: str = "input_ids") -> Array:
    # assume that examples have already been padded to be of equal lengths.
    # pre-allocate the batch
    n_examples = len(examples)
    example_length = len(examples[0][key])

    # since clm use int
    batch = np.zeros((n_examples, example_length)).astype(np.int32)

    # assumes examples are coming from HF `transformers` tokenizer
    for i, example in enumerate(examples):
        batch = batch.at[i, :].set(example[key])

    return batch


class CollatorForCausalLM:
    def __init__(self, tokenizer, include_special_tokens_mask: bool = False):
        self.tokenizer = tokenizer
        self.include_special_tokens_mask = include_special_tokens_mask

    def __call__(self, examples: list[dict[str, list]]) -> JaxBatch:
        input_batch = collate_for_clm(examples, key="input_ids")
        labels = input_batch.copy()

        # the model will handle re-shaping the masks as needed, keep them 2D here
        lookahead_mask = create_lookahead_mask(input_batch.shape[-1])
        padding_mask = create_padding_mask(input_batch, self.tokenizer.pad_token_id)

        batch = JaxBatch(
            inputs=input_batch,
            labels=labels,
            lookahead_mask=lookahead_mask,
            padding_mask=padding_mask,
        )

        # only designates padding tokens for now
        if self.include_special_tokens_mask:
            special_tokens_mask = labels == self.tokenizer.pad_token_id
            batch.update({"special_tokens_mask": special_tokens_mask})

        return batch
