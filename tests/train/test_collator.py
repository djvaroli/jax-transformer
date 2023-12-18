from dataclasses import dataclass

from optimusjx.train.collator import CollatorForCausalLM


@dataclass
class DummyTokenizer:
    pad_token_id: int = 0


def test_clm_collator():
    """Test CollatorForCausalLM"""
    tokenizer = DummyTokenizer()
    collator = CollatorForCausalLM(tokenizer)

    inputs = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [1, 2, 3, 4]}]

    jax_batch = collator(inputs)

    assert "inputs" in jax_batch, "inputs should be in jax_batch"
    assert "labels" in jax_batch, "labels should be in jax_batch"
    assert "lookahead_mask" in jax_batch, "lookahead_mask should be in jax_batch"
    assert "padding_mask" in jax_batch, "padding_mask should be in jax_batch"

    assert jax_batch["inputs"].shape == (2, 4)
    assert jax_batch["labels"].shape == (2, 4)
    assert jax_batch["lookahead_mask"].shape == (4, 4)
    assert jax_batch["padding_mask"].shape == (2, 4)
