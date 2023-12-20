import tempfile

import jax
import torch
from conftest import TestConfig
from torch.utils.data import DataLoader, Dataset

from wheeljax.model import TransformerLM
from wheeljax.train import CollatorForCausalLM, LMTrainer


class RandomIntDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        n_samples: int = 10,
        seed: int = 42,
        padding_amount: int | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.rng = jax.random.PRNGKey(seed)

        # could create samples at __getitem__ call instead
        self._data = jax.random.randint(
            self.rng, (n_samples, seq_len), minval=0, maxval=self.vocab_size
        )

        self.padding_amount = padding_amount
        if self.padding_amount is not None:
            self.pad_token_id = self.vocab_size
            self.vocab_size += 1

            padding = jax.numpy.full(
                (n_samples, self.padding_amount), self.pad_token_id
            )
            self._data = jax.numpy.concatenate([self._data, padding], axis=-1)

    def __getitem__(self, index) -> dict[str, list]:
        if index > self.n_samples - 1:
            raise ValueError("Index larger than length.")

        return {"input_ids": self._data[index, :].tolist()}

    def __len__(self) -> int:
        return self.n_samples


class TokenizerStandin:
    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id


test_config = TestConfig()


def test_clm_trainer():
    """Tests basic operations of CLM Trainer."""
    dataset = RandomIntDataset(
        test_config.seq_len, vocab_size=3, n_samples=2, padding_amount=2
    )

    rng = torch.Generator()
    rng.manual_seed(test_config.random_seed)

    collator = CollatorForCausalLM(TokenizerStandin())

    train_loader = DataLoader(
        dataset, batch_size=test_config.batch_size, generator=rng, collate_fn=collator
    )

    model = TransformerLM(vocab_size=dataset.vocab_size)

    test_batch = next(iter(train_loader))

    with tempfile.TemporaryDirectory() as tmpdir:
        # TODO: test special tokens masking
        trainer = LMTrainer(
            model,
            example_batch=test_batch,
            max_iters=2,
            warmup=1,
            checkpoint_dir=tmpdir,
        )
        trainer.train(1, train_loader)
        trainer.save_model()
        trainer.load_model()

    model.generate(
        trainer.state.params,
        input_tokens=test_batch["inputs"][:1, :],
        max_new_tokens=2,
    )
