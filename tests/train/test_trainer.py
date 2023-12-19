import jax
import torch
from conftest import TestConfig
from torch.utils.data import DataLoader, Dataset

from optimusjx.model import TransformerLM
from optimusjx.train import CollatorForCausalLM, LMTrainer


class RandomIntDataset(Dataset):
    def __init__(
        self, seq_len: int, vocab_size: int, n_batches: int = 10, seed: int = 42
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_batches = n_batches
        self.rng = jax.random.PRNGKey(seed)

    def __getitem__(self, index) -> dict[str, list]:
        if index > self.n_batches - 1:
            raise ValueError("Index larger than length.")

        random_vocab = jax.random.randint(
            self.rng, (self.seq_len,), minval=0, maxval=self.vocab_size
        ).tolist()

        self.rng, _ = jax.random.split(self.rng, 2)

        return {"input_ids": random_vocab}

    def __len__(self) -> int:
        return self.n_batches


class TokenizerStandin:
    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id


test_config = TestConfig()


def test_clm_trainer():
    """Tests basic operations of CLM Trainer."""
    dataset = RandomIntDataset(test_config.seq_len, vocab_size=3, n_batches=2)

    rng = torch.Generator()
    rng.manual_seed(test_config.random_seed)

    collator = CollatorForCausalLM(TokenizerStandin())

    train_loader = DataLoader(
        dataset, batch_size=test_config.batch_size, generator=rng, collate_fn=collator
    )

    model = TransformerLM(vocab_size=dataset.vocab_size)

    test_batch = next(iter(train_loader))
    trainer = LMTrainer(model, example_batch=test_batch, max_iters=101, warmup=100)

    trainer.train(1, train_loader)
