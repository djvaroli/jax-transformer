from pathlib import Path
from typing import Callable, Iterable, Literal

import jax
import numpy as np
import optax
import tqdm
from flax.core.scope import FrozenVariableDict
from flax.training import checkpoints, train_state
from jax import Array

from wheeljax.model import TransformerLM

from .collator import JaxBatch


# TODO: using LM-specific trainer for speed, need to create abstractions
class LMTrainer:
    def __init__(
        self,
        model: TransformerLM,
        example_batch: JaxBatch,
        max_iters: int,
        lr: float = 1e-3,
        warmup: int = 100,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints",
        report_to: Literal["wandb"] | None = None,
    ):
        """
        Inputs:
            model_name - Name of the model. Used for saving and checkpointing
            exmp_batch - Example batch to the model for initialization
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            seed - Seed to use for model init
            log_to: Where to log training progress. Currently only supports "wandb" or no logging.

        """
        super().__init__()
        self.model = model
        self.max_iters = max_iters

        self.lr = lr
        self.warmup = warmup
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

        # Create empty model. Note: no parameters yet
        self.model = model

        # init train state
        state, rng = self.init_model(example_batch)
        self.state = state
        self.rng = rng

        # Create jitted training and eval functions
        train_step, eval_step = self.create_functions()
        self.train_step = train_step
        self.eval_step = eval_step

        # metric history tracker
        self.history = {
            "train": {"loss": [], "perplexity": []},
            "val": {"loss": [], "perplexity": []},
        }

        self.report_to = report_to

    def get_loss_function(
        self,
    ) -> Callable[
        [
            FrozenVariableDict | dict[str, Array],
            jax.random.PRNGKey,
            bool,
            dict[str, Array],
        ],
        tuple[Array, Array, jax.random.PRNGKey],
    ]:
        def compute_batch_loss(
            params: FrozenVariableDict | dict[str, Array],
            rng: jax.random.PRNGKey,
            train: bool,
            **inputs,
        ) -> tuple[Array, Array, jax.random.PRNGKey]:
            rng, dropout_apply_rng = jax.random.split(rng, 2)

            # expected shape (batch_size, seq_len)
            labels: Array = inputs.pop("labels")

            # special token mask indicates which positions are padding
            # expected shape (batch_size, seq_len)
            special_token_mask = inputs.pop("special_token_mask", None)

            # expected shape (batch_size, seq_len, vocab_size)
            logits: Array = self.model.apply(
                {"params": params},
                **inputs,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )

            # mask out logits for special tokens
            if special_token_mask is not None:
                logits = logits * special_token_mask

            # shift by one to predict next token
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # flatten logits and labels (bs, sl, vs) -> (bs * sl, vs)
            # and (bs, sl) -> (bs * sl, )
            shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.reshape(-1)

            # loss batch can contain nans (e.g. if an index is set to -100)
            loss_batch = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )

            loss = jax.numpy.nanmean(loss_batch)
            ppl = jax.numpy.exp(loss)

            return loss, (ppl, rng)

        return compute_batch_loss

    def init_model(self, example_batch: JaxBatch) -> None:
        """Initialize model.

        Args:
            example_batch (example_batch): a dictionary with the following keys:
                "inputs", "labels", "lookahead_mask", "padding_mask"
        """
        # Initialize model
        rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)

        params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            **example_batch,
            train=True,
        )["params"]

        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=self.warmup,
            decay_steps=self.max_iters,
            end_value=0.0,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adam(lr_schedule),
        )

        # Initialize training state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

        return state, rng

    def create_functions(self) -> tuple[Callable, Callable]:
        # TODO: hard to read, simplify
        compute_batch_loss: Callable[
            [
                FrozenVariableDict | dict[str, Array],
                jax.random.PRNGKey,
                bool,
                dict[str, Array],
            ],
            tuple[Array, Array, jax.random.PRNGKey],
        ] = self.get_loss_function()

        def train_step(
            state: train_state.TrainState, rng: jax.random.PRNGKey, **inputs: Array
        ) -> tuple[train_state.TrainState, jax.random.PRNGKey, Array, Array]:
            # TODO: return perplexity here maybe
            loss_fn = lambda params: compute_batch_loss(
                params, rng, train=True, **inputs
            )
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

            # ret is a tuple(loss, (aux_data,))
            loss, ppl, rng = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, ppl

        # TODO: not working and not implemented
        def val_step(rng: jax.random.PRNGKey, **inputs: Array):
            loss, ppl, rng = compute_batch_loss(
                self.state.params, rng, train=False, **inputs
            )
            return loss, ppl, rng

        return jax.jit(train_step), jax.jit(val_step)

    def train_epoch(
        self,
        train_loader: Iterable[JaxBatch],
        wandb_run=None,
    ):
        with tqdm.tqdm(total=len(train_loader), leave=False) as pbar:
            for batch in train_loader:
                self.state, self.rng, loss, ppl = self.train_step(
                    self.state, self.rng, **batch
                )
                self.history["train"]["loss"].append(loss.item())
                self.history["train"]["perplexity"].append(ppl.item())

                pbar.set_postfix(loss=loss.item(), perplexity=ppl.item())
                pbar.update(1)
                if wandb_run is not None:
                    wandb_run.log(
                        {"train/loss": loss.item(), "train/perplexity": ppl.item()}
                    )

    def train(self, n_epochs: int, train_loader: Iterable[JaxBatch]):
        per_epoch_steps = len(train_loader)

        wandb_run = None
        if self.report_to == "wandb":
            import wandb

            wandb_run = wandb.init(project="bumble-jax")

        with tqdm.tqdm(total=n_epochs) as pbar:
            for epoch in range(n_epochs):
                pbar.set_description(f"Epoch {epoch + 1} / {n_epochs}")
                self.train_epoch(train_loader, wandb_run=wandb_run)
                epoch_losses = self.history["train"]["loss"][
                    epoch * per_epoch_steps : (epoch + 1) * per_epoch_steps
                ]
                epoch_perplexity = self.history["train"]["perplexity"][
                    epoch * per_epoch_steps : (epoch + 1) * per_epoch_steps
                ]
                mean_epoch_loss = np.mean(epoch_losses)
                mean_epoch_perplexity = np.mean(epoch_perplexity)
                pbar.set_postfix(loss=mean_epoch_loss, perplexity=mean_epoch_perplexity)
                pbar.update(1)

    def save_model(self, step: int = 0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=str(self.checkpoint_dir), target=self.state.params, step=step
        )

    def load_model(self):
        params = checkpoints.restore_checkpoint(
            str(self.checkpoint_dir), target=self.state.params
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.state.tx
        )
