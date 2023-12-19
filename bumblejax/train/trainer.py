import os
from copy import deepcopy
from typing import Callable, Iterable, Literal

import jax
import numpy as np
import optax
import tqdm
import wandb
from flax.core.scope import FrozenVariableDict
from flax.training import train_state
from jax import Array
from wandb import wandb_sdk

from bumblejax.model import TransformerLM

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

        self.history = {"train_loss": [], "val_loss": []}
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
        tuple[float, jax.random.PRNGKey],
    ]:
        # Return a function that calculates the loss for a batch
        # To be implemented in a task specific sub-class

        def compute_batch_loss(
            params: FrozenVariableDict | dict[str, Array],
            rng: jax.random.PRNGKey,
            train: bool,
            **inputs,
        ) -> tuple[float, jax.random.PRNGKey]:
            rng, dropout_apply_rng = jax.random.split(rng, 2)

            # expected shape (batch_size, seq_len)
            labels: Array = inputs.pop("labels")

            # special token mask indicates which positions are padding
            special_token_mask = inputs.pop("special_token_mask", None)

            # expected shape (batch_size, seq_len, vocab_size)
            logits: Array = self.model.apply(
                {"params": params},
                **inputs,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )

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
            return loss, rng

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
            tuple[float, jax.random.PRNGKey],
        ] = self.get_loss_function()

        def train_step(
            state: train_state.TrainState, rng: jax.random.PRNGKey, **inputs: Array
        ) -> tuple[train_state.TrainState, jax.random.PRNGKey, float]:
            # TODO: return perplexity here maybe
            loss_fn = lambda params: compute_batch_loss(
                params, rng, train=True, **inputs
            )
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, rng = ret
            state = state.apply_gradients(grads=grads)
            return state, rng, loss

        def val_step(rng: jax.random.PRNGKey, **inputs: Array):
            loss, rng = compute_batch_loss(
                self.state.params, rng, train=False, **inputs
            )
            return loss, rng

        return jax.jit(train_step), jax.jit(val_step)

    def train_epoch(
        self,
        train_loader: Iterable[JaxBatch],
        wandb_run: wandb_sdk.wandb_run.Run | None = None,
    ):
        with tqdm.tqdm(total=len(train_loader), leave=False) as pbar:
            for batch in train_loader:
                self.state, self.rng, loss = self.train_step(
                    self.state, self.rng, **batch
                )
                self.history["train_loss"].append(loss.item())

                pbar.set_postfix(loss=loss)
                pbar.update(1)
                if wandb_run is not None:
                    wandb_run.log({"train_loss": loss.item()})

    def train(self, n_epochs: int, train_loader: Iterable[JaxBatch]):
        per_epoch_steps = len(train_loader)

        wandb_run = None
        if self.report_to == "wandb":
            wandb_run = wandb.init(project="bumble-jax")

        with tqdm.tqdm(total=n_epochs) as pbar:
            for epoch in range(n_epochs):
                pbar.set_description(f"Epoch {epoch + 1} / {n_epochs}")
                self.train_epoch(train_loader, wandb_run=wandb_run)
                epoch_losses = self.history["train_loss"][
                    epoch * per_epoch_steps : (epoch + 1) * per_epoch_steps
                ]
                mean_epoch_loss = np.mean(epoch_losses)
                pbar.set_postfix(loss=mean_epoch_loss)
                pbar.update(1)
