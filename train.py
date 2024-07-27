"""Training script for the model."""

import argparse

import lightning as L  # noqa: N812
from lightning.pytorch.loggers import WandbLogger

import torch
from torch.nn import functional as F  # noqa: N812

from utils.model import MultiTokenizerLLM
from utils.utils import get_dataloaders, load_config


class MultiTokenizerModel(L.LightningModule):
    """LightningModule for the model."""

    def __init__(self, config: argparse.Namespace) -> None:
        """Initialize the model."""
        super().__init__()
        self.config = config
        self.model = MultiTokenizerLLM(**vars(config.model_params))

    @staticmethod
    def loss_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss function."""
        y_hat_loss_view = y_hat.view(-1, config.model_params.vocab_size)
        y_loss_view = y.view(-1)
        return F.cross_entropy(y_hat_loss_view, y_loss_view)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self) -> tuple:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.training_args.learning_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss


def train(config: argparse.Namespace, last_checkpoint: str | None = None) -> None:
    """Train the model."""
    train_dataloader, val_dataloader = get_dataloaders(config)
    wandb_logger = WandbLogger(project="multi-tokenizer-llm")
    model = MultiTokenizerModel(config)
    trainer = L.Trainer(
        logger=wandb_logger,
        fast_dev_run=True,
        max_steps=config.training_args.max_steps,
        # TODO: Add more trainer arguments
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=last_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
