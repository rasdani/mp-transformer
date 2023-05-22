"""Hyperparameter optimization with wandb sweep."""
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import wandb
from mp_transformer.config import SWEEP_CONFIG
from mp_transformer.train import setup, setup_wandb


def main():
    run = wandb.init()
    config = wandb.config

    model, train_dataset, val_dataset = setup(config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        drop_last=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        drop_last=True,
        num_workers=4,
    )

    wandb_logger = setup_wandb(model, run=run)
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=wandb_logger,
        log_every_n_steps=1,
        gpus=1,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


sweep_id = wandb.sweep(SWEEP_CONFIG, project="mp-transformer")
# wandb.agent(sweep_id, main, count=40)
wandb.agent(sweep_id, main)
