"""Setups and runs model training. Logging is disabled if --no-log flag is set."""
import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from mp_transformer.config import CONFIG
from mp_transformer.datasets import ToyDataset
from mp_transformer.models import MovementPrimitiveTransformer
from mp_transformer.utils import save_side_by_side_subsequences, save_side_by_side_video


def setup(config):
    """Setup model, datasets, and dataloaders."""
    model = MovementPrimitiveTransformer(config)
    train_dataset = ToyDataset(
        return_segments=True,
        sequence_length=config["sequence_length"],
        N=config["N"],
    )
    val_dataset = ToyDataset(
        path="data/toy/val-set/",
        return_segments=True,
        sequence_length=config["sequence_length"],
        N=config["N"],
    )

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

    return model, train_dataloader, val_dataloader


def setup_wandb(config, model):
    """Setup Weights & Biases logging."""
    wandb_logger = WandbLogger(
        project="mp-transformer",
        name="MP-Transformer",
    )
    wandb_logger.watch(model)
    wandb_logger.experiment.config.update(config)
    return wandb_logger


def log_to_wandb(config, model, val_dataset):
    """Log results to Weights & Biases after training."""
    torch.save(model.state_dict(), "tmp/transformer.pt")
    wandb.save("tmp/transformer.pt")

    # Save multiple reconstruction examples
    for i, idx in enumerate([0, len(val_dataset) // 2, len(val_dataset) - 1]):
        # Log comparison video of whole reconstructed sequence
        save_side_by_side_video(val_dataset, model, dataset_idx=idx)
        wandb.log({f"example{i + 1}_masked_average": wandb.Video("tmp/comp_vid.mp4")})

        # Log videos of movement primitive subsequences
        save_side_by_side_subsequences(
            val_dataset, model, num_subseqs=config["num_primitives"], dataset_idx=idx
        )
        for j in range(config["num_primitives"]):
            wandb.log(
                {f"example{i + 1}_MP{j + 1}": wandb.Video(f"tmp/comp_vid{j}.mp4")}
            )


def main(config, no_log=False, debug=False):
    model, train_dataloader, val_dataloader = setup(config)
    if no_log or debug:
        if debug:
            config["epochs"] = 1
        trainer = pl.Trainer(
            max_epochs=config["epochs"],
            gpus=1,
            logger=False,
            enable_checkpointing=False,
        )
    else:
        wandb_logger = setup_wandb(config, model)
        trainer = pl.Trainer(
            max_epochs=config["epochs"],
            logger=wandb_logger,
            log_every_n_steps=1,
            gpus=1,
        )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    if not no_log and not debug:
        log_to_wandb(config, model, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--debug", action="store_true", help="Disable logging for debugging"
    )
    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    args = parser.parse_args()
    main(CONFIG, no_log=args.no_log, debug=args.debug)
