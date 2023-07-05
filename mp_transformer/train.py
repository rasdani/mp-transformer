"""Setups and runs model training. Logging is disabled if --no-log flag is set."""
import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from mp_transformer.config import CONFIG
from mp_transformer.datasets import ToyDataset
from mp_transformer.models import MovementPrimitiveTransformer
from mp_transformer.utils import save_generation_video, save_side_by_side_strip

CUDA_AVAILABLE = torch.cuda.is_available()


def setup(config):
    """Setup model and datasets."""
    model = MovementPrimitiveTransformer(config)
    train_dataset = ToyDataset(
        return_segments=True,
        sequence_length=config["sequence_length"],
        N=config["N_train"],
    )
    val_dataset = ToyDataset(
        path="data/toy/val-set/",
        return_segments=True,
        sequence_length=config["sequence_length"],
        N=config["N_val"],
    )

    return model, train_dataset, val_dataset


def setup_wandb(config=None, run=None):
    """Setup Weights & Biases logging."""
    if "run_name" in config.keys():  # A run, not a sweep
        run_name = config["run_name"]
    else:
        run_name = "MP-Transformer"
    wandb_logger = WandbLogger(
        # project="mp-transformer", name=run_name, experiment=run, log_model="all"
        project="mp-transformer",
        name=run_name,
        experiment=run,
    )
    if config is not None:
        wandb_logger.experiment.config.update(config)
    os.makedirs("tmp", exist_ok=True)
    return wandb_logger


def log_to_wandb(config, model, val_dataset, model_path):
    """Log results to Weights & Biases after training."""
    model_artifact = wandb.Artifact("model", type="model")
    model_artifact.add_file(model_path)
    # model_artifact.link("model-registry/mp-transformer")
    wandb.log_artifact(model_artifact)

    # Save multiple reconstruction examples
    for i, idx in enumerate([0, len(val_dataset) // 2, len(val_dataset) - 1]):
        item = val_dataset[idx]
        # Log videos of movement primitive subsequences
        save_side_by_side_strip(item, model, num_subseqs=config["num_primitives"])
        wandb.log({f"example{i + 1}": wandb.Video("tmp/comp_strip.mp4")})

    # Save generated movement examples
    for i in range(3):
        save_generation_video(model, path=f"tmp/gen_vid{i}.mp4")
        wandb.log(
            {f"Generations/generation{i + 1}": wandb.Video(f"tmp/gen_vid{i}.mp4")}
        )


def main(config, no_log=False, debug=False, checkpoint_path=None):
    """Initialize PyTorch Lightning Trainer and train the model."""
    model, train_dataset, val_dataset = setup(config)
    num_workers = 4 if not debug else 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        drop_last=True,
        num_workers=num_workers,
    )
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=config["batch_size"],
    #     drop_last=True,
    #     num_workers=num_workers,
    # )
    val_dataloader = train_dataloader
    gpus = 1 if CUDA_AVAILABLE else 0
    if no_log or debug:
        if debug:
            config["epochs"] = 1
        trainer = pl.Trainer(
            max_epochs=config["epochs"],
            gpus=gpus,
            logger=False,
            enable_checkpointing=False,
        )
    else:  # Log normal training run
        wandb_logger = setup_wandb(config=config)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", save_top_k=1, filename="model"
        )
        # checkpoint_callback = ModelCheckpoint()
        trainer = pl.Trainer(
            max_epochs=config["epochs"],
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
            gpus=gpus,
            resume_from_checkpoint=checkpoint_path,
            num_sanity_val_steps=0,
        )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    if not no_log and not debug:
        model_path = checkpoint_callback.best_model_path
        log_to_wandb(config, model, val_dataset, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--debug", action="store_true", help="Disable logging for debugging"
    )
    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    args = parser.parse_args()
    main(CONFIG, no_log=args.no_log, debug=args.debug, checkpoint_path=args.checkpoint)
