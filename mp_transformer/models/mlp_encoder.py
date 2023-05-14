"""For potential end-to-end training of MP-Transformer on toy data images."""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from mp_transformer.datasets.toy_dataset import ToyDataset
from mp_transformer.utils import render_side_by_side_images


class MultiLayerPeceptronEncoder(pl.LightningModule):
    def __init__(self, layers_sizes):
        super().__init__()
        self.save_hyperparameters()

        layers = nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layers_sizes[i + 1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)

        return self.layers[-1:](x)

    def train_loss(self, y, y_hat):
        return F.mse_loss(y, y_hat)

    def val_loss(self, y, y_hat):
        return self.train_loss(y, y_hat)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["pose"]
        x = torch.flatten(x, 1)  # flatten into a vector
        y_hat = self(x)
        loss = self.train_loss(y, y_hat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["pose"]
        x = torch.flatten(x, 1)  # flatten into a vector
        y_hat = self(x)
        loss = self.val_loss(y, y_hat)
        self.log("val_loss", loss)
        return loss

    def infer(self, x):
        x = torch.flatten(x, 1)
        y_hat = self(x)
        return y_hat


# Define the layer sizes for the MLP
layer_sizes = [4096, 1024, 64, 3]

# Create an instance of the model
model = MultiLayerPeceptronEncoder(layer_sizes)
train_dataset = ToyDataset()
val_dataset = ToyDataset(path="data/toy/val-set")
train_dataloader = DataLoader(
    train_dataset, batch_size=128, drop_last=True, num_workers=4
)
val_dataloader = DataLoader(val_dataset, batch_size=128, drop_last=True, num_workers=4)

# Create a PyTorch Lightning Trainer and fit the model
# wandb_logger = WandbLogger(project="mp-transformer", name="n1000")
# wandb_logger.watch(model)
# trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, log_every_n_steps=1, gpus=1)
# trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# imgs = render_side_by_side_images(val_dataset, model, n=12)
# wandb_logger.log_image(key="ground truth vs prediction", images=imgs)

# # Log the model file to wandb
# torch.save(model.state_dict(), "model.pt")
# wandb.save("model.pt")
