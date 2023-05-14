"""Adapted from Marsot et. al. (2022)"""
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn


class PositionalEncodingLayer(pl.LightningModule):
    """This module creates a conventional positional encoding for transformers
    with timestamps as positional tokens instead."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config["latent_dim"]

        # sinusoidal frequencies for positional encoding
        # linearly spaced 1D tensor ranging from 0 to self.latent_dim a size of self.latent_dim // 2
        linear_space = torch.linspace(0, self.latent_dim, self.latent_dim // 2)
        normalized_space = linear_space / self.latent_dim
        # see thesis for advantages of exponential spacing, 10000 is used in Vaswani et al.
        exponential_space = 10000**normalized_space
        # divide 2*pi by the exponential tensor for a tensor of sinusoidal frequencies
        # frequencies decrease as position index increases
        self.freqs = 2 * torch.pi / exponential_space
        # add batch dimension
        self.freqs = self.freqs.unsqueeze(0)

    def forward(self, timestamps):
        """Takes a series of BSxN timestamps as input, return a BSxNxD positional encoding
        BS, batch size, N : number of timestamps, D : latent dimension

        Args:
            timestamps (torch.tensor): BSxN tensor of normalized timestamps

        Returns:
            torch.tensor : positional encoding of dimension BSxNxD
        """
        timestamps = timestamps.unsqueeze(-1)  # (batch_size, num_timestamps, 1)
        freqs = self.freqs.unsqueeze(0)  # (1, latent_dim // 2)

        # Repeat the frequencies along the first two dimensions
        # to match the batch size and number of timestamps
        freqs_repeated = freqs.repeat(timestamps.shape[0], timestamps.shape[1], 1).to(
            self.device
        )  # (batch_size, num_timestamps, latent_dim // 2)

        # Multiply the expanded timestamp tensor with the repeated frequency tensor
        positional_freqs = (
            timestamps * freqs_repeated
        )  # Shape: (batch_size, num_timestamps, latent_dim // 2)

        # element-wise cosine
        cosines = torch.cos(
            positional_freqs
        )  # (batch_size, num_timestamps, latent_dim // 2)
        # element-wise sine
        sines = torch.sin(
            positional_freqs
        )  # (batch_size, num_timestamps, latent_dim // 2)

        # Concatenate along the last dimension
        positional_encoding = torch.cat(
            [cosines, sines], dim=-1
        )  # (batch_size, num_timestamps, latent_dim)
        return positional_encoding


class MovementPrimitiveEncoder(pl.LightningModule):
    """Maps a sequence of pose parameters (i.e.joint angles) to a sequence of latent primitives."""

    def __init__(self, config):
        super().__init__()

        self.pose_dim = config["pose_dim"]
        self.num_primitives = config["num_primitives"]
        self.latent_dim = config["latent_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_transformer_layers = config["num_transformer_layers"]
        self.num_primitives = config["num_primitives"]

        self.positional_encoding = PositionalEncodingLayer(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=self.num_attention_heads
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim, nhead=self.num_attention_heads
        )
        self.encoder_segments = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.num_transformer_layers
        )
        self.decoder_segments = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=self.num_transformer_layers
        )

        self.embedding = nn.Linear(self.pose_dim, self.latent_dim)

        self.mean_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.logvar_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # positional encoding used as input for the transformer decoder
        # TODO: keep or move?
        self.initial_encoding = self.get_positional_encoding(
            self.num_primitives, self.latent_dim
        )

    # TODO: use PositionalEncodingLayer instead?
    def get_positional_encoding(self, num_primitives, latent_dim):
        p_enc = torch.zeros((num_primitives, latent_dim))
        for pos in range(num_primitives):
            for i in range(0, latent_dim, 2):
                p_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / latent_dim)))
                p_enc[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / latent_dim))
                )

        p_enc = p_enc.unsqueeze(1)

        return p_enc

    def compute_primitive_features(self, poses, timestamps):
        """Compute features which characterize the sequence of primitive distribution using the transformer

        Args:
            poses (torch.tensor): tensor of size BSxNxD of pose (and translation) parameters
            with N number timestamps and D frame dim

        Returns:
            torch.tensor: tensor of size BSxMxD of features with M the number of primitives
        """

        # Get the batch size from the input data
        batch_size = poses.shape[0]

        # Pass the frame representations through the embedding layer
        pose_embeddings = self.embedding(poses)

        # Add the positional encoding to the projections based on the input timestamps
        embeddings = pose_embeddings + self.positional_encoding(timestamps)

        # The PyTorch transformer expects the batch size on the second dimension, so transpose the tensor
        embeddings = embeddings.transpose(0, 1)

        # Pass the transposed tensor through the encoder segments of the transformer
        context = self.encoder_segments(embeddings)

        # Create the initial input for the decoder segments
        initial_target = self.initial_encoding.repeat(1, batch_size, 1).to(self.device)

        # Pass the initial input and the memory from the encoder segments through the decoder segments
        primitive_features = self.decoder_segments(tgt=initial_target, memory=context)

        # Transpose the primitive features tensor to have the batch size on the first dimension
        primitive_features = primitive_features.transpose(0, 1)

        return primitive_features

    def forward(self, poses, timestamps):
        """Encode an input motion to a series of primitives
        in latent space

        Args:
            poses (torch.tensor): tensor of size BSxNxD of pose (and translation) parameters with N number timestamps and D frame dim
            timestamps (torch.tensor): tensor of size BSxN of timestamps
        Returns:
            torch.tensor: tensor of size BSxMxD of latent primitives with M the number of primitives
            torch.tensor: tensor of size BSxMxD of mean of the latent primitives
            torch.tensor: tensor of size BSxMxD of logvar of the latent primitives
        """
        primitive_features = self.compute_primitive_features(poses, timestamps)

        mus = self.mean_encoder(primitive_features)
        logvars = self.logvar_encoder(primitive_features)

        latent_primitives = self.reparameterize(mus, logvars)
        return {"latent_primitives": latent_primitives, "mus": mus, "logvars": logvars}

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon
