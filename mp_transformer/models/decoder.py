"""Adapted from Marsot et. al. (2022)"""
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn


def apply_rigid_transformation(sequence, transformation):
    """A learned rigid transformation for translation/rotation invariance.

    Latent space should capture intrinsic movement not
    orientation in space. With 2D toy limb simply adjust for root joint angle
    to make the toy limb representation rotation agnostic.
    """
    ret = sequence.clone()
    ret[..., 0] += transformation.squeeze(-1)
    return ret


class MovementPrimitiveDecoder(pl.LightningModule):
    """Multiple decoders for various learned features needed for reconstruction."""

    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()  # PyTorch Lightning

        self.pose_dim = config["pose_dim"]
        self.num_primitives = config["num_primitives"]
        self.latent_dim = config["latent_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_transformer_layers = config["num_transformer_layers"]
        self.feat_time = config["num_primitives"]
        # self.feat_time = 8  # hardcoded in Marsot et al.
        self.slope = config["masking_slope"]
        self.hidden_dim = config["hidden_dim"]
        self.learn_segmentation = config["learn_segmentation"]

        # Decodes latent primtives and timestamps into subsequences of poses
        self.decoder = nn.Sequential(
            # self.feat_time: time feature dimension
            nn.Linear(self.latent_dim + 2 * self.feat_time, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.pose_dim),
            nn.Sigmoid(),
        )

        # Learns intermediate features from positionally encoded latent primitives
        implicit_decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=self.num_attention_heads
        )
        self.implicit_decoder = torch.nn.TransformerEncoder(
            implicit_decoder_layer, num_layers=self.num_transformer_layers
        )

        # Learns the duration of each Movement Primitive subsequence
        self.duration_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

        # Learns a rigid transformation to adjust for rotation/translation
        self.rigid_transformation_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
        )

        # Intial positional encoding
        # TODO: consider moving where this is initialized
        # TODO: reuse .encoder.PositionalEncodingLayer?
        self.init_positional_encoding = self.get_positional_encoding(
            self.num_primitives, self.latent_dim
        )

    def create_gaussian_masks(self, cumsum_and_durations, timestamps, eps=1e-7):
        """Creates a gaussian mask for each subsequence.

        This function generates a Gaussian mask (bell curve) based on the given durations
        and current time step. This mask can be used to weight parts of a sequence differently
        depending on their position.
        """
        # Calculate the midpoint and half length of each subsequence
        durations_accum = cumsum_and_durations[..., 0:1]  # Cumulative sum of durations
        durations = cumsum_and_durations[..., 1:2]  # slicing keeps last dim
        mid_points = durations_accum + durations / 2
        half_lengths = durations / 2 + eps  # avoid division by 0

        # Normalize the current time based on the midpoint and half length
        normalized_timestamps = (timestamps - mid_points) / half_lengths

        # Apply a Gaussian function to the normalized time
        gaussian_mask = torch.exp(-self.slope * normalized_timestamps**2)
        return gaussian_mask

    def masked_average(self, subseqs, transformation, masks, eps=1e-7):
        """Applies gaussian masks to the subsequences and averages them."""
        adjusted_subseqs = apply_rigid_transformation(subseqs, transformation)
        masked = adjusted_subseqs * masks
        summed = torch.sum(masked, dim=1)
        normed = summed / (torch.sum(masks, dim=1) + eps)  # avoid division by 0
        return normed

    def get_positional_encoding(self, num_primitives, latent_dim):
        """Get positional encoding for the latent primitives."""
        p_enc = torch.zeros((num_primitives, latent_dim))
        for pos in range(num_primitives):
            for i in range(0, latent_dim, 2):
                p_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / latent_dim)))
                p_enc[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / latent_dim))
                )

        return p_enc.unsqueeze(1)

    def get_time_encoding(self):
        """Get time encoding for the latent primitives."""
        return 2 ** (torch.arange(0, self.feat_time)) * torch.pi

    def compute_segment_transformations(self, latent_primitives):
        """Learns rigid transformation and duration of each subsequence."""
        batch_size = latent_primitives.shape[0]

        positional_encoding = self.init_positional_encoding.repeat(1, batch_size, 1).to(
            latent_primitives.device
        )
        # Add positional encoding to the latent primitives
        positional_latent_primitives = latent_primitives.transpose(
            0, 1
        ) + positional_encoding.to(latent_primitives.device)

        # Obtain features from the implicit decoder transformer
        intermediate_features = self.implicit_decoder(
            positional_latent_primitives
        ).transpose(0, 1)

        # If learn_segmentation is true, use the segmentation_adjust module to compute segment parameters
        if self.learn_segmentation:
            durations = self.duration_decoder(intermediate_features)
        # Otherwise, make segments equal length
        else:
            uniform_distribution = torch.ones(
                (
                    latent_primitives.shape[0],
                    latent_primitives.shape[1],
                    1,
                ),
                requires_grad=False,
            )
            durations = uniform_distribution / self.num_primitives
            durations = durations.to(latent_primitives.device)

        # Create a tensor with zeros to accumulate segment durations along the time dimension
        zeros_like_durations = torch.zeros_like(durations[..., 0:1, :])
        durations_no_last = durations[..., :-1, :]
        durations_cumsum = torch.cumsum(durations_no_last, dim=1)

        # Concatenate zeros and accumulated segment durations
        durations_accum = torch.cat([zeros_like_durations, durations_cumsum], dim=1)

        # Concatenate accumulated segment durations and original segment durations
        cumsum_and_durations = torch.cat([durations_accum, durations], dim=-1)

        rigid_transformation = self.rigid_transformation_decoder(intermediate_features)
        return {
            "cumsum_and_durations": cumsum_and_durations,
            "rigid_transformation": rigid_transformation,
        }

    def forward(self, timestamps, latent_primitives):
        """Compute features and positional encoding and feed them to the decoder."""
        out = self.compute_segment_transformations(latent_primitives)
        cumsum_and_durations, rigid_transformation = (
            out["cumsum_and_durations"],
            out["rigid_transformation"],
        )

        batch_size, num_timestamps = timestamps.shape[0], timestamps.shape[1]

        # Repeat timestamps for each primitive
        repeated_timestamps = (
            timestamps.unsqueeze(1).unsqueeze(3).repeat(1, self.num_primitives, 1, 1)
        )

        # Repeat latent primitives for each timestamp
        repeated_latents = latent_primitives.unsqueeze(2).repeat(
            1, 1, num_timestamps, 1
        )

        # Repeat segment parameters for each timestamp
        repeated_cumsum_and_durations = cumsum_and_durations.unsqueeze(2).repeat(
            1, 1, num_timestamps, 1
        )

        # accumulated durations
        repeated_durations_accum = repeated_cumsum_and_durations[..., 0:1]
        # Calculate time difference between repeated timestamps and segment parameters
        time_diff = repeated_timestamps - repeated_durations_accum

        # zero out the past
        # warped_timestamps = nn.ReLU()(time_diff)  # TODO: readup on nn. vs F.
        warped_timestamps = time_diff

        ## debug = torch.cat([repeated_timestamps, repeated_durations_accum, time_diff, warped_timestamps], dim=-1)

        # Repeat and tile the time encoding coefficients
        # TODO: check self.feat_time=8 in get_time_encoding
        coefs = (
            self.get_time_encoding()
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(timestamps.device)
        )
        coefs = coefs.repeat(batch_size, self.num_primitives, num_timestamps, 1)

        # Calculate higher dimensional timestamps using the sine and cosine of the warped timestamps
        higher_dim_timestamp = torch.cat(
            [
                torch.sin(coefs * warped_timestamps.repeat(1, 1, 1, self.feat_time)),
                torch.cos(coefs * warped_timestamps.repeat(1, 1, 1, self.feat_time)),
            ],
            dim=-1,
        )

        # Repeat global transformations for each timestamp
        repeated_transfo = rigid_transformation.unsqueeze(2).repeat(
            1, 1, num_timestamps, 1
        )

        # Concatenate repeated latents and higher dimensional timestamps, and pass them through the decoder
        recons_subseqs = self.decoder(
            torch.cat([repeated_latents, higher_dim_timestamp], dim=-1)
        )

        # Create masks using repeated segment parameters and repeated timestamps
        gaussian_masks = self.create_gaussian_masks(
            repeated_cumsum_and_durations, repeated_timestamps
        )

        # Calculate the reconstructed sequence using masked average
        recons_sequence = self.masked_average(
            recons_subseqs, repeated_transfo, gaussian_masks
        )
        print(f"{recons_sequence.max()=}")
        print(f"{recons_sequence.min()=}")

        return {
            "recons_sequence": recons_sequence,
            "rigid_transformation": rigid_transformation,
            "gaussian_masks": gaussian_masks,
            "cumsum_and_durations": cumsum_and_durations,
            "recons_subseqs": recons_subseqs,
        }
