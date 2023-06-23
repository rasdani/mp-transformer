"""Adapted from Marsot et. al. (2022)"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from mp_transformer.models import MovementPrimitiveDecoder, MovementPrimitiveEncoder
from mp_transformer.models.decoder import apply_rigid_transformation


class MovementPrimitiveTransformer(pl.LightningModule):
    """This Transformer architecture maps a sequence of poses i.e movement to a
    (smaller) sequence of latent primitives. These act as low-dimensional
    building blocks of movement and are encoded as distributions in latent space.
    It then samples from these like a VAE and feeds them to a sequence decoder.
    After applying gaussian masks, the subsequences are averaged
    to form the final reconstructed movement sequence.
    """

    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()  # PyTorch Lightning

        self.encoder = MovementPrimitiveEncoder(config)
        self.decoder = MovementPrimitiveDecoder(config)

        self.num_primitives = self.encoder.num_primitives
        self.sequence_length = config["sequence_length"]
        self.kl_weight = config["kl_weight"]
        self.current_kl_weight = 0.0
        self.anneal_start = config["anneal_start"]
        self.anneal_end = config["anneal_end"]
        self.cycle_length = config["cycle_length"]
        self.durations_weight = config["durations_weight"]
        self.lr = config["lr"]  # learning rate
        self.best_val_loss = float("inf")

    def forward(self, poses, timestamps):
        """Recieves a sequence of latent primitives from the encoder and feeds
        them to the decoder."""
        enc_out = self.encoder(poses, timestamps)
        latent_primitives, mus, logvars = (
            enc_out["latent_primitives"],
            enc_out["mus"],
            enc_out["logvars"],
        )
        dec_out = self.decoder(timestamps, latent_primitives)
        (
            recons_sequence,
            rigid_transformation,
            gaussian_masks,
            cumsum_and_durations,
            recons_subseqs,
        ) = (
            dec_out["recons_sequence"],
            dec_out["rigid_transformation"],
            dec_out["gaussian_masks"],
            dec_out["cumsum_and_durations"],
            dec_out["recons_subseqs"],
        )
        return {
            "recons_sequence": recons_sequence,
            "rigid_transformation": rigid_transformation,
            "gaussian_masks": gaussian_masks,
            "mus": mus,
            "logvars": logvars,
            "cumsum_and_durations": cumsum_and_durations,
            "recons_subseqs": recons_subseqs,
        }

    def kl_divergence(self, mu, logvar):
        # TODO: decide on docstring format
        """Returns kl divergence between mu and logvar and a normal distribution.

        Args:
            mu (tensor): The means of the distribution
            logvar (tensor): The logarithm of the variance of the distribution

        Shape:
            mu : N x D with N the batch size
            logvar : N x D with N the batch size

        Returns:
            tensor: The averaged KL divergence over the entire batch
        """
        kl_divergences = -0.5 * (1 + logvar - mu**2 - logvar.exp())
        return torch.mean(kl_divergences)

    def pose_loss(self, gt, recons_sequence):
        """Global pose reconstruction loss."""
        return F.mse_loss(gt, recons_sequence)

    def align_ground_truth(self, gt, rigid_transformation, gaussian_masks):
        """So ground truth can be compared with single subsequences."""
        num_timestamps = gt.shape[1]
        repeated_gt = gt.unsqueeze(1).repeat(1, self.num_primitives, 1, 1)
        repeated_transfo = rigid_transformation.unsqueeze(2).repeat(
            1, 1, num_timestamps, 1
        )
        repeated_mask = gaussian_masks.repeat(1, 1, 1, gt.shape[-1])
        aligned_gt = apply_rigid_transformation(repeated_gt, repeated_transfo)
        return aligned_gt, repeated_mask

    def pose_loss_segments(
        self, gt, recons_subseqs, rigid_transformation, gaussian_masks
    ):
        """Per-segment pose reconstruction loss."""
        aligned_gt, repeated_mask = self.align_ground_truth(
            gt, rigid_transformation, gaussian_masks
        )
        masked_and_summed = torch.sum(
            (recons_subseqs - aligned_gt) ** 2 * repeated_mask, dim=1
        )
        return torch.mean(masked_and_summed)

    def kl_loss(self, means, logvars):
        """KL divergence between the latent primitives and a normal distribution."""
        self.log("kl_raw", self.kl_divergence(means, logvars))
        return self.kl_divergence(means, logvars) * self.current_kl_weight

    def durations_loss(self, cumsum_and_durations):
        """Penalizes deviating from equal length segments too much."""
        durations = cumsum_and_durations[..., 1]
        return (
            torch.mean((durations - 1 / self.num_primitives) ** 2)
            * self.durations_weight
        )

    def train_loss(
        self,
        gt,
        recons_sequence,
        rigid_transformation,
        gaussian_masks,
        mus,
        logvars,
        cumsum_and_durations,
        recons_subseqs,
    ):
        """Comprised of pose reconstruction loss (global and in subsequences),
        KL loss and segment duration loss."""
        l_pose = self.pose_loss(gt, recons_sequence)
        l_pose_segments = self.pose_loss_segments(
            gt, recons_subseqs, rigid_transformation, gaussian_masks
        )
        kl_loss = self.kl_loss(mus, logvars)
        l_durations = self.durations_loss(cumsum_and_durations)
        loss = l_pose + l_pose_segments + kl_loss + l_durations
        self.log("l_pose", l_pose)
        self.log("l_pose_segmentation", l_pose_segments)
        self.log("kl_segments", kl_loss)
        self.log("l_durations", l_durations)
        self.log("Diagnostics/min logvar", logvars.min())
        self.log("Diagnostics/max logvar", logvars.max())
        self.log("Diagnostics/median logvar", logvars.median())
        return loss

    def val_loss(self, gt, recons_sequence):
        """Validation loss is just global pose reconstruction loss."""
        return self.pose_loss(gt, recons_sequence)

    def training_step(self, batch, _):
        "Pytorch Lightning training step."
        poses, timestamps = (
            batch["poses"],
            batch["timestamps"],
        )
        out = self.forward(poses, timestamps)
        (
            recons_sequence,
            rigid_transformation,
            gaussian_masks,
            mus,
            logvars,
            cumsum_and_durations,
            recons_subseqs,
        ) = (
            out["recons_sequence"],
            out["rigid_transformation"],
            out["gaussian_masks"],
            out["mus"],
            out["logvars"],
            out["cumsum_and_durations"],
            out["recons_subseqs"],
        )
        loss = self.train_loss(
            gt=poses,
            recons_sequence=recons_sequence,
            rigid_transformation=rigid_transformation,
            gaussian_masks=gaussian_masks,
            mus=mus,
            logvars=logvars,
            cumsum_and_durations=cumsum_and_durations,
            recons_subseqs=recons_subseqs,
        )
        self.log("train_loss", loss)
        # joint_maxs, _ = torch.max(recons_subseqs.view(-1, 3), dim=0)
        # joint_mins, _ = torch.min(recons_subseqs.view(-1, 3), dim=0)
        # self.log("joint0_max", torch.max(joint_maxs[0]))
        # self.log("joint0_min", torch.min(joint_mins[0]))
        # self.log("joint1_max", torch.max(joint_maxs[1]))
        # self.log("joint1_min", torch.min(joint_mins[1]))
        # self.log("joint2_max", torch.max(joint_maxs[2]))
        # self.log("joint2_min", torch.min(joint_mins[2]))
        return loss

    def on_train_epoch_end(self):
        """Cyclical KL annealing with PyTorch Lightning."""
        epoch_in_cycle = (
            self.current_epoch % self.cycle_length
            if self.cycle_length
            else self.current_epoch
        )
        if epoch_in_cycle >= self.anneal_start:  # start annealing
            if epoch_in_cycle >= self.anneal_end:  # finish annealing
                self.current_kl_weight = self.kl_weight
            else:  # anneal
                period = self.anneal_end - self.anneal_start
                rate = (epoch_in_cycle - self.anneal_start) / period
                self.current_kl_weight = rate * self.kl_weight

    def validation_step(self, batch, _):
        "Pytorch Lightning validation step."
        poses, timestamps = (
            batch["poses"],
            batch["timestamps"],
        )
        out = self.forward(poses, timestamps)
        # TODO: simplify when you settle on val_loss
        recons_sequence = out["recons_sequence"]
        loss = self.val_loss(gt=poses, recons_sequence=recons_sequence)
        self.log("val_loss", loss)
        if loss < self.best_val_loss and self.current_epoch > self.anneal_end:
            self.best_val_loss = loss
        return loss

    def on_train_end(self):
        """Callback by PyTorch Lightning."""
        # log directly to wandb because PyTorch Lightning complains
        if self.logger is not None:
            self.logger.experiment.log({"best_val_loss": self.best_val_loss})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def infer(self, poses, timestamps):
        """Reconstructs a sequence of poses."""
        poses = poses.unsqueeze(0)
        timestamps = timestamps.unsqueeze(0)
        out = self.forward(poses, timestamps)
        return out["recons_sequence"]

    def infer_subsequence(self, poses, timestamps, subseq_idx=0, return_mask=False):
        """Returns a subsequence defined by a single movement primitive."""
        poses = poses.unsqueeze(0)
        timestamps = timestamps.unsqueeze(0)
        out = self.forward(poses, timestamps)
        ret = {"recons_subseq": out["recons_subseqs"][:, subseq_idx, ...]}
        if return_mask:
            ret["gaussian_mask"] = out["gaussian_masks"][:, subseq_idx, ...]
        return ret

    def generate(self):
        """Generates a sequence of poses."""
        random_latents = torch.randn(
            1, self.latent_dim, self.sequence_length, device=self.device
        )
        dec_out = self.decoder(random_latents)
        return dec_out["recons_sequence"]

    def complete(self, poses, timestamps, from_idx, to_idx=-1):
        """Completes a sequence of poses, filling in with random latents from from_idx to to_idx."""
        to_idx = (
            self.num_primitives if to_idx == -1 else to_idx
        )  # no empty slices with relative indices
        poses = poses.unsqueeze(0)
        print(f"poses range: [{poses.min()}, {poses.max()}]")
        random_poses = torch.rand_like(poses)
        timestamps = timestamps.unsqueeze(0)
        enc_out = self.encoder(poses, timestamps)
        gt_latents = enc_out["latent_primitives"]
        # mus, logvars = enc_out["mus"], enc_out["logvars"]
        random_out = self.encoder(random_poses, timestamps)
        mus, logvars = random_out["mus"], random_out["logvars"]
        random_latents = self.encoder.reparameterize(mus, logvars)
        print(f"mus range: [{mus.min()}, {mus.max()}]")
        print(f"average mu: {mus.mean()}")
        print(f"logvars range: [{logvars.min()}, {logvars.max()}]")
        print(f"median logvar: {logvars.median()}")
        print(f"gt_latents range: [{gt_latents.min()}, {gt_latents.max()}]")
        print(f"average gt_latents: {gt_latents.mean()}")
        print(f"random_latents range: [{random_latents.min()}, {random_latents.max()}]")
        print(f"average random_latents: {random_latents.mean()}")
        latents = torch.cat(
            [
                gt_latents[:, :from_idx, :],
                random_latents[:, from_idx:to_idx, :],
                gt_latents[:, to_idx:, :],
            ],
            dim=1,
        )
        dec_out = self.decoder(timestamps, latent_primitives=latents)
        return dec_out["recons_sequence"]
