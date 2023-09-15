"""Defines the ToyDataset class for the toy dataset."""
import os

import numpy as np
import PIL
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PIL.PILLOW_VERSION = PIL.__version__  # torchvision bug


TOY_DATA_PATH = "data/toy/train-set"


def normalize_pose(pose):
    """Transform [-1, 1] to [0, 1]"""
    # print(pose)
    # print(np.where(pose < -1))
    # print(np.where(pose > 1))
    assert np.all(pose >= -1) and np.all(pose <= 1)
    pose = (pose + 1) / 2
    assert np.all(pose >= 0) and np.all(pose <= 1)

    return pose


def unnormalize_pose(pose):
    """Transform [0, 1] to [-1, 1]"""
    # assert np.all(pose >= 0) and np.all(pose <= 1)
    pose = pose * 2 - 1
    # assert np.all(pose >= -1) and np.all(pose <= 1)
    return pose


class ToyDataset(Dataset):
    """Toy data for pose estimation"""

    def __init__(
        self,
        path=TOY_DATA_PATH,
        transform=True,
        return_segments=False,
        sequence_length=64,
        N=40000,
        sparsity=1,
        return_images=False,
    ):
        super().__init__()

        if return_images:
            image_paths = [
                os.path.join(path, "images/", f)
                for f in os.listdir(os.path.join(path, "images"))
            ]
            self.image_paths = natsorted(image_paths)
            # Gets too sparse for fast motion
            self.image_paths = self.image_paths[:N:sparsity]

        self.poses = [
            np.load(os.path.join(path, f))
            for f in natsorted(os.listdir(path))
            if f.endswith(".npy")
        ]
        self.num_pose_files = len(self.poses)
        # calculate boundary indices of each source array in the concatenated array
        self.boundary_indices = np.cumsum([0] + [len(pose) for pose in self.poses[:-1]])

        self.poses = np.concatenate(self.poses)  # concats dimensions if not list!
        self.poses = self.poses[:N:sparsity]

        self.transform = transform
        self.return_segments = return_segments
        self.sequence_length = sequence_length
        self.return_images = return_images

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        if self.return_segments:  # Adjust the length based on 'return_segments'
            return len(self.poses) // self.sequence_length
            # return sum((boundary_idx // self.sequence_length for boundary_idx in self.boundary_indices))
        return len(self.poses)

    def __getitem__(self, idx):
        if self.return_segments:
            ret = self._get_segment(idx)
        else:
            ret = self._get_single_item(idx)
        return ret

    def _get_single_item(self, idx):
        pose = self.poses[idx]
        # print(f"{idx=}")
        pose = normalize_pose(pose)
        pose = torch.tensor(pose).to(torch.float32)
        ret = {"pose": pose}

        if self.return_images:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("L")
            image = np.array(image)

            if self.transform:
                image = self.transforms(image)

            ret["image"] = image

        return ret

    def _get_segment(self, idx):
        # print(f"{idx=}")
        images = []
        poses = []
        timestamps = []

        # Compute the start and end indices of the segment
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length

        # adjust start index if a boundary fall within the segment
        for boundary_idx in self.boundary_indices:
            if start_idx < boundary_idx < end_idx:
                start_idx = boundary_idx
                end_idx = start_idx + self.sequence_length
                break

        for i in range(start_idx, end_idx):
            item = self._get_single_item(i)
            pose = item["pose"]
            poses.append(pose)

            if self.return_images:
                image = item["image"]
                if self.transform:
                    image = self.transforms(image)

                images.append(image)

        poses = torch.stack(poses)
        timestamps = torch.linspace(0, 1, self.sequence_length)
        ret = {"poses": poses, "timestamps": timestamps}

        if self.return_images:
            images = torch.stack(images)
            ret["images"] = images

        return ret
