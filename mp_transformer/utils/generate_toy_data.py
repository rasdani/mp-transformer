"""Adapted from Benjamin. Generates data for a toy limb with 3 joints."""
import os
from multiprocessing import Pool

import imageio
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from sklearn import gaussian_process
import torch

BONE_LENGTHS = [6.86943196, 7.83778524, 9.5505643]

# def make_bonelengths_and_width(n_bones=3, img_dim_sqrt=64):
#     """
#     Returns bone lengths and keymarker width with sensible defaults.
#     """
#     # eps = np.random.rand(n_bones)
#     # bone_lengths = img_dim_sqrt // 6 * (eps / 2 + 1 - 1 / n_bones)
#     # key_marker_width = 1.5 * img_dim_sqrt / 32
#     bone_lengths = np.array([6.86943196, 7.83778524, 9.5505643])
#     key_marker_width = 3.0
#     return bone_lengths, key_marker_width


def generate_gaussian_process_poses(N, bone_lengths, train_or_val):
    """Uses gaussian processes of two speeds to animate different parts of the limb.
    To avoid discontinuities angles are represented as sin and cos."""

    print(f"Drawing {N} samples, might take a while...")

    if train_or_val == "train":
        rbf = gaussian_process.kernels.RBF(length_scale=0.25)
        rbf_slow = gaussian_process.kernels.RBF(length_scale=0.6)
    elif train_or_val in ["val", "test"]:
        rbf = gaussian_process.kernels.RBF(length_scale=0.15)  # val-set
        rbf_slow = gaussian_process.kernels.RBF(length_scale=0.55)  # val-set
    else:
        raise ValueError(
            f"{train_or_val} must be 'train' or 'val' in Gaussian Process."
        )

    GP = gaussian_process.GaussianProcessRegressor(kernel=rbf)
    GP_slow = gaussian_process.GaussianProcessRegressor(kernel=rbf_slow)

    t = np.linspace(0, 120, N)
    y = np.empty(
        (N, len(bone_lengths) * 2)
    )  # Multiply by 2 for sine and cosine representation

    # Generate the slow GP for the first joint
    y_samples = 3 * GP_slow.sample_y(t[:, None], random_state=None)[:, 0]
    y[:, 0:2] = np.column_stack([np.sin(y_samples), np.cos(y_samples)])

    # Generate the fast GP for the remaining joints
    for i in range(1, len(bone_lengths)):
        y_samples = (
            0.7 * GP.sample_y(t[:, None], n_samples=1, random_state=None).T[0]
        )  # Extract the 1D array
        y[:, i * 2 : i * 2 + 2] = np.column_stack(
            [np.sin(y_samples), np.cos(y_samples)]
        )

    print("RAW")
    print(f"y.min() = {y.min()}, y.max() = {y.max()}")

    # Check for significant changes
    y_diff = np.diff(y, axis=0)
    print("DIFF")
    print(y_diff)
    print(f"y_diff.min() = {y_diff.min()}, y_diff.max() = {y_diff.max()}")

    max_diff = 0.5  # Choose a suitable maximum difference
    indices_of_large_diffs = np.where(np.abs(y_diff) > max_diff)
    print(f"indices_of_large_diffs = {indices_of_large_diffs}")

    # Display values around large diffs
    window_size = 2  # Adjust as needed

    for idx in indices_of_large_diffs[0]:
        print(
            f"Large difference at index {idx} in column {indices_of_large_diffs[1][0]}:"
        )
        print(f"Values before: {y[idx-window_size:idx+1]}")
        print(f"Values after: {y[idx+1:idx+window_size+1]}")
        print(f"Difference: {y_diff[idx]}")
        print("---")

    return y


def forward(angles, bone_lengths=BONE_LENGTHS):
    """
    Compute forward kinematics
    angles --> cartesian coordinates

    :param angles:
      List of pairs of sine and cosine values for each bone_length in the hierarchy
      relative to its parent bone_length
    :param bone_lengths: List of bone_length lengths
    """
    # print(f"angles = {angles}")
    # print(f"bone_lengths = {bone_lengths}")
    bone_lengths = np.array(bone_lengths)

    coordinates = [(0, 0)]
    cumulative_angle = 0
    angles_sin = angles[::2]
    angles_cos = angles[1::2]
    # breakpoint()
    for angle_sin, angle_cos, bone_length in zip(angles_sin, angles_cos, bone_lengths):
    # for angle_sin_cos, bone_length in zip(angles, bone_lengths):
        # angle_sin, angle_cos = angle_sin_cos
        angle = np.arctan2(angle_sin, angle_cos)
        offs = coordinates[-1]
        cumulative_angle += angle
        coordinates += [
            (
                bone_length * np.cos(cumulative_angle) + offs[0],
                bone_length * np.sin(cumulative_angle) + offs[1],
            )
        ]
    return coordinates

def forward_kinematics(angles, bone_lengths=BONE_LENGTHS):
    """
    Compute forward kinematics using PyTorch
    angles --> cartesian coordinates

    :param angles: Tensor of shape [batch_size, sequence_length, num_joints * 2]
      Pairs of sine and cosine values for each bone_length in the hierarchy
      relative to its parent bone_length
    :param bone_lengths: List or tensor of bone_length lengths
    """
    bone_lengths = torch.tensor(bone_lengths, device=angles.device, dtype=angles.dtype)

    batch_size, seq_length, _ = angles.shape
    num_joints = len(bone_lengths)

    coordinates = torch.zeros(batch_size, seq_length, num_joints+1, 2, device=angles.device)
    cumulative_angle = torch.zeros(batch_size, seq_length, device=angles.device)

    for i in range(num_joints):
        angle_sin = angles[:, :, 2*i]
        angle_cos = angles[:, :, 2*i + 1]
        angle = torch.atan2(angle_sin, angle_cos)
        cumulative_angle += angle

        offsets = coordinates[:, :, i]
        coordinates[:, :, i+1, 0] = bone_lengths[i] * torch.cos(cumulative_angle) + offsets[:, :, 0]
        coordinates[:, :, i+1, 1] = bone_lengths[i] * torch.sin(cumulative_angle) + offsets[:, :, 1]

    return coordinates

def convert_px_to_mm(error_in_px, assumed_limb_length_mm=600, bone_lengths=BONE_LENGTHS):
    """
    Convert the error from pixels to millimeters based on an assumed real-world length of the limb.

    Parameters:
    - error_in_px: Error value in pixels.
    - assumed_limb_length_mm: Assumed length of the toy limb in real-world millimeters.
    - total_limb_length_px: Total length of the toy limb in pixels (sum of BONE_LENGTHS in your case).

    Returns:
    - error_in_mm: Error value converted to millimeters.
    """
    
    total_limb_length_px = sum(bone_lengths)
    conversion_factor = assumed_limb_length_mm / total_limb_length_px
    error_in_mm = error_in_px * conversion_factor
    return error_in_mm

def coordinates_to_image(coords, size=(64, 64), fwhm_joints=5, fwhm_limbs=2):
    h, w = size
    img = np.zeros(size)

    def gaussian(x, y, fwhm):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        X, Y = np.meshgrid(np.arange(-w // 2, w // 2), np.arange(-h // 2, h // 2))
        return np.exp(-0.5 / sigma**2 * ((x - X) ** 2 + (y - Y) ** 2))

    for i, (x, y) in enumerate(coords[:-1]):
        img += gaussian(x, y, fwhm_joints)

        # Draw limb
        x2, y2 = coords[i + 1]
        n_points = int(np.hypot(x2 - x, y2 - y))
        x_values = np.linspace(x, x2, n_points)
        y_values = np.linspace(y, y2, n_points)

        for x_limb, y_limb in zip(x_values, y_values):
            img += gaussian(x_limb, y_limb, fwhm_limbs)

    # Draw last joint
    img += gaussian(coords[-1][0], coords[-1][1], fwhm_joints)

    return img


def render_image(pose, bone_lengths=BONE_LENGTHS):
    coordinates = forward(pose, bone_lengths)
    img = coordinates_to_image(coordinates)

    # Apply a colormap (viridis) to the image
    cmap = plt.get_cmap("viridis")
    colored_data = cmap(img)

    # Convert the colored_data to a PIL image
    pil_img = Image.fromarray(np.uint8(colored_data * 255))

    return pil_img


def save_image(i, pose, bone_lengths, train_or_val):
    img = render_image(pose, bone_lengths)
    file_path = f"data/toy/{train_or_val}-set/images/{i:05d}.png"
    img.save(file_path)
    # print(f"{file_path} written")


def create_video(image_folder, output_video, fps=20):
    images = natsorted(
        [
            os.path.join(image_folder, img)
            for img in os.listdir(image_folder)
            if img.endswith(".png")
        ]
    )

    with imageio.get_writer(output_video, fps=fps) as writer:
        for img_path in images:
            img = imageio.imread(img_path)
            writer.append_data(img)

    print(f"Video saved to {output_video}")


def _generate(iteration=0, N=5000, train_or_val="train", gen_images=True):
    """Gaussian Process too slow for large N"""
    bone_lengths = BONE_LENGTHS
    poses_file = f"data/toy/{train_or_val}-set/poses{iteration}.npy"

    if not os.path.exists(poses_file):
        os.makedirs(f"data/toy/{train_or_val}-set", exist_ok=True)
        poses = generate_gaussian_process_poses(
            N, bone_lengths=bone_lengths, train_or_val=train_or_val
        )
        np.save(poses_file, poses)
        print(f"{poses_file} written")
    else:
        print(f"{poses_file} already exists.")
        poses = np.load(poses_file)

    if gen_images:
        print(f"Generating images for {train_or_val} set")
        os.makedirs(f"data/toy/{train_or_val}-set/images", exist_ok=True)
        print(f"poses before pool = {poses}")
        with Pool() as pool:
            # Generate list of arguments for save_image outside of starmap
            idxs_and_poses = [
                (i + iteration * N, pose.reshape(-1, 2)) for i, pose in enumerate(poses)
            ]
            save_image_args = [
                (idx, pose, bone_lengths, train_or_val) for idx, pose in idxs_and_poses
            ]
            pool.starmap(save_image, save_image_args)

        print(f"Generating video for {train_or_val} set")
        create_video(
            f"data/toy/{train_or_val}-set/images",
            f"data/toy/{train_or_val}-set/video.mp4",
        )


def main(iterations=1, train_or_val="train", gen_images=True, N=5000):
    options = ["train", "val"] if train_or_val == "both" else [train_or_val]
    for option in options:
        for i in range(iterations):
            _generate(iteration=i, train_or_val=option, gen_images=gen_images, N=N)


if __name__ == "__main__":
    # TODO: hardcode the setting you end up with in the end
    N = 5000
    # N = 1000
    # N = 4000
    GEN_IMAGES = True  # Learning and reconstructing only poses at the moment
    # GEN_IMAGES = False  # Learning and reconstructing only poses at the moment
    # ITERATIONS = 1  # Run muliple times on smaller N and concatenate
    # ITERATIONS = 16
    ITERATIONS = 20
    # TRAIN_OR_VAL = "both"
    # TRAIN_OR_VAL = "train"
    # TRAIN_OR_VAL = "val"
    TRAIN_OR_VAL = "test"
    main(iterations=ITERATIONS, train_or_val=TRAIN_OR_VAL, gen_images=GEN_IMAGES, N=N)
