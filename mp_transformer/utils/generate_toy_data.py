"""Adapted from Benjamin. Generates data for a toy limb with 3 joints."""
import os
from multiprocessing import Pool

import imageio
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from sklearn import gaussian_process

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
    """Uses gaussian processes to animate the limb with movements of two speeds."""

    print(f"Drawing {N} samples, might take a while...")

    if train_or_val == "train":
        rbf = gaussian_process.kernels.RBF(length_scale=0.25)
        rbf_slow = gaussian_process.kernels.RBF(length_scale=0.6)
    elif train_or_val == "val":
        rbf = gaussian_process.kernels.RBF(length_scale=0.15)  # val-set
        rbf_slow = gaussian_process.kernels.RBF(length_scale=0.55)  # val-set
    else:
        raise ValueError(f"train_or_val must be 'train' or 'val' in Gaussian Process.")

    GP = gaussian_process.GaussianProcessRegressor(kernel=rbf)
    GP_slow = gaussian_process.GaussianProcessRegressor(kernel=rbf_slow)

    t = np.linspace(0, 120, N)
    # TODO: hardcode if I end up using only 3 joints
    y = np.empty((N, len(bone_lengths)))

    y[:, 0] = GP_slow.sample_y(t[:, None], random_state=None)[:, 0] * 3

    y_samples = (
        GP.sample_y(t[:, None], n_samples=len(bone_lengths) - 1, random_state=None).T
        * 0.7
    )
    y[:, 1:] = np.column_stack(y_samples)

    idx = abs(y) > np.pi
    y[idx] = y[idx] - 2 * np.sign(y[idx]) * np.pi

    return y


def forward(angles, bone_lenghts=BONE_LENGTHS):
    """forward
    Compute forward kinematics
    angles --> cartesian coordinates

    :param angles:
      List of angles for each bone_length in the hierarchy
      relative to its parent bone_length
    :param bone_lengths: List of bone_length lengths
    """
    bone_lengths = np.array(bone_lenghts)
    if bone_lengths is None:
        bone_lengths = np.ones_like(angles)
    elif len(bone_lengths) == 2:
        bone_lengths = bone_lengths * np.ones_like(angles)
    else:
        try:
            assert len(angles) == len(bone_lengths)
            # assert angles.shape == bone_lengths.shape
        except AssertionError as excp:
            raise Exception(
                f"Number of angles and bone_lengths should be the same"
                f" but: {len(angles)} is not {len(bone_lengths)}"
                # f" but: {angles.shape} is not {bone_lengths.shape}"
            ) from excp

    coordinates = [(0, 0)]
    cumulative_angle = 0
    for angle, bone_length in zip(angles, bone_lengths):
        offs = coordinates[-1]
        cumulative_angle += angle
        coordinates += [
            (
                bone_length * np.cos(cumulative_angle) + offs[0],
                bone_length * np.sin(cumulative_angle) + offs[1],
            )
        ]
    return coordinates


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
    print(f"{file_path} written")


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
        with Pool() as pool:
            pool.starmap(
                save_image,
                [
                    (i + iteration * N, pose, bone_lengths, train_or_val)
                    for i, pose in enumerate(poses)
                ],
            )

        print(f"Generating video for {train_or_val} set")
        create_video(
            f"data/toy/{train_or_val}-set/images",
            f"data/toy/{train_or_val}-set/video.mp4",
        )


def main(iterations=1, train_or_val="train", gen_images=True):
    options = ["train", "val"] if train_or_val == "both" else [train_or_val]
    for option in options:
        for i in range(iterations):
            _generate(iteration=i, train_or_val=option, gen_images=gen_images)


if __name__ == "__main__":
    # TODO: hardcode the setting you end up with in the end
    GEN_IMAGES = False  # Learning and reconstructing only poses at the moment
    ITERATIONS = 4  # Run muliple times on smaller N and concatenate
    # ITERATIONS = 16
    # ITERATIONS = 20
    # TRAIN_OR_VAL = "both"
    # TRAIN_OR_VAL = "train"
    TRAIN_OR_VAL = "val"
    main(iterations=ITERATIONS, train_or_val=TRAIN_OR_VAL, gen_images=GEN_IMAGES)
