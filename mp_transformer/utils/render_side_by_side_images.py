"""Utilities for rendering reconstructions."""
import imageio
import numpy as np
from PIL import Image

from mp_transformer.datasets.toy_dataset import unnormalize_pose
from mp_transformer.utils.generate_toy_data import BONE_LENGTHS, render_image


def render_side_by_side(gt_pose, pred_pose, bone_lengths=BONE_LENGTHS):
    gt_image = render_image(gt_pose, bone_lengths)
    pred_image = render_image(pred_pose, bone_lengths)

    # Create a new PIL image with double width
    side_by_side = Image.new("RGB", (gt_image.width * 2, gt_image.height))

    # Paste the ground truth and predicted images side by side
    side_by_side.paste(gt_image, (0, 0))
    side_by_side.paste(pred_image, (gt_image.width, 0))

    return side_by_side


# def render_side_by_side(gt_image, gt_pose, pred_pose, bone_lengths=BONE_LENGTHS):
#     gt_image = gt_image.squeeze(0)  # Remove extra dimension
#     gt_image = Image.fromarray(
#         (gt_image.numpy() * 255).astype(np.uint8)
#     )  # Convert tensor to PIL Image
#     gt_pose_image = render_image(gt_pose, bone_lengths)
#     pred_image = render_image(pred_pose, bone_lengths)

#     # Create a new PIL image with triple width
#     side_by_side = Image.new(
#         "RGB", (gt_image.width + gt_pose_image.width * 2, gt_image.height)
#     )

#     # Paste the ground truth image, ground truth pose image, and predicted pose image side by side
#     side_by_side.paste(gt_image, (0, 0))
#     side_by_side.paste(gt_pose_image, (gt_image.width, 0))
#     side_by_side.paste(pred_image, (gt_image.width + gt_pose_image.width, 0))

#     return side_by_side


def render_side_by_side_images(toy_dataset, model, n=6):
    # Generate random indices from the dataset
    random_indices = np.random.choice(len(toy_dataset), size=n, replace=False)

    # Get the random samples and create side-by-side images
    side_by_side_images = []
    for index in random_indices:
        sample = toy_dataset[index]
        x, y = sample["image"], sample["pose"]
        y = y.detach().numpy()
        y_hat = model.infer(x.unsqueeze(0))
        y_hat = y_hat.detach().numpy()[0]
        y = unnormalize_pose(y)
        y_hat = unnormalize_pose(y_hat)
        side_by_side_images.append(render_side_by_side(y, y_hat))

    return side_by_side_images


def render_side_by_side_sequence(toy_dataset, model, subseq_idx=None, dataset_idx=0):
    # random_index = np.random.choice(len(toy_dataset), size=1, replace=False)[0]
    random_index = dataset_idx
    sample = toy_dataset[random_index]  # Returns a segment
    xs, ys, timestamps = sample["images"], sample["poses"], sample["timestamps"]
    if subseq_idx is not None:
        ys_hat = model.infer_subsequence(ys, timestamps, subseq_idx=subseq_idx)
    else:
        ys_hat = model.infer(ys, timestamps)
    ys = ys.detach().numpy()
    ys_hat = ys_hat.squeeze(0).detach().numpy()  # Remove batch dimension
    side_by_side_sequence = []
    for y, y_hat in zip(ys, ys_hat):
        # for x, y, y_hat in zip(xs, ys, ys_hat):
        y = unnormalize_pose(y)
        y_hat = unnormalize_pose(y_hat)
        img = render_side_by_side(y, y_hat)
        # img = render_side_by_side(x, y, y_hat)
        side_by_side_sequence.append(img)

    return side_by_side_sequence


def save_side_by_side_video(toy_dataset, model, fps=20, subseq_idx=None, dataset_idx=0):
    side_by_side_sequence = render_side_by_side_sequence(
        toy_dataset, model, subseq_idx=subseq_idx, dataset_idx=dataset_idx
    )

    i = "" if subseq_idx is None else subseq_idx
    output_file = f"tmp/comp_vid{i}.mp4"
    with imageio.get_writer(output_file, fps=fps) as writer:
        for img in side_by_side_sequence:
            img_array = np.array(img)  # Convert PIL Image object to NumPy array
            writer.append_data(img_array)

    print(f"Video saved to {output_file}")


def save_side_by_side_subsequences(
    toy_dataset, model, num_subseqs, dataset_idx=0, fps=20
):
    for subseq_idx in range(num_subseqs):
        save_side_by_side_video(
            toy_dataset, model, fps=fps, subseq_idx=subseq_idx, dataset_idx=dataset_idx
        )
