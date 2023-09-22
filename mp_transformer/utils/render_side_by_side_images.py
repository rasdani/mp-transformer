"""Utilities for rendering reconstructions."""
import imageio
import numpy as np
from moviepy.editor import VideoFileClip, clips_array
from PIL import Image

from mp_transformer.datasets.toy_dataset import unnormalize_pose
from mp_transformer.utils.generate_toy_data import render_image


def render_side_by_side(gt_pose, pred_pose):
    gt_image = render_image(gt_pose)
    pred_image = render_image(pred_pose)

    # Create a new PIL image with double width
    side_by_side = Image.new("RGB", (gt_image.width * 2, gt_image.height))

    # Paste the ground truth and predicted images side by side
    side_by_side.paste(gt_image, (0, 0))
    side_by_side.paste(pred_image, (gt_image.width, 0))

    return side_by_side


### For inspecting toy data or learning on images later on
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


# def render_side_by_side_images(toy_dataset, model, n=6):
#     # Generate random indices from the dataset
#     random_indices = np.random.choice(len(toy_dataset), size=n, replace=False)

#     # Get the random samples and create side-by-side images
#     side_by_side_images = []
#     for index in random_indices:
#         sample = toy_dataset[index]
#         x, y = sample["image"], sample["pose"]
#         y = y.detach().numpy()
#         y_hat = model.infer(x.unsqueeze(0))
#         y_hat = y_hat.detach().numpy()[0]
#         y = unnormalize_pose(y)
#         y_hat = unnormalize_pose(y_hat)
#         side_by_side_images.append(render_side_by_side(y, y_hat))

#     return side_by_side_images


def render_side_by_side_sequence(item, model, subseq_idx=None):
    ys, timestamps = item["poses"], item["timestamps"]
    if subseq_idx is not None:
        out = model.infer_subsequence(
            ys, timestamps, subseq_idx=subseq_idx, return_mask=True
        )
        ys_hat = out["recons_subseq"]
        mask = out["gaussian_mask"].squeeze(0, 2).detach().numpy()
    else:
        ys_hat = model.infer(ys, timestamps)
    ys = ys.detach().numpy()
    ys_hat = ys_hat.squeeze(0).detach().numpy()  # Remove batch dimension
    side_by_side_sequence = []
    if subseq_idx is None:
        for y, y_hat in zip(ys, ys_hat):
            y = unnormalize_pose(y)
            y_hat = unnormalize_pose(y_hat)
            img = render_side_by_side(y, y_hat)
            side_by_side_sequence.append(img)
    else:
        for y, y_hat, m in zip(ys, ys_hat, mask):
            y = unnormalize_pose(y)
            y_hat = unnormalize_pose(y_hat)
            img = render_side_by_side(y, y_hat)
            brightness = 1.5 if m >= 0.5 else 0.5
            img = img.point(lambda p: p * brightness)
            side_by_side_sequence.append(img)

    return side_by_side_sequence


def render_side_by_side_completion(item, model, from_idx, to_idx=-1):
    ys, timestamps = item["poses"], item["timestamps"]
    ys_hat = model.complete(ys, timestamps, from_idx=from_idx, to_idx=to_idx)
    ys = ys.detach().numpy()
    ys_hat = ys_hat.squeeze(0).detach().numpy()  # Remove batch dimension
    side_by_side_sequence = []
    for y, y_hat in zip(ys, ys_hat):
        y = unnormalize_pose(y)
        y_hat = unnormalize_pose(y_hat)
        img = render_side_by_side(y, y_hat)
        side_by_side_sequence.append(img)
    return side_by_side_sequence


def render_generation(model):
    ys_hat = model.generate()
    ys_hat = ys_hat.squeeze(0).detach().numpy()  # Remove batch dimension
    imgs = []
    for y_hat in ys_hat:
        y_hat = unnormalize_pose(y_hat)
        img = render_image(y_hat)
        imgs.append(img)
    return imgs


def save_side_by_side_video(
    item,
    model,
    fps=20,
    subseq_idx=None,
    from_idx=None,
    to_idx=-1,
    path="tmp/comp_vid.mp4",
):
    if from_idx is None:
        side_by_side_sequence = render_side_by_side_sequence(
            item, model, subseq_idx=subseq_idx
        )
    else:
        side_by_side_sequence = render_side_by_side_completion(
            item, model, from_idx, to_idx
        )

    i = "" if subseq_idx is None else subseq_idx
    output_file = f"{path[:-4]}{i}.mp4"
    with imageio.get_writer(output_file, fps=fps) as writer:
        for img in side_by_side_sequence:
            img_array = np.array(img)  # Convert PIL Image object to NumPy array
            writer.append_data(img_array)

    print(f"Video saved to {output_file}")


def save_side_by_side_strip(
    item, model, num_subseqs=6, fps=20, from_idx=None, to_idx=-1
):
    # Whole sequence
    save_side_by_side_video(
        item, model, fps=fps, subseq_idx=None, from_idx=from_idx, to_idx=to_idx
    )
    clips = [VideoFileClip("tmp/comp_vid.mp4")]
    # Subsequences
    for subseq_idx in range(num_subseqs):
        save_side_by_side_video(
            item,
            model,
            fps=fps,
            subseq_idx=subseq_idx,
            from_idx=from_idx,
            to_idx=to_idx,
        )
        clip = VideoFileClip(f"tmp/comp_vid{subseq_idx}.mp4")
        clips.append(clip)

    final_clip = clips_array([clips])  # stack horizontally
    final_clip.write_videofile("tmp/comp_strip.mp4")


def save_generation_video(
    model,
    fps=20.0,
    path="tmp/gen_vid.mp4",
):
    imgs = render_generation(model)
    with imageio.get_writer(path, fps=fps) as writer:
        for img in imgs:
            img_array = np.array(img)  # Convert PIL Image object to NumPy array
            writer.append_data(img_array)

    print(f"Video saved to {path}")


def unroll_from_video(image_path, video_path, step=18):
    # Read the video and extract every nth frame
    clip = VideoFileClip(video_path)
    frames = [frame for idx, frame in enumerate(clip.iter_frames()) if idx % step == 0]

    # Convert frames to PIL Images
    images = [Image.fromarray(frame) for frame in frames]

    # Stitch images vertically
    width = images[0].width
    total_height = len(images) * images[0].height
    stitched_image = Image.new("RGB", (width, total_height))

    y_offset = 0
    for img in images:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.height

    stitched_image.save(image_path)
    print(f"Image saved as {image_path}.png")
