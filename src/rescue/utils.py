import os
import tempfile

import torch
import tqdm
import xarray as xr
import geopandas as gpd
import numpy as np
import cv2
from shapely.geometry import box
import trimesh
import pyrender
from pathlib import Path
from PIL import Image
from IPython.display import Image as IPyImage, display

def sample_video(video_path, fps, output_dir=None):
    """
    Sample frames from a video at a given fps and save them as JPEGs.

    Args:
        video_path (str): Path to the input video file.
        fps (float): Frames per second to sample at.
        output_dir (str, optional): Directory to save frames. If None, a
            temporary directory is created. The caller is responsible for
            cleanup in that case.

    Returns:
        tuple:
            - str: Path to the directory containing the sampled frames (frame_0000.jpg, ...).
            - list[int]: Source frame indices that were sampled, usable as a numpy index array.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = video_fps / fps   # read every Nth source frame

    frame_idx = 0    # source frame counter
    saved_idx = 0    # output file counter
    sampled_indices = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % round(frame_interval) == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            sampled_indices.append(frame_idx)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return output_dir, sampled_indices


def get_png_from_naip(naip_path, in_utm_crs = False):
    sample_img = xr.open_dataset(naip_path)
    sample_img = sample_img.rio.write_crs("EPSG:4326")

    if in_utm_crs:
        minx, miny, maxx, maxy = sample_img.rio.bounds()

        utm_crs = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326").estimate_utm_crs()
        print (utm_crs)
        sample_img = sample_img.rio.reproject("EPSG:26914")

    rgb = sample_img.to_array()[0].values[:3, :, :]
    rgb = np.moveaxis(rgb, 0, -1).astype("uint8")

    return rgb


def plot_sam3_detections(rgb, masks, bboxes, scores, labels):
    """
    Plots SAM 3 detections on the RGB image using OpenCV.

    Args:
        rgb (np.ndarray): Original image (H, W, 3).
        masks (torch.Tensor or np.ndarray): Boolean masks (N, H, W).
        bboxes (torch.Tensor or np.ndarray): Bounding boxes (N, 4) in [x1, y1, x2, y2].
        scores (torch.Tensor or np.ndarray): Confidence scores (N,).
        labels (list of str): Class labels for each detection.
    """
    # Convert to numpy if necessary
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    if hasattr(bboxes, "cpu"):
        bboxes = bboxes.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    output = rgb.copy()

    # Unique labels and their colors
    unique_labels = sorted(list(set(labels)))
    np.random.seed(42)
    label_colors = {
        label: np.random.randint(0, 255, 3, dtype=np.uint8).tolist()
        for label in unique_labels
    }

    # 1. Group and draw combined masks by label
    mask_overlay = output.copy()
    for label in unique_labels:
        # Get all masks for this label
        indices = [i for i, l in enumerate(labels) if l == label]
        if not indices:
            continue

        # Combine masks for this label
        combined_label_mask = np.logical_or.reduce([masks[i] for i in indices])

        # Draw this label's combined mask
        mask_overlay[combined_label_mask] = label_colors[label]

    # Apply weighted blending for all masks at once
    cv2.addWeighted(mask_overlay, 0.4, output, 0.6, 0, output)

    # 2. Draw individual bounding boxes and labels
    for i, (bbox, score, label) in tqdm.tqdm(
        enumerate(zip(bboxes, scores, labels)), total=len(bboxes), desc="Drawing boxes"
    ):
        color = label_colors[label]

        # # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        # cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw label and score
        text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw label background
        cv2.rectangle(
            output, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1
        )
        # Draw text
        cv2.putText(
            output,
            text,
            (x1, y1 - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return output


def collate_sam3_results(results, prompts):
    labels = []
    masks = []
    bboxes = []
    scores = []

    for i, prompt in enumerate(prompts):
        if results[i]["scores"].shape[0] == 0:
            continue
        masks.append(results[i]["masks"].cpu())
        bboxes.append(results[i]["boxes"].cpu())
        scores.append(results[i]["scores"].cpu())
        labels.extend([prompt] * masks[-1].shape[0])

    masks = torch.cat(masks).numpy()
    bboxes = torch.cat(bboxes).numpy()
    scores = torch.cat(scores).numpy()

    return masks, bboxes, scores, labels


def look_at(eye, target, up):
    zaxis = eye - target
    zaxis /= np.linalg.norm(zaxis)
    xaxis = np.cross(up, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)

    view_matrix = np.eye(4)
    view_matrix[:3, 0] = xaxis
    view_matrix[:3, 1] = yaxis
    view_matrix[:3, 2] = zaxis
    view_matrix[:3, 3] = eye
    return view_matrix


def render_3d_plot_from_above(
    recon_path,
    bg_color=[255, 255, 255, 255],
    ambient_light=np.ones(4, dtype="uint8") * 200,
):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    scene = trimesh.load(recon_path)
    geometry_names = list(scene.geometry.keys())

    print(f"Found num geometries: {len(geometry_names)}")

    render_scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255], ambient_light=np.ones(4, dtype="uint8") * 200
    )

    # 1. Collect all trimesh objects in a list
    meshes = [scene.geometry[name] for name in geometry_names]
    # 2. Combine them into one single trimesh

    combined_trimesh = trimesh.util.concatenate(meshes)

    scene_to_render = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    render_mesh = pyrender.Mesh.from_trimesh(combined_trimesh)
    scene_to_render.add(render_mesh)

    center = combined_trimesh.centroid
    eye = np.array([center[0], center[1], 0.0])
    target = center

    up = np.array([0, 1, 0])
    camera_pose = look_at(eye, target, up)

    mag = max(combined_trimesh.extents[:2]) / 2.0

    camera = pyrender.OrthographicCamera(xmag=mag, ymag=mag, znear=0.01, zfar=100.0)
    scene_to_render.add(camera, pose=camera_pose)

    r = pyrender.OffscreenRenderer(1024, 1024)
    color, depth = r.render(scene_to_render)

    return color, depth

def save_images_as_gif(images, output_path, duration=100, loop=0):
    """
    Save a list of images as an animated GIF.

    Args:
        images: list of PIL.Image objects or numpy arrays
        output_path: path to save the GIF
        duration: frame duration in milliseconds
        loop: 0 means infinite loop
    """
    if not images:
        raise ValueError("images list is empty")

    pil_images = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img.convert("RGB"))
        else:
            pil_images.append(Image.fromarray(img).convert("RGB"))

    output_path = str(Path(output_path))
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=loop
    )
    return output_path

def display_gif_jupyter(gif_path):
    """
    Display a GIF in a Jupyter notebook.
    """
    gif_path = str(Path(gif_path))
    display(IPyImage(filename=gif_path))