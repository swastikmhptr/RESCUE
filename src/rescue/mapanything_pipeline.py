import os
import gc
import torch
import shutil
from mapanything.models import MapAnything, init_model_from_config
from mapanything.utils.image import load_images
from mapanything.utils.inference import postprocess_model_outputs_for_inference
from huggingface_hub import snapshot_download
from rescue.utils import sample_video
from rescue import ges_utils
from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.colmap_export import export_predictions_to_colmap
import numpy as np
from safetensors.torch import save_file, load_file
import clip
from rescue.feature_reduction import TorchIncrementalPCA

# MapAnything config name (for init_model_from_config) -> (resolution_set, norm_type for load_images)
# See map-anything README "Running External Models" for resolution / normalization.
_MODULAR_IMAGE_PRESETS = {
    "mapanything": (518, "dinov2"),
    "mapanything_v1": (518, "dinov2"),
    "mapanything_ablations": (518, "dinov2"),
    "modular_dust3r": (512, "dust3r"),
    "vggt": (518, "identity"),
    "vggt_commercial": (518, "identity"),
    "vggt_non_pretrained": (518, "identity"),
    "pi3": (518, "identity"),
    "pi3x": (518, "identity"),
    "moge_1": (518, "identity"),
    "moge_2": (518, "identity"),
    "dust3r": (512, "dust3r"),
    "mast3r": (512, "dust3r"),
    "must3r": (512, "dust3r"),
    "pow3r": (512, "dust3r"),
    "pow3r_ba": (512, "dust3r"),
    "da3": (504, "dinov2"),
    "da3_nested": (504, "dinov2"),
}


def _resolve_image_load_kwargs(modular_model, image_resolution_set, image_norm_type):
    """Defaults match HF MapAnything path: 518 + dinov2."""
    if modular_model and modular_model in _MODULAR_IMAGE_PRESETS:
        rs, nt = _MODULAR_IMAGE_PRESETS[modular_model]
    elif modular_model:
        rs, nt = 518, "dinov2"
        print(
            f"[mapanything_pipeline] No preset for modular_model={modular_model!r}; "
            f"using resolution_set={rs}, norm_type={nt!r}. "
            "Override with image_resolution_set / image_norm_type if needed."
        )
    else:
        rs, nt = 518, "dinov2"
    if image_resolution_set is not None:
        rs = image_resolution_set
    if image_norm_type is not None:
        nt = image_norm_type
    return rs, nt


def _ensure_prediction_masks(predictions):
    """``postprocess_model_outputs_for_inference`` may omit ``mask`` if no non_ambiguous_mask; GLB helpers expect it."""
    fixed = []
    for p in predictions:
        p = dict(p)
        if "mask" not in p and "depth_z" in p:
            p["mask"] = torch.ones_like(p["depth_z"], dtype=torch.float32)
        fixed.append(p)
    return fixed


def run_mapanything(
    video_path,
    ges_path,
    fps=2,
    num_views=-1,
    temp_dir="../generated/temp",
    model_local_dir="../generated/map-anything",
    device="cuda" if torch.cuda.is_available() else "cpu",
    hf_model_id="facebook/map-anything",
    modular_model=None,
    modular_machine="default",
    image_resolution_set=None,
    image_norm_type=None,
    modular_postprocess_apply_mask=False,
    modular_postprocess_mask_edges=False,
):
    """
    Run reconstruction on sampled video frames.

    Parameters
    ----------
    hf_model_id : str
        Hugging Face repo id (or local folder) for ``MapAnything.from_pretrained`` when ``modular_model`` is None.
    modular_model : str or None
        If set (e.g. ``\"vggt\"``, ``\"pi3\"``, ``\"dust3r\"``), load that model via ``init_model_from_config``
        and run ``model(views)`` instead of MapAnything ``infer``. Requires optional map-anything dependencies.
        External wrappers typically **ignore** GES poses and use poses/geometry from the model itself.
    modular_machine : str
        Hydra ``machine=`` override for ``init_model_from_config`` (default ``\"default\"``).
    image_resolution_set / image_norm_type : optional
        Override ``load_images`` resizing/normalization; otherwise chosen from ``_MODULAR_IMAGE_PRESETS`` or 518/dinov2.
    modular_postprocess_apply_mask / modular_postprocess_mask_edges : bool
        Passed to ``postprocess_model_outputs_for_inference`` for the modular path. External models often lack
        ``non_ambiguous_mask``; keep these False unless you know the model provides it.
    """
    dev = torch.device(device) if isinstance(device, str) else device

    if modular_model:
        print(f"Loading modular model {modular_model!r} via init_model_from_config...")
        model = init_model_from_config(
            modular_model, device=str(dev), machine=modular_machine
        )
        model.eval()
    else:
        print("Loading MapAnything model...")
        save_path = snapshot_download(
            hf_model_id, repo_type="model", local_dir=model_local_dir
        )
        model = MapAnything.from_pretrained(save_path).to(dev)
        model.eval()

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir, exist_ok=True)
    files_dir, sampled_indices = sample_video(video_path, fps=fps, output_dir=temp_dir)

    rs, nt = _resolve_image_load_kwargs(modular_model, image_resolution_set, image_norm_type)
    print(f"Loading images (resolution_set={rs}, norm_type={nt!r})...")
    if num_views == -1:
        views = load_images(files_dir, resolution_set=rs, norm_type=nt)
    else:
        views = load_images(files_dir, resolution_set=rs, norm_type=nt)[:num_views]

    poses, c2w_list, K_list, orig_width, orig_height = ges_utils.convert_ges_to_mapanything_from_file(
        ges_path, ref_frame=0
    )
    if num_views == -1:
        poses = [poses[i] for i in sampled_indices]
        c2w_list = [c2w_list[i] for i in sampled_indices]
        K_list = [K_list[i] for i in sampled_indices]
    else:
        poses = [poses[i] for i in sampled_indices[:num_views]]
        c2w_list = [c2w_list[i] for i in sampled_indices[:num_views]]
        K_list = [K_list[i] for i in sampled_indices[:num_views]]

    assert len(views) == len(poses) == len(c2w_list) == len(K_list), (
        "Lengths of views, poses, c2w_list, and K_list must be equal"
    )

    for i in range(len(views)):
        h_new, w_new = views[i]["true_shape"][0]
        K = K_list[i].copy()
        scale_x = w_new / orig_width
        scale_y = h_new / orig_height
        K[0, 0] *= scale_x  # fl_x
        K[0, 2] *= scale_x  # cx
        K[1, 1] *= scale_y  # fl_y
        K[1, 2] *= scale_y  # cy

        # load_images() yields CPU tensors; MapAnything.infer() moves views to the model device.
        # Modular model.forward() does not, so for that path we put geometry + img on ``dev`` here.
        view_device = dev if modular_model else views[i]["img"].device
        views[i]["intrinsics"] = torch.tensor(K, dtype=torch.float32, device=view_device)[
            None
        ]
        views[i]["camera_poses"] = torch.tensor(
            c2w_list[i], dtype=torch.float32, device=view_device
        )[None]
        views[i]["is_metric_scale"] = False
        if modular_model:
            views[i]["img"] = views[i]["img"].to(dev, non_blocking=True)

    if modular_model:
        print(f"Running modular model {modular_model!r}...")
        with torch.inference_mode():
            raw = model(views)
        predictions = postprocess_model_outputs_for_inference(
            raw,
            views,
            apply_mask=modular_postprocess_apply_mask,
            mask_edges=modular_postprocess_mask_edges,
            apply_confidence_mask=False,
        )
        predictions = _ensure_prediction_masks(predictions)
        print("Modular model run complete...")
    else:
        print("Running MapAnything model...")
        predictions = model.infer(
            views,
            memory_efficient_inference=False,
            apply_mask=True,
            mask_edges=True,
            use_amp=True,
            amp_dtype="bf16",
        )
        print("MapAnything model run complete...")

    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return predictions

def save_points_to_glb(predictions, output_path, show_cam=False):
    world_points_list = []
    images_list = []
    extrinsic_list = []
    mask_list = []

    for pred in predictions:
        pts3d, valid_mask = depthmap_to_world_frame(
            pred["depth_z"][0].squeeze(-1),
            pred["intrinsics"][0],
            pred["camera_poses"][0],
        )

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()

        world_points_list.append(pts3d.cpu().numpy())
        images_list.append(pred["img_no_norm"][0].cpu().numpy())
        extrinsic_list.append(pred["camera_poses"][0].cpu().numpy())
        mask_list.append(mask)

    pred_dict = {
        "world_points": np.stack(world_points_list),  # (S, H, W, 3)
        "images": np.stack(images_list),  # (S, H, W, 3)
        "extrinsic": np.stack(extrinsic_list),  # (S, 3, 4)
        "final_mask": np.stack(mask_list),  # (S, H, W)
    }

    # Export to GLB
    glbscene = predictions_to_glb(
        pred_dict,
        show_cam=show_cam,
        as_mesh=False,  # True = mesh, False = point cloud
        conf_percentile=0,  # 0 = keep all points
    )
    glbscene.export(output_path)

def save_mesh_to_glb(predictions, output_path, conf_percentile=0, show_cam=False):
    world_points_list = []
    images_list = []
    extrinsic_list = []
    mask_list = []
    conf_list = []

    for pred in predictions:
        pts3d, valid_mask = depthmap_to_world_frame(
            pred["depth_z"][0].squeeze(-1),
            pred["intrinsics"][0],
            pred["camera_poses"][0],
        )

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()

        world_points_list.append(pts3d.cpu().numpy())
        images_list.append(pred["img_no_norm"][0].cpu().numpy())
        extrinsic_list.append(pred["camera_poses"][0].cpu().numpy())
        mask_list.append(mask)
        conf_list.append(pred["conf"][0].squeeze(-1).cpu().numpy())

    pred_dict = {
        "world_points": np.stack(world_points_list),  # (S, H, W, 3)
        "images": np.stack(images_list),  # (S, H, W, 3)
        "extrinsic": np.stack(extrinsic_list),  # (S, 3, 4)
        "final_mask": np.stack(mask_list),  # (S, H, W)
        "conf": np.stack(conf_list),  # (S, H, W)
    }

    # Export to GLBs
    glbscene = predictions_to_glb(
        pred_dict,
        show_cam=show_cam,
        as_mesh=True,  # True = mesh, False = point cloud
        conf_percentile=conf_percentile,  # 0 = keep all points
    )
    glbscene.export(output_path)

def save_colmap(
    predictions,
    output_dir,
    export_points=True,
    export_images=True,
    image_names=None,
    **kwargs,
):
    """
    Save predictions to COLMAP-compatible folder structure using MapAnything's built-in exporter.

    Args:
        predictions: list from run_mapanything
        output_dir: target directory for COLMAP output
        export_points: if True, save ``points.ply`` under ``sparse/`` (same as ``save_ply``)
        export_images: if True, save denormalized frames under ``images/`` (same as ``save_images``)
        image_names: one basename per frame for ``images/``; if None, uses ``frame_000000.jpg``, ...
        kwargs: forwarded to ``export_predictions_to_colmap`` (e.g. ``voxel_fraction``, ``skip_point2d``)
    """
    print(f"[COLMAP Export] Saving predictions to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    n = len(predictions)
    if image_names is None:
        image_names = [f"frame_{i:06d}.jpg" for i in range(n)]
    elif len(image_names) != n:
        raise ValueError(
            f"image_names length ({len(image_names)}) must match predictions ({n})"
        )
    # ``processed_views`` is part of the upstream API but unused inside ``export_predictions_to_colmap``.
    export_predictions_to_colmap(
        outputs=predictions,
        processed_views=predictions,
        image_names=image_names,
        output_dir=output_dir,
        save_ply=export_points,
        save_images=export_images,
        **kwargs,
    )
    print(f"[COLMAP Export] COLMAP files saved in {output_dir}")

def save_language_features(predictions, language_features, output_path, ipca=None):
    """
    Save per-point language features and 3D positions for semantic querying.
    Optionally, also saves the ipca matrix if provided.

    Args:
        predictions       : raw predictions list from run_mapanything()
        language_features : np.ndarray (S, H, W, D) from langseg,
                            resized to match MapAnything's (H, W) dims
        output_path       : path for the .safetensors file
        ipca              : Optional; if provided, the ipca matrix object to save
    """
    world_points_list, mask_list = [], []

    for pred in predictions:
        pts3d, valid_mask = depthmap_to_world_frame(
            pred["depth_z"][0].squeeze(-1),
            pred["intrinsics"][0],
            pred["camera_poses"][0],
        )
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()
        world_points_list.append(pts3d.cpu().numpy())
        mask_list.append(mask)

    world_points = np.stack(world_points_list)  # (S, H, W, 3)
    final_mask = np.stack(mask_list)  # (S, H, W)

    mask_flat = final_mask.reshape(-1)  # (S*H*W,) numpy bool
    pts_flat = world_points.reshape(-1, 3)  # (S*H*W, 3) numpy

    # language_features may be a torch Tensor or numpy array
    if isinstance(language_features, torch.Tensor):
        feat_flat = language_features.reshape(-1, language_features.shape[-1]).cpu()
        mask_t = torch.from_numpy(mask_flat)
        feats_out = feat_flat[mask_t].to(torch.float16)
    else:
        feat_flat = language_features.reshape(-1, language_features.shape[-1])
        feats_out = torch.from_numpy(feat_flat[mask_flat]).to(torch.float16)

    save_dict = {
        "features": feats_out,
        "world_points": torch.from_numpy(pts_flat[mask_flat]).to(torch.float32),
    }

    save_dict["has_ipca"] = torch.tensor(ipca is not None)

    if ipca is not None:
        save_dict["ipca_components"] = ipca.components
        save_dict["ipca_singular_values"] = ipca.singular_values
        save_dict["ipca_n_samples_seen"] = torch.tensor(ipca.n_samples_seen)
        save_dict["ipca_n_components"] = torch.tensor(ipca.n_components)

    save_file(
        save_dict,
        output_path,
    )
    print(f"[MapAnything] Saved {mask_flat.sum()} points to {output_path}")


def integrate_tsdf(
    predictions,
    voxel_length: float = 0.2,
    sdf_trunc: float = 0.8,
    depth_trunc: float = 30.0,
    block_count: int = 50000,
    conf_percentile: float = 10.0,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
):
    """
    Fuse per-frame depth maps from run_mapanything() into a TSDF volume
    and extract a mesh. Uses Open3D's tensor VoxelBlockGrid API (>=0.18).

    Args:
        predictions         : list of dicts from run_mapanything()
        voxel_length        : voxel side length in world units (meters). Smaller = finer mesh.
        sdf_trunc           : truncation band in meters; typically 4-6x voxel_length.
        depth_trunc         : depths beyond this are ignored (meters).
        block_count         : max voxel blocks to allocate; increase for large scenes.
        conf_percentile     : per-frame confidence percentile below which depth is zeroed out.
                              0 keeps all pixels; 10 drops the bottom 10% least-confident.
                              Only applied when pred["conf"] is present.
        outlier_nb_neighbors: neighbors to consider for statistical outlier removal on the mesh vertices.
        outlier_std_ratio   : std-dev threshold for outlier removal; lower = more aggressive.

    Returns:
        mesh : open3d.geometry.TriangleMesh
    """
    try:
        import open3d as o3d
        import open3d.core as o3c
    except ImportError:
        raise ImportError("pip install open3d")

    cpu = o3c.Device("CPU:0")

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_length,
        block_resolution=16,
        block_count=block_count,
        device=cpu,
    )

    for pred in predictions:
        depth_np = np.ascontiguousarray(pred["depth_z"][0].squeeze(-1).cpu().float().numpy())
        rgb_np = np.ascontiguousarray((pred["img_no_norm"][0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        mask_np = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        depth_np[~mask_np] = 0.0

        if conf_percentile > 0 and "conf" in pred:
            conf_np = pred["conf"][0].squeeze(-1).cpu().numpy()
            valid_conf = conf_np[mask_np]
            if valid_conf.size > 0:
                threshold = np.percentile(valid_conf, conf_percentile)
                depth_np[mask_np & (conf_np < threshold)] = 0.0

        K = pred["intrinsics"][0].cpu().numpy()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        c2w = pred["camera_poses"][0].cpu().numpy()
        w2c = np.linalg.inv(c2w)

        depth_t = o3d.t.geometry.Image(depth_np.astype(np.float32)).to(cpu)
        color_t = o3d.t.geometry.Image(rgb_np).to(cpu)
        K_t = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3c.float64)
        T_t = o3c.Tensor(w2c, dtype=o3c.float64)

        block_coords = vbg.compute_unique_block_coordinates(
            depth_t, K_t, T_t, depth_scale=1.0, depth_max=depth_trunc
        )
        vbg.integrate(
            block_coords, depth_t, color_t, K_t, K_t, T_t,
            depth_scale=1.0, depth_max=depth_trunc
        )

    mesh = vbg.extract_triangle_mesh().to_legacy()
    mesh.compute_vertex_normals()

    if outlier_nb_neighbors > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        _, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio
        )
        mesh = mesh.select_by_index(inlier_idx)
        mesh.compute_vertex_normals()

    return mesh


class SceneQueryer:
    """
    Loads a scene's language features once and supports fast repeated text queries.

    Usage:
        queryер = SceneQueryer("reconstruction_langfeats.safetensors", device="cuda")
        points, sims = queryер.query("red fire truck", threshold=0.25)
        points, sims = queryер.query("green tree", top_k=5000)
    """

    def __init__(self, features_path, clip_model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.ipca = None

        print(f"[SceneQueryer] Loading CLIP {clip_model_name}...")
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.tokenizer = clip.tokenize
        print(f"[SceneQueryer] CLIP loaded.")

        print(f"[SceneQueryer] Loading features from {features_path}...")
        data = load_file(features_path)
        self.features = data["features"].to(torch.float32).to(device)
        self.world_points = data["world_points"].to(device)

        if data["has_ipca"].item():
            print(data['ipca_components'].shape)
            print(data['ipca_singular_values'].shape)
            print(data['ipca_n_samples_seen'])
            print(data['ipca_n_components'])
            # exit()
            self.ipca = TorchIncrementalPCA(
                n_components=data["ipca_n_components"].item(),
                components=data["ipca_components"].to(device),
                singular_values=data["ipca_singular_values"].to(device),
                n_samples_seen=data["ipca_n_samples_seen"].item(),
                device=device,
            )

        print(f"[SceneQueryer] Loaded {len(self.features):,} points, feature dim={self.features.shape[1]}")

    def query(self, text, threshold=None, top_k=None):
        """
        Args:
            text      : text query string
            threshold : return all points with similarity >= threshold
            top_k     : return the top-K most similar points
                        (one of threshold or top_k must be set)

        Returns:
            points : torch.Tensor (K, 3)
            sims   : torch.Tensor (K,)
        """
        assert threshold is not None or top_k is not None, "Provide threshold or top_k"

        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(tokens).float()      # (1, D)
        text_emb = torch.nn.functional.normalize(text_emb, dim=1).float()

        if self.ipca is not None:
            text_emb = torch.nn.functional.normalize(self.ipca.transform(text_emb), dim=1)

        sims = (self.features @ text_emb.T).squeeze(1)                 # (N,)

        if top_k is not None:
            idx = sims.topk(min(top_k, len(sims))).indices
        else:
            idx = sims >= threshold

        return self.world_points[idx], sims[idx]