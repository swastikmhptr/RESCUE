import os
import gc
import torch
import shutil
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from huggingface_hub import snapshot_download
from rescue.utils import sample_video
from rescue import ges_utils
from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.geometry import depthmap_to_world_frame
import numpy as np
from safetensors.torch import save_file, load_file
from safetensors.torch import load_file
import clip
from rescue.feature_reduction import TorchIncrementalPCA

def run_mapanything(video_path, ges_path, fps = 2, num_views = -1, temp_dir = '../generated/temp', model_local_dir = '../generated/map-anything', device = "cuda" if torch.cuda.is_available() else "cpu"):
    print ("Loading MapAnything model...")
    save_path = snapshot_download("facebook/map-anything", repo_type="model", local_dir=model_local_dir)
    model = MapAnything.from_pretrained(save_path).to(device)

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir, exist_ok=True)
    files_dir, sampled_indices = sample_video(video_path, fps=fps, output_dir =temp_dir)

    print ("Loading images...")
    if num_views == -1:
        views = load_images(files_dir)
    else:
        views = load_images(files_dir)[:num_views]

    poses, c2w_list, K_list, orig_width, orig_height = ges_utils.convert_ges_to_mapanything_from_file('../generated/long_pattern.json', ref_frame = 0)
    if num_views == -1:
        poses = [poses[i] for i in sampled_indices]
        c2w_list = [c2w_list[i] for i in sampled_indices]
        K_list = [K_list[i] for i in sampled_indices]
    else:
        poses = [poses[i] for i in sampled_indices[:num_views]]
        c2w_list = [c2w_list[i] for i in sampled_indices[:num_views]]
        K_list = [K_list[i] for i in sampled_indices[:num_views]]

    assert len(views) == len(poses) == len(c2w_list) == len(K_list), 'Lengths of views, poses, c2w_list, and K_list must be equal'

    for i in range(len(views)):
        h_new, w_new = views[i]['true_shape'][0]   
        K = K_list[i].copy()
        scale_x = w_new / orig_width            
        scale_y = h_new / orig_height           
        K[0, 0] *= scale_x   # fl_x
        K[0, 2] *= scale_x   # cx
        K[1, 1] *= scale_y   # fl_y
        K[1, 2] *= scale_y   # cy

        device = views[i]['img'].device
        views[i]['intrinsics']    = torch.tensor(K, dtype=torch.float32, device=device)[None]
        views[i]['camera_poses']  = torch.tensor(c2w_list[i], dtype=torch.float32, device=device)[None]
        views[i]['is_metric_scale'] = False

    print ("Running MapAnything model...")
    predictions = model.infer(
    views,
    memory_efficient_inference=False,   # ← change from True
    apply_mask=True,
    mask_edges=True,
    use_amp=True,
    amp_dtype="bf16",
    )
    print ("MapAnything model run complete...")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return predictions

def save_points_to_glb(predictions, output_path, show_cam = False):
    world_points_list = []
    images_list       = []
    extrinsic_list    = []
    mask_list         = []

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
        extrinsic_list.append(pred["camera_poses"][0].cpu().numpy())   # (4, 4) cam2world
        mask_list.append(mask)

    pred_dict = {
        "world_points": np.stack(world_points_list),  # (S, H, W, 3)
        "images":       np.stack(images_list),         # (S, H, W, 3)
        "extrinsic":    np.stack(extrinsic_list),      # (S, 3, 4)
        "final_mask":   np.stack(mask_list),           # (S, H, W)
    }

    # Export to GLB
    glbscene = predictions_to_glb(
        pred_dict,
        show_cam=show_cam,
        as_mesh=False,       # True = mesh, False = point cloud
        conf_percentile=0,   # 0 = keep all points
    )
    glbscene.export(output_path)

def save_mesh_to_glb(predictions, output_path, conf_percentile = 0, show_cam = False):
    world_points_list = []
    images_list       = []
    extrinsic_list    = []
    mask_list         = []
    conf_list         = []

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
        extrinsic_list.append(pred["camera_poses"][0].cpu().numpy())   # (4, 4) cam2world
        mask_list.append(mask)
        conf_list.append(pred["conf"][0].squeeze(-1).cpu().numpy())

    pred_dict = {
        "world_points": np.stack(world_points_list),  # (S, H, W, 3)
        "images":       np.stack(images_list),         # (S, H, W, 3)
        "extrinsic":    np.stack(extrinsic_list),      # (S, 3, 4)
        "final_mask":   np.stack(mask_list),           # (S, H, W)
        "conf":         np.stack(conf_list),           # (S, H, W)
    }

    # Export to GLBs
    glbscene = predictions_to_glb(
        pred_dict,
        show_cam=show_cam,
        as_mesh=True,       # True = mesh, False = point cloud
        conf_percentile=conf_percentile,   # 0 = keep all points
    )
    glbscene.export(output_path)


def save_language_features(predictions, language_features, output_path, ipca = None):
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
    final_mask   = np.stack(mask_list)          # (S, H, W)

    mask_flat = final_mask.reshape(-1)                      # (S*H*W,) numpy bool
    pts_flat  = world_points.reshape(-1, 3)                  # (S*H*W, 3) numpy

    # language_features may be a torch Tensor or numpy array
    if isinstance(language_features, torch.Tensor):
        feat_flat = language_features.reshape(-1, language_features.shape[-1]).cpu()
        mask_t    = torch.from_numpy(mask_flat)
        feats_out = feat_flat[mask_t].to(torch.float16)
    else:
        feat_flat = language_features.reshape(-1, language_features.shape[-1])
        feats_out = torch.from_numpy(feat_flat[mask_flat]).to(torch.float16)

    save_dict = {
        "features":     feats_out,
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

        print(f"[SceneQueryer] Loading CLIP {clip_model_name}...")
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.tokenizer = clip.tokenize
        print(f"[SceneQueryer] CLIP loaded.")

        print(f"[SceneQueryer] Loading features from {features_path}...")
        data              = load_file(features_path)
        self.features     = data["features"].to(torch.float32).to(device)
        self.world_points = data["world_points"].to(device)

        if data["has_ipca"].item():
            print (data['ipca_components'].shape)
            print (data['ipca_singular_values'].shape)
            print (data['ipca_n_samples_seen'])
            print (data['ipca_n_components'])
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