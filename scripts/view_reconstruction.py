"""
View a MapAnything reconstruction in viser.

Automatically detects mesh vs point cloud from the filename suffix
produced by run_mapanything.py (_only_mesh.glb / _only_points.glb).

Usage:
    # Auto-detect from filename
    python scripts/view_reconstruction.py --glb generated/reconstructiononly_mesh.glb
    python scripts/view_reconstruction.py --glb generated/reconstructiononly_points.glb

    # Force mode
    python scripts/view_reconstruction.py --glb generated/reconstruction.glb --mode mesh
    python scripts/view_reconstruction.py --glb generated/reconstruction.glb --mode points
"""

import argparse
import sys
import os
import time

import numpy as np
import trimesh
import torch
import viser
from safetensors.torch import load_file

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from rescue.mapanything_pipeline import SceneQueryer

# mapanything/utils/viz.py applies this after building the scene; hf_utils/viz.py
# (used by RESCUE for GLB export) does not — so we fix orientation here for display.
_R_GLB_UPRIGHT = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])[:3, :3]


def _apply_glb_orientation(pts: np.ndarray) -> np.ndarray:
    """Rotate centered points so +Z matches gravity like the non-hf MapAnything exporter."""
    return np.asarray(pts, dtype=np.float64) @ _R_GLB_UPRIGHT


def show_mesh(server, path, apply_orientation_fix: bool):
    scene    = trimesh.load(path)
    combined = trimesh.util.concatenate(list(scene.geometry.values()))
    center   = combined.centroid
    combined.vertices -= center
    if apply_orientation_fix:
        combined.vertices[:, :] = _apply_glb_orientation(combined.vertices)
    vmin = np.asarray(combined.vertices.min(axis=0))
    vmax = np.asarray(combined.vertices.max(axis=0))

    print(f"Mesh: {len(combined.vertices):,} vertices, {len(combined.faces):,} faces")
    server.scene.add_mesh_trimesh(name="reconstruction", mesh=combined)
    print(f"Centered at {center}")
    return center, vmin, vmax


def show_points(
    server,
    path,
    max_points=500_000,
    point_size=0.05,
    center_override=None,
    apply_orientation_fix: bool = True,
):
    scene = trimesh.load(path)
    all_pts, all_colors = [], []
    for geom in scene.geometry.values():
        if isinstance(geom, trimesh.PointCloud):
            all_pts.append(np.array(geom.vertices))
            c = np.array(geom.colors)[:, :3] if geom.colors is not None \
                else np.full((len(geom.vertices), 3), 180, dtype=np.uint8)
            all_colors.append(c)
        elif isinstance(geom, trimesh.Trimesh):
            all_pts.append(np.array(geom.vertices))
            vc = geom.visual.vertex_colors
            c = np.array(vc)[:, :3] if vc is not None \
                else np.full((len(geom.vertices), 3), 180, dtype=np.uint8)
            all_colors.append(c)
    pts    = np.concatenate(all_pts,    axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if len(pts) > max_points:
        # Uniformly sample indices for downsampling
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[idx]
        colors = colors[idx]
        print(f"Uniformly downsampled to {max_points:,} points")

    center = center_override if center_override is not None else pts.mean(axis=0)
    pts -= center
    if apply_orientation_fix:
        pts = _apply_glb_orientation(pts)

    print(f"Points: {len(pts):,}")
    print(f"X: [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]")
    print(f"Y: [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")
    print(f"Z: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")

    server.scene.add_point_cloud(
        name="reconstruction",
        points=pts.astype(np.float32),
        colors=(colors.astype(np.float32) / 255.0),
        point_size=point_size,
        precision="float32",
    )
    print(f"Centered at {center}")
    return center, pts.min(axis=0), pts.max(axis=0)


def show_query(
    server,
    features_path,
    center,
    point_size=0.05,
    apply_orientation_fix: bool = True,
):
    scene_query = SceneQueryer(features_path)

    query_text = server.gui.add_text("Query",        initial_value="building")
    threshold  = server.gui.add_slider("Threshold",  min=0.0, max=1.0, step=0.01, initial_value=0.85)
    opacity    = server.gui.add_slider("Opacity",    min=0.0, max=1.0, step=0.01, initial_value=0.5)
    run_btn    = server.gui.add_button("Search")

    def do_query(_):
        # Always clear the previous result first to avoid stale/corrupted node
        try:
            server.scene.remove("/query")
        except Exception:
            pass

        points, sims = scene_query.query(query_text.value, threshold=threshold.value)
        points = points.cpu().numpy() - center
        if apply_orientation_fix:
            points = _apply_glb_orientation(points)
        N = len(points)
        print(f"Query '{query_text.value}': {N:,} matched points (threshold={threshold.value})")

        if N == 0:
            return

        rgbs        = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(np.float32)
        opacities   = np.full((N, 1), opacity.value, dtype=np.float32)
        cov_scale   = (point_size * 0.5) ** 2
        covariances = np.tile(np.eye(3) * cov_scale, (N, 1, 1)).astype(np.float32)

        server.scene.add_gaussian_splats(
            name="query",
            centers=points.astype(np.float32),
            covariances=covariances,
            rgbs=rgbs,
            opacities=opacities,
        )

    run_btn.on_click(do_query)


def detect_mode(path):
    if path.endswith("mesh.glb"):
        return "mesh"
    if path.endswith("points.glb"):
        return "points"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb",        required=True, help="Path to .glb file")
    parser.add_argument("--features",   default=None,  help="Path to .safetensors language features file")
    parser.add_argument("--mode",       choices=["mesh", "points"], default=None,
                        help="Display mode. Auto-detected from filename if not set.")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.05)
    parser.add_argument("--max_points", type=int, default=5000000)
    parser.add_argument("--view",       choices=["top", "side"], default="top",
                        help="Initial camera: 'top' nadir from +Z toward origin; 'side' from +Y toward origin (ENU: Z up).")
    parser.add_argument(
        "--no-orientation-fix",
        action="store_true",
        help="Skip 180° X rotation (hf_utils GLB export omits it; other GLBs may already be upright).",
    )
    args = parser.parse_args()

    mode = args.mode or detect_mode(args.glb)
    if mode is None:
        parser.error("Could not detect mode from filename. Pass --mode mesh or --mode points.")

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser running at http://{args.host}:{args.port}  (mode={mode})")

    # Derive center from safetensors world_points when available so that the
    # displayed cloud and query results share the same coordinate origin.
    center_override = None
    if args.features is not None:
        data = load_file(args.features)
        wp   = data["world_points"].numpy()          # (N, 3)
        center_override = wp.mean(axis=0)
        print(f"Center derived from features: {center_override}")

    fix = not args.no_orientation_fix
    print(f"GLB 180° X orientation fix: {'on' if fix else 'off'}")

    if mode == "mesh":
        center, pts_min, pts_max = show_mesh(server, args.glb, fix)
    else:
        center, pts_min, pts_max = show_points(
            server, args.glb, args.max_points, args.point_size,
            center_override=center_override,
            apply_orientation_fix=fix,
        )

    if args.features is not None:
        show_query(server, args.features, center, args.point_size, apply_orientation_fix=fix)

    # MapAnything / RESCUE reconstructions follow ENU-style coords: X/Y span the
    # site, Z is vertical (+Z up). Viser's default scene up is +Z — match that so
    # the horizontal "ground" plane (XY) and any grid read correctly in nadir.
    server.scene.set_up_direction("+z")
    z_extent = float(pts_max[2] - pts_min[2])
    y_extent = float(pts_max[1] - pts_min[1])

    if args.view == "top":
        # Nadir: camera above origin on +Z, look straight down; screen-up = +Y (North).
        cam_pos = (0.0, 0.0, float(pts_max[2] + z_extent * 2.0))
        cam_up = (0.0, 1.0, 0.0)
    else:  # side
        cam_pos = (0.0, float(pts_max[1] + y_extent * 2.0), 0.0)
        cam_up = (0.0, 0.0, 1.0)

    # Use initial_camera (initial=True wire messages) instead of mutating
    # client.camera on_connect. The latter leaves drei CameraControls out of sync
    # with the orbit target until the first click, which shows up as a zoom jump.
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.initial_camera.up = cam_up
    server.initial_camera.position = cam_pos
    max_extent = float(np.linalg.norm(pts_max - pts_min))
    server.initial_camera.far = max(1000.0, max_extent * 50.0)

    print(f"initial_camera position={cam_pos}  up={cam_up}  far={server.initial_camera.far}")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
