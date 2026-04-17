"""
View a MapAnything (or other) reconstruction in viser.

Supports **mesh** GLBs (triangulated surface) and **point cloud** GLBs.

Auto-detection (in order):
  1. Filename ends with ``mesh.glb`` → mesh; ``points.glb`` → points
     (e.g. ``*_only_mesh.glb`` / ``*_only_points.glb`` from the RESCUE pipeline).
  2. Otherwise inspects the file: any triangle mesh with faces → mesh; else points
     (e.g. ``tsdf_mesh.glb``, ``reconstruction.glb``).

Usage:
    # Auto-detect from filename
    python scripts/view_reconstruction.py --glb generated/reconstruction_only_mesh.glb
    python scripts/view_reconstruction.py --glb generated/reconstruction_only_points.glb

    # Mesh from a generic path (auto from contents, or force)
    python scripts/view_reconstruction.py --glb generated/tsdf_mesh.glb
    python scripts/view_reconstruction.py --glb generated/foo.glb --mode mesh

    # Force point mode on a file that also contains mesh geometry
    python scripts/view_reconstruction.py --glb generated/foo.glb --mode points

Troubleshooting (Viser stuck on "Loading" / no control panel):
    Use http://127.0.0.1:PORT in the browser when the script prints host 0.0.0.0.
    Large GLBs block the first paint until load finishes. Language-query CLIP loads
    on first **Search** click so the 3D view can appear first.
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


def show_mesh(server, path, apply_orientation_fix: bool, center_override=None):
    """
    Center mesh for display. If ``center_override`` is set (e.g. mean of ``world_points``
    from ``--features``), use it so mesh, point cloud, and query overlays share one origin.
    Otherwise use the mesh centroid (same idea as ``show_points`` default).
    """
    scene    = trimesh.load(path)
    combined = trimesh.util.concatenate(list(scene.geometry.values()))
    center = (
        np.asarray(center_override, dtype=np.float64)
        if center_override is not None
        else np.asarray(combined.centroid, dtype=np.float64)
    )
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
    # Lazy-load CLIP + safetensors on first Search. Loading them at startup blocks the
    # main thread so the Viser web client can sit on "Loading…" with no GUI/scene.
    scene_query_holder = [None]  # lazy SceneQueryer

    def _scene_query():
        if scene_query_holder[0] is None:
            print(
                "[query] First search: loading CLIP + language features (may take a minute)…"
            )
            scene_query_holder[0] = SceneQueryer(features_path)
        return scene_query_holder[0]

    # Viser expects either handle.remove() or scene.remove_by_name(...).
    # scene.remove(...) is not reliable across versions and was silently failing here.
    last_query_handle = [None]  # holds returned SceneNodeHandle from last add_* call

    def _clear_query_visual() -> None:
        h = last_query_handle[0]
        if h is not None:
            try:
                h.remove()
            except Exception as e:
                print(f"(query) could not remove previous handle: {e}")
            last_query_handle[0] = None
        scene = server.scene
        for nm in ("query", "/query"):
            if hasattr(scene, "remove_by_name"):
                try:
                    scene.remove_by_name(nm)
                except Exception:
                    pass
            elif hasattr(scene, "remove"):
                try:
                    scene.remove(nm)
                except Exception:
                    pass

    query_text = server.gui.add_text("Query",        initial_value="building")
    threshold  = server.gui.add_slider("Threshold",  min=0.0, max=1.0, step=0.01, initial_value=0.85)
    opacity    = server.gui.add_slider("Opacity",    min=0.0, max=1.0, step=0.01, initial_value=0.85)
    render_as  = server.gui.add_dropdown("Render as", options=["points", "splats"], initial_value="splats")
    size       = server.gui.add_slider("Size", min=0.001, max=0.25, step=0.001, initial_value=float(point_size) * 0.25)
    stable     = server.gui.add_checkbox("Stable splats preset", initial_value=True)
    max_query  = server.gui.add_slider("Max query points", min=1_000, max=2_000_000, step=1_000, initial_value=250_000)
    run_btn    = server.gui.add_button("Search")

    def do_query(_):
        _clear_query_visual()

        points, sims = _scene_query().query(query_text.value, threshold=threshold.value)
        points = points.cpu().numpy() - center
        if apply_orientation_fix:
            points = _apply_glb_orientation(points)
        N = len(points)
        print(f"Query '{query_text.value}': {N:,} matched points (threshold={threshold.value})")

        if N == 0:
            return

        # Deterministic downsample to reduce overlap flicker + frontend load.
        max_n = int(max_query.value)
        if N > max_n:
            idx = np.linspace(0, N - 1, max_n, dtype=int)
            points = points[idx]
            N = len(points)
            print(f"Downsampled query to {N:,} points")

        # Gaussian splats are view-dependent (alpha compositing + depth sorting),
        # so a point-cloud overlay is often a better "mask" visualization.
        if render_as.value == "points":
            last_query_handle[0] = server.scene.add_point_cloud(
                name="query",
                points=points.astype(np.float32),
                colors=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (N, 1)),
                point_size=float(size.value),
                precision="float32",
            )
        else:
            # "Stable" preset: tighter splats + higher opacity.
            if stable.value:
                splat_opacity = 0.95
                splat_size = float(size.value)
                # Make splats tighter than point-size suggests.
                cov_scale = (splat_size * 0.35) ** 2
            else:
                splat_opacity = float(opacity.value)
                splat_size = float(size.value)
                cov_scale = splat_size ** 2

            rgbs        = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(np.float32)
            opacities   = np.full((N, 1), float(splat_opacity), dtype=np.float32)
            covariances = np.tile(np.eye(3) * cov_scale, (N, 1, 1)).astype(np.float32)
            last_query_handle[0] = server.scene.add_gaussian_splats(
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


def detect_mode_from_glb_contents(path: str) -> str | None:
    """
    Choose mesh vs points from GLB contents when the filename is ambiguous.

    If any geometry is a ``Trimesh`` with at least one face, use mesh mode
    (``show_mesh`` / ``add_mesh_trimesh``). Otherwise use point mode.
    """
    try:
        scene = trimesh.load(path, force=None)
    except Exception as e:
        print(f"(auto-detect) could not load GLB for inspection: {e}")
        return None

    geoms: list = []
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
    elif isinstance(scene, trimesh.Trimesh):
        geoms = [scene]

    for g in geoms:
        if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0:
            return "mesh"
    if geoms:
        return "points"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb",        required=True, help="Path to .glb file")
    parser.add_argument("--features",   default=None,  help="Path to .safetensors language features file")
    parser.add_argument("--mode",       choices=["mesh", "points"], default=None,
                        help="Display mode. Auto: filename (*mesh.glb / *points.glb), else GLB contents.")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.05)
    parser.add_argument("--max_points", type=int, default=5000000)
    parser.add_argument("--view",       choices=["top", "side"], default="side",
                        help="Initial camera: 'top' nadir from +Z toward origin; 'side' from +Y toward origin (ENU: Z up).")
    parser.add_argument(
        "--no-orientation-fix",
        action="store_true",
        help="Skip 180° X rotation (hf_utils GLB export omits it; other GLBs may already be upright).",
    )
    args = parser.parse_args()

    mode = args.mode or detect_mode(args.glb) or detect_mode_from_glb_contents(args.glb)
    if mode is None:
        parser.error(
            "Could not detect mesh vs points. Pass --mode mesh (triangulated surface) "
            "or --mode points (point cloud)."
        )

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser listening on http://{args.host}:{args.port}  (mode={mode})")
    if args.host in ("0.0.0.0", "::"):
        print(f"  → Open in your browser: http://127.0.0.1:{args.port}  (not 0.0.0.0)")
    print("Loading geometry (large GLBs can take a while)…")

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
        center, pts_min, pts_max = show_mesh(
            server, args.glb, fix, center_override=center_override
        )
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

    # --- Camera auto-rotate controls ---
    auto_rotate = server.gui.add_checkbox("Auto-rotate (Z)", initial_value=False)
    rotate_speed = server.gui.add_slider(
        "Rotate speed (deg/s)", min=0.0, max=60.0, step=0.5, initial_value=8.0
    )

    # Track connected clients and per-client orbit state.
    clients: dict[int, viser.ClientHandle] = {}
    orbit_state: dict[int, dict[str, float]] = {}

    x_extent = float(pts_max[0] - pts_min[0])
    y_extent = float(pts_max[1] - pts_min[1])
    default_radius = max(1e-6, max(x_extent, y_extent) * 1.5)

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle):
        clients[client.client_id] = client

    @server.on_client_disconnect
    def _on_disconnect(client: viser.ClientHandle):
        clients.pop(client.client_id, None)
        orbit_state.pop(client.client_id, None)

    def _ensure_orbit_state(client: viser.ClientHandle) -> dict[str, float] | None:
        """Initialize orbit state from the current camera pose."""
        try:
            pos = np.asarray(client.camera.position, dtype=np.float64)
        except Exception:
            # Camera state not ready yet.
            return None

        r = float(np.hypot(pos[0], pos[1]))
        # If we're on the Z axis (top-down default), give it a reasonable orbit radius.
        if r < 1e-6:
            r = default_radius
            pos[0] = r
            pos[1] = 0.0
            try:
                client.camera.position = (float(pos[0]), float(pos[1]), float(pos[2]))
                client.camera.look_at = (0.0, 0.0, 0.0)
            except Exception:
                return None

        return {"r": r, "z": float(pos[2]), "theta": float(np.arctan2(pos[1], pos[0]))}

    while True:
        dt = 1.0 / 30.0
        time.sleep(dt)
        if not auto_rotate.value:
            continue

        omega = float(rotate_speed.value) * np.pi / 180.0  # rad/s
        dtheta = omega * dt

        for client_id, client in list(clients.items()):
            state = orbit_state.get(client_id)
            if state is None:
                state = _ensure_orbit_state(client)
                if state is None:
                    continue
                orbit_state[client_id] = state

            state["theta"] += dtheta
            x = state["r"] * float(np.cos(state["theta"]))
            y = state["r"] * float(np.sin(state["theta"]))
            z = state["z"]
            try:
                client.camera.position = (x, y, z)
                client.camera.look_at = (0.0, 0.0, 0.0)
            except Exception:
                # Client may have disconnected or not be ready.
                orbit_state.pop(client_id, None)


if __name__ == "__main__":
    main()
