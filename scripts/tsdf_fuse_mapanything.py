#!/usr/bin/env python3
"""
Run MapAnything (or a modular model) on a video + GES, then TSDF-fuse depths to one mesh.

Example:
  PYTHONPATH=src python scripts/tsdf_fuse_mapanything.py \\
    --video ../data/clip.mp4 --ges ../data/poses.ges \\
    --output ../generated/tsdf_mesh.ply \\
    --voxel-length 0.03 --sdf-trunc 0.12 --modular-model vggt
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root without install
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def main():
    parser = argparse.ArgumentParser(description="TSDF fuse MapAnything predictions")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--ges", required=True, help="GES pose file path")
    parser.add_argument("--output", default="../generated/tsdf_mesh.ply", help="Output mesh path (.ply)")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--num-views", type=int, default=-1, help="-1 = all sampled frames")
    parser.add_argument("--voxel-length", type=float, default=0.02, help="Voxel size (depth units)")
    parser.add_argument("--sdf-trunc", type=float, default=0.08, help="SDF truncation distance")
    parser.add_argument("--depth-max", type=float, default=50.0)
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Multiply depth before fusion")
    parser.add_argument("--modular-model", default=None, help="e.g. vggt; omit for HF MapAnything")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    import torch
    from rescue.mapanything_pipeline import run_mapanything
    from rescue.tsdf_fusion import predictions_to_tsdf_mesh

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    predictions = run_mapanything(
        args.video,
        args.ges,
        fps=args.fps,
        num_views=args.num_views,
        modular_model=args.modular_model,
        device=device,
    )

    mesh, _vol = predictions_to_tsdf_mesh(
        predictions,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_max=args.depth_max,
        depth_scale=args.depth_scale,
        use_mask=True,
    )

    import open3d as o3d

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"Wrote {args.output} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")


if __name__ == "__main__":
    main()
