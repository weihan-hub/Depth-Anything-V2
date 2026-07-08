"""Back-project predicted metric depth maps into per-frame colored point clouds.

Reads the 16-bit millimetre depth PNGs written by infer_depth.py, the matching
RGB frames, and the pinhole intrinsics from the pipeline's frames_meta.json,
then unprojects every valid pixel:

    Z = depth(u, v)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

Output is one binary PLY per frame in the camera coordinate frame (+Z forward),
colored from the RGB image. Intrinsics are scaled if the depth map resolution
differs from the calibration resolution. Use --stride to thin dense 4K frames.

    cd metric_depth
    uv run python -m calder.app.infer.depth_to_pointcloud \
        --depth-dir   /mnt/data/.../nurec_run1/mono_depth_pred \
        --image-dir   /mnt/data/.../nurec_run1/extracted/iphone_camera \
        --frames-meta /mnt/data/.../nurec_run1/extracted/frames_meta.json \
        --out-dir     /mnt/data/.../nurec_run1/pointclouds \
        --max-depth 20 --stride 4
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def load_intrinsics(frames_meta_path):
    """Return {camera_params_id(str): dict(fx,fy,cx,cy,W,H)} and a
    {image_stem: camera_params_id} lookup from frames_meta.json."""
    meta = json.load(open(frames_meta_path))
    cam_params = meta["camera_params_id_to_camera_params"]
    intr = {}
    for cid, cp in cam_params.items():
        calib = cp["calibration_parameters"]
        K = calib["projection_matrix"]["data"]  # row-major 3x4
        intr[str(cid)] = {
            "fx": K[0], "fy": K[5], "cx": K[2], "cy": K[6],
            "W": calib["image_width"], "H": calib["image_height"],
        }
    stem_to_cid = {}
    for kf in meta.get("keyframes_metadata", []):
        stem = os.path.splitext(os.path.basename(kf["image_name"]))[0]
        stem_to_cid[stem] = str(kf["camera_params_id"])
    return intr, stem_to_cid


def write_ply_binary(path, xyz, rgb):
    """Write a binary-little-endian PLY (float32 xyz + uchar rgb)."""
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", required=True, help="16-bit mm depth PNGs (infer_depth output)")
    ap.add_argument("--image-dir", required=True, help="matching RGB frames (for color)")
    ap.add_argument("--frames-meta", required=True, help="frames_meta.json with intrinsics")
    ap.add_argument("--out-dir", required=True, help="where to write per-frame PLYs")
    ap.add_argument("--max-depth", type=float, default=20.0, help="drop points beyond this (m)")
    ap.add_argument("--min-depth", type=float, default=0.01, help="drop points closer than this (m)")
    ap.add_argument("--stride", type=int, default=4, help="pixel subsample step (1 = full density)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    intr, stem_to_cid = load_intrinsics(args.frames_meta)
    only_cid = next(iter(intr)) if len(intr) == 1 else None

    depth_files = sorted(glob.glob(os.path.join(args.depth_dir, "*.png")))
    if args.limit:
        depth_files = depth_files[:args.limit]
    if not depth_files:
        raise FileNotFoundError(f"No depth PNGs under {args.depth_dir}")
    print(f"found {len(depth_files)} depth maps | stride={args.stride}")

    total_pts = 0
    for i, dpath in enumerate(depth_files):
        stem = os.path.splitext(os.path.basename(dpath))[0]

        # locate the matching RGB frame (jpeg/jpg/png)
        rgb_path = next((p for ext in (".jpeg", ".jpg", ".png")
                         for p in [os.path.join(args.image_dir, stem + ext)]
                         if os.path.isfile(p)), None)
        if rgb_path is None:
            raise FileNotFoundError(f"No RGB frame for {stem} in {args.image_dir}")

        depth_mm = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        depth = depth_mm.astype(np.float32) / 1000.0  # -> meters
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        H, W = depth.shape

        cid = only_cid or stem_to_cid[stem]
        k = intr[cid]
        # scale intrinsics if depth resolution differs from calibration resolution
        sx, sy = W / k["W"], H / k["H"]
        fx, fy, cx, cy = k["fx"] * sx, k["fy"] * sy, k["cx"] * sx, k["cy"] * sy

        s = args.stride
        d = depth[::s, ::s]
        rgb_s = rgb[::s, ::s]
        vv, uu = np.meshgrid(np.arange(0, H, s), np.arange(0, W, s), indexing="ij")

        valid = (d >= args.min_depth) & (d <= args.max_depth)
        z = d[valid]
        x = (uu[valid] - cx) * z / fx
        y = (vv[valid] - cy) * z / fy
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        col = rgb_s[valid].astype(np.uint8)

        write_ply_binary(os.path.join(args.out_dir, f"{stem}.ply"), xyz, col)
        total_pts += xyz.shape[0]
        if i % 50 == 0 or i == len(depth_files) - 1:
            print(f"  [{i + 1}/{len(depth_files)}] {stem}  {xyz.shape[0]} pts")

    print(f"done -> {args.out_dir} ({len(depth_files)} PLYs, {total_pts} pts total)")


if __name__ == "__main__":
    main()
