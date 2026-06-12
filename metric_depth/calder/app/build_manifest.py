"""Build a manifest.jsonl for the Calder NuRec session.

Reads the keyframe metadata (poses + intrinsics) and emits one JSONL line per
(camera, frame) that has a corresponding GT depth PNG on disk. Only the three
``*_left`` cameras have GT depth, so most keyframe entries are skipped.

Run from the ``metric_depth/`` directory:

    uv run python -m calder.app.build_manifest    # data-root / out default to calder.config.paths
"""
import argparse
import json
import os

from calder.config import paths
from calder.lib.geometry import (
    axis_angle_to_matrix, pose_to_4x4, projection_to_K, rig_and_side,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=paths.DATA_ROOT,
                    help="Path to .../nurec_run1")
    ap.add_argument("--out", default=paths.MANIFEST)
    ap.add_argument("--meta",
                    default="pycusfm_poses/keyframes/frames_meta.json",
                    help="Path to frames_meta.json relative to data-root")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    session_id = os.path.basename(os.path.dirname(root))  # .../<session-uuid>/nurec_run1
    meta_path = os.path.join(root, args.meta)
    with open(meta_path) as f:
        meta = json.load(f)

    cam_params = meta["camera_params_id_to_camera_params"]
    # Reverse map sensor_name -> camera_params_id. Some keyframes omit
    # 'camera_params_id' (the left_left camera), so we resolve via image_name.
    name_to_cpid = {
        cp["sensor_meta_data"]["sensor_name"]: cpid
        for cpid, cp in cam_params.items()
    }

    n_written = 0
    n_skip_depth = 0
    n_skip_rgb = 0
    with open(args.out, "w") as out:
        for kf in meta["keyframes_metadata"]:
            image_name = kf["image_name"]            # '<cam>/<ts_ns>.jpeg'
            cam_name = image_name.split("/")[0]
            cpid = kf.get("camera_params_id") or name_to_cpid[cam_name]
            cp = cam_params[cpid]
            ts_ns = os.path.splitext(os.path.basename(image_name))[0]

            rgb_path = os.path.join(root, "pycusfm_input", image_name)
            gt_depth_path = os.path.join(root, "depth", cam_name, f"{ts_ns}.png")

            if not os.path.exists(gt_depth_path):
                n_skip_depth += 1
                continue
            if not os.path.exists(rgb_path):
                n_skip_rgb += 1
                continue

            K = projection_to_K(cp["calibration_parameters"]["projection_matrix"]["data"])
            world_T_cam = pose_to_4x4(kf["camera_to_world"])
            body_T_cam = pose_to_4x4(
                cp["sensor_meta_data"]["sensor_to_vehicle_transform"])
            rig, side = rig_and_side(cam_name)

            entry = {
                "sample_id": f"{session_id[:8]}_{cam_name}_{ts_ns}",
                "session_id": session_id,
                "rig_id": rig,
                "camera_id": cam_name,  # Changed: camera_id 应该和camera_name一样
                "camera_name": cam_name,
                "frame_id": ts_ns,
                "timestamp": int(kf["timestamp_microseconds"]) / 1e6,
                "rgb_path": rgb_path,
                "gt_depth_path": gt_depth_path,
                "foundation_depth_path": None,
                "valid_mask_path": None,
                "confidence_path": None,
                "K": K,
                "world_T_cam": world_T_cam,
                "body_T_cam": body_T_cam,
                "neighbors": [],
            }
            out.write(json.dumps(entry) + "\n")
            n_written += 1

    print(f"manifest written: {args.out}")
    print(f"  entries: {n_written}")
    print(f"  skipped (no GT depth): {n_skip_depth}")
    print(f"  skipped (no RGB):      {n_skip_rgb}")


if __name__ == "__main__":
    main()
