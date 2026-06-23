"""Ingest a flat pre-rendered *synthetic* delivery into one frame-level-split shard.

Unlike a real Calder session (per-camera ``stereo_camera_*/`` subdirs + a
``pycusfm_poses/.../frames_meta.json``), a synthetic delivery is a single flat
folder::

    <delivery>/rgb/<name>.png        8-bit RGB
    <delivery>/depth/<name>.png      uint16 depth in millimeters
    <delivery>/frames_meta.json      CMN-rig-schema intrinsics + per-frame poses

with ``name = {pass}_{frame:06d}_{eye}`` (e.g. ``eye_forward_000200_front_left``).
The camera is recovered from each keyframe's ``camera_params_id`` ->
``sensor_meta_data.sensor_name`` (``stereo_camera_front_left`` ...).

Because the whole delivery is one logical dataset, it cannot be split at the
session level (that would dump all of it into a single split). Instead each frame
is assigned a split. To avoid train<->test leakage between the highly-correlated
stereo eyes and temporally-adjacent frames, the split unit is the *waypoint*
``(pass, frame_index)`` -- all 6 cameras of a waypoint, and a contiguous block of
neighboring waypoints within a pass, share one split.

The materialized layout and per-row shard schema match the real-session builder
(:mod:`calder.lib.shard`), so the dataloader ingests it unchanged. The only
difference is that each shard row already carries its own ``split`` field.
"""
import json
import os
import re
import shutil

import cv2
import numpy as np

from calder.lib.geometry import pose_to_4x4, projection_to_K, rig_and_side
from calder.lib import quality as quality_mod
from calder.lib.diversity import rotation_angle_deg
from calder.lib.shard import BUILDER_VERSION, DEPTH_SCALE, _materialize, _write_json_atomic

SPLITS = ["train", "val", "test"]
DEFAULT_BLOCK_SIZE = 20          # waypoints per contiguous split block (leakage guard)

# name = {pass}_{frame:06d}_{eye}; pass and eye are each two tokens, frame one.
#   eye_forward_000200_front_left -> ("eye_forward", 200, "front_left")
_NAME_RE = re.compile(r"^(?P<pass>[a-z]+_[a-z]+)_(?P<frame>\d+)_(?P<eye>[a-z]+_[a-z]+)$")


def parse_name(stem):
    """``eye_forward_000200_front_left`` -> (pass, frame_index, eye) or None."""
    m = _NAME_RE.match(stem)
    if not m:
        return None
    return m.group("pass"), int(m.group("frame")), m.group("eye")


def _projection_data(proj):
    """Accept both the real-session ``{"data": [...]}`` and the synthetic flat list."""
    return proj["data"] if isinstance(proj, dict) else proj


def _load_meta(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    cam_params = meta["camera_params_id_to_camera_params"]
    kf = meta.get("keyframes_metadata", [])
    return cam_params, kf


def assign_block_splits(waypoints_by_pass, ratios, block_size):
    """Assign each ``(pass, frame_index)`` waypoint to a split in contiguous blocks.

    Deterministic, quota-greedy at the block level: blocks of up to ``block_size``
    consecutive waypoints (within a single pass) are kept together and the block is
    handed to the split that is currently most under its target ratio. Keeping whole
    blocks together means stereo eyes (same waypoint) and temporal neighbors (same
    block) never straddle the train/val/test boundary.
    """
    counts = {s: 0 for s in SPLITS}
    assignment = {}
    for pass_name in sorted(waypoints_by_pass):
        idxs = sorted(waypoints_by_pass[pass_name])
        for b in range(0, len(idxs), block_size):
            block = idxs[b:b + block_size]
            total = sum(counts.values()) + len(block)

            def deficit(s):
                r = ratios.get(s, 0.0)
                return (counts[s] / total) / r if r > 0 else float("inf")

            best = min(SPLITS, key=lambda s: (deficit(s), SPLITS.index(s)))
            for idx in block:
                assignment[(pass_name, idx)] = best
            counts[best] += len(block)
    return assignment


def _index_keyframes(cam_params, keyframes):
    """Group keyframes by camera_name, recovering (K, sensor_name) per cpid.

    Returns: (by_cam: {cam_name: [(stem, pass, frame_idx, eye, kf)]},
              K_by_cam: {cam_name: 3x3}, waypoints_by_pass: {pass: set(frame_idx)}).
    Keyframes whose name doesn't parse, or whose cpid is unknown, are skipped.
    """
    name_by_cpid = {cpid: cp["sensor_meta_data"]["sensor_name"]
                    for cpid, cp in cam_params.items()}
    K_by_cpid = {cpid: projection_to_K(
                     _projection_data(cp["calibration_parameters"]["projection_matrix"]))
                 for cpid, cp in cam_params.items()}

    by_cam, K_by_cam, waypoints_by_pass = {}, {}, {}
    for kf in keyframes:
        stem = os.path.splitext(kf["image_name"])[0]
        parsed = parse_name(stem)
        if parsed is None:
            continue
        pass_name, frame_idx, eye = parsed
        cpid = str(kf.get("camera_params_id"))
        cam = name_by_cpid.get(cpid)
        if cam is None:
            continue
        by_cam.setdefault(cam, []).append((stem, pass_name, frame_idx, eye, kf))
        K_by_cam.setdefault(cam, K_by_cpid[cpid])
        waypoints_by_pass.setdefault(pass_name, set()).add(frame_idx)
    return by_cam, K_by_cam, waypoints_by_pass


def build_synthetic_delivery(delivery_dir, name, args, data_dir, shards_dir):
    """Build one frame-level-split shard from a flat synthetic delivery.

    Returns a status dict (also written to ``shards/<name>.status.json``) with a
    per-split ``counts`` breakdown. ``args`` reuses the build_finetune_dataset
    namespace (min_depth, max_depth, min_valid_ratio, min_translation,
    min_rotation, dedup_mode, no_dedup, train/val/test_frac, dry_run, ...).
    """
    meta_path = os.path.join(delivery_dir, "frames_meta.json")
    rgb_dir = os.path.join(delivery_dir, "rgb")
    depth_dir = os.path.join(delivery_dir, "depth")
    cam_params, keyframes = _load_meta(meta_path)
    by_cam, K_by_cam, waypoints_by_pass = _index_keyframes(cam_params, keyframes)

    ratios = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    block_size = getattr(args, "block_size", DEFAULT_BLOCK_SIZE)
    split_of = assign_block_splits(waypoints_by_pass, ratios, block_size)

    counts = {s: 0 for s in SPLITS}                 # kept frames per split
    tot = {"frames": 0, "kept": 0, "dropped": 0, "missing_rgb": 0, "missing_depth": 0}

    # dry-run: report planned coverage without reading depth / writing anything.
    if getattr(args, "dry_run", False):
        planned = {s: 0 for s in SPLITS}
        for cam, items in by_cam.items():
            for stem, pass_name, frame_idx, eye, kf in items:
                planned[split_of[(pass_name, frame_idx)]] += 1
        n_wp = sum(len(v) for v in waypoints_by_pass.values())
        return {"name": name, "state": "dry-run", "builder_version": BUILDER_VERSION,
                "delivery_dir": delivery_dir, "n_cameras": len(by_cam),
                "n_waypoints": n_wp, "n_frames": sum(len(v) for v in by_cam.values()),
                "planned_split_pre_dedup": planned, "block_size": block_size}

    rows = []
    cam_stats = {}
    for cam in sorted(by_cam):
        rig, side = rig_and_side(cam)
        K = K_by_cam[cam]
        # dedup is a *sequential* chain along one continuous trajectory, so reset the
        # kept-pose history at each pass boundary (the 4 passes are separate runs).
        items = sorted(by_cam[cam], key=lambda it: (it[1], it[2]))   # (pass, frame_idx)
        cstat = {"frames": len(items), "kept": 0, "dropped": 0,
                 "depth_min_m": None, "depth_max_m": None}
        dmin, dmax = np.inf, -np.inf
        kept_poses = []
        cur_pass = None

        for stem, pass_name, frame_idx, eye, kf in items:
            if pass_name != cur_pass:
                kept_poses = []
                cur_pass = pass_name
            tot["frames"] += 1

            rgb_src = os.path.join(rgb_dir, stem + ".png")
            gt_src = os.path.join(depth_dir, stem + ".png")
            if not os.path.exists(rgb_src):
                tot["missing_rgb"] += 1
                continue
            if not os.path.exists(gt_src):
                tot["missing_depth"] += 1
                continue

            depth_mm = cv2.imread(gt_src, cv2.IMREAD_UNCHANGED)
            depth_m = depth_mm.astype(np.float32) / DEPTH_SCALE
            valid_mask = ((depth_m > 0) & (depth_m >= args.min_depth) &
                          (depth_m <= args.max_depth))
            q = quality_mod.compute_quality(depth_m, valid_mask)
            kept_q, reason = quality_mod.passes_filter(q, args.min_valid_ratio)

            world_T_cam = pose_to_4x4(kf["camera_to_world"])
            R = np.asarray(world_T_cam)[:3, :3]
            t = np.asarray(world_T_cam)[:3, 3]

            # pose-diversity dedup (same logic as the real-session shard builder)
            trans_delta = rot_delta = None
            keep_div = True
            if kept_q and not args.no_dedup and kept_poses:
                deltas = [(float(np.linalg.norm(t - kt)), rotation_angle_deg(kR, R))
                          for (kR, kt) in kept_poses]
                if getattr(args, "dedup_mode", "sequential") == "coverage":
                    is_dup = any(td <= args.min_translation and rd <= args.min_rotation
                                 for td, rd in deltas)
                    keep_div = not is_dup
                    trans_delta, rot_delta = min(deltas, key=lambda d: d[0])
                else:
                    trans_delta, rot_delta = deltas[-1]
                    keep_div = (trans_delta > args.min_translation) or (rot_delta > args.min_rotation)

            kept = kept_q and keep_div
            drop_reason = None if kept else (reason if not kept_q else "duplicate_pose")
            split = split_of[(pass_name, frame_idx)]

            if q["depth_min_m"] is not None:
                dmin, dmax = min(dmin, q["depth_min_m"]), max(dmax, q["depth_max_m"])

            if kept:
                kept_poses.append((R, t))
                dst_cam_dir = os.path.join(data_dir, name, cam)
                rgb_dst, gt_dst, mask_dst, conf_dst = _materialize(
                    dst_cam_dir, stem, rgb_src, gt_src, valid_mask)
                tot["kept"] += 1; cstat["kept"] += 1
                counts[split] += 1
            else:
                rgb_dst = gt_dst = mask_dst = conf_dst = None
                tot["dropped"] += 1; cstat["dropped"] += 1

            rows.append({
                "sample_id": f"{name}_{cam}_{stem}",
                "session_id": name,
                "rig_id": rig, "camera_id": side, "camera_name": cam,
                "frame_id": stem, "timestamp": int(kf["timestamp_microseconds"]) / 1e6,
                "rgb_path": rgb_dst,
                "gt_depth_path": gt_dst,
                "foundation_depth_path": gt_src,         # original synthetic GT source
                "gsplat_depth_path": None,
                "valid_mask_path": mask_dst,
                "confidence_path": conf_dst,
                "confidence_kind": "valid_mask",
                "K": K, "world_T_cam": world_T_cam,
                "neighbors": [],
                "quality": q,
                "kept": kept, "drop_reason": drop_reason,
                "trans_delta_m": trans_delta, "rot_delta_deg": rot_delta,
                "split": split,                          # frame-level split (per row)
                "builder_version": BUILDER_VERSION,
            })

        cstat["depth_min_m"] = None if dmin == np.inf else dmin
        cstat["depth_max_m"] = None if dmax == -np.inf else dmax
        cam_stats[cam] = cstat

    shard_path = os.path.join(shards_dir, f"{name}.jsonl")
    shard_tmp = shard_path + ".tmp"
    with open(shard_tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(shard_tmp, shard_path)

    st = {"name": name, "state": "built", "builder_version": BUILDER_VERSION,
          "kind": "synthetic", "split_mode": "frame-level", "block_size": block_size,
          "delivery_dir": delivery_dir, "cameras": cam_stats,
          "totals": tot, "counts": counts}
    _write_json_atomic(os.path.join(shards_dir, f"{name}.status.json"), st)
    return st
