"""Per-session shard build + materialization — the idempotent unit of the dataset build.

For one session: iterate its GT depth frames, join RGB + pose + K, run the quality
(§8.2) and pose-diversity (§8.3) filters, materialize the kept frames'
RGB/GT/valid_mask/confidence into ``data/<session>/<cam>/...``, and emit one §3
sample per frame into ``shards/<session>.jsonl`` (+ a ``<session>.status.json``).

Re-runs skip a session whose shard+status already exist at the current builder
version (unless ``--force``), so adding newly-downloaded sessions is cheap.
"""
import json
import os
import shutil

import cv2
import numpy as np

from calder.lib.geometry import pose_to_4x4, projection_to_K, rig_and_side
from calder.lib import quality as quality_mod
from calder.lib.diversity import rotation_angle_deg

BUILDER_VERSION = "1"
DEPTH_SCALE = 1000.0          # uint16 millimeters -> meters
MIN_MATCH_RATE = 0.5          # R2 guard: below this a session is marked "suspect"


def _status_path(shards_dir, uuid):
    return os.path.join(shards_dir, f"{uuid}.status.json")


def _shard_path(shards_dir, uuid):
    return os.path.join(shards_dir, f"{uuid}.jsonl")


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _load_meta(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    cam_params = meta["camera_params_id_to_camera_params"]
    name_to_cpid = {cp["sensor_meta_data"]["sensor_name"]: cpid
                    for cpid, cp in cam_params.items()}
    kf_index = {kf["image_name"]: kf for kf in meta.get("keyframes_metadata", [])}
    return cam_params, name_to_cpid, kf_index


def is_cached(shards_dir, uuid):
    """True if a shard+status at the current builder version already exist."""
    sp, stp = _shard_path(shards_dir, uuid), _status_path(shards_dir, uuid)
    if not (os.path.exists(sp) and os.path.exists(stp)):
        return False
    try:
        with open(stp) as f:
            return json.load(f).get("builder_version") == BUILDER_VERSION
    except (json.JSONDecodeError, OSError):
        return False


def _materialize(dst_cam_dir, ts, rgb_src, gt_src, valid_mask):
    """Copy RGB + GT and write valid_mask/confidence PNGs. Returns the 4 dst paths."""
    subdirs = {k: os.path.join(dst_cam_dir, k)
               for k in ("rgb", "gt_depth", "valid_mask", "confidence")}
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    rgb_ext = os.path.splitext(rgb_src)[1] or ".jpeg"
    rgb_dst = os.path.join(subdirs["rgb"], ts + rgb_ext)
    gt_dst = os.path.join(subdirs["gt_depth"], ts + ".png")
    mask_dst = os.path.join(subdirs["valid_mask"], ts + ".png")
    conf_dst = os.path.join(subdirs["confidence"], ts + ".png")
    shutil.copy2(rgb_src, rgb_dst)
    shutil.copy2(gt_src, gt_dst)
    mask_u8 = (valid_mask.astype(np.uint8) * 255)
    cv2.imwrite(mask_dst, mask_u8)
    cv2.imwrite(conf_dst, mask_u8)          # V0: confidence == valid_mask
    return rgb_dst, gt_dst, mask_dst, conf_dst


def build_shard(layout, args, data_dir, shards_dir):
    """Build (or skip) one session's shard. Returns the status dict."""
    uuid = layout.uuid
    os.makedirs(shards_dir, exist_ok=True)

    if not args.force and is_cached(shards_dir, uuid):
        with open(_status_path(shards_dir, uuid)) as f:
            st = json.load(f)
        st["cached"] = True
        return st

    # Non-buildable session: empty shard + status, skip.
    if not layout.buildable:
        open(_shard_path(shards_dir, uuid), "w").close()   # empty shard
        st = {"uuid": uuid, "state": layout.state, "builder_version": BUILDER_VERSION,
              "run_dir": layout.run_dir, "cameras": {}, "totals": {}}
        _write_json_atomic(_status_path(shards_dir, uuid), st)
        return st

    cam_params, name_to_cpid, kf_index = _load_meta(layout.meta_path)
    cam_filter = set(args.cameras) if getattr(args, "cameras", None) else None
    rgb_ext = getattr(layout, "rgb_ext", ".jpeg")   # .jpeg pycusfm / .jpg batch_slam

    rows = []
    cam_stats = {}
    tot = {"frames": 0, "kept": 0, "dropped": 0, "missing_rgb": 0, "missing_kf": 0}

    for cam in layout.cameras:
        if cam_filter and cam not in cam_filter:
            continue
        gt_cam_dir = os.path.join(layout.gt_depth_root, cam)
        ts_list = sorted(
            (os.path.splitext(f)[0] for f in os.listdir(gt_cam_dir) if f.endswith(".png")),
            key=lambda s: int(s),
        )
        rig, side = rig_and_side(cam)
        cpid_default = name_to_cpid.get(cam)
        cstat = {"frames": len(ts_list), "kept": 0, "dropped": 0,
                 "depth_min_m": None, "depth_max_m": None}
        dmin, dmax = np.inf, -np.inf
        kept_poses = []          # (R, t) of frames kept so far in this (session, camera)

        for ts in ts_list:
            tot["frames"] += 1
            rgb_src = os.path.join(layout.rgb_dir, cam, ts + rgb_ext)
            if not os.path.exists(rgb_src):
                tot["missing_rgb"] += 1
                continue

            image_name = f"{cam}/{ts}{rgb_ext}"
            kf = kf_index.get(image_name)
            if kf is None:
                tot["missing_kf"] += 1
            cpid = kf.get("camera_params_id") if kf else None
            if cpid is None:                      # pycusfm left_left omits it
                cpid = cpid_default
            cp = cam_params[str(cpid)]            # JSON keys are strings; colmap stores int cpid
            K = projection_to_K(cp["calibration_parameters"]["projection_matrix"]["data"])
            world_T_cam = pose_to_4x4(kf["camera_to_world"]) if kf else None

            gt_src = os.path.join(layout.gt_depth_root, cam, ts + ".png")
            depth_mm = cv2.imread(gt_src, cv2.IMREAD_UNCHANGED)
            depth_m = depth_mm.astype(np.float32) / DEPTH_SCALE
            valid_mask = ((depth_m > 0) & (depth_m >= args.min_depth) &
                          (depth_m <= args.max_depth))
            q = quality_mod.compute_quality(depth_m, valid_mask)
            kept_q, reason = quality_mod.passes_filter(q, args.min_valid_ratio)

            # pose-diversity (only among quality-passing frames)
            #   sequential: compare vs the LAST kept frame (greedy chain, O(n))
            #   coverage:   compare vs ALL kept frames (loop-closure aware, O(n^2))
            trans_delta = rot_delta = None
            keep_div = True
            R = t = None
            if world_T_cam is not None:
                R = np.asarray(world_T_cam)[:3, :3]
                t = np.asarray(world_T_cam)[:3, 3]
            if kept_q and not args.no_dedup and R is not None and kept_poses:
                deltas = [(float(np.linalg.norm(t - kt)), rotation_angle_deg(kR, R))
                          for (kR, kt) in kept_poses]
                if getattr(args, "dedup_mode", "sequential") == "coverage":
                    # near-duplicate of ANY kept frame => drop; report nearest-by-translation
                    is_dup = any(td <= args.min_translation and rd <= args.min_rotation
                                 for td, rd in deltas)
                    keep_div = not is_dup
                    trans_delta, rot_delta = min(deltas, key=lambda d: d[0])
                else:  # sequential: vs the last kept frame
                    trans_delta, rot_delta = deltas[-1]
                    keep_div = (trans_delta > args.min_translation) or (rot_delta > args.min_rotation)

            kept = kept_q and keep_div
            if kept:
                drop_reason = None
            elif not kept_q:
                drop_reason = reason
            else:
                drop_reason = "duplicate_pose"

            if q["depth_min_m"] is not None:
                dmin, dmax = min(dmin, q["depth_min_m"]), max(dmax, q["depth_max_m"])

            gsplat_src = None
            if layout.gsplat_depth_root:
                cand = os.path.join(layout.gsplat_depth_root, cam, ts + ".png")
                gsplat_src = cand if os.path.exists(cand) else None
            foundation_src = (os.path.join(layout.gt_depth_root, cam, ts + ".png")
                              if args.gt_source == "depth" else None)

            if kept:
                if R is not None:
                    kept_poses.append((R, t))
                dst_cam_dir = os.path.join(data_dir, uuid, cam)
                rgb_dst, gt_dst, mask_dst, conf_dst = _materialize(
                    dst_cam_dir, ts, rgb_src, gt_src, valid_mask)
                tot["kept"] += 1; cstat["kept"] += 1
            else:
                rgb_dst = gt_dst = mask_dst = conf_dst = None
                tot["dropped"] += 1; cstat["dropped"] += 1

            rows.append({
                "sample_id": f"{uuid[:8]}_{cam}_{ts}",
                "session_id": uuid,
                "rig_id": rig, "camera_id": side, "camera_name": cam,
                "frame_id": ts, "timestamp": int(ts) / 1e9,
                "rgb_path": rgb_dst,
                "gt_depth_path": gt_dst,                 # back-compat alias (materialized GT)
                "foundation_depth_path": foundation_src, # original GT source (depth/<cam>)
                "gsplat_depth_path": gsplat_src,         # future GT (original path or null)
                "valid_mask_path": mask_dst,
                "confidence_path": conf_dst,
                "confidence_kind": "valid_mask",
                "K": K, "world_T_cam": world_T_cam,
                "neighbors": [],
                "quality": q,
                "kept": kept, "drop_reason": drop_reason,
                "trans_delta_m": trans_delta, "rot_delta_deg": rot_delta,
                "builder_version": BUILDER_VERSION,
            })

        cstat["depth_min_m"] = None if dmin == np.inf else dmin
        cstat["depth_max_m"] = None if dmax == -np.inf else dmax
        cam_stats[cam] = cstat

    # R2 guard: low RGB-match rate => suspect (don't assign unless --force)
    matched = tot["frames"] - tot["missing_rgb"]
    match_rate = (matched / tot["frames"]) if tot["frames"] else 0.0
    state = "built"
    if match_rate < MIN_MATCH_RATE and not args.force:
        state = "suspect"

    # write shard atomically (kept-only rows are materialized; dropped rows kept for audit)
    shard_tmp = _shard_path(shards_dir, uuid) + ".tmp"
    with open(shard_tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(shard_tmp, _shard_path(shards_dir, uuid))

    st = {"uuid": uuid, "state": state, "builder_version": BUILDER_VERSION,
          "run_dir": layout.run_dir, "rgb_kind": layout.rgb_kind,
          "gt_source": args.gt_source, "match_rate": round(match_rate, 4),
          "cameras": cam_stats, "totals": tot}
    _write_json_atomic(_status_path(shards_dir, uuid), st)
    return st
