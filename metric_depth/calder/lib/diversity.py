"""Pose-based near-duplicate suppression (§8.3).

Deterministic (no RNG): within one (session, camera) stream sorted by timestamp,
keep a frame iff it moved enough vs the LAST KEPT frame — translation > min_trans
OR rotation > min_rot. This is the dataset-build diversity filter; it is NOT the
§18 training sampler (that's a later round).
"""
import numpy as np


def _R_t(pose4x4):
    T = np.asarray(pose4x4, dtype=np.float64)
    return T[:3, :3], T[:3, 3]


def rotation_angle_deg(Ra, Rb):
    """Geodesic angle (degrees) between two 3x3 rotation matrices."""
    R = Ra.T @ Rb
    cos = (np.trace(R) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, float(cos)))
    return float(np.degrees(np.arccos(cos)))


def suppress_near_duplicates(poses, min_translation=0.2, min_rotation=5.0):
    """``poses``: list of 4x4 (temporal order). Returns list of dicts per frame:
    {kept: bool, trans_delta_m: float|None, rot_delta_deg: float|None}.

    First frame is always kept (deltas None). A frame is kept iff translation or
    rotation vs the last kept frame exceeds its threshold; deltas are always
    reported (vs last kept) for auditability.
    """
    out = []
    last_R = last_t = None
    for pose in poses:
        if pose is None:                       # no pose -> can't dedup; keep, deltas null
            out.append({"kept": True, "trans_delta_m": None, "rot_delta_deg": None})
            continue
        R, t = _R_t(pose)
        if last_R is None:
            out.append({"kept": True, "trans_delta_m": None, "rot_delta_deg": None})
            last_R, last_t = R, t
            continue
        td = float(np.linalg.norm(t - last_t))
        rd = rotation_angle_deg(last_R, R)
        kept = (td > min_translation) or (rd > min_rotation)
        out.append({"kept": kept, "trans_delta_m": td, "rot_delta_deg": rd})
        if kept:
            last_R, last_t = R, t
    return out
