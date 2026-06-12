"""Per-frame quality metadata + hard filter (§8.2).

Only signals actually available today are computed: valid-depth ratio and depth
statistics from the GT depth map. Slots for absent signals (alpha / support /
reproj) are present but null, so V1/V2 can fill them later without a schema change.
"""
import numpy as np


def compute_quality(depth_m, valid_mask):
    """Quality dict for one frame. ``depth_m`` meters float32, ``valid_mask`` bool."""
    valid_mask = valid_mask.astype(bool)
    ratio = float(valid_mask.mean()) if valid_mask.size else 0.0
    v = depth_m[valid_mask]
    if v.size == 0:
        stats = dict(depth_min_m=None, depth_median_m=None, depth_max_m=None, depth_mean_m=None)
    else:
        stats = dict(
            depth_min_m=float(v.min()),
            depth_median_m=float(np.median(v)),
            depth_max_m=float(v.max()),
            depth_mean_m=float(v.mean()),
        )
    return {
        "valid_depth_ratio": ratio,
        **stats,
        # V1/V2 slots — populated when alpha/support/reproj maps exist on disk.
        "alpha_mean": None,
        "support_count_mean": None,
        "reproj_error_mean": None,
    }


def passes_filter(quality, min_valid_ratio):
    """Return (kept: bool, reason: str|None)."""
    if quality["depth_min_m"] is None:
        return False, "degenerate_depth"
    if quality["valid_depth_ratio"] < min_valid_ratio:
        return False, "low_valid_ratio"
    # all-equal / non-finite depth => degenerate
    if quality["depth_max_m"] <= quality["depth_min_m"]:
        return False, "degenerate_depth"
    if not np.isfinite([quality["depth_min_m"], quality["depth_max_m"]]).all():
        return False, "degenerate_depth"
    return True, None
