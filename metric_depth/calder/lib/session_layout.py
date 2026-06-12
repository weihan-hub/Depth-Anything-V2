"""Session discovery + on-disk layout detection for the finetune-dataset build.

Two on-disk layouts exist across Calder sessions, so each session is probed:
  - RGB dir:  ``extracted/<cam>/`` (new) or ``pycusfm_input/<cam>/`` (old)
  - GT depth: ``depth/<cam>/`` (FoundationStereo, default) or
              ``gsplat_w_depth/renders_full/depth/<cam>/`` (GSplat, future)
"""
import os
import re
from dataclasses import dataclass, field

# RGB directory candidates, in preference order.
RGB_DIR_CANDIDATES = ["extracted", "pycusfm_input"]

# GT depth root relative to the run dir, per --gt-source.
GT_DEPTH_RELPATH = {
    "depth": "depth",
    "gsplat": os.path.join("gsplat_w_depth", "renders_full", "depth"),
}
GSPLAT_DEPTH_RELPATH = GT_DEPTH_RELPATH["gsplat"]

META_RELPATH = os.path.join("pycusfm_poses", "keyframes", "frames_meta.json")

_SESSION_RE = re.compile(r"session=([0-9a-fA-F-]{36})")
_CAM_RE = re.compile(r"^stereo_camera_")


def parse_selection(selection_path):
    """Parse sessions.txt -> ordered list of unique session UUIDs.

    Each line looks like ``program=calder-rig/project=default/session=<uuid>``.
    Lines without a session token (blank/comment) are skipped.
    """
    uuids = []
    seen = set()
    if not os.path.exists(selection_path):
        return uuids
    with open(selection_path) as f:
        for line in f:
            m = _SESSION_RE.search(line)
            if not m:
                continue
            u = m.group(1).lower()
            if u not in seen:
                seen.add(u)
                uuids.append(u)
    return uuids


def resolve_run_dir(uuid, output_bases, run_subdir="nurec_run1"):
    """First existing ``<base>/<uuid>/<run_subdir>`` over the search bases, else None."""
    for base in output_bases:
        cand = os.path.join(base, uuid, run_subdir)
        if os.path.isdir(cand):
            return cand
    return None


def _cam_dirs(root):
    """*_left camera subdirs under a depth root (skips foundation_stereo_v2 etc.)."""
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if _CAM_RE.match(name) and os.path.isdir(os.path.join(root, name)):
            out.append(name)
    return out


@dataclass
class SessionLayout:
    uuid: str
    run_dir: str | None = None
    rgb_dir: str | None = None          # absolute path to the RGB parent (extracted/ ...)
    rgb_kind: str | None = None         # "extracted" | "pycusfm_input"
    gt_depth_root: str | None = None    # absolute path to chosen GT depth parent
    gsplat_depth_root: str | None = None
    meta_path: str | None = None
    cameras: list = field(default_factory=list)   # *_left cams that have GT depth
    state: str = "absent"               # absent|ok|skipped_no_rgb|skipped_no_gt_depth|skipped_no_meta

    @property
    def buildable(self):
        return self.state == "ok"


def detect_layout(uuid, output_bases, gt_source="depth", run_subdir="nurec_run1"):
    """Resolve a session's run dir and probe its RGB / GT-depth / meta layout."""
    run_dir = resolve_run_dir(uuid, output_bases, run_subdir)
    lay = SessionLayout(uuid=uuid, run_dir=run_dir)
    if run_dir is None:
        lay.state = "absent"
        return lay

    # RGB dir
    for cand in RGB_DIR_CANDIDATES:
        p = os.path.join(run_dir, cand)
        if os.path.isdir(p):
            lay.rgb_dir, lay.rgb_kind = p, cand
            break

    # GT depth root (per source) + always record gsplat root if present
    gt_root = os.path.join(run_dir, GT_DEPTH_RELPATH[gt_source])
    lay.gt_depth_root = gt_root if os.path.isdir(gt_root) else None
    gsplat_root = os.path.join(run_dir, GSPLAT_DEPTH_RELPATH)
    lay.gsplat_depth_root = gsplat_root if os.path.isdir(gsplat_root) else None

    # meta
    meta = os.path.join(run_dir, META_RELPATH)
    lay.meta_path = meta if os.path.exists(meta) else None

    # cameras with GT depth
    if lay.gt_depth_root:
        lay.cameras = _cam_dirs(lay.gt_depth_root)

    # state
    if lay.rgb_dir is None:
        lay.state = "skipped_no_rgb"
    elif lay.gt_depth_root is None or not lay.cameras:
        lay.state = "skipped_no_gt_depth"
    elif lay.meta_path is None:
        lay.state = "skipped_no_meta"
    else:
        lay.state = "ok"
    return lay
