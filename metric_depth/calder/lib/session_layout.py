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
# batch_slam fallback: convert_tum_to_colmap writes a pycusfm-format meta here
# (same schema, poses in keyframes_metadata.camera_to_world).
COLMAP_META_RELPATH = os.path.join("colmap", "kpmap", "keyframes", "frames_meta.json")

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


def _detect_rgb_ext(rgb_dir):
    """Sample the first image under the first stereo_camera_* subdir to learn the
    RGB extension: .jpeg (pycusfm) or .jpg (batch_slam). Defaults to .jpeg."""
    if not rgb_dir or not os.path.isdir(rgb_dir):
        return ".jpeg"
    for name in sorted(os.listdir(rgb_dir)):
        sub = os.path.join(rgb_dir, name)
        if not (_CAM_RE.match(name) and os.path.isdir(sub)):
            continue
        for f in os.listdir(sub):
            ext = os.path.splitext(f)[1].lower()
            if ext in (".jpeg", ".jpg"):
                return ext
    return ".jpeg"


@dataclass
class SessionLayout:
    uuid: str
    run_dir: str | None = None
    rgb_dir: str | None = None          # absolute path to the RGB parent (extracted/ ...)
    rgb_kind: str | None = None         # "extracted" | "pycusfm_input"
    rgb_ext: str = ".jpeg"              # actual RGB ext on disk (.jpeg pycusfm / .jpg batch_slam)
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

    # RGB dir + actual extension (.jpeg pycusfm / .jpg batch_slam)
    for cand in RGB_DIR_CANDIDATES:
        p = os.path.join(run_dir, cand)
        if os.path.isdir(p):
            lay.rgb_dir, lay.rgb_kind = p, cand
            break
    if lay.rgb_dir:
        lay.rgb_ext = _detect_rgb_ext(lay.rgb_dir)

    # GT depth root (per source) + always record gsplat root if present
    gt_root = os.path.join(run_dir, GT_DEPTH_RELPATH[gt_source])
    lay.gt_depth_root = gt_root if os.path.isdir(gt_root) else None
    gsplat_root = os.path.join(run_dir, GSPLAT_DEPTH_RELPATH)
    lay.gsplat_depth_root = gsplat_root if os.path.isdir(gsplat_root) else None

    # pose/intrinsics meta: pycusfm layout first, colmap (batch_slam) fallback
    lay.meta_path = None
    for relpath in (META_RELPATH, COLMAP_META_RELPATH):
        cand = os.path.join(run_dir, relpath)
        if os.path.exists(cand):
            lay.meta_path = cand
            break

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
