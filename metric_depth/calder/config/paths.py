"""Central path config for the calder DA-V2 metric finetune project.

All paths are derived from this file's location so scripts run regardless of the
current working directory (as long as they're launched as ``python -m calder.*``
from ``metric_depth/`` so the repo packages import).
"""
import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CALDER_ROOT = os.path.dirname(_CONFIG_DIR)          # .../metric_depth/calder
METRIC_ROOT = os.path.dirname(CALDER_ROOT)          # .../metric_depth
REPO_ROOT = os.path.dirname(METRIC_ROOT)            # .../Depth-Anything-V2

# --- external data (single legacy Calder NuRec session, used by build_manifest) ---
DATA_ROOT = ("/mnt/data/data/calder/output/dev_sessions/"
             "c391d83a-2958-40b6-a637-7f963e64f07f/nurec_run1")

# --- multi-session finetune dataset build ---
# Base dirs searched (in order) to resolve <session>/<run-subdir> on disk.
CALDER_OUTPUT = "/mnt/data/data/calder/output"
OUTPUT_BASES = [
    CALDER_OUTPUT,
    os.path.join(CALDER_OUTPUT, "dev_sessions"),   # legacy location (e.g. c391d83a)
]
SESSION_SELECTION = os.path.join(CALDER_OUTPUT, "_finetune_session_selection", "sessions.txt")

FINETUNE_DATASET_ROOT = "/mnt/data/data/calder/fine_tune_dataset"
FT_DATA_DIR = os.path.join(FINETUNE_DATASET_ROOT, "data")              # data/<session>/<cam>/...
FT_MANIFESTS_DIR = os.path.join(FINETUNE_DATASET_ROOT, "manifests")
FT_SHARDS_DIR = os.path.join(FT_MANIFESTS_DIR, "shards")              # shards/<session>.jsonl
FT_TRAIN_MANIFEST = os.path.join(FT_MANIFESTS_DIR, "train.jsonl")
FT_VAL_MANIFEST = os.path.join(FT_MANIFESTS_DIR, "val.jsonl")
FT_TEST_MANIFEST = os.path.join(FT_MANIFESTS_DIR, "test.jsonl")
FT_SPLIT_ASSIGNMENT = os.path.join(FINETUNE_DATASET_ROOT, "split_assignment.json")
FT_DATASET_META = os.path.join(FINETUNE_DATASET_ROOT, "dataset_meta.json")

# --- pretrained checkpoints ---
CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")
METRIC_HYPERSIM_VITS = os.path.join(CHECKPOINTS, "depth_anything_v2_metric_hypersim_vits.pth")
RELATIVE_VITS = os.path.join(CHECKPOINTS, "depth_anything_v2_vits.pth")

# --- datasets (manifest + splits) ---
DATASETS = os.path.join(CALDER_ROOT, "datasets")
MANIFEST = os.path.join(DATASETS, "manifest.jsonl")
SPLITS = os.path.join(DATASETS, "splits")

# --- results (training / eval / overfit outputs) ---
RESULTS = os.path.join(CALDER_ROOT, "results")
RESULTS_OVERFIT = os.path.join(RESULTS, "overfit")
RESULTS_FINETUNE = os.path.join(RESULTS, "finetune")
RESULTS_EVAL = os.path.join(RESULTS, "eval")
