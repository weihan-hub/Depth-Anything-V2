"""Deterministic, incremental session -> train/val/test assignment (§7).

Quota-greedy: pure hashing does not guarantee the 70/15/15 ratios (small N can be
all-train), so each newly-built session is assigned to the split that is currently
MOST under its quota. Properties:
  - tracks the target ratios even at small N (N=2 -> train+val, N=3 -> all three),
  - STABLE: already-assigned sessions are never reassigned (no leakage as data grows),
  - reproducible: new sessions are processed in a fixed hash order, independent of
    download sequence.
"""
import hashlib
import json
import os

SPLITS = ["train", "val", "test"]


def _hash(seed, uuid):
    return int(hashlib.sha256((seed + uuid).encode()).hexdigest(), 16)


def load_assignment(path, seed, ratios, version=1):
    """Load split_assignment.json, or a fresh structure if absent."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"_header": {"seed": seed, "ratios": ratios, "version": version},
            "assignments": {}}


def save_assignment(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def assign_new_sessions(data, new_uuids, ratios, seed):
    """Assign each not-yet-assigned uuid to the most-under-quota split.

    Mutates and returns ``data`` (the loaded assignment structure).
    """
    assignments = data["assignments"]
    counts = {s: 0 for s in SPLITS}
    for s in assignments.values():
        if s in counts:
            counts[s] += 1

    todo = [u for u in new_uuids if u not in assignments]
    todo.sort(key=lambda u: _hash(seed, u))     # deterministic, download-order-independent

    for u in todo:
        total = sum(counts.values()) + 1

        def deficit(s):
            r = ratios.get(s, 0.0)
            return (counts[s] / total) / r if r > 0 else float("inf")

        # lowest deficit wins; ties broken by SPLITS order (train > val > test)
        best = min(SPLITS, key=lambda s: (deficit(s), SPLITS.index(s)))
        assignments[u] = best
        counts[best] += 1

    return data
