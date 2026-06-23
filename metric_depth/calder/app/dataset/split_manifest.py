"""19.3 Train/test split of the Calder manifest.

Take manifest.jsonl, keep one camera (default front_left), and write a 70/30
train/test split. Two modes:

  - random:     fixed-seed shuffle then slice. Matches "70% train / 30% test"
                literally, but because these are consecutive video frames the
                test frames sit next to train frames in time and look nearly
                identical -> optimistic (leaky) metrics.
  - contiguous: sort by timestamp, first 70% -> train, last 30% -> test. No
                adjacent-frame leakage; a more honest generalization test
                (test is a different stretch of the trajectory).

    cd metric_depth
    uv run python -m calder.app.dataset.split_manifest --camera-name "" --split-mode contiguous \
        --train-out calder/datasets/splits/all_cams/train_contiguous.jsonl \
        --test-out  calder/datasets/splits/all_cams/test_contiguous.jsonl
"""
import argparse
import json
import random

from calder.config import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=paths.MANIFEST)
    ap.add_argument("--camera-name", default="stereo_camera_front_left")
    ap.add_argument("--split-mode", choices=["random", "contiguous"], default="random")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--test-out", required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    if args.camera_name:
        entries = [e for e in entries if e["camera_name"] == args.camera_name]
    if not entries:
        raise SystemExit(f"no entries for camera {args.camera_name!r}")

    # frame_id is the ns timestamp string -> sort chronologically
    entries.sort(key=lambda e: int(e["frame_id"]))
    n = len(entries)
    n_train = int(round(n * args.train_frac))

    if args.split_mode == "random":
        idx = list(range(n))
        random.Random(args.seed).shuffle(idx)
        train_idx = sorted(idx[:n_train])
        test_idx = sorted(idx[n_train:])
        train = [entries[i] for i in train_idx]
        test = [entries[i] for i in test_idx]
    else:  # contiguous
        train = entries[:n_train]
        test = entries[n_train:]

    for path, rows in [(args.train_out, train), (args.test_out, test)]:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"camera={args.camera_name}  mode={args.split_mode}  total={n}")
    print(f"  train={len(train)} -> {args.train_out}")
    print(f"  test ={len(test)} -> {args.test_out}")


if __name__ == "__main__":
    main()
