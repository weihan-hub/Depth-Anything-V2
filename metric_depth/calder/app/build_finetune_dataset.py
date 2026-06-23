"""Build the curated, materialized multi-session finetune dataset (§7–§8).

Pipeline: discover sessions -> per-session shard (+ materialize) -> deterministic
quota-greedy session split -> assemble train/val/test manifests + dataset_meta.

Incremental & idempotent: re-running after downloading more sessions builds only the
new sessions and never reshuffles prior split assignments.

    cd metric_depth
    # dry-run discovery
    uv run python -m calder.app.build_finetune_dataset --dry-run --include-unlisted
    # build the two test sessions
    uv run python -m calder.app.build_finetune_dataset --include-unlisted \
        --sessions 6dcb9a95-0d73-4f13-ad5e-c377b7564fbf \
                   c391d83a-2958-40b6-a637-7f963e64f07f
"""
import argparse
import datetime
import json
import os
import re

from calder.config import paths
from calder.lib.session_layout import detect_layout, parse_selection
from calder.lib import shard as shard_mod
from calder.lib import split as split_mod
from calder.lib import synthetic_shard as synth_mod

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def discover_present_unlisted(output_bases, known, run_subdir):
    """Scan base dirs for present <uuid>/<run_subdir> sessions not already in `known`."""
    found = []
    for base in output_bases:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            u = name.lower()
            if not _UUID_RE.match(u) or u in known:
                continue
            if os.path.isdir(os.path.join(base, name, run_subdir)):
                found.append(u)
                known.add(u)
    return found


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default=paths.SESSION_SELECTION)
    ap.add_argument("--session-root", action="append", default=None,
                    help="extra base dir(s) to search; repeatable")
    ap.add_argument("--run-subdir", default="nurec_run1")
    ap.add_argument("--gt-source", choices=["depth", "gsplat"], default="depth")
    ap.add_argument("--out-root", default=paths.FINETUNE_DATASET_ROOT)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="explicit uuid allowlist (overrides the selection list)")
    ap.add_argument("--include-unlisted", action="store_true",
                    help="also build present sessions not in the selection list")
    ap.add_argument("--force", action="store_true", help="rebuild cached shards")
    # flat pre-rendered synthetic deliveries (rgb/ + depth/ + frames_meta.json),
    # split at the frame level instead of the session level
    ap.add_argument("--synthetic-delivery", action="append", default=None,
                    help="path to a flat synthetic delivery dir; repeatable")
    ap.add_argument("--synthetic-name", action="append", default=None,
                    help="dataset name for the matching --synthetic-delivery "
                         "(default: the delivery dir's basename); repeatable, positional")
    ap.add_argument("--block-size", type=int, default=synth_mod.DEFAULT_BLOCK_SIZE,
                    help="synthetic frame-level split: consecutive waypoints per "
                         "leakage-safe block")
    # split
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-seed", default="calder-v1")
    # depth / quality
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=100.0)
    ap.add_argument("--min-valid-ratio", type=float, default=0.5)
    ap.add_argument("--min-frames-per-cam", type=int, default=50)
    # diversity
    ap.add_argument("--min-translation", type=float, default=0.2)
    ap.add_argument("--min-rotation", type=float, default=5.0)
    ap.add_argument("--dedup-mode", choices=["sequential", "coverage"], default="sequential",
                    help="sequential=compare vs last kept frame (O(n)); "
                         "coverage=compare vs all kept frames, loop-closure aware (O(n^2))")
    ap.add_argument("--no-dedup", action="store_true")
    # control
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()

    output_bases = list(paths.OUTPUT_BASES)
    if args.session_root:
        output_bases = args.session_root + output_bases

    # derived output paths (honor --out-root)
    data_dir = os.path.join(args.out_root, "data")
    manifests_dir = os.path.join(args.out_root, "manifests")
    shards_dir = os.path.join(manifests_dir, "shards")
    split_path = os.path.join(args.out_root, "split_assignment.json")
    meta_path = os.path.join(args.out_root, "dataset_meta.json")
    split_outputs = {"train": os.path.join(manifests_dir, "train.jsonl"),
                     "val": os.path.join(manifests_dir, "val.jsonl"),
                     "test": os.path.join(manifests_dir, "test.jsonl")}

    # --- Phase 0: discover ---
    if args.sessions:
        universe = [u.lower() for u in args.sessions]
    else:
        universe = parse_selection(args.selection)
    known = set(universe)
    if args.include_unlisted:
        unlisted = discover_present_unlisted(output_bases, known, args.run_subdir)
        universe = universe + unlisted

    layouts = {u: detect_layout(u, output_bases, args.gt_source, args.run_subdir)
               for u in universe}
    present = [u for u, l in layouts.items() if l.run_dir]
    absent = [u for u, l in layouts.items() if not l.run_dir]
    print(f"discovery: selected/listed universe={len(universe)}  "
          f"present={len(present)}  absent={len(absent)}  (gt_source={args.gt_source})")
    for u in present:
        l = layouts[u]
        print(f"  PRESENT {u}  state={l.state:18s} rgb={l.rgb_kind} cams={len(l.cameras)}")
    if absent and args.verbose:
        print(f"  ABSENT ({len(absent)}): not downloaded yet")
    if not present:
        print("!! no present sessions to build "
              "(none downloaded, or all off the selection list — try --include-unlisted "
              "or --sessions)")

    # pair each --synthetic-delivery with a name (positional --synthetic-name,
    # else the delivery dir's basename)
    synth_pairs = []
    for i, d in enumerate(args.synthetic_delivery or []):
        d = os.path.abspath(d)
        names = args.synthetic_name or []
        name = names[i] if i < len(names) else os.path.basename(d.rstrip("/"))
        synth_pairs.append((d, name))
    if synth_pairs:
        print(f"synthetic deliveries: {[n for _, n in synth_pairs]}")

    if args.dry_run:
        for d, name in synth_pairs:
            s = synth_mod.build_synthetic_delivery(d, name, args, data_dir, shards_dir)
            print(f"  SYNTHETIC {name}: cams={s['n_cameras']} waypoints={s['n_waypoints']} "
                  f"frames={s['n_frames']} planned(pre-dedup)={s['planned_split_pre_dedup']}")
        print("dry-run: stopping before shard build.")
        return

    # --- Phase 1–3: per-session shards ---
    os.makedirs(shards_dir, exist_ok=True)
    statuses = {}
    for u in present:
        st = shard_mod.build_shard(layouts[u], args, data_dir, shards_dir)
        statuses[u] = st
        tag = " (cached)" if st.get("cached") else ""
        tot = st.get("totals", {})
        print(f"  shard {u}: state={st['state']}{tag}  "
              f"kept={tot.get('kept', 0)} dropped={tot.get('dropped', 0)} "
              f"missing_rgb={tot.get('missing_rgb', 0)}")

    # --- Phase 3b: synthetic deliveries (frame-level split, own shard) ---
    synth_statuses = {}
    for d, name in synth_pairs:
        st = synth_mod.build_synthetic_delivery(d, name, args, data_dir, shards_dir)
        synth_statuses[name] = st
        print(f"  synthetic {name}: state={st['state']}  kept={st['totals']['kept']} "
              f"dropped={st['totals']['dropped']}  counts={st['counts']}")

    # --- Phase 4: deterministic quota-greedy split (only 'built' sessions) ---
    ratios = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    built = [u for u, st in statuses.items() if st["state"] == "built"]
    data = split_mod.load_assignment(split_path, args.split_seed, ratios)
    split_mod.assign_new_sessions(data, built, ratios, args.split_seed)
    # synthetic deliveries are split per frame, not per session: record a sentinel
    # so the whole-session assembler skips them (their rows carry their own split).
    for name in synth_statuses:
        data["assignments"][name] = "frame-level"
    split_mod.save_assignment(split_path, data)
    assignments = data["assignments"]

    # --- Phase 5: assemble global manifests from assigned shards ---
    os.makedirs(manifests_dir, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    writers = {s: open(p + ".tmp", "w") for s, p in split_outputs.items()}
    # deterministic order: by (split, session, camera, frame)
    for u in sorted(assignments):
        split = assignments[u]
        sp = os.path.join(shards_dir, f"{u}.jsonl")
        if split not in writers or not os.path.exists(sp):
            continue   # "frame-level" synthetic shards are assembled below
        with open(sp) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("kept"):
                    continue
                row["split"] = split
                writers[split].write(json.dumps(row) + "\n")
                counts[split] += 1
    # synthetic shards: each kept row already carries its own frame-level split
    for name in sorted(synth_statuses):
        sp = os.path.join(shards_dir, f"{name}.jsonl")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("kept"):
                    continue
                split = row.get("split")
                if split not in writers:
                    continue
                writers[split].write(json.dumps(row) + "\n")
                counts[split] += 1
    for s, w in writers.items():
        w.close()
        os.replace(split_outputs[s] + ".tmp", split_outputs[s])

    # --- dataset_meta.json ---
    sessions_meta = {}
    for u, st in statuses.items():
        sessions_meta[u] = {"state": st["state"],
                            "kept": st.get("totals", {}).get("kept", 0),
                            "dropped": st.get("totals", {}).get("dropped", 0),
                            "split": assignments.get(u)}
    for name, st in synth_statuses.items():
        sessions_meta[name] = {"state": st["state"], "kind": "synthetic",
                               "kept": st.get("totals", {}).get("kept", 0),
                               "dropped": st.get("totals", {}).get("dropped", 0),
                               "split": "frame-level",
                               "split_counts": st.get("counts", {})}
    meta = {
        "builder_version": shard_mod.BUILDER_VERSION,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "counts": counts,
        "n_sessions": {"present": len(present), "built": len(built), "absent": len(absent)},
        "sessions": sessions_meta,
        "absent": absent,
    }
    split_mod.save_assignment(meta_path, meta)   # reuse atomic json writer

    print(f"\nassembled: train={counts['train']}  val={counts['val']}  test={counts['test']}")
    empty = [s for s, c in counts.items() if c == 0]
    if empty:
        print(f"!! empty split(s): {empty} (expected at small session count; "
              f"fills in as more sessions are added)")
    print(f"outputs under {args.out_root}/ "
          f"(manifests/, data/, split_assignment.json, dataset_meta.json)")


if __name__ == "__main__":
    main()
