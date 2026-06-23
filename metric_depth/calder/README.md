# calder — DA-V2 Metric Depth Finetune on Calder NuRec

Fine-tune **Depth-Anything-V2 metric (ViT-S)** on the Calder NuRec stereo session
and compare it against the off-the-shelf Hypersim metric checkpoint. GT depth =
`depth/<cam>/<ts>.png` (uint16 mm), RGB = `pycusfm_input/<cam>/<ts>.jpeg`.

## Project structure

```
calder/
├── README.md                      # this file
├── config/
│   └── paths.py                   # DATA_ROOT, CHECKPOINTS, default manifest/splits/results paths
├── lib/                           # reusable library code (imported, no side effects)
│   ├── dataset.py                 # CalderDepthDataset
│   └── model.py                   # MODEL_CONFIGS + build_model / load_state_flexible
├── app/                           # CLI entrypoints (uv run python -m calder.app.<x>)
│   ├── build_manifest.py          #   frames_meta.json -> manifest.jsonl
│   ├── split_manifest.py          #   manifest -> 70/30 train/test (random | contiguous)
│   ├── finetune.py                #   single-GPU finetune + per-epoch eval
│   ├── evaluate.py                #   eval a checkpoint; GT vs baseline vs finetuned panels
│   └── visualize_dataloader.py    #   dataloader sanity viz
├── test/
│   └── overfit_one_batch.py       # sanity gate: overfit ONE fixed batch to ~0 loss
├── datasets/                      # all jsonl
│   ├── manifest.jsonl
│   └── splits/
│       ├── front_left/{train,test}_{random,contiguous}.jsonl
│       └── all_cams/{train,test}_{random,contiguous}.jsonl
└── results/                       # all outputs
    ├── overfit/{relative_init,metric_init}/
    ├── finetune/<experiment>/<split>/     # best.pth, latest.pth, history.json, curves.png, train.log
    └── eval/<experiment>/                 # metrics_*.json, qual_*.png
```

`config/paths.py` derives every path from its own location, so scripts run from
`metric_depth/` regardless of the working directory. Edit it to point at a
different session / checkpoints.

## How to run

Run everything from the `metric_depth/` directory with `uv run` (see the repo's
uv environment). Module paths are `calder.app.<name>` / `calder.test.<name>`.

```bash
cd metric_depth

# 1. Build manifest (defaults to config.paths.DATA_ROOT / MANIFEST)
uv run python -m calder.app.build_manifest

# 2. Sanity gate (optional): overfit a single fixed batch to ~0 loss
uv run python -m calder.test.overfit_one_batch \
    --checkpoint ../checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
    --metric-checkpoint --max-depth 20 --steps 1500 --bs 2 --lr 1e-5

# 3. Split (all 3 left cams; --camera-name "" disables the single-camera filter)
uv run python -m calder.app.split_manifest --camera-name "" --split-mode contiguous \
    --train-out calder/datasets/splits/all_cams/train_contiguous.jsonl \
    --test-out  calder/datasets/splits/all_cams/test_contiguous.jsonl

# 4. Finetune (init from full Hypersim metric ckpt; dual-LR + poly decay + hflip)
#    best.pth is selected on --val-manifest; TEST is left untouched for step 5.
uv run python -m calder.app.finetune \
    --train-manifest calder/datasets/splits/all_cams/train_contiguous.jsonl \
    --val-manifest   calder/datasets/splits/all_cams/val_contiguous.jsonl \
    --max-depth 20 --epochs 10 --bs 4 --lr 5e-6 \
    --out-dir calder/results/finetune/all_cams/contiguous

# 5. Compare baseline vs finetuned on the held-out TEST set (metrics + qual panels)
uv run python -m calder.app.evaluate \
    --test-manifest calder/datasets/splits/all_cams/test_contiguous.jsonl \
    --checkpoint calder/results/finetune/all_cams/contiguous/best.pth \
    --baseline-checkpoint ../checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
    --max-depth 20 --tag finetuned_all_contig
```

## Key conventions

- **max_depth=20** is used for both baseline and finetuned so the comparison is
  fair (head output = `sigmoid * max_depth`; changing it rescales the head).
- Two splits: `random` (shuffled — optimistic, adjacent-frame leakage) and
  `contiguous` (first 70% train / last 30% test — honest generalization).
- Depth: uint16 mm -> meters; `valid_mask = depth > 0`; loss/metrics only on
  valid pixels within `[min_depth, max_depth]`.

## Results so far (all 3 left cams, 2061 frames, test set, max_depth=20)

| init | split | abs_rel | rmse (m) | d1 |
|---|---|---|---|---|
| Original Hypersim metric | — | ~0.84–0.87 | ~2.6 | ~0.03 |
| Finetuned | random | 0.071 | 0.37 | 0.969 |
| Finetuned | contiguous (honest) | 0.070 | 0.38 | 0.968 |

Off-the-shelf metric model is unusable on this automotive domain; finetuning
drops abs_rel ~92% and lifts d1 from 3% to ~97%.
