"""19.5 Evaluate a checkpoint on a Calder test split.

Loads a metric checkpoint (original Hypersim metric OR a finetuned best.pth --
both are raw model state_dicts), runs it over the test manifest at native depth
resolution (image-only resize + bilinear upsample of the prediction, exactly
like train.py's val loop), and reports the full eval_depth metric table. Run it
once per checkpoint to compare original vs finetuned on the SAME test set.

    cd metric_depth
    uv run python -m calder.app.evaluate \
        --test-manifest calder/datasets/splits/all_cams/test_contiguous.jsonl \
        --checkpoint calder/results/finetune/all_cams/contiguous/best.pth \
        --baseline-checkpoint <original metric ckpt> \
        --max-depth 20 --tag finetuned_all_contig --n-viz 4
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calder.config import paths
from calder.lib.dataset import CalderDepthDataset
from calder.lib.model import MODEL_CONFIGS, build_model, load_state_flexible
from util.metric import eval_depth

METRIC_KEYS = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

# GT-depth ranges for the per-bucket error breakdown (pixel-pooled across the set).
BUCKETS = [(0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    ap.add_argument("--checkpoint", required=True,
                    help="primary checkpoint metrics are computed on (e.g. finetuned best.pth)")
    ap.add_argument("--baseline-checkpoint", default=None,
                    help="optional 2nd checkpoint (e.g. original metric ckpt); when set, "
                         "the qualitative panel becomes GT vs baseline vs finetuned")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--min-depth", type=float, default=0.001)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--out-dir", default=paths.RESULTS_EVAL)
    ap.add_argument("--n-viz", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = (args.img_size, args.img_size)

    model = build_model(args.encoder, args.max_depth)
    info = model.load_state_dict(load_state_flexible(args.checkpoint), strict=False)
    print(f"[{args.tag}] loaded {args.checkpoint}: "
          f"{len(info.missing_keys)} missing / {len(info.unexpected_keys)} unexpected")
    model = model.to(device).eval()

    baseline_model = None
    if args.baseline_checkpoint:
        baseline_model = build_model(args.encoder, args.max_depth)
        binfo = baseline_model.load_state_dict(load_state_flexible(args.baseline_checkpoint), strict=False)
        print(f"[{args.tag}] baseline {args.baseline_checkpoint}: "
              f"{len(binfo.missing_keys)} missing / {len(binfo.unexpected_keys)} unexpected")
        baseline_model = baseline_model.to(device).eval()

    testset = CalderDepthDataset(args.test_manifest, mode='val', size=size)
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    agg = {k: 0.0 for k in METRIC_KEYS}
    nsamples = 0
    viz = []
    # pixel-pooled accumulators per GT-depth bucket
    bacc = [{'abs_rel': 0.0, 'se': 0.0, 'd1': 0.0, 'n': 0} for _ in BUCKETS]
    with torch.no_grad():
        for idx, sample in enumerate(loader):
            img = sample['image'].to(device).float()
            depth = sample['depth'].to(device)[0]
            valid = sample['valid_mask'].to(device)[0]
            pred = model(img)
            pred = F.interpolate(pred[:, None], depth.shape[-2:],
                                 mode='bilinear', align_corners=True)[0, 0]
            mask = (valid == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            if mask.sum() < 10:
                continue
            cur = eval_depth(pred[mask], depth[mask])
            for k in METRIC_KEYS:
                agg[k] += cur[k]
            nsamples += 1

            # per-bucket pixel-pooled accumulation (abs_rel / rmse / d1 by GT range)
            p_all, g_all = pred[mask], depth[mask]
            ratio = torch.maximum(g_all / p_all, p_all / g_all)
            for bi, (lo, hi) in enumerate(BUCKETS):
                bm = (g_all >= lo) & (g_all < hi)
                cnt = int(bm.sum())
                if cnt == 0:
                    continue
                pb, gb = p_all[bm], g_all[bm]
                bacc[bi]['abs_rel'] += (torch.abs(pb - gb) / gb).sum().item()
                bacc[bi]['se'] += ((pb - gb) ** 2).sum().item()
                bacc[bi]['d1'] += (ratio[bm] < 1.25).sum().item()
                bacc[bi]['n'] += cnt
            if len(viz) < args.n_viz:
                base_pr = None
                if baseline_model is not None:
                    bp = baseline_model(img)
                    bp = F.interpolate(bp[:, None], depth.shape[-2:],
                                       mode='bilinear', align_corners=True)[0, 0]
                    base_pr = bp.cpu().numpy()
                viz.append((sample['image'][0].cpu().numpy(),
                            depth.cpu().numpy(), pred.cpu().numpy(),
                            mask.cpu().numpy(), cur['abs_rel'], base_pr))

    metrics = {k: agg[k] / max(nsamples, 1) for k in METRIC_KEYS}
    print(f"\n[{args.tag}]  n={nsamples}  max_depth={args.max_depth}")
    print(f"  abs_rel={metrics['abs_rel']*100:.2f}%  d1={metrics['d1']*100:.2f}%  "
          f"rmse={metrics['rmse']:.4f} m  silog={metrics['silog']*100:.2f}%")
    print("  (raw: " + "  ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_KEYS) + ")")

    # per-bucket error characterization (pixel-pooled across the test set)
    total_px = sum(b['n'] for b in bacc)
    buckets = []
    for (lo, hi), b in zip(BUCKETS, bacc):
        if b['n'] == 0:
            buckets.append({'range_m': f'{lo:g}-{hi:g}', 'n_px': 0, 'pct_px': 0.0,
                            'abs_rel': None, 'rmse': None, 'd1': None})
        else:
            buckets.append({
                'range_m': f'{lo:g}-{hi:g}', 'n_px': b['n'],
                'pct_px': b['n'] / total_px,
                'abs_rel': b['abs_rel'] / b['n'],
                'rmse': (b['se'] / b['n']) ** 0.5,
                'd1': b['d1'] / b['n'],
            })
    print(f"  per-bucket (GT range, pixel-pooled):")
    print(f"    {'range(m)':>10} {'%px':>6} {'abs_rel':>9} {'rmse(m)':>8} {'d1':>8}")
    for bk in buckets:
        if bk['n_px'] == 0:
            print(f"    {bk['range_m']:>10} {0.0:>6.1f} {'--':>9} {'--':>8} {'--':>8}")
        else:
            print(f"    {bk['range_m']:>10} {bk['pct_px']*100:>6.1f} "
                  f"{bk['abs_rel']*100:>8.2f}% {bk['rmse']:>8.4f} {bk['d1']*100:>7.2f}%")

    with open(os.path.join(args.out_dir, f"metrics_{args.tag}.json"), "w") as f:
        json.dump({'tag': args.tag, 'checkpoint': args.checkpoint,
                   'n': nsamples, 'max_depth': args.max_depth, **metrics,
                   'buckets': buckets}, f, indent=2)

    # qualitative panels
    if viz:
        n = len(viz)
        compare = baseline_model is not None
        ncol = 6 if compare else 4
        fig, ax = plt.subplots(n, ncol, figsize=(5 * ncol, 5 * n), squeeze=False)

        def show(a, data, title, cmap, vmin, vmax):
            im = a.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            a.set_title(title); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)

        for r, (im, gt, pr, mk, ar, base_pr) in enumerate(viz):
            rgb = np.clip(im.transpose(1, 2, 0) * STD + MEAN, 0, 1)
            gtm = np.where(mk, gt, np.nan)
            errm = np.where(mk, np.abs(pr - gt), np.nan)
            vmax = float(np.nanmax(gtm))
            ax[r][0].imshow(rgb); ax[r][0].set_title("RGB"); ax[r][0].axis("off")
            show(ax[r][1], gtm, "GT (m)", "turbo", 0, vmax)
            if compare:
                base_err = np.where(mk, np.abs(base_pr - gt), np.nan)
                base_ar = float(np.nanmean(np.where(mk, np.abs(base_pr - gt) / np.maximum(gt, 1e-6), np.nan)))
                # shared depth scale (GT vmax) so baseline scale error is visible
                show(ax[r][2], base_pr, "baseline pred (m)", "turbo", 0, vmax)
                show(ax[r][3], base_err, f"baseline |err| (abs_rel={base_ar:.3f})", "magma", None, None)
                show(ax[r][4], pr, "finetuned pred (m)", "turbo", 0, vmax)
                show(ax[r][5], errm, f"finetuned |err| (abs_rel={ar:.3f})", "magma", None, None)
            else:
                show(ax[r][2], pr, "pred (m)", "turbo", 0, vmax)
                show(ax[r][3], errm, f"|err| (abs_rel={ar:.3f})", "magma", None, None)
        title = (f"{args.tag}: GT vs baseline vs finetuned  (max_depth={args.max_depth})"
                 if compare else f"{args.tag}  (max_depth={args.max_depth})")
        fig.suptitle(title)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"qual_{args.tag}.png")
        fig.savefig(out, dpi=70); plt.close(fig)
        print(f"  qualitative -> {out}")


if __name__ == "__main__":
    main()
