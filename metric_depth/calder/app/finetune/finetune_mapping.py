"""Train ONLY a relative->metric mapping head on the Calder split.

The released *relative* DA-V2 checkpoint (encoder + relative head) stays fully
frozen; we learn a tiny mapping (see ``relative_to_metric.py``) from its
disparity-like output to metric depth.  This recovers metric *scale* while
preserving the relative structure that full metric finetuning was degrading.

Mirrors ``finetune.py`` (same dataset / SiLog loss / eval / checkpointing) but:
  * model = RelativeToMetricModel (frozen backbone + trainable head),
  * optimizer only sees the head params,
  * a data-driven warm start fits the affine head on the first batch,
  * ``--data-root-old/new`` rewrites manifest paths (dataset moved /mnt -> /Data).

    cd metric_depth
    uv run python -m calder.app.finetune.finetune_mapping \
        --train-manifest /home/weihanwang/workspace/exp/train_real.jsonl \
        --val-manifest   /home/weihanwang/workspace/exp/val_real_sub.jsonl \
        --encoder vitl --checkpoint ../checkpoints/depth_anything_v2_vitl.pth \
        --max-depth 20 --epochs 10 --bs 16 --lr 1e-3 \
        --data-root-old /mnt/data/data/calder/fine_tune_dataset \
        --data-root-new /Data/wwh/fine_tune_dataset \
        --out-dir calder/results/finetune/mapping_affine
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calder.config import paths
from calder.lib.dataset import CalderDepthDataset
from calder.app.finetune.relative_to_metric import RelativeToMetricModel
from util.loss import SiLogLoss
from util.metric import eval_depth

METRIC_KEYS = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']


@torch.no_grad()
def evaluate(model, loader, min_depth, max_depth, device):
    model.eval()
    agg = {k: 0.0 for k in METRIC_KEYS}
    nsamples = 0
    for sample in loader:
        img = sample['image'].to(device).float()
        depth = sample['depth'].to(device)[0]
        valid = sample['valid_mask'].to(device)[0]
        pred = model(img)
        pred = F.interpolate(pred[:, None], depth.shape[-2:],
                             mode='nearest', align_corners=None)[0, 0]
        mask = (valid == 1) & (depth >= min_depth) & (depth <= max_depth)
        if mask.sum() < 10:
            continue
        cur = eval_depth(pred[mask], depth[mask])
        for k in METRIC_KEYS:
            agg[k] += cur[k]
        nsamples += 1
    model.train()
    return {k: agg[k] / max(nsamples, 1) for k in METRIC_KEYS}, nsamples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True,
                    help="held-out VAL split; drives best.pth selection.")
    ap.add_argument("--encoder", default="vitl", choices=list(_choices()))
    ap.add_argument("--checkpoint", default=paths.RELATIVE_VITL,
                    help="RELATIVE DA-V2 checkpoint (frozen backbone).")
    ap.add_argument("--hidden", type=int, default=256, help="conditioning MLP hidden width")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="head LR (can be large -- tiny head, frozen backbone)")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--min-depth", type=float, default=0.001)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--data-root-old", default=None,
                    help="path prefix in the manifest to rewrite (dataset move)")
    ap.add_argument("--data-root-new", default=None,
                    help="replacement path prefix on this machine")
    ap.add_argument("--out-dir", default=os.path.join(paths.RESULTS_FINETUNE, "mapping"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = (args.img_size, args.img_size)
    remap = _remap(args.data_root_old, args.data_root_new)

    # --- data ---
    trainset = CalderDepthDataset(args.train_manifest, mode='train', size=size,
                                  path_remap=remap)
    valset = CalderDepthDataset(args.val_manifest, mode='val', size=size,
                                path_remap=remap)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    print(f"train={len(trainset)}  val={len(valset)}  iters/epoch={len(trainloader)}")

    # --- model: FROZEN relative backbone + trainable mapping head ---
    model = RelativeToMetricModel(encoder=args.encoder, checkpoint=args.checkpoint,
                                  max_depth=args.max_depth,
                                  hidden=args.hidden).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"per-frame conditioned mapping  trainable {n_train:,} / {n_total:,} params "
          f"({100 * n_train / n_total:.4f}%)")

    # --- data-driven warm start of the global affine anchor from one batch ---
    # (in conditioned mode the MLP starts at ~0 residual, so training begins at
    #  this global affine and learns per-image adjustments from there)
    model.eval()
    s0 = next(iter(trainloader))
    with torch.no_grad():
        rel0 = model.relative_depth(s0['image'].to(device).float())
    model.head.init_from_batch(rel0, s0['depth'].to(device),
                               s0['valid_mask'].to(device).bool())
    model.train()
    print(f"warm-start global anchor: scale={F.softplus(model.head.scale_raw).item():.5f} "
          f"shift={model.head.shift.item():.5f}")

    criterion = SiLogLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    total_iters = args.epochs * len(trainloader)

    # --- baseline eval before any training step ---
    base_metrics, base_n = evaluate(model, valloader, args.min_depth, args.max_depth, device)
    print(f"[epoch -1 / warm-start] abs_rel={base_metrics['abs_rel']:.4f} "
          f"rmse={base_metrics['rmse']:.4f} d1={base_metrics['d1']:.4f} (n={base_n})")

    history = {'train_loss': [], 'eval': [{'epoch': -1, **base_metrics, 'n': base_n}]}
    best_abs_rel = base_metrics['abs_rel']

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img = sample['image'].to(device).float()
            depth = sample['depth'].to(device)
            valid = sample['valid_mask'].to(device)
            if random.random() < 0.5:
                img = img.flip(-1); depth = depth.flip(-1); valid = valid.flip(-1)

            pred = model(img)
            mask = (valid == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            loss = criterion(pred, depth, mask)
            loss.backward()
            optimizer.step()
            running += loss.item()

            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            for g in optimizer.param_groups:
                g["lr"] = lr

            if i % 50 == 0:
                print(f"  epoch {epoch} iter {i}/{len(trainloader)} "
                      f"lr {lr:.2e} loss {loss.item():.4f}")

        avg_loss = running / len(trainloader)
        history['train_loss'].append(avg_loss)

        metrics, n = evaluate(model, valloader, args.min_depth, args.max_depth, device)
        history['eval'].append({'epoch': epoch, **metrics, 'n': n})
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f}  "
              f"abs_rel={metrics['abs_rel']:.4f} rmse={metrics['rmse']:.4f} "
              f"d1={metrics['d1']:.4f}")

        torch.save(model.state_dict(), os.path.join(args.out_dir, "latest.pth"))
        # also save just the head (tiny, portable)
        torch.save(model.head.state_dict(), os.path.join(args.out_dir, "head_latest.pth"))
        if metrics['abs_rel'] < best_abs_rel:
            best_abs_rel = metrics['abs_rel']
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            torch.save(model.head.state_dict(), os.path.join(args.out_dir, "head_best.pth"))
            print(f"  -> new best abs_rel={best_abs_rel:.4f}, saved best.pth")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history['train_loss'], marker='o'); ax[0].set_title("train SiLog loss")
    ax[0].set_xlabel("epoch")
    ep = [e['epoch'] for e in history['eval']]
    ax[1].plot(ep, [e['abs_rel'] for e in history['eval']], marker='o', label='abs_rel')
    ax[1].plot(ep, [e['rmse'] for e in history['eval']], marker='s', label='rmse (m)')
    ax[1].axvline(-1, ls='--', c='gray'); ax[1].set_xlabel("epoch (-1=warm-start)")
    ax[1].legend(); ax[1].set_title("val metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "curves.png"), dpi=90)
    print(f"\nwarm-start abs_rel={base_metrics['abs_rel']:.4f} -> best abs_rel={best_abs_rel:.4f}")
    print(f"artifacts in {args.out_dir}/ (best.pth, head_best.pth, history.json, curves.png)")


def _choices():
    from calder.app.finetune.relative_to_metric import _REL_CONFIGS
    return _REL_CONFIGS


def _remap(old, new):
    if old and new:
        return (old, new)
    return None


if __name__ == "__main__":
    main()
