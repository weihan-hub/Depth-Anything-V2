"""19.4 Single-GPU finetune of DA-V2 metric on the Calder split.

Adapted from metric_depth/train.py with DDP/tensorboard stripped out:
  - init from the FULL Hypersim metric checkpoint (encoder + metric head),
  - dual-LR AdamW (head 10x) with polynomial LR decay (train.py recipe),
  - random horizontal-flip augmentation,
  - per-epoch eval on the held-out VAL split (util.metric.eval_depth) for
    checkpoint/hyperparameter selection,
  - save best.pth (lowest val abs_rel) and latest.pth as raw model state_dicts.

The TEST split is NOT touched here -- it stays untouched until you run
app/evaluate.py once on best.pth for the final, unbiased report.

    cd metric_depth
    uv run python -m calder.app.finetune \
        --train-manifest calder/datasets/splits/all_cams/train_contiguous.jsonl \
        --val-manifest   calder/datasets/splits/all_cams/val_contiguous.jsonl \
        --max-depth 20 --epochs 10 --bs 4 --lr 5e-6 \
        --out-dir calder/results/finetune/all_cams/contiguous
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
from calder.lib.model import MODEL_CONFIGS, build_model, load_state_flexible
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
        depth = sample['depth'].to(device)[0]            # [H, W] original res
        valid = sample['valid_mask'].to(device)[0]
        pred = model(img)
        pred = F.interpolate(pred[:, None], depth.shape[-2:],
                             mode='bilinear', align_corners=True)[0, 0]
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
                    help="held-out VAL split; drives best.pth / hyperparameter "
                         "selection. Keep TEST out of this -- eval it once via "
                         "app/evaluate.py at the end.")
    ap.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    ap.add_argument("--checkpoint", default=paths.METRIC_HYPERSIM_VITS)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--min-depth", type=float, default=0.001)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(paths.RESULTS_FINETUNE, "run"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = (args.img_size, args.img_size)

    # --- data ---
    trainset = CalderDepthDataset(args.train_manifest, mode='train', size=size)
    valset = CalderDepthDataset(args.val_manifest, mode='val', size=size)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False,
                           num_workers=4, pin_memory=True)
    print(f"train={len(trainset)}  val={len(valset)}  "
          f"iters/epoch={len(trainloader)}")

    # --- model: inherit full metric checkpoint (encoder + metric head) ---
    model = build_model(args.encoder, args.max_depth)
    info = model.load_state_dict(load_state_flexible(args.checkpoint), strict=False)
    print(f"loaded metric checkpoint: {len(info.missing_keys)} missing / "
          f"{len(info.unexpected_keys)} unexpected")
    model = model.to(device)

    criterion = SiLogLoss()
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'pretrained' in n], 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if 'pretrained' not in n], 'lr': args.lr * 10.0},
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = args.epochs * len(trainloader)

    # --- baseline eval before any training step ---
    base_metrics, base_n = evaluate(model, valloader, args.min_depth, args.max_depth, device)
    print(f"[epoch -1 / pretrained] abs_rel={base_metrics['abs_rel']:.4f} "
          f"rmse={base_metrics['rmse']:.4f} d1={base_metrics['d1']:.4f} (n={base_n})")

    history = {'train_loss': [], 'eval': [{'epoch': -1, **base_metrics, 'n': base_n}]}
    best_abs_rel = base_metrics['abs_rel']

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img = sample['image'].to(device)
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
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

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
        if metrics['abs_rel'] < best_abs_rel:
            best_abs_rel = metrics['abs_rel']
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"  -> new best abs_rel={best_abs_rel:.4f}, saved best.pth")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)

    # --- curves ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history['train_loss'], marker='o'); ax[0].set_title("train SiLog loss")
    ax[0].set_xlabel("epoch")
    ep = [e['epoch'] for e in history['eval']]
    ax[1].plot(ep, [e['abs_rel'] for e in history['eval']], marker='o', label='abs_rel')
    ax[1].plot(ep, [e['rmse'] for e in history['eval']], marker='s', label='rmse (m)')
    ax[1].axvline(-1, ls='--', c='gray'); ax[1].set_xlabel("epoch (-1=pretrained)")
    ax[1].legend(); ax[1].set_title("val metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "curves.png"), dpi=90)
    print(f"\nbaseline abs_rel={base_metrics['abs_rel']:.4f} -> best abs_rel={best_abs_rel:.4f}")
    print(f"artifacts in {args.out_dir}/ (best.pth, latest.pth, history.json, curves.png)")


if __name__ == "__main__":
    main()
