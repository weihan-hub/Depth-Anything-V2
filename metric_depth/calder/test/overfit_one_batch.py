"""19.2 One-batch overfit (the sanity gate).

Take ONE fixed batch (a few frames from a single camera), reuse it every step,
and drive the SiLog loss toward ~0. If it doesn't go to ~0 with predictions
visually near-identical to GT, the bug is in one of: lr, output activation,
depth clamp, mask, depth scale, or loss impl -- fix before any real training.

    cd metric_depth
    uv run python -m calder.test.overfit_one_batch --steps 1500 --bs 2 --lr 1e-5
"""
import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calder.config import paths
from calder.lib.dataset import CalderDepthDataset
from calder.lib.model import MODEL_CONFIGS, build_model
from util.loss import SiLogLoss

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(img_chw):
    img = img_chw.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(img, 0, 1)


def save_pred_vs_gt(out_dir, step, imgs, gts, preds, masks):
    n = imgs.shape[0]
    fig, ax = plt.subplots(n, 4, figsize=(20, 5 * n), squeeze=False)
    for r in range(n):
        gt = np.where(masks[r], gts[r], np.nan)
        pr = preds[r]
        err = np.where(masks[r], np.abs(preds[r] - gts[r]), np.nan)
        vmax = np.nanmax(gt)
        ax[r][0].imshow(denorm(imgs[r])); ax[r][0].set_title("RGB"); ax[r][0].axis("off")
        im1 = ax[r][1].imshow(gt, cmap="turbo", vmin=0, vmax=vmax)
        ax[r][1].set_title("GT (m)"); ax[r][1].axis("off"); fig.colorbar(im1, ax=ax[r][1], fraction=0.046)
        im2 = ax[r][2].imshow(pr, cmap="turbo", vmin=0, vmax=vmax)
        ax[r][2].set_title("pred (m)"); ax[r][2].axis("off"); fig.colorbar(im2, ax=ax[r][2], fraction=0.046)
        im3 = ax[r][3].imshow(err, cmap="magma")
        ax[r][3].set_title("|pred-GT| (m)"); ax[r][3].axis("off"); fig.colorbar(im3, ax=ax[r][3], fraction=0.046)
    fig.suptitle(f"step {step}")
    fig.tight_layout()
    out = os.path.join(out_dir, f"pred_vs_gt_step{step}.png")
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=paths.MANIFEST)
    ap.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    ap.add_argument("--checkpoint", default=paths.RELATIVE_VITS)
    ap.add_argument("--camera-name", default="stereo_camera_front_left")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-depth", type=float, default=30.0)
    ap.add_argument("--min-depth", type=float, default=0.001)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--out-dir", default=os.path.join(paths.RESULTS_OVERFIT, "run"))
    ap.add_argument("--viz-every", type=int, default=250)
    ap.add_argument("--metric-checkpoint", action="store_true",
                    help="Checkpoint is a pre-trained METRIC model (encoder + "
                         "metric head): load the full state_dict so the metric "
                         "head is inherited, not reinitialized. Without this flag "
                         "we treat it as a relative model and load encoder only "
                         "(metric head trained from scratch, like train.py).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model: metric head is sigmoid * max_depth (depth in meters) ---
    model = build_model(args.encoder, args.max_depth)
    state = torch.load(args.checkpoint, map_location='cpu')
    if args.metric_checkpoint:
        # Pre-trained metric model: inherit encoder AND metric head.
        missing = model.load_state_dict(state, strict=False)
        print(f"loaded FULL metric checkpoint (encoder + metric head); "
              f"{len(missing.missing_keys)} missing / "
              f"{len(missing.unexpected_keys)} unexpected keys")
        if args.max_depth != 20.0:
            print(f"  NOTE: Hypersim metric model was trained with max_depth=20; "
                  f"you set max_depth={args.max_depth}. The head output rescales "
                  f"with max_depth, so it will need to re-adapt during finetune.")
    else:
        # Relative model: load only the pretrained encoder, like train.py:91.
        # The relative head (disparity/ReLU) is dropped; metric head starts random.
        missing = model.load_state_dict(
            {k: v for k, v in state.items() if 'pretrained' in k}, strict=False)
        print(f"loaded encoder weights only; {len(missing.missing_keys)} keys left "
              f"at init (metric depth_head trained from scratch)")
    model = model.to(device).train()

    # --- one FIXED batch, grabbed once, reused every step ---
    ds = CalderDepthDataset(args.manifest, mode='train',
                            size=(args.img_size, args.img_size),
                            camera_name=args.camera_name)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    img = batch['image'].to(device)
    depth = batch['depth'].to(device)
    valid_mask = batch['valid_mask'].to(device)
    print(f"fixed batch: {img.shape[0]} frames from {args.camera_name} "
          f"-> {[s for s in batch['sample_id']]}")

    # --- optimizer: dual lr like train.py, but CONSTANT (no decay) ---
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'pretrained' in n], 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if 'pretrained' not in n], 'lr': args.lr * 10.0},
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    criterion = SiLogLoss()

    losses = []
    for step in range(args.steps + 1):
        optimizer.zero_grad()
        pred = model(img)
        mask = valid_mask & (depth >= args.min_depth) & (depth <= args.max_depth)
        loss = criterion(pred, depth, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.6f}")
        if step % args.viz_every == 0 or step == args.steps:
            with torch.no_grad():
                p = model(img)
            out = save_pred_vs_gt(
                args.out_dir, step,
                img, depth.cpu().numpy(), p.cpu().numpy(), mask.cpu().numpy())
            if step == args.steps:
                print(f"final viz -> {out}")

    # --- loss curve ---
    fig, axc = plt.subplots(figsize=(8, 5))
    axc.plot(losses)
    axc.set_yscale("log")
    axc.set_xlabel("step"); axc.set_ylabel("SiLog loss (log scale)")
    axc.set_title("one-batch overfit loss")
    fig.tight_layout()
    curve = os.path.join(args.out_dir, "loss_curve.png")
    fig.savefig(curve, dpi=90); plt.close(fig)
    print(f"loss curve -> {curve}")

    # --- final metrics on the batch ---
    with torch.no_grad():
        pred = model(img)
    m = mask
    p, g = pred[m], depth[m]
    abs_rel = (torch.abs(p - g) / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    print(f"\nGATE  start_loss={losses[0]:.6f}  final_loss={losses[-1]:.6f}"
          f"  abs_rel={abs_rel:.4f}  rmse={rmse:.4f} m")


if __name__ == "__main__":
    main()
