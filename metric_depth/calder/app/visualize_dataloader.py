"""19.1 Dataloader visualization.

Pull N random samples from CalderDepthDataset and save a panel per sample:
RGB | GT depth | valid mask | depth histogram. Prints K + depth stats so we can
eyeball that RGB/depth are aligned and depths are in plausible meters.

    uv run python -m calder.app.visualize_dataloader --n 6
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calder.config import paths
from calder.lib.dataset import CalderDepthDataset

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(img_chw):
    """CHW normalized tensor -> HWC uint8-ish float in [0,1]."""
    img = img_chw.numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=paths.MANIFEST)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--mode", default="train", choices=["train", "val"])
    ap.add_argument("--camera-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(paths.RESULTS, "viz", "dataloader"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    ds = CalderDepthDataset(args.manifest, mode=args.mode,
                            camera_name=args.camera_name)
    # load K per sample_id for printing
    k_by_id = {json.loads(l)["sample_id"]: json.loads(l)["K"]
               for l in open(args.manifest)}

    idxs = random.sample(range(len(ds)), min(args.n, len(ds)))
    for i in idxs:
        s = ds[i]
        rgb = denorm(s["image"])
        depth = s["depth"].numpy()
        mask = s["valid_mask"].numpy()
        valid = depth[mask]

        K = k_by_id[s["sample_id"]]
        print(f"[{s['sample_id']}]")
        print(f"  K fx={K[0][0]:.1f} fy={K[1][1]:.1f} cx={K[0][2]:.1f} cy={K[1][2]:.1f}")
        print(f"  depth(m) min={valid.min():.3f} max={valid.max():.3f} "
              f"mean={valid.mean():.3f}  valid_frac={mask.mean():.4f}")

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(rgb); ax[0].set_title("RGB"); ax[0].axis("off")
        dvis = np.where(mask, depth, np.nan)
        im = ax[1].imshow(dvis, cmap="turbo")
        ax[1].set_title("GT depth (m)"); ax[1].axis("off")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        ax[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax[2].set_title(f"valid mask ({mask.mean()*100:.1f}%)"); ax[2].axis("off")
        ax[3].hist(valid.ravel(), bins=60); ax[3].set_title("depth histogram (m)")

        fig.suptitle(s["sample_id"], fontsize=9)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{s['sample_id']}.png")
        fig.savefig(out, dpi=80)
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
