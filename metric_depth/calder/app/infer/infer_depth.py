"""Run a finetuned DA-V2 depth model on a directory of RGB frames.

Supports two checkpoint kinds:
  * default: a finetuned *metric* DA-V2 (sigmoid * max_depth head);
  * ``--mapping``: our RelativeToMetricModel (frozen relative backbone +
    per-frame scale/offset head). Both expose ``forward(image) -> metres``.

Pure inference (no GT needed): builds the model, loads a checkpoint, and for
every image writes a 16-bit millimetre depth PNG (same format as the training
GT) plus an optional colormap visualization.

The encoder size is auto-detected from the checkpoint, so the same command works
for vits / vitb / vitl finetunes. Keep --max-depth identical to training
(default 20) -- the metric head output is sigmoid * max_depth, so a mismatch
silently rescales every prediction.

    cd metric_depth
    uv run python -m calder.app.infer.infer_depth \
        --checkpoint ../checkpoints/best.pth \
        --image-dir  /mnt/data/.../nurec_run1/extracted/iphone_camera \
        --out-dir    /mnt/data/.../nurec_run1/mono_depth_pred \
        --max-depth 20
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from calder.lib.model import MODEL_CONFIGS, build_model, load_state_flexible
from calder.app.finetune.relative_to_metric import RelativeToMetricModel
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

IMG_EXTS = ("*.jpeg", "*.jpg", "*.png")
_EMBED_DIM_TO_ENCODER = {384: "vits", 768: "vitb", 1024: "vitl"}
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def infer_image_nearest(model, raw_bgr, input_size, device):
    """Like model.infer_image but every RESOLUTION resize uses NEAREST:
    the 4K->518-long-edge input downscale and the 518-space->4K depth upscale.

    Nearest never blends depth across object boundaries, so unprojecting the
    result to a point cloud produces no interpolated "flying pixels" at edges.
    (The DPT head's internal feature upsampling in dpt.py is left as-is; it is
    part of the network, not the input/output resolution resize.)
    """
    h, w = raw_bgr.shape[:2]
    transform = Compose([
        Resize(width=input_size, height=input_size, resize_target=False,
               keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method="lower_bound",
               image_interpolation_method=cv2.INTER_NEAREST),   # 4K -> 518, nearest
        NormalizeImage(mean=_MEAN, std=_STD),
        PrepareForNet(),
    ])
    image = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0
    image = torch.from_numpy(transform({"image": image})["image"]).unsqueeze(0).to(device)
    depth = model.forward(image)                                # [1, h', w'] at 518-space
    depth = F.interpolate(depth[:, None], (h, w), mode="nearest")[0, 0]  # 518 -> 4K, nearest
    return depth.cpu().numpy()


def detect_encoder(state_dict):
    """Infer the encoder variant from the patch-embed projection width."""
    proj = next(v for k, v in state_dict.items()
                if "patch_embed.proj.weight" in k)
    embed_dim = proj.shape[0]
    if embed_dim not in _EMBED_DIM_TO_ENCODER:
        raise ValueError(f"Unknown embed_dim={embed_dim}; cannot map to an encoder. "
                         f"Known: {_EMBED_DIM_TO_ENCODER}")
    return _EMBED_DIM_TO_ENCODER[embed_dim]


def colorize(depth_m, max_depth):
    """Map a metric depth map (meters) to an 8-bit INFERNO color image."""
    d = np.clip(depth_m, 0, max_depth) / max(max_depth, 1e-6)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="finetuned state_dict (e.g. best.pth)")
    ap.add_argument("--image-dir", required=True, help="directory of RGB frames")
    ap.add_argument("--out-dir", required=True, help="where to write depth PNGs")
    ap.add_argument("--encoder", default=None, choices=list(MODEL_CONFIGS),
                    help="override the auto-detected encoder")
    ap.add_argument("--max-depth", type=float, default=20.0,
                    help="MUST match training; metric head output = sigmoid * max_depth, "
                         "and it clamps the mapping-head depth")
    ap.add_argument("--mapping", action="store_true",
                    help="checkpoint is a RelativeToMetricModel (frozen relative backbone + "
                         "per-frame scale/offset head), not a metric DA-V2")
    ap.add_argument("--input-size", type=int, default=518,
                    help="network input long-side (multiple of 14)")
    ap.add_argument("--no-viz", action="store_true", help="skip colormap visualization")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "viz")
    if not args.no_viz:
        os.makedirs(viz_dir, exist_ok=True)

    # --- model ---
    state = load_state_flexible(args.checkpoint)
    encoder = args.encoder or detect_encoder(state)
    if args.mapping:
        model = RelativeToMetricModel(encoder=encoder, checkpoint=None,
                                      max_depth=args.max_depth)
        info = model.load_state_dict(state, strict=True)
        kind = "mapping (frozen relative + per-frame affine head)"
    else:
        model = build_model(encoder, args.max_depth)
        info = model.load_state_dict(state, strict=False)
        kind = "metric"
    print(f"{kind} | encoder={encoder} max_depth={args.max_depth} device={device} | "
          f"loaded ckpt: {len(info.missing_keys)} missing / "
          f"{len(info.unexpected_keys)} unexpected")
    model = model.to(device).eval()

    # --- frames (sorted by timestamp filename) ---
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(args.image_dir, ext)))
    files = sorted(files)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No images ({IMG_EXTS}) under {args.image_dir}")
    print(f"found {len(files)} frames")

    for i, path in enumerate(files):
        raw = cv2.imread(path)  # BGR; infer_image_nearest does BGR->RGB internally
        depth_m = infer_image_nearest(model, raw, args.input_size, device)  # [H, W] meters, nearest resize

        stem = os.path.splitext(os.path.basename(path))[0]
        # 16-bit millimetres, 0 = (here just clipped at 0); matches GT format.
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}.png"), depth_mm)
        if not args.no_viz:
            cv2.imwrite(os.path.join(viz_dir, f"{stem}.png"),
                        colorize(depth_m, args.max_depth))

        if i % 50 == 0 or i == len(files) - 1:
            print(f"  [{i + 1}/{len(files)}] {stem}  "
                  f"depth min/median/max = {depth_m.min():.2f}/"
                  f"{np.median(depth_m):.2f}/{depth_m.max():.2f} m")

    print(f"done -> {args.out_dir} (16-bit mm PNGs"
          f"{'' if args.no_viz else ' + viz/'})")


if __name__ == "__main__":
    main()
