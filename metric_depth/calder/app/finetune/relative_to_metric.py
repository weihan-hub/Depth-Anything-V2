"""Learnable relative-depth -> metric-depth mapping (frozen relative backbone).

Motivation
----------
Full metric finetuning of DA-V2 fixed the *scale* on the Calder split but
degraded the *relative* depth structure (the thing DA-V2 is best at).  This
module keeps the released **relative** DA-V2 checkpoint completely frozen and
learns only a tiny mapping from its (affine-invariant, disparity-like) output
to metric depth.

Why a *conditioned* affine (not a single global one)
----------------------------------------------------
DA-V2's relative head is trained scale-and-shift-invariant, so its output is
disparity **up to a per-image affine** -- the scale/shift drift from image to
image.  Empirically (see ``probe.py``) a *per-image* affine recovers metric
depth well (abs_rel ~0.12), but a *single global* affine does not (abs_rel
~0.55): the backbone is not scale-consistent across images.

So we predict the affine **per image**, conditioned on the frozen encoder's
pooled CLS features (which encode scene scale cues), and apply it to the frozen
disparity map::

    scale, shift = MLP(cls_features)          # one (scale>0, shift) per image
    disparity    = scale * rel + shift
    metric       = 1 / disparity

Because the affine is *shared across all pixels of an image* and monotonic
(scale > 0), the relative ordering of the frozen prediction is preserved
exactly -- only the per-image scale/offset is learned.  The MLP is initialised
so its prediction starts at a data-fit *global* affine and learns per-image
residuals from there.

The heavy encoder runs under ``no_grad`` (frozen); training touches only the
tiny head, so memory is small and batches can be large.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from calder.config import paths

_REL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def build_relative_model(encoder='vitl'):
    """Construct the released *relative* DepthAnythingV2 backbone (ReLU disparity
    head), NOT the metric one (Sigmoid * max_depth).

    We run from ``metric_depth/`` so a bare ``import depth_anything_v2`` resolves
    to the metric variant; prepending REPO_ROOT makes it resolve to the relative
    variant.  The repo root has no ``dataset`` / ``util`` / ``calder`` package,
    so those still resolve to metric_depth -- no collateral shadowing.
    """
    if paths.REPO_ROOT not in sys.path:
        sys.path.insert(0, paths.REPO_ROOT)
    from depth_anything_v2.dpt import DepthAnythingV2 as RelativeDepthAnythingV2
    return RelativeDepthAnythingV2(**_REL_CONFIGS[encoder])


def load_relative_state(checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    return {k.replace('module.', ''): v for k, v in state.items()}


def _inv_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


class RelToMetricHead(nn.Module):
    """Frozen relative (disparity-like) prediction -> metric depth.

    Modes:
      * ``conditioned`` (default): per-image affine (scale, shift) predicted from
        pooled CLS features -- adapts to the backbone's per-image scale drift.
      * ``affine``: a single global (scale, shift). Baseline / reference only;
        the backbone is not globally scale-consistent so this underperforms.
    """

    def __init__(self, mode='conditioned', cond_dim=4096, max_depth=20.0,
                 hidden=256, eps=1e-3):
        super().__init__()
        self.mode = mode
        self.max_depth = float(max_depth)
        self.eps = float(eps)
        self._min_disp = 1.0 / self.max_depth

        # global affine "anchor" (also the whole model in ``affine`` mode); the
        # conditioned MLP predicts per-image residuals around this anchor.
        self.scale_raw = nn.Parameter(_inv_softplus(1.0))   # scale = softplus(.)
        self.shift = nn.Parameter(torch.tensor(0.0))

        if mode == 'conditioned':
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 2),
            )
            # start at ~0 residual so the initial mapping == the global anchor
            with torch.no_grad():
                self.mlp[-1].weight.mul_(0.0)
                self.mlp[-1].bias.zero_()
        elif mode != 'affine':
            raise ValueError(f"unknown mode: {mode}")

    def _scale_shift(self, cond):
        """Return per-sample (scale>0, shift), shape (B,) each."""
        if self.mode == 'affine' or cond is None:
            scale = F.softplus(self.scale_raw).expand(1)
            shift = self.shift.expand(1)
            return scale, shift
        res = self.mlp(cond)                                 # (B,2) residuals
        scale = F.softplus(self.scale_raw + res[:, 0])       # (B,)
        shift = self.shift + res[:, 1]                        # (B,)
        return scale, shift

    def forward(self, rel, cond=None):
        """rel: (B,H,W) disparity-like; cond: (B,cond_dim) or None -> metric (B,H,W)."""
        scale, shift = self._scale_shift(cond)
        if scale.numel() == 1:                               # global broadcast
            disp = scale * rel + shift
        else:
            disp = scale[:, None, None] * rel + shift[:, None, None]
        disp = disp.clamp(min=self._min_disp)                # depth <= max_depth
        return (1.0 / disp).clamp(self.eps, self.max_depth)

    @torch.no_grad()
    def init_from_batch(self, rel, depth, valid_mask):
        """Warm start the global anchor: least-squares fit of disparity=1/depth
        against rel over valid pixels (pooled across the batch)."""
        m = valid_mask & (depth > self.eps)
        if m.sum() < 100:
            return
        r = rel[m].float()
        target = 1.0 / depth[m].float()
        rm, tm = r.mean(), target.mean()
        var = ((r - rm) ** 2).mean().clamp(min=1e-8)
        cov = ((r - rm) * (target - tm)).mean()
        s = (cov / var).clamp(min=1e-6)
        t = tm - s * rm
        self.scale_raw.copy_(_inv_softplus(s))
        self.shift.copy_(t)


class RelativeToMetricModel(nn.Module):
    """Frozen relative DA-V2 backbone + trainable RelToMetricHead.

    ``forward(image) -> metric depth (B,H,W)`` matches the metric-model API used
    by the finetune / evaluate loops.  The backbone runs under ``no_grad``; only
    the head has gradients.
    """

    def __init__(self, encoder='vitl', checkpoint=None, mode='conditioned',
                 max_depth=20.0, hidden=256):
        super().__init__()
        self.encoder = encoder
        self.relative = build_relative_model(encoder)
        if checkpoint is not None:
            info = self.relative.load_state_dict(load_relative_state(checkpoint),
                                                 strict=False)
            print(f"[relative backbone] loaded {checkpoint}: "
                  f"{len(info.missing_keys)} missing / "
                  f"{len(info.unexpected_keys)} unexpected")
        for p in self.relative.parameters():
            p.requires_grad = False
        self.relative.eval()

        embed = self.relative.pretrained.embed_dim
        n_layers = len(self.relative.intermediate_layer_idx[encoder])
        self.head = RelToMetricHead(mode=mode, cond_dim=embed * n_layers,
                                    max_depth=max_depth, hidden=hidden)

    def train(self, mode=True):
        super().train(mode)
        self.relative.eval()                                 # backbone always eval
        return self

    @torch.no_grad()
    def backbone(self, image):
        """Run the frozen backbone -> (rel disparity (B,H,W), cls cond (B,C))."""
        idx = self.relative.intermediate_layer_idx[self.encoder]
        feats = self.relative.pretrained.get_intermediate_layers(
            image, idx, return_class_token=True)
        ph, pw = image.shape[-2] // 14, image.shape[-1] // 14
        rel = torch.relu(self.relative.depth_head(feats, ph, pw)).squeeze(1)
        cond = torch.cat([f[1] for f in feats], dim=-1)      # concat CLS tokens
        return rel, cond

    @torch.no_grad()
    def relative_depth(self, image):
        return self.backbone(image)[0]

    def forward(self, image):
        rel, cond = self.backbone(image)
        return self.head(rel.detach(), cond.detach())
