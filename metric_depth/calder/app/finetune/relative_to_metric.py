"""Learnable relative-depth -> metric-depth mapping (frozen relative backbone).

Motivation
----------
Full metric finetuning of DA-V2 fixed the *scale* on the Calder split but
degraded the *relative* depth structure (the thing DA-V2 is best at).  This
module keeps the released **relative** DA-V2 checkpoint completely frozen and
learns only a tiny mapping from its (affine-invariant, disparity-like) output
to metric depth.

Per-frame affine, conditioned on the image
------------------------------------------
DA-V2's relative head is trained scale-and-shift-invariant, so its output is
disparity **up to a per-image affine** -- the scale/shift drift from image to
image.  Empirically (see ``probe.py``) a *per-image* affine recovers metric
depth well (abs_rel ~0.12), but a *single global* affine does not (abs_rel
~0.55): the backbone is not scale-consistent across images.

So we predict the affine **per frame**, conditioned on the frozen encoder's
pooled CLS features (which encode scene scale cues), and apply it to the frozen
disparity map::

    scale, shift = affine_predictor(cls_features)   # one (scale>0, shift) per frame
    disparity    = scale * rel + shift
    metric       = 1 / disparity

Because the affine is *shared across all pixels of a frame* and monotonic
(scale > 0), the relative ordering of the frozen prediction is preserved
exactly -- only the per-frame scale/offset is learned.

The predictor is parameterised as a learnable **global anchor** (scale_raw,
shift) plus a per-frame **residual** from a small MLP.  The anchor is warm
started (see ``RelToMetricHead.init_from_batch``) to a data-fit global affine
and the MLP starts at ~0 residual, so training begins at that global affine and
learns per-frame corrections from there.

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

    We run from ``metric_depth/`` where ``depth_anything_v2`` normally resolves to
    the *metric* variant.  Prepending REPO_ROOT is not enough on its own: if the
    metric ``depth_anything_v2`` is already in ``sys.modules`` (e.g. imported by
    ``calder.lib.model``), a plain ``import`` returns that cached metric module,
    silently giving a Sigmoid-headed backbone whose disparity is squashed to
    [0, 1] -- which then loads fine (identical param keys!) but is wrong.

    So we stash any cached ``depth_anything_v2*`` modules, import the relative
    package from REPO_ROOT under a clean slate, then restore the originals so the
    rest of the process still sees the metric variant.
    """
    import importlib

    prefix = 'depth_anything_v2'
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == prefix or k.startswith(prefix + '.')}
    sys.path.insert(0, paths.REPO_ROOT)
    try:
        RelativeDepthAnythingV2 = importlib.import_module(prefix + '.dpt').DepthAnythingV2
        model = RelativeDepthAnythingV2(**_REL_CONFIGS[encoder])
    finally:
        try:
            sys.path.remove(paths.REPO_ROOT)
        except ValueError:
            pass
        # drop our freshly-imported relative modules, restore the cached ones
        for k in [k for k in sys.modules if k == prefix or k.startswith(prefix + '.')]:
            del sys.modules[k]
        sys.modules.update(saved)
    return model


def load_relative_state(checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    return {k.replace('module.', ''): v for k, v in state.items()}


def _inv_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


class RelToMetricHead(nn.Module):
    """Per-frame affine mapping: frozen disparity-like ``rel`` -> metric depth.

    The affine ``(scale > 0, shift)`` is predicted per frame from pooled CLS
    features -- a learnable global *anchor* plus a small-MLP per-frame *residual*.
    Applying a single monotonic affine per frame preserves the frozen
    prediction's relative ordering exactly.
    """

    def __init__(self, cond_dim=4096, max_depth=20.0, hidden=256, eps=1e-3):
        super().__init__()
        self.max_depth = float(max_depth)
        self.eps = float(eps)
        self._min_disp = 1.0 / self.max_depth

        # global affine anchor (warm started from data); the MLP predicts a
        # per-frame residual around it (starts at ~0 -> begins as global affine).
        self.scale_raw = nn.Parameter(_inv_softplus(1.0))   # scale = softplus(.)
        self.shift = nn.Parameter(torch.tensor(0.0))
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        with torch.no_grad():                                # start at 0 residual
            self.mlp[-1].weight.mul_(0.0)
            self.mlp[-1].bias.zero_()

    def _scale_shift(self, cond):
        """cond: (B, cond_dim) -> per-frame (scale>0, shift), each (B,)."""
        res = self.mlp(cond)                                 # (B, 2) residuals
        scale = F.softplus(self.scale_raw + res[:, 0])
        shift = self.shift + res[:, 1]
        return scale, shift

    def forward(self, rel, cond):
        """rel: (B,H,W) disparity-like; cond: (B,cond_dim) -> metric (B,H,W)."""
        scale, shift = self._scale_shift(cond)
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

    def __init__(self, encoder='vitl', checkpoint=None, max_depth=20.0, hidden=256):
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
        self.head = RelToMetricHead(cond_dim=embed * n_layers,
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
