"""Model construction + checkpoint loading helpers for DA-V2 metric.

Factors out the MODEL_CONFIGS table and checkpoint-loading logic that was
previously duplicated across the finetune / evaluate / overfit scripts.
"""
import torch

from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def build_model(encoder='vits', max_depth=20.0):
    """Build a DepthAnythingV2 metric model (head output = sigmoid * max_depth)."""
    return DepthAnythingV2(**{**MODEL_CONFIGS[encoder], 'max_depth': max_depth})


def load_state_flexible(checkpoint_path):
    """Load any of our checkpoints into a plain state_dict.

    Handles raw state_dicts (original metric ckpt, our saved best.pth) and
    train.py-style ``{'model': ..., 'optimizer': ...}`` checkpoints, and strips
    any ``module.`` prefix left by DistributedDataParallel.
    """
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    return {k.replace('module.', ''): v for k, v in state.items()}
