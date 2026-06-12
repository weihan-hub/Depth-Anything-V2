"""Dataset for the Calder NuRec stereo-depth manifest.

Modeled on ``dataset/hypersim.py`` so it plugs into the same training loop and
reuses the repo transform pipeline. GT depth PNGs are uint16 millimeters with
0 = invalid; we convert to meters and derive the valid mask from depth > 0.
"""
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class CalderDepthDataset(Dataset):
    def __init__(self, manifest_path, mode='train', size=(518, 518),
                 camera_name=None):
        """
        manifest_path: path to manifest.jsonl produced by build_manifest.py
        mode: 'train' (resizes depth + random crop) or 'val' (image-only resize)
        size: network input (h, w), both multiples of 14
        camera_name: if set, keep only rows with this camera (e.g. for overfit)
        """
        self.mode = mode
        self.size = size

        with open(manifest_path) as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        if camera_name is not None:
            self.entries = [e for e in self.entries
                            if e['camera_name'] == camera_name]

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))

    def __getitem__(self, item):
        entry = self.entries[item]

        image = cv2.imread(entry['rgb_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # uint16 PNG in millimeters; 0 = invalid. IMREAD_UNCHANGED keeps 16-bit.
        depth_mm = cv2.imread(entry['gt_depth_path'], cv2.IMREAD_UNCHANGED)
        depth = depth_mm.astype(np.float32) / 1000.0  # -> meters

        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])

        sample['valid_mask'] = sample['depth'] > 0
        sample['depth'][sample['valid_mask'] == 0] = 0

        sample['image_path'] = entry['rgb_path']
        sample['sample_id'] = entry['sample_id']

        return sample

    def __len__(self):
        return len(self.entries)
