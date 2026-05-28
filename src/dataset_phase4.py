"""Phase-4 dataset extension: depth as a genuine 4th input channel.

Phase 3 found depth-as-blue HURT object-wise accuracy by ~2 pp (0.730 → 0.710)
and hypothesised the *cause* was the substitution destroying the blue colour
channel, not the depth signal itself being useless. The clean way to falsify
that is to keep all three RGB channels and append depth as a 4th channel, with
the backbone's first conv re-initialised to accept 4 inputs (RGB weights copied
from ImageNet pretraining, the depth filter seeded from the mean RGB filter).

``CornellGraspDatasetP4`` is a thin subclass of ``CornellGraspDatasetP3`` that
only changes the channel-assembly step when ``use_depth_4ch=True``. Everything
else (rotation aug, hflip, multi-positive routing, the cached depth loader) is
inherited unchanged so the ONLY variable vs E3.3 (depth-as-blue) is whether the
blue channel is preserved. Depth is loaded with the exact same cached,
un-rotated path as Phase-3's depth-blue so the comparison isolates channel
placement and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dataset import INPUT_SIZE, _imagenet_normalize
from dataset_phase3 import CornellGraspDatasetP3, MULTI_POS_MAX

# Depth is per-image min-max normalised to [0, 1] (foreground-high) by the
# Phase-3 loader. We standardise it to roughly zero-mean/unit-ish-scale so it
# enters the network on a comparable footing to the ImageNet-normalised RGB
# channels rather than dominating or vanishing.
_DEPTH_MEAN = 0.5
_DEPTH_STD = 0.225


@dataclass
class CornellGraspDatasetP4(CornellGraspDatasetP3):
    """Phase-3 dataset + an optional 4th depth channel.

    When ``use_depth_4ch`` is True the item is a 4xHxW tensor
    (R, G, B, depth). ``use_depth_blue`` must be False in that case (the two
    are mutually exclusive ways to inject depth). When ``use_depth_4ch`` is
    False the behaviour is byte-for-byte identical to ``CornellGraspDatasetP3``.
    """
    use_depth_4ch: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.use_depth_4ch and self.use_depth_blue:
            raise ValueError("use_depth_4ch and use_depth_blue are mutually exclusive")

    def __getitem__(self, idx: int):
        if not self.use_depth_4ch:
            return super().__getitem__(idx)

        s = self.samples[idx]
        rgb = self._load_rgb(s.rgb_path)
        use_rot_aug = self.augment and self.augment_rot_deg > 0

        if use_rot_aug:
            from dataset_phase3 import _rotate_image_array, _rotate_grasp
            positives = self._resize_grasps(s.positives)
            encode = self._encode_224
            rot_deg = self._rng.uniform(-self.augment_rot_deg, self.augment_rot_deg)
            rgb = _rotate_image_array(rgb, rot_deg)
            pivot = INPUT_SIZE / 2.0
            positives = [_rotate_grasp(g, rot_deg, pivot, pivot) for g in positives]
        else:
            from dataset import encode_target
            positives = list(s.positives)
            encode = lambda g: encode_target(g, self._scale_x, self._scale_y)

        if self.augment and self._rng.random() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
            if use_rot_aug:
                positives = [self._hflip_grasp(g) for g in positives]
            else:
                positives = [self._hflip_grasp_original(g) for g in positives]

        # depth loaded with the SAME cached, un-rotated path as the Phase-3
        # depth-blue trick — the only thing that differs from E3.3 is that we
        # *append* it rather than overwrite the blue channel.
        depth = self._load_depth(s.pcd_path)                       # (H, W) in [0, 1]
        depth_norm = (depth - _DEPTH_MEAN) / _DEPTH_STD             # standardise
        rgb_chw = _imagenet_normalize(rgb)                         # (3, H, W)
        x4 = np.concatenate(
            [rgb_chw, depth_norm[None, :, :].astype(np.float32)], axis=0
        )                                                          # (4, H, W)
        x = torch.from_numpy(x4)

        primary = self._central(positives)
        target_primary = encode(primary)
        t = torch.from_numpy(target_primary)

        if not self.multi_positive:
            return x, t, idx

        targets = [encode(g) for g in positives][:MULTI_POS_MAX]
        n_real = len(targets)
        while len(targets) < MULTI_POS_MAX:
            targets.append(target_primary)
        targets_arr = np.stack(targets, axis=0)
        return x, t, torch.from_numpy(targets_arr), int(n_real), idx
