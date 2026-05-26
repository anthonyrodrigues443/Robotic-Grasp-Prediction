"""PyTorch Dataset for Cornell grasps.

Phase-2 training target choice: one positive rectangle per image. Cornell has
multiple positives per image (median ~6); picking just one keeps the task a
clean single-shot regression (matching Redmon & Angelova 2015) without
forcing a multi-output assignment problem. We pick the *most central* positive
(closest to the centroid of all positives in the image) — empirically more
stable than picking-one-at-random because the network sees a consistent target
per image across epochs.

Augmentations are deliberately minimal for the Phase-2 head-to-head: each
architecture gets the same crops/flips so the comparison is about the model,
not the augmentation pipeline. Phase 3 will sweep augmentations once the
champion is known.

Coordinate convention: the network outputs 6 numbers per image:
    (cx_norm, cy_norm, w_norm, h_norm, sin(2*theta), cos(2*theta))
where (cx, cy) are in the resized 224x224 frame (so /224), w in [0, 224]
(/224), h in [0, 224] (/224), and the doubled-angle representation handles
Cornell's 180-degree grasp symmetry without a discontinuity at +/- pi/2.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cornell import CornellSample, GraspRect, IMG_W, IMG_H, make_rect


INPUT_SIZE = 224  # ImageNet-standard, lets us reuse pretrained backbones unchanged


def _imagenet_normalize(arr: np.ndarray) -> np.ndarray:
    """arr: HxWx3 in [0, 1]. Returns CxHxW float32 ImageNet-normalised."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _central_positive(positives: list[GraspRect]) -> GraspRect:
    if len(positives) == 1:
        return positives[0]
    cx_mean = float(np.mean([g.cx for g in positives]))
    cy_mean = float(np.mean([g.cy for g in positives]))
    return min(positives, key=lambda g: (g.cx - cx_mean) ** 2 + (g.cy - cy_mean) ** 2)


def encode_target(g: GraspRect, scale_x: float, scale_y: float) -> np.ndarray:
    """Resized-frame target. scale_x = INPUT_SIZE / IMG_W, etc."""
    cx = g.cx * scale_x
    cy = g.cy * scale_y
    # use mean scale for size so wide-but-thin rectangles don't get distorted
    s = (scale_x + scale_y) / 2.0
    w = g.width * s
    h = g.height * s
    a = g.angle_rad
    return np.array([
        cx / INPUT_SIZE,
        cy / INPUT_SIZE,
        w / INPUT_SIZE,
        h / INPUT_SIZE,
        math.sin(2.0 * a),
        math.cos(2.0 * a),
    ], dtype=np.float32)


def decode_prediction(y: np.ndarray, out_w: int = IMG_W, out_h: int = IMG_H) -> GraspRect:
    """Convert a 6-vector network output back into a GraspRect in the ORIGINAL
    640x480 image frame. ``y`` is in the resized 224x224 frame."""
    cx_r = y[0] * INPUT_SIZE
    cy_r = y[1] * INPUT_SIZE
    w_r = max(y[2], 1e-3) * INPUT_SIZE
    h_r = max(y[3], 1e-3) * INPUT_SIZE
    # Project resized coords back to original frame
    cx = cx_r * (out_w / INPUT_SIZE)
    cy = cy_r * (out_h / INPUT_SIZE)
    s = ((out_w / INPUT_SIZE) + (out_h / INPUT_SIZE)) / 2.0
    w = w_r * s
    h = h_r * s
    # angle from sin(2θ), cos(2θ)
    s2, c2 = float(y[4]), float(y[5])
    # if both ~0, atan2 returns 0; that's fine as a fallback
    theta = 0.5 * math.atan2(s2, c2)
    return make_rect(cx, cy, w, h, theta)


@dataclass
class CornellGraspDataset(Dataset):
    """One example per image, encoding the *most central positive* grasp.

    ``augment=True`` adds horizontal-flip with grasp transform. We deliberately
    avoid rotations/crops in the Phase-2 baseline so all models compete on the
    same inputs; Phase-3 will sweep augmentations on top of the winner.
    """
    samples: list[CornellSample]
    augment: bool = False
    seed: int = 42

    def __post_init__(self):
        # Keep only samples with at least one positive grasp
        self.samples = [s for s in self.samples if s.positives]
        self._rng = random.Random(self.seed)
        self._scale_x = INPUT_SIZE / IMG_W
        self._scale_y = INPUT_SIZE / IMG_H

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _flip_grasp(self, g: GraspRect) -> GraspRect:
        """Horizontal flip: cx → IMG_W - cx, angle → -angle. Width/height unchanged."""
        return make_rect(
            cx=IMG_W - g.cx,
            cy=g.cy,
            w=g.width,
            h=g.height,
            angle_rad=-g.angle_rad,
        )

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load_image(s.rgb_path)
        g = _central_positive(s.positives)
        # horizontal-flip augmentation (50% chance) — keeps the model from
        # baking in left/right bias of the small training set
        if self.augment and self._rng.random() < 0.5:
            img = img[:, ::-1, :].copy()
            g = self._flip_grasp(g)
        target = encode_target(g, self._scale_x, self._scale_y)
        return torch.from_numpy(_imagenet_normalize(img)), torch.from_numpy(target), idx


# ---------------------------------------------------------------------------
# Helper for object-wise splitting (Phase 1 used folders 01,02 train / 03 test)
# ---------------------------------------------------------------------------

def split_object_wise(samples: list[CornellSample], test_folders: Sequence[int]) -> tuple[list, list]:
    test_set = set(test_folders)
    train, test = [], []
    for s in samples:
        (test if s.object_id in test_set else train).append(s)
    return train, test
