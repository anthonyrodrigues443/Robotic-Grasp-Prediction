"""Phase-3 dataset extensions: rotation augmentation, depth-as-blue-channel,
and multi-positive label exposure.

We deliberately keep these as a separate module so the Phase-2 ``dataset.py``
stays frozen as the apples-to-apples Phase-2 baseline. Each option here is a
*flag* on the Phase-3 dataset wrapper so the ablation in the notebook can flip
one knob at a time.

Design notes:

  * **Rotation augmentation**: a random rotation in [-rot_deg, +rot_deg] is
    applied to the *image* (around its center) and the *same* transform is
    applied to all positive GT rectangles. We don't rotate the negatives
    because they're not used in supervised training; only positives are picked
    by ``_central_positive`` / multi-positive routing. Rotation is the right
    augmentation for grasp detection because (a) Cornell's tabletop has no
    canonical 'up' so the prior is rotation-invariant, and (b) the Phase-1
    finding that *orientation* is *not* the hard part means we can rotate
    freely without breaking the implicit angle prior — we want the model to
    *generalise* over orientation, not memorise the tabletop's narrow band.

  * **Depth-as-blue-channel** (Lenz 2014 trick): the depth value at each
    pixel is normalised to [0, 1] using the *image's own* depth range
    (min-max), then substituted into the blue channel of the RGB image.
    This lets us keep ResNet-18 / ViT pretraining unchanged (still 3-channel
    input). It uses the same depth loader as the Phase-1 depth baseline so
    no new parsing code is needed.

  * **Multi-positive supervision**: instead of always using the *most central*
    positive as the supervision target (Phase-2 behaviour), the dataset
    returns *all* positives' 6-vector targets. The training step in
    ``trainer_phase3.py`` then computes per-example loss against each
    candidate target and uses the *minimum* — a 'closest-GT' loss. This
    matches what real grasps look like (an image has many valid grasps)
    without requiring assignment heuristics.

All three flags compose. The Phase-2 frozen dataset behaviour is
``CornellGraspDatasetP3(augment_rot_deg=0, use_depth_blue=False,
multi_positive=False)``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cornell import CornellSample, GraspRect, IMG_W, IMG_H, make_rect
from dataset import INPUT_SIZE, encode_target, _imagenet_normalize
from baselines import DepthAntipodalBaseline


# ---------------------------------------------------------------------------
# Process-level depth cache. Cornell .pcd files are 5+ MB of ASCII; parsing
# them per __getitem__ once-per-epoch is what made the Phase-3 notebook hit
# the 1-hour cell timeout on the depth-as-blue experiments. We cache the
# normalised (224, 224) float32 channel keyed by pcd_path so it's parsed at
# most once per process.
# ---------------------------------------------------------------------------

_DEPTH_CACHE: dict[str, np.ndarray] = {}
_DEPTH_LOADER = DepthAntipodalBaseline()


def _cached_depth(pcd_path: Path) -> np.ndarray:
    key = str(pcd_path)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    raw = _DEPTH_LOADER._load_depth_image(pcd_path)
    norm = _depth_to_normalised_channel(raw, INPUT_SIZE)
    _DEPTH_CACHE[key] = norm
    return norm


def prewarm_depth_cache(samples: list[CornellSample]) -> int:
    """Force-load every sample's depth map into the in-process cache.
    Returns the number of unique files now cached. Call once at notebook
    start so the first epoch of a depth-enabled experiment doesn't pay the
    ASCII-parse tax."""
    for s in samples:
        _cached_depth(s.pcd_path)
    return len(_DEPTH_CACHE)


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def _depth_to_normalised_channel(depth: np.ndarray, target_size: int = INPUT_SIZE) -> np.ndarray:
    """Convert a (480, 640) depth map in mm with NaNs to a (target_size,
    target_size) float32 in [0, 1].

    Normalisation: per-image min-max within the *finite* values, with NaNs
    set to 0 (= 'background' / max-distance side). We then invert so that
    *closer* objects = higher value (matches the convention that depth
    'pops out' the foreground)."""
    if depth is None or not np.isfinite(depth).any():
        return np.zeros((target_size, target_size), dtype=np.float32)
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    d_min = float(np.nanmin(d))
    d_max = float(np.nanmax(d))
    if d_max - d_min < 1.0:
        out = np.zeros_like(d)
    else:
        out = np.where(finite, (d - d_min) / (d_max - d_min), 1.0)
        out = 1.0 - out  # closer = higher
    out_img = Image.fromarray((out * 255).astype(np.uint8))
    out_img = out_img.resize((target_size, target_size), Image.BILINEAR)
    return (np.asarray(out_img, dtype=np.float32) / 255.0)


# ---------------------------------------------------------------------------
# Rotation augmentation helpers
# ---------------------------------------------------------------------------

def _rotate_image_array(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a HxWx3 float [0,1] image around its center by angle_deg
    (CCW). Out-of-frame pixels are filled with the edge value."""
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.rotate(angle_deg, resample=Image.BILINEAR, expand=False,
                     fillcolor=(0, 0, 0))
    return np.asarray(pil, dtype=np.float32) / 255.0


def _rotate_grasp(g: GraspRect, angle_deg: float, cx_pivot: float, cy_pivot: float) -> GraspRect:
    """Rotate the grasp rectangle around (cx_pivot, cy_pivot) by angle_deg
    *CCW in image coordinates* (PIL rotates CCW with positive angles around
    image centre, but the image coordinate y-axis points DOWN, so a CCW
    image rotation is a CW geometric rotation — we account for this by
    negating sin in the rotation matrix below)."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    # PIL's positive angle = CCW with y-up; in image coords (y-down) this
    # is equivalent to applying R = [[cos, sin], [-sin, cos]] around the
    # pivot. The grasp's own orientation angle decreases by `angle_deg`.
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    pivot = np.array([cx_pivot, cy_pivot])
    new_corners = (g.corners - pivot) @ R.T + pivot
    new_angle = g.angle_rad - rad
    # wrap to [-pi/2, pi/2)
    while new_angle >= math.pi / 2:
        new_angle -= math.pi
    while new_angle < -math.pi / 2:
        new_angle += math.pi
    return GraspRect(corners=new_corners, label=g.label)


# ---------------------------------------------------------------------------
# Phase-3 dataset
# ---------------------------------------------------------------------------

MULTI_POS_MAX = 8  # cap the number of positives exposed per item for batching


@dataclass
class CornellGraspDatasetP3(Dataset):
    """Phase-3 superset of ``CornellGraspDataset``.

    Flags (all default to False / 0 → exact Phase-2 frozen behaviour):

    * ``augment_rot_deg``: if > 0, apply a random rotation in
      [-augment_rot_deg, +augment_rot_deg] each ``__getitem__`` call.
      Horizontal-flip aug is always applied when ``augment=True`` for parity
      with Phase 2.
    * ``use_depth_blue``: substitute the (normalised, foreground-high) depth
      channel for the blue channel of the input image. Keeps the 3-channel
      input shape so pretrained backbones load unchanged.
    * ``multi_positive``: in addition to ``target``, return a
      ``targets_all`` tensor of shape (MULTI_POS_MAX, 6) with the encoded
      targets for *all* positives (padded with the most-central positive
      repeated if there are fewer than MULTI_POS_MAX). ``n_targets`` says
      how many are real. The closest-GT loss in trainer_phase3.py uses
      ``targets_all[:n_targets]``.
    * ``augment``: must be True for any aug to happen at all.
    """
    samples: list[CornellSample]
    augment: bool = False
    augment_rot_deg: float = 0.0
    use_depth_blue: bool = False
    multi_positive: bool = False
    seed: int = 42

    def __post_init__(self):
        self.samples = [s for s in self.samples if s.positives]
        self._rng = random.Random(self.seed)
        self._scale_x = INPUT_SIZE / IMG_W
        self._scale_y = INPUT_SIZE / IMG_H
        self._depth_loader = DepthAntipodalBaseline()

    def __len__(self) -> int:
        return len(self.samples)

    # ---- image / depth I/O ------------------------------------------------

    def _load_rgb(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _load_depth(self, pcd_path: Path) -> np.ndarray:
        # Cached: O(1) after the first call per file in this process.
        return _cached_depth(pcd_path)

    # ---- aug helpers ------------------------------------------------------

    @staticmethod
    def _hflip_grasp(g: GraspRect) -> GraspRect:
        # mirror in the resized 224 frame
        cx_new = INPUT_SIZE - 1 - g.corners[:, 0]
        cy = g.corners[:, 1]
        new_corners = np.stack([cx_new, cy], axis=1)
        # Reverse winding so the (p0→p1) jaw-opening direction convention holds
        new_corners = new_corners[[1, 0, 3, 2]]
        return GraspRect(corners=new_corners, label=g.label)

    def _resize_grasps(self, grasps: list[GraspRect]) -> list[GraspRect]:
        """Project each grasp from 640x480 → 224x224 (same as encode_target
        does for the single chosen positive, but applied to every positive so
        we can rotate/flip uniformly with the image)."""
        out: list[GraspRect] = []
        for g in grasps:
            corners224 = np.stack([
                g.corners[:, 0] * self._scale_x,
                g.corners[:, 1] * self._scale_y,
            ], axis=1)
            out.append(GraspRect(corners=corners224, label=g.label))
        return out

    def _encode_224(self, g: GraspRect) -> np.ndarray:
        """Encode a grasp that's *already* in the 224-frame to the 6-vec."""
        cx = g.corners[:, 0].mean()
        cy = g.corners[:, 1].mean()
        w = float(np.linalg.norm(g.corners[1] - g.corners[0]))
        h = float(np.linalg.norm(g.corners[2] - g.corners[1]))
        a = g.angle_rad
        return np.array([
            cx / INPUT_SIZE,
            cy / INPUT_SIZE,
            w / INPUT_SIZE,
            h / INPUT_SIZE,
            math.sin(2.0 * a),
            math.cos(2.0 * a),
        ], dtype=np.float32)

    @staticmethod
    def _central(positives: list[GraspRect]) -> GraspRect:
        if len(positives) == 1:
            return positives[0]
        cx = float(np.mean([p.corners[:, 0].mean() for p in positives]))
        cy = float(np.mean([p.corners[:, 1].mean() for p in positives]))
        return min(positives, key=lambda g: (g.corners[:, 0].mean() - cx) ** 2 + (g.corners[:, 1].mean() - cy) ** 2)

    # ---- main getter ------------------------------------------------------

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rgb = self._load_rgb(s.rgb_path)
        positives_224 = self._resize_grasps(s.positives)

        # optional rotation
        rot_deg = 0.0
        if self.augment and self.augment_rot_deg > 0:
            rot_deg = self._rng.uniform(-self.augment_rot_deg, self.augment_rot_deg)
            rgb = _rotate_image_array(rgb, rot_deg)
            pivot = INPUT_SIZE / 2.0
            positives_224 = [_rotate_grasp(g, rot_deg, pivot, pivot) for g in positives_224]

        # optional hflip (Phase-2 parity)
        if self.augment and self._rng.random() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
            positives_224 = [self._hflip_grasp(g) for g in positives_224]

        # optional depth-blue substitution
        if self.use_depth_blue:
            depth = self._load_depth(s.pcd_path)
            rgb[:, :, 2] = depth  # replace blue

        # primary target = most central positive
        primary = self._central(positives_224)
        target_primary = self._encode_224(primary)
        x = torch.from_numpy(_imagenet_normalize(rgb))
        t = torch.from_numpy(target_primary)

        if not self.multi_positive:
            return x, t, idx

        # multi-positive: pad/truncate to MULTI_POS_MAX
        targets = [self._encode_224(g) for g in positives_224][:MULTI_POS_MAX]
        n_real = len(targets)
        while len(targets) < MULTI_POS_MAX:
            targets.append(target_primary)  # pad with primary (a real target)
        targets_arr = np.stack(targets, axis=0)
        return x, t, torch.from_numpy(targets_arr), int(n_real), idx


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def image_wise_split(samples: list[CornellSample], test_frac: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """Random 80/20 across all samples — the canonical 'image-wise' split.

    This is the split published Cornell numbers commonly report (Lenz 2014,
    Redmon 2015 each report image-wise + object-wise)."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_frac))
    return shuffled[n_test:], shuffled[:n_test]


def object_wise_split(samples: list[CornellSample], test_object_ids: Sequence[int]) -> tuple[list, list]:
    test_set = set(test_object_ids)
    train, test = [], []
    for s in samples:
        (test if s.object_id in test_set else train).append(s)
    return train, test
