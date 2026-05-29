"""Phase-5 per-pixel grasp-map dataset, target painter and decoder.

Phase 2 trained a from-scratch GG-CNN (``M4_GGCNNTiny``) and it scored **0.0**.
Two things were wrong with that run, not the paradigm:

  1. **Data starvation** — only 200 training images (Phase 2 used the 300-image
     subset). Phase 3 later showed the project was data-starved, not
     model-starved (0.55 -> 0.73 from more data alone).
  2. **Single-blob targets** — Phase 2's ``ggcnn_target_maps`` painted a quality
     blob for *one* (the most-central) positive. The entire point of the
     per-pixel paradigm is that it natively represents *every* valid grasp in
     the scene. Supervising it with a single blob throws that away and asks a
     fully-convolutional map predictor to behave like a global regressor.

This module fixes both: it paints a **multi-grasp** quality map from *all*
positive rectangles (Morrison & Corke 2018, GG-CNN), with per-pixel angle
(cos2θ, sin2θ) and gripper-width maps assigned from the nearest contributing
grasp. The output resolution is configurable so the same target painter feeds a
tiny from-scratch GG-CNN (218²), a GR-ConvNet-lite (112²) and a
ResNet-FCN (56²).

Coordinate convention matches the rest of the project: grasps are encoded in the
224×224 input frame; widths are stored normalised by ``INPUT_SIZE`` so they are
resolution-independent; decode projects back to the original 640×480 frame and
hands a 6-vec to ``dataset.decode_prediction`` so the field-standard Jiang
metric in ``cornell.py`` scores per-pixel and global models identically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cornell import CornellSample, GraspRect, IMG_W, IMG_H
from dataset import INPUT_SIZE, _imagenet_normalize, decode_prediction


# ---------------------------------------------------------------------------
# Target painting
# ---------------------------------------------------------------------------

def _grasp_to_224(g: GraspRect) -> tuple[float, float, float, float, float]:
    """Project a 640×480-frame grasp to the 224 input frame and return
    (cx, cy, w_px, h_px, angle_rad) in that frame. Width/height use the mean
    axis scale so an oriented rectangle is not distorted (same convention as
    ``dataset.encode_target``)."""
    sx = INPUT_SIZE / IMG_W
    sy = INPUT_SIZE / IMG_H
    s = (sx + sy) / 2.0
    cx = g.cx * sx
    cy = g.cy * sy
    w = g.width * s
    h = g.height * s
    return cx, cy, w, h, g.angle_rad


def paint_target_maps(positives: list[GraspRect], out_size: int,
                      sigma_frac: float = 0.0,
                      multi_grasp: bool = True) -> np.ndarray:
    """Build the (4, out_size, out_size) target stack (quality, cos2θ, sin2θ,
    width_norm) from a list of positive grasps (in the original 640×480 frame).

    * ``quality`` is the max over per-grasp Gaussian blobs centred on each
      grasp centre. σ scales with the grasp's own width (a bigger grasp gets a
      bigger blob), so the painted footprint roughly tracks the object size.
    * ``cos2θ / sin2θ / width`` at each pixel are taken from the grasp whose
      blob is *largest* there (per-pixel nearest-grasp assignment), and are 0
      outside any blob (masked in the loss).
    * ``multi_grasp=False`` paints only the most-central positive — the Phase-2
      behaviour, kept here so the ablation can isolate the multi-grasp lever.
    """
    H = W = out_size
    q = np.zeros((H, W), dtype=np.float32)
    cos = np.zeros((H, W), dtype=np.float32)
    sin = np.zeros((H, W), dtype=np.float32)
    wid = np.zeros((H, W), dtype=np.float32)
    if not positives:
        return np.stack([q, cos, sin, wid], axis=0)

    scale = out_size / INPUT_SIZE
    grasps = list(positives)
    if not multi_grasp:
        # most-central positive only
        mx = float(np.mean([g.cx for g in grasps]))
        my = float(np.mean([g.cy for g in grasps]))
        grasps = [min(grasps, key=lambda g: (g.cx - mx) ** 2 + (g.cy - my) ** 2)]

    ys = np.arange(H, dtype=np.float32).reshape(H, 1)
    xs = np.arange(W, dtype=np.float32).reshape(1, W)
    for g in grasps:
        cx224, cy224, w224, h224, ang = _grasp_to_224(g)
        cx_o = cx224 * scale
        cy_o = cy224 * scale
        # blob radius proportional to the grasp's *plate* (short) axis, with a
        # floor so tiny grasps still leave a learnable footprint.
        sig = max(h224 * scale * 0.5, 2.0)
        if sigma_frac > 0:  # ablation override: fixed σ as a fraction of out_size
            sig = max(sigma_frac * out_size, 1.0)
        blob = np.exp(-((xs - cx_o) ** 2 + (ys - cy_o) ** 2) / (2.0 * sig * sig))
        upd = blob > q
        q = np.where(blob > q, blob, q)
        cos = np.where(upd, math.cos(2.0 * ang), cos)
        sin = np.where(upd, math.sin(2.0 * ang), sin)
        wid = np.where(upd, min(w224 / INPUT_SIZE, 1.0), wid)
    return np.stack([q, cos, sin, wid], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Decoding a predicted map stack back to a GraspRect (640×480 frame)
# ---------------------------------------------------------------------------

def _smooth(q: np.ndarray, k: int = 5) -> np.ndarray:
    """Light Gaussian smoothing of the quality map before argmax (Morrison
    2018 does this to suppress single-pixel noise). Uses scipy when available,
    falls back to a separable box blur."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(q, sigma=k / 3.0)
    except Exception:
        pad = k // 2
        qp = np.pad(q, pad, mode="edge")
        out = np.zeros_like(q)
        for dy in range(k):
            for dx in range(k):
                out += qp[dy:dy + q.shape[0], dx:dx + q.shape[1]]
        return out / (k * k)


def decode_grasp_map(q: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                     wid: np.ndarray, out_size: int,
                     smooth: bool = True) -> GraspRect:
    """argmax the (smoothed) quality map → grasp centre; read angle from
    cos2θ/sin2θ and width at that pixel; fix the plate height at the Cornell
    median. Returns a GraspRect in the original 640×480 frame."""
    qq = _smooth(q) if smooth else q
    idx = int(np.argmax(qq))
    yy, xx = divmod(idx, out_size)
    back = INPUT_SIZE / out_size
    cx_n = (xx * back) / INPUT_SIZE
    cy_n = (yy * back) / INPUT_SIZE
    w_n = max(float(wid[yy, xx]), 1e-3)
    s_v = float(sin[yy, xx])
    c_v = float(cos[yy, xx])
    h_n = 25.0 / INPUT_SIZE  # gripper-fixed plate height (Cornell median)
    vec = np.array([cx_n, cy_n, w_n, h_n, s_v, c_v], dtype=np.float32)
    return decode_prediction(vec)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class GraspMapDataset(Dataset):
    """Returns (image[3,224,224], target_maps[4,out,out], idx).

    * ``out_size`` is the model's native output resolution.
    * ``augment`` enables 50%-chance horizontal flip (image + every positive),
      the one augmentation Phase 2 used; rotation is intentionally omitted
      because Phases 3–4 showed it hurts on Cornell.
    * ``multi_grasp`` / ``sigma_frac`` are forwarded to ``paint_target_maps``
      so the ablation can flip the multi-grasp lever and the blob size.
    """
    samples: list[CornellSample]
    out_size: int
    augment: bool = False
    multi_grasp: bool = True
    sigma_frac: float = 0.0
    seed: int = 42

    def __post_init__(self):
        import random
        self.samples = [s for s in self.samples if s.positives]
        self._rng = random.Random(self.seed)
        # Target maps are identical every epoch (only the random hflip varies),
        # so paint them once up front. This turns each training epoch from
        # painting-bound (~17 s) into pure GPU compute (~4 s) — a ~4x speedup
        # that matters across the 7 Phase-5 trainings. hflip is applied on the
        # cached tensors in __getitem__: flip horizontally and negate sin2θ
        # (θ → -θ under a mirror), which is exact and avoids re-painting.
        self._imgs = [self._load_rgb(s.rgb_path) for s in self.samples]
        self._maps = [paint_target_maps(s.positives, self.out_size,
                                        sigma_frac=self.sigma_frac,
                                        multi_grasp=self.multi_grasp)
                      for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        rgb = self._imgs[idx]
        maps = self._maps[idx]
        if self.augment and self._rng.random() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
            maps = maps[:, :, ::-1].copy()
            maps[2] = -maps[2]  # sin2θ flips sign under horizontal mirror
        x = torch.from_numpy(_imagenet_normalize(rgb))
        t = torch.from_numpy(maps)
        return x, t, idx
