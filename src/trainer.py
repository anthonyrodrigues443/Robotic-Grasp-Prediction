"""Training + evaluation harness shared across all Phase-2 models.

Two flavours of training step are implemented:
  - ``train_regression(...)``      6-vec MSE loss (M1, M2, M5).
  - ``train_hybrid(...)``          4-vec MSE for (cx,cy,w,h) + cross-entropy
                                   for 18-way angle bin (M3).
  - ``train_ggcnn(...)``           per-pixel quality MSE + masked angle/width
                                   MSE (M4).

All three return the same shape of training-curve dict so the leaderboard can
plot them on a single chart.

Eval is identical for everything: predict the 6-vec → decode to a GraspRect →
score under Jiang IoU>0.25 / angle<30° against ALL positives in the test
image. (Decoded prediction lives in the original 640x480 frame; the Jiang
metric is independent of input resolution.)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cornell import is_correct_jiang, iou as poly_iou, angle_diff_deg
from dataset import CornellGraspDataset, decode_prediction
from torch_models import (
    ANGLE_BINS, GGCNN_OUT_SIZE, GGCNNTiny, ResNet18HybridHead,
    angle_to_bin_idx, bin_idx_to_angle_rad, ggcnn_decode, ggcnn_target_maps,
)


@dataclass
class TrainResult:
    name: str
    n_params: int
    train_seconds: float
    train_curve: list[float]
    val_curve: list[float] = field(default_factory=list)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Standard 6-vec regression loop (M1, M2, M5)
# ---------------------------------------------------------------------------

def train_regression(model, train_loader: DataLoader, *, epochs: int = 30,
                     lr: float = 1e-3, weight_decay: float = 1e-4,
                     device: torch.device | None = None) -> TrainResult:
    device = device or _pick_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    name = getattr(model, "name", model.__class__.__name__)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    losses: list[float] = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0; nb = 0
        for x, y, _ in train_loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        scheduler.step()
        losses.append(ep_loss / max(nb, 1))
    return TrainResult(name=name, n_params=n_params,
                       train_seconds=time.time() - t0, train_curve=losses)


# ---------------------------------------------------------------------------
# Hybrid (regression + classification) loop (M3)
# ---------------------------------------------------------------------------

def train_hybrid(model: ResNet18HybridHead, train_loader: DataLoader, *,
                 epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-4,
                 ce_weight: float = 1.0, device: torch.device | None = None) -> TrainResult:
    device = device or _pick_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    name = getattr(model, "name", model.__class__.__name__)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    losses: list[float] = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0; nb = 0
        for x, y, _ in train_loader:
            x = x.to(device); y = y.to(device)
            # extract angle from sin2θ/cos2θ → bin idx
            s2 = y[:, 4].cpu().numpy()
            c2 = y[:, 5].cpu().numpy()
            angles = 0.5 * np.arctan2(s2, c2)
            bin_idx = np.array([angle_to_bin_idx(float(a)) for a in angles], dtype=np.int64)
            bin_idx_t = torch.from_numpy(bin_idx).to(device)
            reg, cls = model(x)
            # reg targets: cx, cy, w, h
            reg_target = y[:, :4]
            loss_reg = F.mse_loss(reg, reg_target)
            loss_cls = F.cross_entropy(cls, bin_idx_t)
            loss = loss_reg + ce_weight * loss_cls
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        scheduler.step()
        losses.append(ep_loss / max(nb, 1))
    return TrainResult(name=name, n_params=n_params,
                       train_seconds=time.time() - t0, train_curve=losses)


# ---------------------------------------------------------------------------
# GG-CNN loop (M4)
# ---------------------------------------------------------------------------

def train_ggcnn(model: GGCNNTiny, train_loader: DataLoader, *,
                epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-4,
                device: torch.device | None = None) -> TrainResult:
    device = device or _pick_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    name = getattr(model, "name", model.__class__.__name__)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    losses: list[float] = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0; nb = 0
        for x, y, _ in train_loader:
            x = x.to(device); y = y.to(device)
            q_t, s_t, c_t, w_t = ggcnn_target_maps(y, out_size=GGCNN_OUT_SIZE)
            q_p, s_p, c_p, w_p = model(x)
            # quality MSE everywhere
            loss = F.mse_loss(q_p, q_t)
            # masked angle/width MSE (only where quality target is high)
            mask = (q_t > 0.05).float()
            denom = mask.sum().clamp(min=1.0)
            loss = loss + ((s_p - s_t) ** 2 * mask).sum() / denom
            loss = loss + ((c_p - c_t) ** 2 * mask).sum() / denom
            loss = loss + ((w_p - w_t) ** 2 * mask).sum() / denom
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        scheduler.step()
        losses.append(ep_loss / max(nb, 1))
    return TrainResult(name=name, n_params=n_params,
                       train_seconds=time.time() - t0, train_curve=losses)


# ---------------------------------------------------------------------------
# Evaluation — runs the model on the test set, returns the same dict shape
# as Phase-1's evaluate() so the leaderboard merges cleanly.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, samples, *, device: torch.device | None = None,
             model_kind: str = "reg", iou_thresh: float = 0.25,
             angle_thresh_deg: float = 30.0) -> dict:
    """``model_kind`` ∈ {'reg', 'hybrid', 'ggcnn'}. ``samples`` is a list of
    CornellSample. Returns {accuracy, median_iou, median_angle_err, n,
    per_sample, inference_ms_per_image}."""
    device = device or _pick_device()
    model = model.to(device).eval()
    ds = CornellGraspDataset(samples, augment=False)
    # Use a DataLoader with batch_size=1 so we keep the sample-index mapping
    # trivial; throughput isn't the point of eval.
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    correct = 0; total = 0
    best_ious: list[float] = []
    best_angle_errs: list[float] = []
    per_sample = []
    t_total = 0.0
    for x, _, idx in loader:
        x = x.to(device)
        idx = int(idx.item())
        s = ds.samples[idx]
        t0 = time.time()
        if model_kind == "reg":
            pred = model(x).cpu().numpy()[0]
        elif model_kind == "hybrid":
            reg, cls = model(x)
            reg = reg.cpu().numpy()[0]
            bin_idx = int(cls.argmax(dim=1).item())
            ang = bin_idx_to_angle_rad(bin_idx)
            pred = np.array([reg[0], reg[1], reg[2], reg[3],
                             math.sin(2 * ang), math.cos(2 * ang)], dtype=np.float32)
        elif model_kind == "ggcnn":
            q, s_m, c_m, w_m = model(x)
            pred = ggcnn_decode(q[0, 0].cpu().numpy(), s_m[0, 0].cpu().numpy(),
                                c_m[0, 0].cpu().numpy(), w_m[0, 0].cpu().numpy())
        else:
            raise ValueError(model_kind)
        t_total += time.time() - t0
        rect = decode_prediction(pred)
        gts = s.positives
        ok = is_correct_jiang(rect, gts, iou_thresh, angle_thresh_deg)
        cands = [(poly_iou(rect, g), angle_diff_deg(rect, g)) for g in gts]
        best = max(cands, key=lambda t: t[0]) if cands else (0.0, 90.0)
        best_ious.append(best[0])
        best_angle_errs.append(best[1])
        per_sample.append({
            "pcd_id": s.pcd_id,
            "object_id": s.object_id,
            "correct": int(ok),
            "best_iou": float(best[0]),
            "best_angle_err": float(best[1]),
        })
        correct += int(ok); total += 1
    return {
        "n": total,
        "accuracy": correct / max(total, 1),
        "median_iou": float(np.median(best_ious)) if best_ious else 0.0,
        "median_angle_err": float(np.median(best_angle_errs)) if best_angle_errs else 0.0,
        "per_sample": per_sample,
        "inference_ms_per_image": 1000.0 * t_total / max(total, 1),
    }
