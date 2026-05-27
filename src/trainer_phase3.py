"""Phase-3 trainer with optional multi-positive 'closest-GT' loss.

This is a thin extension of ``trainer.train_regression`` that supports:

  * single-positive MSE (identical to Phase-2 behaviour) when the loader
    yields ``(x, target, idx)``;
  * multi-positive closest-GT MSE when the loader yields
    ``(x, target_primary, targets_all (B, K, 6), n_real (B,), idx)``.
    Per-example loss is the MSE against the candidate GT that minimises the
    6-vec distance to the *current* prediction. Padded entries (>= n_real)
    are masked out so they don't influence the argmin.

The optimizer / schedule / epochs interface matches ``trainer.train_regression``
so the Phase-3 notebook can swap the two cleanly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


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


def _closest_gt_loss(pred: torch.Tensor, targets_all: torch.Tensor, n_real: torch.Tensor) -> torch.Tensor:
    """pred: (B, 6); targets_all: (B, K, 6); n_real: (B,) int.

    For each row b, find the k* in [0, n_real[b]) that minimises
    ||pred[b] - targets_all[b, k]||^2 and use that as the supervision target.
    Mean MSE over the batch.
    """
    B, K, D = targets_all.shape
    pred_e = pred.unsqueeze(1).expand(B, K, D)  # (B, K, 6)
    sq = (pred_e - targets_all).pow(2).sum(dim=2)  # (B, K) squared distance per candidate
    # mask out padded slots so they can't be picked
    idx_range = torch.arange(K, device=pred.device).unsqueeze(0).expand(B, K)
    valid = idx_range < n_real.unsqueeze(1)
    sq_masked = sq.masked_fill(~valid, float('inf'))
    k_star = sq_masked.argmin(dim=1)                # (B,)
    chosen = targets_all[torch.arange(B, device=pred.device), k_star]  # (B, 6)
    return F.mse_loss(pred, chosen)


def train_regression_p3(model, train_loader: DataLoader, *, epochs: int = 30,
                        lr: float = 1e-3, weight_decay: float = 1e-4,
                        multi_positive: bool = False,
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
        ep_loss = 0.0
        nb = 0
        for batch in train_loader:
            if multi_positive:
                x, t_primary, targets_all, n_real, _idx = batch
                x = x.to(device)
                targets_all = targets_all.to(device)
                n_real = n_real.to(device)
                pred = model(x)
                loss = _closest_gt_loss(pred, targets_all, n_real)
            else:
                x, t, _idx = batch
                x = x.to(device); t = t.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, t)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        scheduler.step()
        losses.append(ep_loss / max(nb, 1))
    return TrainResult(name=name, n_params=n_params,
                       train_seconds=time.time() - t0, train_curve=losses)
