"""Five candidate architectures for Phase-2 Cornell grasp detection.

Every model takes a 3x224x224 ImageNet-normalised image and returns a 6-vector
(cx_norm, cy_norm, w_norm, h_norm, sin(2θ), cos(2θ)) in the resized 224x224
frame. The dataset/loss/decoding is identical across models so the leaderboard
is a clean apples-to-apples comparison of *architectural inductive biases*.

The five candidates were chosen along independent axes:

  M1  TinyRedmonCNN     — from-scratch, AlexNet-shaped (no ImageNet prior)
  M2  ResNet18Regressor — ImageNet-pretrained CNN, vanilla regression head
  M3  ResNet18HybridHead — same backbone as M2, but angle as 18-way *classifier*
                          (Redmon-style discretised angle) + 4-D regression
                          for (cx, cy, w, h). Tests the regression-vs-class
                          tradeoff for orientation.
  M4  GGCNNTiny         — fully-convolutional encoder-decoder predicting
                          per-pixel (quality, sin2θ, cos2θ, width) maps;
                          argmax-pick the grasp at inference.
  M5  TinyViT           — patch16 / 6-layer ViT (from scratch). Tests whether
                          a global-attention prior beats convolutional locality
                          on a small dataset.

Reasoning for the axis choices:
  - M1 vs M2 isolates the ImageNet transfer effect (same shape, only weights differ).
  - M2 vs M3 isolates the angle-as-regression vs angle-as-classification choice.
  - M2 vs M4 isolates the global-regression vs per-pixel-quality paradigm.
  - M2 vs M5 isolates the convolutional vs transformer inductive bias.
  - M1 is the from-scratch floor; if any pretrained model fails to beat it
    decisively, the pretraining wasn't transferring.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataset import INPUT_SIZE


# ---------------------------------------------------------------------------
# M1 — TinyRedmonCNN. 5 conv blocks (AlexNet-shaped) + 2 FC. Small enough to
# train from scratch on 200 images without immediate collapse.
# ---------------------------------------------------------------------------

class TinyRedmonCNN(nn.Module):
    name = "M1_TinyRedmonCNN"

    def __init__(self, out_dim: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),  # 224 → 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                                     # 55 → 27
            nn.Conv2d(32, 96, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                                     # 27 → 13
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                                     # 13 → 6
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# M2 — ResNet-18 pretrained, swap fc for 6-vec regression. Vanilla baseline.
# ---------------------------------------------------------------------------

class ResNet18Regressor(nn.Module):
    name = "M2_ResNet18Regressor"

    def __init__(self, out_dim: int = 6, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_dim, out_dim),
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# M3 — ResNet-18 with hybrid head: 4D regression (cx, cy, w, h) + 18-way
# classification for angle bin (each 10 deg). Inference picks argmax bin
# centre as the final angle. This tests Redmon's original discretisation
# choice vs M2's continuous angle output.
# ---------------------------------------------------------------------------

ANGLE_BINS = 18
ANGLE_BIN_WIDTH_DEG = 180.0 / ANGLE_BINS   # 10°


def angle_to_bin_idx(angle_rad: float) -> int:
    """Map angle in [-pi/2, pi/2) → integer bin in [0, ANGLE_BINS)."""
    deg = math.degrees(angle_rad)
    # shift to [0, 180)
    while deg < 0:
        deg += 180.0
    while deg >= 180.0:
        deg -= 180.0
    idx = int(deg / ANGLE_BIN_WIDTH_DEG)
    return min(idx, ANGLE_BINS - 1)


def bin_idx_to_angle_rad(idx: int) -> float:
    deg = (idx + 0.5) * ANGLE_BIN_WIDTH_DEG  # bin centre
    if deg >= 90.0:
        deg -= 180.0
    return math.radians(deg)


class ResNet18HybridHead(nn.Module):
    name = "M3_ResNet18HybridHead"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.reg_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_dim, 4),  # cx, cy, w, h
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_dim, ANGLE_BINS),
        )

    def forward(self, x):
        feats = self.backbone(x)
        reg = self.reg_head(feats)
        cls = self.cls_head(feats)
        return reg, cls


# ---------------------------------------------------------------------------
# M4 — GG-CNN tiny. Per-pixel quality + sin(2θ) + cos(2θ) + width prediction.
# Encoder-decoder: 4 down + 4 up, no skip connections (to keep it tiny).
# Inference picks the pixel with highest quality, decodes angle from sin/cos
# at that pixel, decodes width from the width channel at that pixel.
# ---------------------------------------------------------------------------

class _ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class GGCNNTiny(nn.Module):
    name = "M4_GGCNNTiny"

    def __init__(self):
        super().__init__()
        # encoder
        self.e1 = _ConvBNReLU(3, 32, k=9, s=3, p=3)   # 224 → 75
        self.e2 = _ConvBNReLU(32, 16, k=5, s=2, p=2)  # 75 → 38
        self.e3 = _ConvBNReLU(16, 8, k=3, s=2, p=1)   # 38 → 19
        # decoder (transposed conv)
        self.d1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=0)  # 19 → 37
        self.d2 = nn.ConvTranspose2d(16, 32, kernel_size=5, stride=2, padding=2, output_padding=0)  # 37 → 73
        self.d3 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=3, padding=3, output_padding=0)  # 73 → 219
        # heads (4 per-pixel maps: quality, sin2θ, cos2θ, width)
        self.h_q = nn.Conv2d(32, 1, kernel_size=2)
        self.h_s = nn.Conv2d(32, 1, kernel_size=2)
        self.h_c = nn.Conv2d(32, 1, kernel_size=2)
        self.h_w = nn.Conv2d(32, 1, kernel_size=2)

    def forward(self, x):
        x = self.e1(x); x = self.e2(x); x = self.e3(x)
        x = F.relu(self.d1(x), inplace=True)
        x = F.relu(self.d2(x), inplace=True)
        x = F.relu(self.d3(x), inplace=True)
        # output resolution is 218x218 — close to but not exactly 224; we crop the
        # GT target maps to match in the loss (see ggcnn_target_maps below).
        q = torch.sigmoid(self.h_q(x))
        s = self.h_s(x)
        c = self.h_c(x)
        w = self.h_w(x)
        return q, s, c, w


GGCNN_OUT_SIZE = 218  # output resolution after the conv stack above


def ggcnn_target_maps(targets: torch.Tensor, out_size: int = GGCNN_OUT_SIZE, sigma_px: int = 4) -> tuple[torch.Tensor, ...]:
    """Build per-pixel (quality, sin2θ, cos2θ, width) target maps from the
    flat 6-vec targets used by every other model. ``targets`` is (B, 6) in the
    224-frame; we project to ``out_size`` and paint a Gaussian-shaped quality
    blob centred on (cx, cy) plus constant angle/width maps inside that blob.
    """
    B = targets.shape[0]
    device = targets.device
    H = W = out_size
    q = torch.zeros(B, 1, H, W, device=device)
    s = torch.zeros(B, 1, H, W, device=device)
    c = torch.zeros(B, 1, H, W, device=device)
    wmap = torch.zeros(B, 1, H, W, device=device)
    scale = out_size / INPUT_SIZE
    ys = torch.arange(H, device=device).view(1, 1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, 1, W)
    for b in range(B):
        cx = float(targets[b, 0]) * INPUT_SIZE * scale
        cy = float(targets[b, 1]) * INPUT_SIZE * scale
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        blob = torch.exp(-d2[0, 0] / (2 * sigma_px ** 2))
        q[b, 0] = blob
        # only fill the angle/width inside the blob region (where quality is high)
        mask = (blob > 0.05).float()
        s[b, 0] = mask * float(targets[b, 4])
        c[b, 0] = mask * float(targets[b, 5])
        wmap[b, 0] = mask * float(targets[b, 2])  # width
    return q, s, c, wmap


def ggcnn_decode(q: np.ndarray, s: np.ndarray, c: np.ndarray, w: np.ndarray,
                 out_size: int = GGCNN_OUT_SIZE) -> np.ndarray:
    """Per-image decode: argmax pixel → (cx, cy) in 224-frame, angle from
    sin/cos at that pixel, width from width channel, height = median Cornell
    plate height (in normalised units)."""
    flat = q.reshape(-1)
    idx = int(np.argmax(flat))
    yy, xx = divmod(idx, out_size)
    scale_back = INPUT_SIZE / out_size
    cx_n = (xx * scale_back) / INPUT_SIZE
    cy_n = (yy * scale_back) / INPUT_SIZE
    w_n = max(float(w[yy, xx]), 1e-3)
    s_v = float(s[yy, xx])
    c_v = float(c[yy, xx])
    # Fix height at the Cornell median (~25 px / 224 ≈ 0.112) — GG-CNN
    # canonically predicts only the width; the plate height is gripper-fixed.
    h_n = 25.0 / INPUT_SIZE
    return np.array([cx_n, cy_n, w_n, h_n, s_v, c_v], dtype=np.float32)


# ---------------------------------------------------------------------------
# M5 — TinyViT. Patch16 ViT-tiny from scratch (no pretrained transformer for
# this size in torchvision). 6-layer transformer, head_dim 32. Tests whether
# a global-attention prior can compete with convs on 200 training images.
# ---------------------------------------------------------------------------

class TinyViT(nn.Module):
    name = "M5_TinyViT"

    def __init__(self, image_size: int = INPUT_SIZE, patch_size: int = 16,
                 dim: int = 192, depth: int = 6, heads: int = 6,
                 mlp_ratio: float = 2.0, out_dim: int = 6):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)          # B, dim, H/p, W/p
        x = x.flatten(2).transpose(1, 2) # B, N, dim
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# ---------------------------------------------------------------------------
# Param counter for the leaderboard
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
