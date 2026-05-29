"""Phase-5 per-pixel grasp-map architectures.

Two new fully-convolutional grasp-quality predictors, plus we reuse the
Phase-2 ``GGCNNTiny`` (out 218²) unchanged as the "rescued from-scratch
baseline". Every model here returns four per-pixel maps as a tuple
``(q, cos2θ, sin2θ, width)`` where ``q`` is sigmoid-squashed in [0,1] and the
others are raw. The decoder lives in ``dataset_phase5.decode_grasp_map``.

  GRConvNetLite  — Kumra et al. 2020 (GR-ConvNet) shrunk: conv down ×3, 5
                   residual blocks, transposed-conv up ×2 with a skip
                   connection, 4 heads at 112². From scratch (tests whether a
                   *better* per-pixel architecture rescues the paradigm without
                   ImageNet weights). ``use_skip=False`` drops the skip add for
                   the ablation.

  ResNet18FCN    — an ImageNet-pretrained ResNet-18 encoder + an FPN-style
                   decoder producing 4 maps at 56². This is the clean
                   paradigm-isolating model: it shares M2/E3.x's backbone and
                   pretraining and changes ONLY the output head (per-pixel maps
                   instead of a global 6-vec). ``pretrained=False`` is the
                   from-scratch ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# GR-ConvNet-lite (from scratch), out = 112
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(x + y, inplace=True)


class GRConvNetLite(nn.Module):
    name = "GRConvNetLite"
    out_size = 112

    def __init__(self, n_res: int = 5, use_skip: bool = True):
        super().__init__()
        self.use_skip = use_skip
        self.c1 = nn.Sequential(nn.Conv2d(3, 32, 9, 1, 4, bias=False),
                                nn.BatchNorm2d(32), nn.ReLU(inplace=True))   # 224
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(64), nn.ReLU(inplace=True))   # 112
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(128), nn.ReLU(inplace=True))  # 56
        self.res = nn.Sequential(*[_ResBlock(128) for _ in range(n_res)])
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                 nn.BatchNorm2d(64), nn.ReLU(inplace=True))  # 112
        self.h_q = nn.Conv2d(64, 1, 1)
        self.h_c = nn.Conv2d(64, 1, 1)
        self.h_s = nn.Conv2d(64, 1, 1)
        self.h_w = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)   # 64 @ 112
        x3 = self.c3(x2)   # 128 @ 56
        x3 = self.res(x3)
        u = self.up1(x3)   # 64 @ 112
        if self.use_skip:
            u = u + x2     # skip connection from the matching-resolution encoder map
        q = torch.sigmoid(self.h_q(u))
        return q, self.h_c(u), self.h_s(u), self.h_w(u)


# ---------------------------------------------------------------------------
# ResNet-18 FCN (pretrained encoder + FPN-style decoder), out = 56
# ---------------------------------------------------------------------------

class ResNet18FCN(nn.Module):
    name = "ResNet18FCN"
    out_size = 112  # final maps upsampled to 112² so output resolution is not
                    # the binding constraint (a 56² grid caps GT-roundtrip
                    # accuracy at ~0.90; 112² lifts that ceiling to ~0.99).

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)  # -> 64 @ 56
        self.layer1 = bb.layer1   # 64  @ 56
        self.layer2 = bb.layer2   # 128 @ 28
        self.layer3 = bb.layer3   # 256 @ 14
        self.layer4 = bb.layer4   # 512 @ 7
        # lateral 1x1 projections to a common 64-d decoder width (FPN-lite)
        self.p4 = nn.Conv2d(512, 64, 1)
        self.p3 = nn.Conv2d(256, 64, 1)
        self.p2 = nn.Conv2d(128, 64, 1)
        self.p1 = nn.Conv2d(64, 64, 1)
        self.smooth = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.h_q = nn.Conv2d(64, 1, 1)
        self.h_c = nn.Conv2d(64, 1, 1)
        self.h_s = nn.Conv2d(64, 1, 1)
        self.h_w = nn.Conv2d(64, 1, 1)

    @staticmethod
    def _up_add(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x + skip

    def forward(self, x):
        s = self.stem(x)
        c1 = self.layer1(s)    # 64  @ 56
        c2 = self.layer2(c1)   # 128 @ 28
        c3 = self.layer3(c2)   # 256 @ 14
        c4 = self.layer4(c3)   # 512 @ 7
        d = self.p4(c4)                      # 64 @ 7
        d = self._up_add(d, self.p3(c3))     # 64 @ 14
        d = self._up_add(d, self.p2(c2))     # 64 @ 28
        d = self._up_add(d, self.p1(c1))     # 64 @ 56
        d = F.interpolate(d, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)  # -> 112
        d = self.smooth(d)
        q = torch.sigmoid(self.h_q(d))
        return q, self.h_c(d), self.h_s(d), self.h_w(d)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
