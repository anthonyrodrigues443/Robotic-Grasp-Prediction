"""Phase-4 model variant: ResNet-18 regressor with a 4-channel input stem.

Identical to ``torch_models.ResNet18Regressor`` except the first conv accepts
4 input channels (RGB + depth). The standard "channel inflation" init is used:
the ImageNet-pretrained 3-channel conv1 weights are copied into channels 0-2,
and the 4th (depth) filter is seeded with the *mean* of the RGB filters so the
depth channel starts life as an averaged-grayscale edge detector rather than
random noise. This preserves the pretrained feature hierarchy for layers 2+.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ResNet18Regressor4ch(nn.Module):
    name = "ResNet18Regressor4ch"

    def __init__(self, out_dim: int = 6, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)

        old_conv = backbone.conv1  # Conv2d(3, 64, k7, s2, p3, bias=False)
        new_conv = nn.Conv2d(4, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        with torch.no_grad():
            if pretrained:
                w = old_conv.weight  # (64, 3, 7, 7)
                new_conv.weight[:, :3] = w
                new_conv.weight[:, 3:4] = w.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        backbone.conv1 = new_conv

        in_dim = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_dim, out_dim),
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)
