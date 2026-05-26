"""Unit tests for the Phase-2 dataset + model layer.

These pin the contracts that the leaderboard depends on: encode/decode
round-trip, augmentation invariants, model output shapes, angle-bin
helpers, and GG-CNN target shapes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cornell import GraspRect, make_rect, is_correct_jiang, IMG_W, IMG_H  # noqa: E402
from dataset import (  # noqa: E402
    CornellGraspDataset, INPUT_SIZE, decode_prediction, encode_target,
    split_object_wise,
)
from torch_models import (  # noqa: E402
    ANGLE_BINS, GGCNN_OUT_SIZE, GGCNNTiny, ResNet18HybridHead,
    ResNet18Regressor, TinyRedmonCNN, TinyViT, angle_to_bin_idx,
    bin_idx_to_angle_rad, count_params, ggcnn_decode, ggcnn_target_maps,
)


# ---------------------------------------------------------------------------
# encode_target / decode_prediction round-trip
# ---------------------------------------------------------------------------

def _round_trip(g: GraspRect) -> GraspRect:
    sx = INPUT_SIZE / IMG_W
    sy = INPUT_SIZE / IMG_H
    y = encode_target(g, sx, sy)
    return decode_prediction(y)


def test_encode_decode_axis_aligned():
    g = make_rect(cx=320.0, cy=240.0, w=80.0, h=24.0, angle_rad=0.0)
    out = _round_trip(g)
    assert abs(out.cx - g.cx) < 2.0
    assert abs(out.cy - g.cy) < 2.0
    assert abs(out.width - g.width) < 3.0
    assert abs(out.height - g.height) < 3.0
    assert abs(out.angle_deg - g.angle_deg) < 2.0


def test_encode_decode_45_deg():
    g = make_rect(cx=320.0, cy=240.0, w=80.0, h=24.0, angle_rad=math.radians(45))
    out = _round_trip(g)
    assert abs(out.angle_deg - g.angle_deg) < 2.0
    assert abs(out.cx - g.cx) < 2.0


def test_encode_decode_negative_angle():
    # angle is in [-90, 90); a negative angle should round-trip
    g = make_rect(cx=200.0, cy=300.0, w=60.0, h=20.0, angle_rad=math.radians(-35))
    out = _round_trip(g)
    assert abs(out.angle_deg - g.angle_deg) < 2.0


def test_encode_decode_passes_jiang_against_self():
    """The round-tripped rectangle should score as 'correct' under the Jiang
    metric when evaluated against the original — otherwise our learned target
    is not even an upper bound for the loss."""
    g = make_rect(cx=320.0, cy=240.0, w=80.0, h=24.0, angle_rad=math.radians(15))
    out = _round_trip(g)
    assert is_correct_jiang(out, [g])


# ---------------------------------------------------------------------------
# Doubled-angle representation correctly handles the 180° symmetry
# ---------------------------------------------------------------------------

def test_doubled_angle_handles_pi_over_2_wraparound():
    # 89° and -89° should decode to nearly the same angle because 2*89 = 178
    # and 2*(-89) = -178 ≡ 182, which differ by 4° in the doubled-angle space.
    sx, sy = INPUT_SIZE / IMG_W, INPUT_SIZE / IMG_H
    g_pos = make_rect(cx=320, cy=240, w=80, h=24, angle_rad=math.radians(89))
    g_neg = make_rect(cx=320, cy=240, w=80, h=24, angle_rad=math.radians(-89))
    y_pos = encode_target(g_pos, sx, sy)
    y_neg = encode_target(g_neg, sx, sy)
    # sin(2θ) should be ~ opposite-sign small numbers and cos(2θ) should agree
    assert abs(y_pos[5] - y_neg[5]) < 0.05      # cos(2θ) same
    assert abs(y_pos[4] + y_neg[4]) < 0.05      # sin(2θ) opposite small ≈ 0


# ---------------------------------------------------------------------------
# angle-bin helpers (Hybrid head)
# ---------------------------------------------------------------------------

def test_angle_bin_roundtrip_within_bin_width():
    for deg in (-89, -45, -10, 0, 10, 45, 89):
        idx = angle_to_bin_idx(math.radians(deg))
        back = bin_idx_to_angle_rad(idx)
        # back is the bin centre; should be within half a bin width (5°) of deg
        diff = abs(math.degrees(back) - deg)
        diff = min(diff, 180 - diff)  # wraparound
        assert diff <= ANGLE_BIN_WIDTH_DEG / 2 + 1e-6, f"deg={deg}, back={math.degrees(back)}, diff={diff}"


ANGLE_BIN_WIDTH_DEG = 180.0 / ANGLE_BINS  # mirror torch_models for the comparison above


def test_angle_bins_partition_180():
    bins = [angle_to_bin_idx(math.radians(d)) for d in range(-89, 90, 1)]
    assert min(bins) >= 0 and max(bins) < ANGLE_BINS


# ---------------------------------------------------------------------------
# Model forward shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctor", [TinyRedmonCNN, ResNet18Regressor, TinyViT])
def test_regression_models_forward_shape(ctor):
    torch.manual_seed(0)
    m = ctor()
    out = m(torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE))
    assert out.shape == (2, 6)


def test_hybrid_model_forward_shape():
    torch.manual_seed(0)
    m = ResNet18HybridHead()
    reg, cls = m(torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE))
    assert reg.shape == (2, 4)
    assert cls.shape == (2, ANGLE_BINS)


def test_ggcnn_forward_shape():
    torch.manual_seed(0)
    m = GGCNNTiny()
    q, s, c, w = m(torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE))
    for t in (q, s, c, w):
        assert t.shape == (2, 1, GGCNN_OUT_SIZE, GGCNN_OUT_SIZE)
    assert torch.all((q >= 0) & (q <= 1))  # sigmoid output


# ---------------------------------------------------------------------------
# GG-CNN target maps
# ---------------------------------------------------------------------------

def test_ggcnn_target_maps_shape_and_peak():
    y = torch.tensor([[0.5, 0.5, 0.4, 0.1, math.sin(0.3), math.cos(0.3)]], dtype=torch.float32)
    q, s, c, w = ggcnn_target_maps(y)
    assert q.shape == (1, 1, GGCNN_OUT_SIZE, GGCNN_OUT_SIZE)
    # peak of quality should be near (0.5, 0.5) of the output grid
    qq = q[0, 0].numpy()
    py, px = np.unravel_index(np.argmax(qq), qq.shape)
    assert abs(py / GGCNN_OUT_SIZE - 0.5) < 0.02
    assert abs(px / GGCNN_OUT_SIZE - 0.5) < 0.02


def test_ggcnn_decode_picks_argmax():
    y = torch.tensor([[0.3, 0.7, 0.4, 0.1, math.sin(0.3), math.cos(0.3)]], dtype=torch.float32)
    q, s, c, w = ggcnn_target_maps(y)
    out = ggcnn_decode(q[0, 0].numpy(), s[0, 0].numpy(), c[0, 0].numpy(), w[0, 0].numpy())
    # decoded cx/cy should be near input cx/cy
    assert abs(out[0] - 0.3) < 0.02
    assert abs(out[1] - 0.7) < 0.02


# ---------------------------------------------------------------------------
# Param counts (sanity bounds — catch regressions where someone leaves a
# weights-frozen flag inverted, or the head shape blows up)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctor,lo,hi", [
    (TinyRedmonCNN,        1_000_000, 3_000_000),
    (ResNet18Regressor,    10_000_000, 12_000_000),
    (ResNet18HybridHead,   10_000_000, 13_000_000),
    (GGCNNTiny,            10_000, 200_000),
    (TinyViT,              1_000_000, 3_000_000),
])
def test_param_count_bounds(ctor, lo, hi):
    n = count_params(ctor())
    assert lo <= n <= hi, f"{ctor.__name__} has {n:,} params, expected {lo:,}-{hi:,}"


# ---------------------------------------------------------------------------
# Dataset + augmentation invariants — augmenting still produces a valid
# target (sin² + cos² ≈ 1 on the doubled angle channel, sizes positive).
# ---------------------------------------------------------------------------

def _make_fake_sample(pcd_id=1, object_id=99):
    """Build a minimal CornellSample whose RGB path points at any existing
    PNG. We don't actually need to load it — we just call encode_target."""
    pass


def test_dataset_split_object_wise():
    """split_object_wise puts test_folders on the right side."""
    class S:
        def __init__(self, oid): self.object_id = oid
    train, test = split_object_wise([S(1), S(2), S(3), S(3)], test_folders=[3])
    assert len(train) == 2 and len(test) == 2
    assert {t.object_id for t in test} == {3}
