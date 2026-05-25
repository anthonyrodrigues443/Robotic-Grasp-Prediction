"""Unit tests for the Cornell loader + geometry primitives.

These are *behavioural* tests — they pin the conventions a misread of the
Cornell README would silently get wrong: rectangle corner ordering, angle
direction, the IoU+angle metric thresholds, and the make_rect → GraspRect
round trip.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cornell import (  # noqa: E402
    GraspRect, make_rect, iou, angle_diff_deg, is_correct_jiang,
)


def test_identical_rects_iou_one():
    a = make_rect(320, 240, 80, 30, 0.0)
    b = make_rect(320, 240, 80, 30, 0.0)
    assert iou(a, b) == pytest.approx(1.0, abs=1e-6)
    assert angle_diff_deg(a, b) == pytest.approx(0.0, abs=1e-9)


def test_disjoint_rects_iou_zero():
    a = make_rect(320, 240, 80, 30, 0.0)
    far = make_rect(50, 50, 80, 30, 0.0)
    assert iou(a, far) == 0.0


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 89.9])
def test_make_rect_round_trip(angle_deg):
    """Constructing a rect at a given angle and reading it back should
    recover the same angle (mod the 180° grasp symmetry)."""
    r = make_rect(100, 200, 60, 25, math.radians(angle_deg))
    assert r.width == pytest.approx(60.0, abs=1e-6)
    assert r.height == pytest.approx(25.0, abs=1e-6)
    assert abs(r.angle_deg - angle_deg) % 180.0 < 1e-3


def test_jiang_metric_thresholds():
    a = make_rect(320, 240, 80, 30, 0.0)
    # 29° rotation, identical size + position: IoU > 0.25, angle err = 29
    c29 = make_rect(320, 240, 80, 30, math.radians(29))
    assert is_correct_jiang(c29, [a])           # both ok
    # 31° rotation: angle constraint fails
    c31 = make_rect(320, 240, 80, 30, math.radians(31))
    assert not is_correct_jiang(c31, [a])
    # tiny offset rectangle (IoU should still be > 0.25)
    near = make_rect(330, 240, 80, 30, 0.0)
    assert is_correct_jiang(near, [a])
    # huge offset rectangle (IoU = 0)
    far = make_rect(100, 100, 80, 30, 0.0)
    assert not is_correct_jiang(far, [a])


def test_jiang_passes_if_any_gt_matches():
    """The Jiang metric is 'best of N' over the ground-truth set — only one GT
    needs to match."""
    a = make_rect(320, 240, 80, 30, 0.0)
    pred = make_rect(100, 100, 80, 30, math.radians(45))
    # GT set: one bad match (a) and one perfect match (pred itself)
    gts = [a, make_rect(100, 100, 80, 30, math.radians(45))]
    assert is_correct_jiang(pred, gts)


def test_real_cornell_rect_orientation():
    """The first positive grasp in pcd0100 is corners
    (253,319.7) (309,324) (307,350) (251,345.7). The jaw opening (p0→p1)
    should be ~56 px and the plate depth (p1→p2) ~26 px. If width<height
    here, the corner-ordering convention has been silently swapped."""
    rect = GraspRect(
        corners=np.array([[253, 319.7], [309, 324], [307, 350], [251, 345.7]]),
        label=1,
    )
    assert rect.width > rect.height, (
        f"jaw opening (width={rect.width:.1f}) should exceed plate depth "
        f"(height={rect.height:.1f}) for this Cornell example"
    )
    assert 50 < rect.width < 60
    assert 22 < rect.height < 30
