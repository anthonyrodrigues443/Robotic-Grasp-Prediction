"""Phase-7 unit coverage for the production helpers that don't need the
(gitignored) Cornell data or the champion checkpoint.

The Phase-6 ``test_production.py`` smoke tests cover the full data->predict->
evaluate path but skip on a clean checkout. These tests exercise the pure logic
— config loading, path resolution, the object-wise split, the shared renderer,
and the ``GraspPrediction`` contract — so the suite has real coverage even with
no artifacts present.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cornell import GraspRect, make_rect  # noqa: E402
from data_pipeline import (  # noqa: E402
    load_config,
    object_wise_split,
    preprocess_image,
    resolve_path,
)
from predict import GraspPrediction  # noqa: E402
from viz import render_grasp  # noqa: E402


# ---------------------------------------------------------------- config / paths

def test_config_has_required_contract():
    cfg = load_config()
    for section in ("data", "model", "train", "metric", "reference"):
        assert section in cfg, f"config missing '{section}'"
    assert cfg["metric"]["iou_thresh"] == pytest.approx(0.25)
    assert cfg["metric"]["angle_thresh_deg"] == pytest.approx(30.0)
    assert cfg["model"]["out_dim"] == 6
    # the shipped artifact number the model card / tests pin to
    assert cfg["reference"]["champion_accuracy"] == pytest.approx(0.77, abs=1e-6)


def test_resolve_path_absolute_and_relative():
    rel = resolve_path("models/P4_tuned_champion.pt")
    assert rel.is_absolute()
    assert rel == REPO_ROOT / "models" / "P4_tuned_champion.pt"
    abs_in = Path("/tmp/x.pt")
    assert resolve_path(abs_in) == abs_in


# ---------------------------------------------------------------- object split

def test_object_wise_split_partitions_by_id_with_no_leakage():
    # lightweight stubs — object_wise_split only reads ``.object_id``
    samples = [SimpleNamespace(object_id=i % 4) for i in range(20)]
    train, test = object_wise_split(samples, test_object_ids=[3])
    assert len(train) + len(test) == len(samples)
    assert all(s.object_id == 3 for s in test)
    assert all(s.object_id != 3 for s in train)
    # held-out object never leaks into train
    assert {s.object_id for s in train}.isdisjoint({3})


def test_object_wise_split_empty_testset():
    samples = [SimpleNamespace(object_id=1) for _ in range(5)]
    train, test = object_wise_split(samples, test_object_ids=[999])
    assert len(train) == 5 and len(test) == 0


# ---------------------------------------------------------------- preprocessing

def test_preprocess_image_accepts_pil_and_ndarray():
    arr = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    x_np = preprocess_image(arr)
    x_pil = preprocess_image(Image.fromarray(arr))
    assert tuple(x_np.shape) == (1, 3, 224, 224)
    assert tuple(x_pil.shape) == (1, 3, 224, 224)
    # identical input through both code paths -> identical tensor
    assert np.allclose(x_np.numpy(), x_pil.numpy(), atol=1e-5)


# ---------------------------------------------------------------- renderer

def test_render_grasp_returns_figure():
    img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
    pred = make_rect(cx=320, cy=240, w=80, h=30, angle_rad=0.4)
    gts = [make_rect(cx=300, cy=250, w=90, h=28, angle_rad=0.35)]
    fig = render_grasp(img, pred, gt_rects=gts, title="unit test")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_render_grasp_without_ground_truth():
    img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
    pred = make_rect(cx=100, cy=100, w=60, h=20, angle_rad=-0.7)
    fig = render_grasp(img, pred)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------- prediction dto

def _make_prediction() -> GraspPrediction:
    rect = make_rect(cx=320, cy=240, w=80, h=30, angle_rad=0.4)
    return GraspPrediction(
        cx=rect.cx,
        cy=rect.cy,
        width=rect.width,
        height=rect.height,
        angle_deg=rect.angle_deg,
        confidence=0.93,
        corners=rect.corners.tolist(),
        latency_ms=7.2,
        raw_6vec=[320.0, 240.0, 80.0, 30.0, 0.7, 0.7],
    )


def test_grasp_prediction_to_dict_is_serialisable():
    pred = _make_prediction()
    d = pred.to_dict()
    for k in ("cx", "cy", "width", "height", "angle_deg", "confidence",
              "corners", "latency_ms", "raw_6vec"):
        assert k in d
    assert isinstance(d["corners"], list) and len(d["corners"]) == 4


def test_grasp_prediction_to_grasp_rect_roundtrips_center():
    pred = _make_prediction()
    rect = pred.to_grasp_rect()
    assert isinstance(rect, GraspRect)
    assert rect.cx == pytest.approx(pred.cx, abs=1e-6)
    assert rect.cy == pytest.approx(pred.cy, abs=1e-6)
