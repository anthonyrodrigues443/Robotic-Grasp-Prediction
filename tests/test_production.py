"""Phase-6 production-path smoke tests.

These guard the serving pipeline (config -> data_pipeline -> predict -> evaluate)
against regressions, independent of the research notebooks. They are skipped
automatically if the champion checkpoint or the Cornell data aren't present, so
the suite stays green on a clean checkout without the (gitignored) artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_pipeline import load_config, load_cornell, object_wise_split, preprocess_image  # noqa: E402

cfg = load_config()
CKPT = REPO_ROOT / cfg["model"]["champion_checkpoint"]
DATA_ROOT = REPO_ROOT / cfg["data"]["cornell_root"]

needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="champion checkpoint not present")
needs_data = pytest.mark.skipif(not DATA_ROOT.exists(), reason="Cornell data not present")


def test_preprocess_shape_and_norm():
    arr = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    x = preprocess_image(arr)
    assert tuple(x.shape) == (1, 3, 224, 224)
    assert x.dtype.is_floating_point
    # ImageNet-normalised values comfortably exceed [0, 1]
    assert x.min() < 0.0 < x.max()


@needs_ckpt
@needs_data
def test_predictor_returns_valid_grasp():
    from predict import GraspPredictor

    predictor = GraspPredictor(config=cfg)
    samples = load_cornell(cfg["data"]["cornell_root"])
    _, test = object_wise_split(samples, cfg["data"]["test_object_ids"])
    pred = predictor.predict(test[0].rgb_path)

    assert np.isfinite(pred.cx) and np.isfinite(pred.cy)
    assert pred.width > 0 and pred.height > 0
    assert -90.0 <= pred.angle_deg <= 90.0
    assert 0.0 <= pred.confidence <= 1.0
    corners = np.asarray(pred.corners)
    assert corners.shape == (4, 2)
    # decoded into the 640x480 frame, center should land inside the image
    assert 0 <= pred.cx <= 640 and 0 <= pred.cy <= 480


@needs_ckpt
@needs_data
def test_evaluate_subset_reproduces_reference():
    from evaluate import evaluate_split
    from predict import GraspPredictor

    predictor = GraspPredictor(config=cfg)
    samples = load_cornell(cfg["data"]["cornell_root"])
    _, test = object_wise_split(samples, cfg["data"]["test_object_ids"])
    res = evaluate_split(
        predictor, test, cfg["metric"]["iou_thresh"], cfg["metric"]["angle_thresh_deg"]
    )
    # full folder-03 reproduces the documented 0.770 reference exactly
    assert res["n"] == 100
    assert res["accuracy"] == pytest.approx(cfg["reference"]["champion_accuracy"], abs=0.02)
