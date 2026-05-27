from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cornell import CornellSample, IMG_H, IMG_W, make_rect  # noqa: E402
from dataset import INPUT_SIZE, _central_positive, encode_target  # noqa: E402
from dataset_phase3 import CornellGraspDatasetP3  # noqa: E402


def _make_sample(tmp_path: Path) -> CornellSample:
    rgb_path = tmp_path / "sample.png"
    Image.new("RGB", (IMG_W, IMG_H), color=(128, 128, 128)).save(rgb_path)
    pcd_path = tmp_path / "sample.txt"
    pcd_path.write_text("dummy")
    g1 = make_rect(cx=320.0, cy=240.0, w=80.0, h=20.0, angle_rad=0.0)
    g2 = make_rect(cx=360.0, cy=260.0, w=70.0, h=22.0, angle_rad=0.3)
    return CornellSample(
        pcd_id=1,
        object_id=1,
        rgb_path=rgb_path,
        pcd_path=pcd_path,
        grasps=[g1, g2],
    )


def test_phase3_default_target_matches_phase2(tmp_path):
    sample = _make_sample(tmp_path)
    ds = CornellGraspDatasetP3(
        [sample],
        augment=False,
        augment_rot_deg=0.0,
        use_depth_blue=False,
        multi_positive=False,
        seed=42,
    )
    _, target, _ = ds[0]
    sx, sy = INPUT_SIZE / IMG_W, INPUT_SIZE / IMG_H
    expected = encode_target(_central_positive(sample.positives), sx, sy)
    assert np.allclose(target.numpy(), expected, atol=1e-6)


def test_phase3_hflip_without_rotation_matches_phase2(tmp_path):
    sample = _make_sample(tmp_path)
    ds = CornellGraspDatasetP3(
        [sample],
        augment=True,
        augment_rot_deg=0.0,
        use_depth_blue=False,
        multi_positive=False,
        seed=1,  # random()=0.134... so hflip triggers
    )
    _, target, _ = ds[0]
    sx, sy = INPUT_SIZE / IMG_W, INPUT_SIZE / IMG_H
    flipped = [
        make_rect(cx=IMG_W - g.cx, cy=g.cy, w=g.width, h=g.height, angle_rad=-g.angle_rad)
        for g in sample.positives
    ]
    expected = encode_target(_central_positive(flipped), sx, sy)
    assert np.allclose(target.numpy(), expected, atol=1e-6)
