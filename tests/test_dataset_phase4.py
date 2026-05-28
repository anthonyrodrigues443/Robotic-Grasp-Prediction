from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cornell import CornellSample, IMG_H, IMG_W, make_rect  # noqa: E402
from dataset_phase4 import CornellGraspDatasetP4  # noqa: E402


def _make_sample(tmp_path: Path) -> CornellSample:
    rgb_path = tmp_path / "sample.png"
    Image.new("RGB", (IMG_W, IMG_H), color=(128, 128, 128)).save(rgb_path)
    pcd_path = tmp_path / "sample.txt"
    pcd_path.write_text("dummy")
    g = make_rect(cx=320.0, cy=240.0, w=80.0, h=20.0, angle_rad=0.0)
    return CornellSample(
        pcd_id=1,
        object_id=1,
        rgb_path=rgb_path,
        pcd_path=pcd_path,
        grasps=[g],
    )


class _FixedRNG:
    def __init__(self, rot_deg: float, flip_rand: float):
        self._rot_deg = rot_deg
        self._flip_rand = flip_rand

    def uniform(self, _a: float, _b: float) -> float:
        return self._rot_deg

    def random(self) -> float:
        return self._flip_rand


def test_phase4_depth_channel_follows_rgb_geometry_aug(tmp_path):
    sample = _make_sample(tmp_path)
    ds = CornellGraspDatasetP4(
        [sample],
        augment=True,
        augment_rot_deg=25.0,
        use_depth_blue=False,
        multi_positive=False,
        use_depth_4ch=True,
        seed=42,
    )

    base = np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
    rgb = np.stack([base, np.zeros_like(base), np.zeros_like(base)], axis=2)
    ds._load_rgb = lambda _path: rgb.copy()  # type: ignore[method-assign]
    ds._load_depth = lambda _path: base.copy()  # type: ignore[method-assign]
    ds._rng = _FixedRNG(rot_deg=17.0, flip_rand=0.0)

    x, _, _ = ds[0]
    rgb_r = x[0].numpy() * 0.229 + 0.485
    depth = x[3].numpy() * 0.225 + 0.5
    assert np.allclose(rgb_r, depth, atol=1e-6)
