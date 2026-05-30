"""Production data pipeline for Robotic Grasp Prediction.

A thin, importable layer over the research modules (``cornell``,
``dataset``) that the Phase-6 production scripts (train / evaluate / predict /
app) all share. Keeping this in one place means the inference preprocessing is
*guaranteed* identical to the training preprocessing — the single most common
source of train/serve skew.

Public surface:
    load_config(path)                     -> dict
    load_cornell(root)                    -> list[CornellSample]
    object_wise_split(samples, test_ids)  -> (train, test)
    preprocess_image(image)               -> torch.Tensor (1, 3, 224, 224)
    INPUT_SIZE, IMG_W, IMG_H
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
import yaml
from PIL import Image

from cornell import IMG_H, IMG_W, CornellSample, load_samples
from dataset import INPUT_SIZE, _imagenet_normalize

# Repo root = parent of this file's directory (src/).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

ImageLike = Union[str, Path, Image.Image, np.ndarray]


def load_config(path: Union[str, Path, None] = None) -> dict:
    """Load the YAML config. Relative paths in the config are resolved against
    the repo root so scripts work regardless of the caller's CWD."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(p: Union[str, Path]) -> Path:
    """Resolve a (possibly repo-relative) config path to an absolute path."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_cornell(root: Union[str, Path]) -> list[CornellSample]:
    """Load every Cornell sample under ``root`` (the dir holding folders 01..10)."""
    return load_samples(resolve_path(root))


def object_wise_split(
    samples: list[CornellSample], test_object_ids: Sequence[int]
) -> tuple[list[CornellSample], list[CornellSample]]:
    """Split by Cornell object/folder id — the held-out objects never appear in
    train, so the test signal measures generalisation to unseen objects."""
    test_set = set(test_object_ids)
    train, test = [], []
    for s in samples:
        (test if s.object_id in test_set else train).append(s)
    return train, test


def _to_pil_rgb(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            # assume float in [0, 1]
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def preprocess_image(image: ImageLike) -> torch.Tensor:
    """Path / PIL / ndarray -> (1, 3, 224, 224) ImageNet-normalised float tensor,
    exactly matching ``dataset.CornellGraspDataset``. Also returns nothing about
    the original size because the decoder projects back to the canonical
    640x480 Cornell frame (the model was trained in that frame)."""
    pil = _to_pil_rgb(image).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    chw = _imagenet_normalize(arr)
    return torch.from_numpy(chw).unsqueeze(0)


__all__ = [
    "INPUT_SIZE",
    "IMG_W",
    "IMG_H",
    "load_config",
    "resolve_path",
    "load_cornell",
    "object_wise_split",
    "preprocess_image",
]
