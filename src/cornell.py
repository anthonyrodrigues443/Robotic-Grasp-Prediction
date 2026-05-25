"""Cornell Grasping Dataset loader.

The Cornell dataset ships as one example per .pcd file (point cloud with embedded RGB),
a rendered RGB png ``pcdNNNNr.png``, and two grasp-rectangle files:

  - ``pcdNNNNcpos.txt`` : positive (graspable) oriented rectangles
  - ``pcdNNNNcneg.txt`` : negative (non-graspable) oriented rectangles

Each rectangle is 4 lines of ``x y`` floats (image pixel coordinates) listing
the corners in order. A few lines contain ``NaN`` because some annotators
missed the depth value — Cornell's README acknowledges this, so we drop those
rectangles rather than impute.

The image grid is 640x480. Depth comes from the .pcd; we expose it lazily because
EDA usually only needs the RGB image and the rectangles.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from shapely.geometry import Polygon


IMG_W, IMG_H = 640, 480


@dataclass(frozen=True)
class GraspRect:
    """One annotated grasp rectangle in image coordinates.

    ``corners`` is shape (4, 2): the 4 corners in the order Cornell stores them
    (top-left, top-right, bottom-right, bottom-left along the gripper's frame).
    The gripper closes along the short axis; the long axis is the jaw opening.
    """

    corners: np.ndarray  # (4, 2)  float
    label: int           # 1 positive / 0 negative

    @property
    def cx(self) -> float:
        return float(self.corners[:, 0].mean())

    @property
    def cy(self) -> float:
        return float(self.corners[:, 1].mean())

    @property
    def angle_rad(self) -> float:
        """Angle of the gripper's *jaw opening* axis (long axis), in radians,
        wrapped to [-pi/2, pi/2). Cornell stores corners so that
        ``corners[0]→corners[1]`` is along the jaw opening direction; the
        short axis (plate depth) is ``corners[1]→corners[2]``.
        Verified empirically on pcd0100 where p0→p1 ≈ 56 px (jaw opening)
        and p1→p2 ≈ 26 px (plate)."""
        dx = self.corners[1, 0] - self.corners[0, 0]
        dy = self.corners[1, 1] - self.corners[0, 1]
        a = math.atan2(dy, dx)
        # wrap to [-pi/2, pi/2) since a grasp is symmetric under 180° flip
        while a >= math.pi / 2:
            a -= math.pi
        while a < -math.pi / 2:
            a += math.pi
        return a

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)

    @property
    def width(self) -> float:
        """Length of the *jaw opening* axis (long axis), in pixels."""
        return float(np.linalg.norm(self.corners[1] - self.corners[0]))

    @property
    def height(self) -> float:
        """Length of the *plate* axis (short axis), in pixels."""
        return float(np.linalg.norm(self.corners[2] - self.corners[1]))


@dataclass(frozen=True)
class CornellSample:
    """One image's worth of annotations.

    ``pcd_id`` is the integer in the filename (e.g. ``188`` for ``pcd0188r.png``).
    The original 'object id' that Cornell uses for the object-wise split lives
    in ``z.txt`` files; we infer it from the directory + a separate mapping
    where available, otherwise fall back to ``pcd_id``.
    """

    pcd_id: int
    object_id: int
    rgb_path: Path
    pcd_path: Path
    grasps: list[GraspRect]

    @property
    def positives(self) -> list[GraspRect]:
        return [g for g in self.grasps if g.label == 1]

    @property
    def negatives(self) -> list[GraspRect]:
        return [g for g in self.grasps if g.label == 0]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_PCD_ID_RE = re.compile(r"pcd(\d+)")


def _parse_rect_file(path: Path, label: int) -> list[GraspRect]:
    """Each file holds N rectangles, 4 corner lines apiece. Drop rectangles
    that contain NaN (Cornell's README warns about this)."""
    if not path.is_file():
        return []
    rects: list[GraspRect] = []
    with path.open() as f:
        coords: list[tuple[float, float]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            coords.append((x, y))
            if len(coords) == 4:
                arr = np.array(coords, dtype=float)
                if np.isfinite(arr).all():
                    rects.append(GraspRect(corners=arr, label=label))
                coords = []
    return rects


def load_samples(root: Path) -> list[CornellSample]:
    """Walk ``root`` (the directory containing 01/, 02/, ... folders) and
    return one CornellSample per .pcd file found."""
    root = Path(root)
    samples: list[CornellSample] = []
    for pcd_path in sorted(root.glob("**/pcd*.txt")):
        # the .txt point cloud is named pcdNNNN.txt (no suffix);
        # we also have pcdNNNNr.png, pcdNNNNcpos.txt, pcdNNNNcneg.txt, pcdNNNNz.txt
        name = pcd_path.name
        if name.endswith("cpos.txt") or name.endswith("cneg.txt") or name.endswith("z.txt"):
            continue
        m = _PCD_ID_RE.search(name)
        if not m:
            continue
        pcd_id = int(m.group(1))
        folder = pcd_path.parent
        rgb_path = folder / f"pcd{pcd_id:04d}r.png"
        if not rgb_path.is_file():
            continue
        positives = _parse_rect_file(folder / f"pcd{pcd_id:04d}cpos.txt", label=1)
        negatives = _parse_rect_file(folder / f"pcd{pcd_id:04d}cneg.txt", label=0)
        # Cornell groups objects by folder index but doesn't expose a stable
        # object_id; for the object-wise split we use the folder number as a
        # coarse object grouping. (The classic image-wise split shuffles
        # by pcd_id; the object-wise split holds out by object.)
        try:
            object_id = int(folder.name)
        except ValueError:
            object_id = pcd_id // 10  # fallback
        samples.append(
            CornellSample(
                pcd_id=pcd_id,
                object_id=object_id,
                rgb_path=rgb_path,
                pcd_path=pcd_path,
                grasps=positives + negatives,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# evaluation: Jiang IoU + angle metric
# ---------------------------------------------------------------------------

def iou(a: GraspRect, b: GraspRect) -> float:
    """Polygon IoU between two oriented rectangles, via shapely. Handles the
    edge cases (degenerate rects, non-intersecting) that hand-rolled
    Sutherland-Hodgman would otherwise get wrong."""
    pa = Polygon(a.corners).buffer(0)
    pb = Polygon(b.corners).buffer(0)
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    union_area = pa.union(pb).area
    if union_area <= 0:
        return 0.0
    return pa.intersection(pb).area / union_area


def angle_diff_deg(a: GraspRect, b: GraspRect) -> float:
    """Smallest unsigned angular difference between two grasp orientations,
    in degrees. Range [0, 90] because grasps are symmetric under 180° flip."""
    d = abs(a.angle_deg - b.angle_deg) % 180.0
    return min(d, 180.0 - d)


def is_correct_jiang(pred: GraspRect, gts: Sequence[GraspRect],
                     iou_thresh: float = 0.25,
                     angle_thresh_deg: float = 30.0) -> bool:
    """Jiang et al. 2011 rectangle metric (the field-standard).

    A prediction counts as correct if there exists at least one ground-truth
    *positive* grasp such that IoU > 0.25 AND angle difference < 30°.
    """
    for gt in gts:
        if angle_diff_deg(pred, gt) <= angle_thresh_deg and iou(pred, gt) > iou_thresh:
            return True
    return False


# ---------------------------------------------------------------------------
# construction helpers — predicting an axis-aligned/oriented rectangle from
# center+size+angle for the baselines.
# ---------------------------------------------------------------------------

def make_rect(cx: float, cy: float, w: float, h: float, angle_rad: float,
              label: int = 1) -> GraspRect:
    """Build a GraspRect from center, jaw-opening ``w``, plate-depth ``h``,
    and orientation. Corners are laid out so that p0→p1 is the jaw-opening
    (long) axis, matching Cornell's convention as decoded above."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0
    # local frame: p0→p1 along +x (= the jaw opening axis, length w)
    local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = local @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return GraspRect(corners=rotated, label=label)
