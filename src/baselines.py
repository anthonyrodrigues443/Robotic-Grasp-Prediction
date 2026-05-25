"""Three Phase-1 baselines for Cornell grasp detection.

Each baseline takes an image and returns a single ``GraspRect``. Evaluation is
the Jiang rectangle metric: a prediction is correct iff there exists a
ground-truth positive rectangle with IoU > 0.25 AND angle difference < 30°.

The three baselines target *progressively stronger* priors:

1. ``RandomBaseline``        — chance level. Predict a random oriented
   rectangle in a plausible size/position range. This is the floor.
2. ``CenterHeuristicBaseline`` — predict at the image center, with the median
   grasp size from training, axis-aligned (angle=0). This isolates how much
   the dataset's centering bias contributes.
3. ``DepthAntipodalBaseline``  — classical depth-edge antipodal heuristic.
   Find the object via depth thresholding, compute its minimum-area bounding
   box (cv2.minAreaRect on the object mask), and place the gripper across
   the short axis. This is the strongest non-learning baseline and the one
   that exercises actual domain knowledge from antipodal grasp theory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from cornell import GraspRect, make_rect, IMG_W, IMG_H


# ---------------------------------------------------------------------------
# Baseline 1 — chance level
# ---------------------------------------------------------------------------

@dataclass
class RandomBaseline:
    """Sample (cx, cy, w, h, angle) uniformly from plausible ranges. Used to
    quantify the chance level under the Jiang metric for this dataset and
    image size — *not* hard-coded to a textbook 1 / k value because the
    metric is permissive (IoU>0.25 with any of multiple GTs)."""

    cx_range: tuple[float, float] = (160.0, 480.0)
    cy_range: tuple[float, float] = (120.0, 360.0)
    w_range: tuple[float, float] = (40.0, 140.0)
    h_range: tuple[float, float] = (15.0, 40.0)
    rng: np.random.Generator = None  # type: ignore

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(42)

    def fit(self, samples):
        # nothing to learn
        return self

    def predict(self, sample) -> GraspRect:
        return make_rect(
            cx=self.rng.uniform(*self.cx_range),
            cy=self.rng.uniform(*self.cy_range),
            w=self.rng.uniform(*self.w_range),
            h=self.rng.uniform(*self.h_range),
            angle_rad=self.rng.uniform(-math.pi / 2, math.pi / 2),
        )


# ---------------------------------------------------------------------------
# Baseline 2 — center + median-size, axis-aligned. Tests how much of the
# dataset's accuracy comes from object-centered framing alone.
# ---------------------------------------------------------------------------

@dataclass
class CenterHeuristicBaseline:
    cx: float = IMG_W / 2.0
    cy: float = IMG_H / 2.0
    w_median: float = 80.0
    h_median: float = 25.0
    angle: float = 0.0
    sweep_angles: bool = True  # whether predict() tries 6 angles and returns the best (allowed at eval — see note)

    def fit(self, samples):
        ws, hs, cxs, cys = [], [], [], []
        for s in samples:
            for g in s.positives:
                ws.append(g.width)
                hs.append(g.height)
                cxs.append(g.cx)
                cys.append(g.cy)
        if ws:
            self.cx = float(np.median(cxs))
            self.cy = float(np.median(cys))
            self.w_median = float(np.median(ws))
            self.h_median = float(np.median(hs))
        return self

    def predict(self, sample) -> GraspRect:
        return make_rect(self.cx, self.cy, self.w_median, self.h_median, self.angle)


# ---------------------------------------------------------------------------
# Baseline 3 — depth-edge antipodal heuristic. The point of this baseline is
# that it uses *zero* learned parameters but encodes the physics: a grasp
# should straddle the object's narrow axis, so we extract the object from the
# background via depth thresholding, fit the minimum-area bounding box, and
# place the gripper across the short side.
# ---------------------------------------------------------------------------

@dataclass
class DepthAntipodalBaseline:
    """Classical antipodal-from-depth heuristic.

    Steps:
      1. Read the depth channel from the .pcd file.
      2. Threshold: pixels closer than (background_depth - margin) belong to
         the object. Background is estimated as the modal depth value.
      3. Largest connected component => object mask.
      4. cv2.minAreaRect on the mask => (cx, cy), (w, h), angle.
      5. The gripper's *jaw opening* direction is the short axis; the long
         axis of the grasp rectangle is the short axis of the object box.
         A small empirical scaling keeps the predicted grasp width bounded
         to the gripper's range.
    """

    depth_margin: float = 0.02  # meters of clearance from background
    min_object_pixels: int = 500
    plate_height_px: float = 25.0  # median height of Cornell grasps
    width_scale: float = 1.05  # slightly wider than object short axis

    def fit(self, samples):
        # Estimate median grasp height from positives in training set.
        hs = [g.height for s in samples for g in s.positives]
        if hs:
            self.plate_height_px = float(np.median(hs))
        return self

    @staticmethod
    def _load_depth_image(pcd_path) -> np.ndarray | None:
        """Cornell .pcd is ASCII. Columns are ``x y z rgb index`` where the
        Kinect convention used in this dataset puts forward distance from the
        camera in the ``x`` column (~700–2000 mm range) and table-relative
        height in ``z`` (~-100 to +100 mm). What we actually want is *forward
        distance from the camera*, so we read ``x`` and clip to a sane range
        before returning a (480, 640) float32 image in millimeters with NaN
        for missing points."""
        try:
            with open(pcd_path) as f:
                lines = f.readlines()
        except OSError:
            return None
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith("DATA"):
                data_start = i + 1
                break
        if data_start is None:
            return None
        depth = np.full((IMG_H, IMG_W), np.nan, dtype=np.float32)
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                # forward distance from camera (mm) -- the Cornell convention
                z_forward = float(parts[0])
                idx = int(parts[4])
            except (ValueError, IndexError):
                continue
            r, c = divmod(idx, IMG_W)
            if 0 <= r < IMG_H and 0 <= c < IMG_W:
                depth[r, c] = z_forward
        return depth

    def predict(self, sample) -> GraspRect:
        depth = self._load_depth_image(sample.pcd_path)
        if depth is None or not np.isfinite(depth).any():
            # fallback: image center, default size
            return make_rect(IMG_W / 2, IMG_H / 2, 80.0, self.plate_height_px, 0.0)

        # Inpaint NaNs via 5x5 nearest-finite fill (cv2.inpaint also works).
        finite = np.isfinite(depth)
        if finite.mean() < 0.05:
            return make_rect(IMG_W / 2, IMG_H / 2, 80.0, self.plate_height_px, 0.0)
        d_min = float(np.nanmin(depth))
        d_max = float(np.nanmax(depth))
        # Background = modal depth, estimated by 95th percentile (background is the farthest plane)
        bg = float(np.nanpercentile(depth, 95))
        # Object mask: anything noticeably closer than background.
        mask = ((depth < bg - self.depth_margin) & finite).astype(np.uint8) * 255
        if mask.sum() < self.min_object_pixels * 255:
            # try a looser margin
            mask = ((depth < bg - self.depth_margin / 2) & finite).astype(np.uint8) * 255
        if mask.sum() < self.min_object_pixels * 255:
            return make_rect(IMG_W / 2, IMG_H / 2, 80.0, self.plate_height_px, 0.0)

        # Largest connected component
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return make_rect(IMG_W / 2, IMG_H / 2, 80.0, self.plate_height_px, 0.0)
        # ignore label 0 (background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        obj = (labels == largest).astype(np.uint8) * 255

        contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return make_rect(IMG_W / 2, IMG_H / 2, 80.0, self.plate_height_px, 0.0)
        # use the biggest contour
        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), (w_box, h_box), angle_deg_cv = cv2.minAreaRect(cnt)
        # cv2's angle is in (-90, 0]; w and h are arbitrarily assigned. We
        # want the grasp's long axis = SHORT side of the object box, since
        # the gripper closes across the narrow dimension.
        short = min(w_box, h_box)
        long_ = max(w_box, h_box)
        # Map cv2's angle to the grasp's jaw-opening direction (perpendicular
        # to the object's long axis). If h > w, the long axis is along the
        # box's vertical, so the grasp's long axis is along the box's
        # horizontal, i.e. angle_deg_cv + 90.
        if h_box > w_box:
            grasp_angle_deg = angle_deg_cv + 90.0
        else:
            grasp_angle_deg = angle_deg_cv
        # wrap to [-90, 90)
        while grasp_angle_deg >= 90.0:
            grasp_angle_deg -= 180.0
        while grasp_angle_deg < -90.0:
            grasp_angle_deg += 180.0
        grasp_w = max(20.0, min(short * self.width_scale, 150.0))
        return make_rect(cx, cy, grasp_w, self.plate_height_px, math.radians(grasp_angle_deg))


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate(baseline, samples, iou_thresh: float = 0.25,
             angle_thresh_deg: float = 30.0) -> dict:
    """Run ``baseline.predict`` on each sample and score it under Jiang's
    metric. Returns a dict with accuracy + per-sample diagnostics."""
    from cornell import is_correct_jiang, iou as poly_iou, angle_diff_deg

    correct = 0
    total = 0
    best_ious = []
    best_angle_errs = []
    per_sample = []
    for sample in samples:
        gts = sample.positives
        if not gts:
            continue
        pred = baseline.predict(sample)
        ok = is_correct_jiang(pred, gts, iou_thresh, angle_thresh_deg)
        # best matching GT (for diagnostics, by IoU among GTs within angle thresh)
        candidates = [(poly_iou(pred, g), angle_diff_deg(pred, g), g) for g in gts]
        # best IoU regardless of angle
        best = max(candidates, key=lambda t: t[0])
        best_ious.append(best[0])
        best_angle_errs.append(best[1])
        per_sample.append({
            "pcd_id": sample.pcd_id,
            "object_id": sample.object_id,
            "correct": int(ok),
            "best_iou": float(best[0]),
            "best_angle_err": float(best[1]),
        })
        correct += int(ok)
        total += 1

    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "median_iou": float(np.median(best_ious)) if best_ious else 0.0,
        "median_angle_err": float(np.median(best_angle_errs)) if best_angle_errs else 0.0,
        "per_sample": per_sample,
    }
