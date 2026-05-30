"""Shared grasp-rectangle rendering — used by both the Streamlit app and the
results-screenshot generator so the demo image and the live UI look identical.

A grasp is drawn the way the robotics literature draws it: the two *jaw plates*
(the sides the gripper fingers press against) in one colour, and the *jaw-opening
axis* (the line the gripper closes along) as a centred arrow. This makes the
predicted orientation legible at a glance instead of just a tilted box.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from cornell import IMG_H, IMG_W, GraspRect

PRED_COLOR = "#00E5A0"   # mint — prediction
GT_COLOR = "#FFB000"     # amber — ground truth
JAW_COLOR = "#FF3B6B"    # the gripper plates (short edges)


def _draw_rect(ax, rect: GraspRect, color: str, label: Optional[str] = None,
               lw: float = 2.5, jaw_color: Optional[str] = None) -> None:
    c = rect.corners
    # close the polygon
    poly = np.vstack([c, c[0]])
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, label=label, zorder=3)
    # highlight the two gripper-plate edges (corners 1->2 and 3->0 are the short
    # plate axis in Cornell's corner convention)
    if jaw_color:
        ax.plot([c[1, 0], c[2, 0]], [c[1, 1], c[2, 1]], color=jaw_color, lw=lw + 2, zorder=4)
        ax.plot([c[3, 0], c[0, 0]], [c[3, 1], c[0, 1]], color=jaw_color, lw=lw + 2, zorder=4)


def render_grasp(
    image,
    prediction_rect: GraspRect,
    gt_rects: Optional[Sequence[GraspRect]] = None,
    title: Optional[str] = None,
    figsize=(7, 5.25),
) -> Figure:
    """Return a matplotlib Figure of the image (resized to the canonical
    640×480 Cornell frame) with the predicted grasp overlaid, plus optional
    ground-truth grasps for context."""
    pil = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
    pil = pil.resize((IMG_W, IMG_H), Image.BILINEAR)

    fig = Figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111)
    ax.imshow(np.asarray(pil))
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.axis("off")

    if gt_rects:
        for i, g in enumerate(gt_rects):
            _draw_rect(ax, g, GT_COLOR, label="Ground truth" if i == 0 else None, lw=1.4)

    _draw_rect(ax, prediction_rect, PRED_COLOR, label="Predicted grasp",
               lw=2.6, jaw_color=JAW_COLOR)
    # center marker
    ax.scatter([prediction_rect.cx], [prediction_rect.cy], s=60,
               color=PRED_COLOR, edgecolors="black", zorder=5)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return fig
