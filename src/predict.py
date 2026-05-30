"""Production inference for Robotic Grasp Prediction.

Loads the Phase-4 tuned champion (global ResNet-18 regressor, 0.780 object-wise
on Cornell folder-03) and predicts a single oriented grasp rectangle for an
input RGB image.

The model regresses a 6-vector ``(cx, cy, w, h, sin2θ, cos2θ)`` in a resized
224×224 frame; ``dataset.decode_prediction`` projects it back to the canonical
640×480 Cornell frame and resolves the doubled-angle representation into a real
gripper orientation.

A regression head has no native class probability, so ``confidence`` is an
explicit heuristic: the L2 norm of the predicted ``(sin2θ, cos2θ)`` vector. A
well-trained model drives that pair onto the unit circle; a norm well below 1
means the orientation output is being pulled toward the origin (the network is
"unsure" which way to align the jaws). It is a *useful relative* signal, not a
calibrated probability — documented as such in the model card.

CLI:
    python src/predict.py --image path/to/img.png
    python src/predict.py --image img.png --json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cornell import GraspRect
from data_pipeline import ImageLike, load_config, preprocess_image, resolve_path
from dataset import decode_prediction
from torch_models import ResNet18Regressor


def _pick_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class GraspPrediction:
    """One predicted grasp in the canonical 640×480 Cornell frame."""

    cx: float
    cy: float
    width: float          # jaw-opening length (px)
    height: float         # plate depth (px)
    angle_deg: float      # gripper orientation, [-90, 90)
    confidence: float     # heuristic: ||(sin2θ, cos2θ)||, clamped to [0, 1]
    corners: list         # 4×2 rectangle corners (px)
    latency_ms: float
    raw_6vec: list

    def to_grasp_rect(self) -> GraspRect:
        return GraspRect(corners=np.asarray(self.corners, dtype=float), label=1)

    def to_dict(self) -> dict:
        return asdict(self)


class GraspPredictor:
    """Stateless-after-construction grasp predictor. Construct once, call
    ``predict`` many times — the model and device are cached."""

    def __init__(
        self,
        checkpoint: Optional[ImageLike] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or load_config()
        ckpt = checkpoint or self.cfg["model"]["champion_checkpoint"]
        self.checkpoint_path = resolve_path(ckpt)
        self.device = _pick_device(device)
        model = ResNet18Regressor(
            out_dim=self.cfg["model"]["out_dim"], pretrained=False
        )
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: ImageLike) -> GraspPrediction:
        x = preprocess_image(image).to(self.device)
        t0 = time.perf_counter()
        out = self.model(x).cpu().numpy()[0]
        latency_ms = 1000.0 * (time.perf_counter() - t0)

        rect = decode_prediction(out)
        s2, c2 = float(out[4]), float(out[5])
        confidence = float(min(1.0, math.hypot(s2, c2)))
        return GraspPrediction(
            cx=rect.cx,
            cy=rect.cy,
            width=rect.width,
            height=rect.height,
            angle_deg=rect.angle_deg,
            confidence=confidence,
            corners=rect.corners.tolist(),
            latency_ms=latency_ms,
            raw_6vec=[float(v) for v in out],
        )

    @torch.no_grad()
    def predict_batch(self, images: list[ImageLike]) -> list[GraspPrediction]:
        return [self.predict(im) for im in images]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict an oriented grasp rectangle.")
    ap.add_argument("--image", required=True, help="Path to an RGB image.")
    ap.add_argument("--checkpoint", default=None, help="Override champion checkpoint.")
    ap.add_argument("--config", default=None, help="Override config.yaml path.")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (auto if unset).")
    ap.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else None
    predictor = GraspPredictor(
        checkpoint=args.checkpoint, config=cfg, device=args.device
    )
    pred = predictor.predict(args.image)

    if args.json:
        print(json.dumps(pred.to_dict()))
        return
    print(f"Image:       {Path(args.image).name}")
    print(f"Center:      ({pred.cx:.1f}, {pred.cy:.1f}) px")
    print(f"Jaw opening: {pred.width:.1f} px")
    print(f"Plate depth: {pred.height:.1f} px")
    print(f"Angle:       {pred.angle_deg:+.1f}°")
    print(f"Confidence:  {pred.confidence:.3f}  (orientation-norm heuristic)")
    print(f"Latency:     {pred.latency_ms:.2f} ms ({predictor.device.type})")


if __name__ == "__main__":
    main()
