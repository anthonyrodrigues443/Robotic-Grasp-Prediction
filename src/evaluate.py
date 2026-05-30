"""Evaluation suite for the production champion.

Reproduces the headline number — Jiang IoU>0.25 / angle<30° accuracy on the
object-wise Cornell folder-03 test set (n=100) — straight from the saved
champion checkpoint, with no training involved. This is the script that turns
"trust me, it's 0.78" into a one-command, re-runnable verification.

It also runs a steady-state inference benchmark (warmup excluded) on both the
serving device and CPU so the model card / README can quote an honest latency.

CLI:
    python src/evaluate.py                 # full folder-03 eval + benchmark
    python src/evaluate.py --no-benchmark  # eval only (faster)
    python src/evaluate.py --limit 20      # quick subset for a smoke check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cornell import angle_diff_deg, is_correct_jiang
from cornell import iou as poly_iou
from data_pipeline import (
    REPO_ROOT,
    load_config,
    load_cornell,
    object_wise_split,
    preprocess_image,
)
from dataset import decode_prediction
from predict import GraspPredictor


def evaluate_split(predictor: GraspPredictor, samples, iou_thresh, angle_thresh):
    """Score the champion over a list of CornellSamples under the Jiang metric."""
    correct = 0
    best_ious, best_angle_errs, per_sample = [], [], []
    for s in samples:
        pred = predictor.predict(s.rgb_path)
        rect = pred.to_grasp_rect()
        gts = s.positives
        ok = is_correct_jiang(rect, gts, iou_thresh, angle_thresh)
        cands = [(poly_iou(rect, g), angle_diff_deg(rect, g)) for g in gts]
        best = max(cands, key=lambda t: t[0]) if cands else (0.0, 90.0)
        best_ious.append(best[0])
        best_angle_errs.append(best[1])
        per_sample.append(
            {
                "pcd_id": s.pcd_id,
                "object_id": s.object_id,
                "correct": int(ok),
                "best_iou": float(best[0]),
                "best_angle_err": float(best[1]),
            }
        )
        correct += int(ok)
    n = len(samples)
    return {
        "n": n,
        "accuracy": correct / max(n, 1),
        "median_iou": float(np.median(best_ious)) if best_ious else 0.0,
        "median_angle_err": float(np.median(best_angle_errs)) if best_angle_errs else 0.0,
        "per_sample": per_sample,
    }


def benchmark_latency(predictor: GraspPredictor, sample_path, runs=50, warmup=5):
    """Steady-state forward-pass latency on the serving device and on CPU.
    Warmup iterations are excluded (first MPS/CUDA call pays a one-off cost)."""
    x_dev = preprocess_image(sample_path)
    results = {}
    for dev_name in dict.fromkeys([predictor.device.type, "cpu"]):
        model = predictor.model.to(dev_name).eval()
        x = x_dev.to(dev_name)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if dev_name in ("mps", "cuda"):
                getattr(torch, dev_name).synchronize() if dev_name == "cuda" else torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                model(x)
            if dev_name == "cuda":
                torch.cuda.synchronize()
            elif dev_name == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - t0
        results[dev_name] = {
            "ms_per_image": 1000.0 * elapsed / runs,
            "images_per_sec": runs / elapsed,
            "runs": runs,
        }
    # restore serving device
    predictor.model.to(predictor.device)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the grasp champion on Cornell test.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Eval on first N test images only.")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "phase6_evaluation.json"),
        help="Where to write the metrics JSON.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    predictor = GraspPredictor(checkpoint=args.checkpoint, config=cfg)

    samples = load_cornell(cfg["data"]["cornell_root"])
    _, test = object_wise_split(samples, cfg["data"]["test_object_ids"])
    if args.limit:
        test = test[: args.limit]

    print(f"Evaluating champion on {len(test)} test images (folders "
          f"{cfg['data']['test_object_ids']})...")
    t0 = time.time()
    metrics = evaluate_split(
        predictor,
        test,
        cfg["metric"]["iou_thresh"],
        cfg["metric"]["angle_thresh_deg"],
    )
    metrics["eval_seconds"] = time.time() - t0
    metrics["device"] = predictor.device.type
    metrics["checkpoint"] = str(predictor.checkpoint_path.name)

    print(f"  accuracy          : {metrics['accuracy']:.3f}")
    print(f"  median IoU        : {metrics['median_iou']:.3f}")
    print(f"  median angle err  : {metrics['median_angle_err']:.1f}°")

    if not args.no_benchmark and test:
        print("Benchmarking inference latency...")
        bench = benchmark_latency(predictor, test[0].rgb_path)
        metrics["latency_benchmark"] = bench
        for dev, b in bench.items():
            print(f"  {dev:>4}: {b['ms_per_image']:.2f} ms/img  "
                  f"({b['images_per_sec']:.0f} img/s)")

    # Drop the heavy per_sample list from the printed summary but keep it in JSON.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
