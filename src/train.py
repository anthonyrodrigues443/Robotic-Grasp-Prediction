"""Config-driven training pipeline that reproduces the production champion.

The champion is the Phase-4 tuned global ResNet-18 regressor: ImageNet-pretrained
backbone, 6-vec regression head, trained on the FULL object-wise train split
(every Cornell folder except the held-out test folder 03) with rotation
augmentation and the multi-positive closest-GT loss, using the Optuna-selected
hyperparameters baked into ``config/config.yaml``.

This is the one command that regenerates ``models/champion_reproduced.pt`` from
raw data. The shipped ``models/P4_tuned_champion.pt`` was produced by the Phase-4
notebook with the identical recipe; this script makes that recipe a clean,
re-runnable artifact rather than notebook cells.

CLI:
    python src/train.py                       # full 40-epoch champion retrain
    python src/train.py --epochs 2 --smoke    # fast smoke test on a small subset
    python src/train.py --evaluate            # evaluate the freshly trained model
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cornell import angle_diff_deg, is_correct_jiang
from cornell import iou as poly_iou
from data_pipeline import (
    REPO_ROOT,
    load_config,
    load_cornell,
    object_wise_split,
    resolve_path,
)
from dataset import decode_prediction
from dataset_phase3 import CornellGraspDatasetP3
from torch_models import ResNet18Regressor
from trainer_phase3 import _pick_device, train_regression_p3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def quick_eval(model, samples, iou_thresh, angle_thresh, device) -> dict:
    from data_pipeline import preprocess_image

    model = model.to(device).eval()
    correct, n = 0, 0
    ious = []
    for s in samples:
        x = preprocess_image(s.rgb_path).to(device)
        out = model(x).cpu().numpy()[0]
        rect = decode_prediction(out)
        ok = is_correct_jiang(rect, s.positives, iou_thresh, angle_thresh)
        best = max((poly_iou(rect, g) for g in s.positives), default=0.0)
        ious.append(best)
        correct += int(ok)
        n += 1
    return {
        "accuracy": correct / max(n, 1),
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/reproduce the grasp champion.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny subset (40 train imgs) for a fast wiring check.")
    ap.add_argument("--evaluate", action="store_true",
                    help="Evaluate on the test folder after training.")
    ap.add_argument("--out", default=None, help="Override output checkpoint path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    device = _pick_device()

    samples = load_cornell(cfg["data"]["cornell_root"])
    train, test = object_wise_split(samples, cfg["data"]["test_object_ids"])
    if args.smoke:
        train = train[:40]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]

    print(f"Device: {device.type} | train={len(train)} test={len(test)} | epochs={epochs}")

    ds = CornellGraspDatasetP3(
        train,
        augment=True,
        augment_rot_deg=tcfg["rotation_deg"],
        use_depth_blue=tcfg["use_depth_blue"],
        multi_positive=tcfg["multi_positive"],
        seed=tcfg["seed"],
    )
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=False)

    model = ResNet18Regressor(out_dim=cfg["model"]["out_dim"], pretrained=True)
    result = train_regression_p3(
        model,
        loader,
        epochs=epochs,
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        multi_positive=tcfg["multi_positive"],
        device=device,
    )
    print(f"Trained {result.name} ({result.n_params:,} params) in "
          f"{result.train_seconds:.0f}s | final loss {result.train_curve[-1]:.4f}")

    out_path = Path(args.out) if args.out else resolve_path(tcfg["out_checkpoint"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint -> {out_path}")

    if args.evaluate:
        m = cfg["metric"]
        res = quick_eval(model, test, m["iou_thresh"], m["angle_thresh_deg"], device)
        print(f"Test (folder {cfg['data']['test_object_ids']}, n={res['n']}): "
              f"acc={res['accuracy']:.3f} medIoU={res['median_iou']:.3f}")


if __name__ == "__main__":
    main()
