# Model Card — Cornell Grasp Champion (`P4_tuned_champion.pt`)

A single-shot oriented grasp-rectangle regressor for parallel-jaw robotic
grasping, trained and evaluated on the Cornell Grasping Dataset.

## Model details

| | |
|---|---|
| **Architecture** | ImageNet-pretrained ResNet-18 backbone + dropout(0.3) + linear regression head |
| **Output** | 6-vector `(cx, cy, w, h, sin2θ, cos2θ)` in a resized 224×224 frame, decoded to an oriented rectangle in the canonical 640×480 Cornell frame |
| **Parameters** | 11,179,590 (~45 MB fp32) |
| **Input** | RGB image, resized to 224×224, ImageNet-normalised |
| **Angle encoding** | Doubled-angle `(sin2θ, cos2θ)` — handles a grasp's 180° symmetry with no wrap-around discontinuity at ±90° |
| **Framework** | PyTorch (`torch>=2.4`, `torchvision>=0.19`) |
| **Checkpoint** | `models/P4_tuned_champion.pt` (plain `state_dict`) |
| **License (code)** | research/portfolio use |

## Intended use

- **Primary:** predict one high-quality parallel-jaw grasp (placement, jaw
  width, orientation) for a single, roughly-centered tabletop object from an
  RGB image — the Cornell single-object setting.
- **Out of scope:** cluttered multi-object scenes, suction/multi-finger
  grippers, 6-DoF grasping, or deployment on a physical robot without a
  depth/force-feedback safety layer. The model predicts one grasp per image and
  has no notion of collision or reachability.

## Training data

- **Dataset:** Cornell Grasping Dataset, folders 01–10 (885 images, downloaded
  via the Wayback Machine — the official `pr.cs.cornell.edu` URL has 404'd since
  2025). One supervised example per image.
- **Split:** *object-wise* — folder **03** (n=100) is held out as the test set;
  the model never sees those objects during training (785 train images).
- **Augmentation:** random rotation ±30° (image + grasp co-transformed),
  horizontal flip, and a **multi-positive "closest-GT" loss** (each prediction
  is supervised against whichever of the image's valid grasps it is nearest to,
  rather than a single hand-picked target).

## Training procedure

Optuna-selected hyperparameters (Phase 4), retrained on the full 785-image
train split for 40 epochs:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, cosine-annealed LR |
| Learning rate | 1.197e-3 |
| Weight decay | 2.607e-4 |
| Rotation aug | ±30° |
| Batch size | 16 |
| Epochs | 40 |
| Loss | 6-vec MSE (multi-positive closest-GT) |

Reproduce from raw data: `python src/train.py --evaluate`.

## Evaluation

**Metric:** Jiang et al. 2011 rectangle metric — a prediction is correct iff
some ground-truth positive grasp has **IoU > 0.25 AND angle error < 30°**.

Verified from the saved artifact via `python src/evaluate.py` (identical under
the research harness `trainer.evaluate` and on both CPU and MPS):

| Metric | Value (folder-03 test, n=100) |
|---|---|
| **Jiang accuracy** | **0.770** |
| Median IoU | 0.396 |
| Median angle error | 5.5° |

> The Phase-4 report logged **0.780** as the in-session number; the persisted
> checkpoint reproduces **0.770** (a single-image difference). The production
> pipeline and the original research harness agree exactly, so 0.770 is the
> honest deployed-artifact figure.

### Versus published Cornell baselines (object-wise)

| Model | Accuracy |
|---|---|
| Lenz et al. 2014 | 0.739 |
| **This model** | **0.770** |
| Redmon & Angelova 2015 | 0.849 |
| Cao et al. 2023 (SOTA, image-wise) | 0.994 |

Beats Lenz 2014; ~7.9 pp under Redmon 2015. Phase 5 established this is a
*paradigm* ceiling for global regressors on Cornell, not a tuning gap.

### Inference latency (steady-state, warmup excluded)

| Device | ms/image | images/sec |
|---|---:|---:|
| Apple MPS | 7.1 | 141 |
| CPU (single image) | 327.7 | 3 |

At ~7 ms on MPS the model is ~**5,800× faster** than a frontier-LLM grasp call
(GPT-5.5 ≈ 41 s/image in the Phase-5 head-to-head) at zero marginal cost.

## Confidence semantics

The regression head emits no class probability. The `confidence` field exposed
by `src/predict.py` is a **heuristic**: the L2 norm of the predicted
`(sin2θ, cos2θ)` pair. A confident model drives that pair onto the unit circle;
a norm well below 1 means the orientation output is being pulled toward the
origin. Treat it as a *relative* sanity signal, **not** a calibrated probability.

## Limitations and ethical considerations

- **One grasp, one object.** No clutter handling, no collision/reachability
  reasoning, no multi-grasp output.
- **Domain shift.** Trained on Cornell's lab tabletop with a top-down-ish RGB
  view. Lighting, camera angle, or object classes far from Cornell will degrade
  accuracy silently — there is no out-of-distribution detector.
- **Safety.** Not validated on hardware. Any physical deployment must add force
  feedback and a human-in-the-loop / e-stop layer; a wrong grasp can damage the
  object or gripper.
- **No depth at inference.** Phase 4 falsified depth injection on this metric;
  the model is RGB-only, so transparent/low-texture objects (where depth would
  help most) are a known weak spot.
