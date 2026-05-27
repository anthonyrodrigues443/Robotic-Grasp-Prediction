# Experiment Log — Robotic Grasp Prediction (DL-3)

Running log of every experiment. Most-recent phase on top. Numbers here are
the *source of truth* — anything that ends up in a LinkedIn post or the
project README is cross-checked against this file.

---

## Phase 2 — 2026-05-26 — Five CNN/ViT architectures head-to-head

**Dataset / split:** unchanged from Phase 1 (Cornell archives 01–03, 300
images, object-wise split: folders 01–02 train / folder 03 test, 200 / 100
images).
**Metric (primary):** Jiang IoU > 0.25 AND angle err < 30°.
**Training:** 40 epochs, batch 16, AdamW + cosine LR schedule, horizontal-flip
augmentation only (no rotations/crops — kept minimal so the 5-way comparison
is about architecture, not augmentation). MPS (Apple Silicon) backend.
**Target encoding:** single positive grasp per image (the most central one),
6-vec `(cx, cy, w, h, sin(2θ), cos(2θ))` in the 224×224 resized frame. The
doubled-angle representation handles Cornell's 180° grasp symmetry without
the wrap discontinuity at ±π/2.

### Leaderboard

| Rank | Model | Acc | Median IoU | Median angle err | Params | Train s | Inf ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **M2 ResNet-18 Regressor** | **0.550** | 0.293 | **6.1°** | 11.2 M | 170.8 | 19.1 |
| 2 | M5 TinyViT (from scratch) | 0.470 | **0.364** | 36.8° | 2.0 M | 165.5 | 14.1 |
| 3 | M1 TinyRedmonCNN (from scratch) | 0.350 | 0.156 | 11.9° | 1.7 M | 126.8 | 8.8 |
| 4 | B3 depth + antipodal (Phase 1) | 0.170 | 0.159 | 37.0° | 0 | 0 | — |
| 5 | M3 ResNet-18 Hybrid (4-D reg + 18-way angle CE) | 0.130 | 0.004 | 8.6° | 11.2 M | 169.2 | 11.6 |
| 6 | B2 centre + median size (Phase 1) | 0.040 | 0.083 | 24.0° | 0 | 0 | — |
| 7 | B1 random rectangle (Phase 1) | 0.010 | 0.000 | 46.0° | 0 | 0 | — |
| 8 | M4 GGCNNTiny | 0.000 | 0.000 | 44.0° | 0.12 M | 201.6 | 12.0 |

### Failure-mode decomposition (Jiang sub-constraints, % of 100 test images)

| Model | both ok | angle ok / IoU low | IoU ok / angle wrong | both wrong |
|---|---:|---:|---:|---:|
| **M2 ResNet-18 Regressor** | **55** | 27 | 8 | 10 |
| M5 TinyViT | 41 | 5 | **38** | 16 |
| M1 TinyRedmonCNN | 35 | 35 | 5 | 25 |
| B3 depth + antipodal | 17 | 31 | 2 | 50 |
| M3 ResNet-18 Hybrid | 13 | **58** | 2 | 27 |
| B2 centre + median size | 4 | 49 | 9 | 38 |
| B1 random | 1 | 35 | 0 | 64 |
| M4 GGCNNTiny | 0 | 41 | 0 | 59 |

### Key findings

1. **Headline — Redmon's 2015 angle-as-classification choice loses 42 pp on
   the same backbone in 2026.** M2 (ResNet-18 + 6-D continuous regression
   including sin/cos of doubled angle) hits 0.550; M3 (ResNet-18 + 4-D
   regression for cx/cy/w/h + 18-way cross-entropy for angle bin) collapses
   to 0.130. The angle head in M3 actually **works** — median angle err
   8.6° (better than M2's 6.1° — class-balanced bins are easier than
   continuous floats). But **the regression head fails completely**: median
   IoU = 0.004 vs M2's 0.293. The cross-entropy loss on 11 examples-per-bin
   dominates the AdamW step direction; the regression branch gets effectively
   no gradient. This is the post-worthy result.

2. **ImageNet transfer is worth +20 pp on 200 training images.** M2
   (ResNet-18 pretrained) = 0.550; M1 (TinyRedmonCNN from scratch, same
   regression head, similar param count to a single ResNet block) = 0.350.
   With 200 grasp images we cannot replicate Redmon & Angelova's 2015
   from-scratch result (84.9 % on the full 885-image dataset); Phase 3's
   archives 04–10 will close that gap by 3×ing the training set.

3. **The ViT and the CNN fail in opposite directions.** M5 TinyViT has the
   highest median IoU (0.36) of any model but the *highest* median angle
   error (37°) — its failure profile is 38 % "IoU ok, angle wrong" vs M2's
   8 %. TinyViT learns *where* to grasp but not *how to orient*. M2 ResNet-18
   has the inverted profile: 27 % "angle ok, IoU too low" vs M5's 5 %. The
   Phase-1 framing "localisation is the bottleneck" is upheld for CNNs and
   **inverted for transformers** at this scale.

4. **Phase-1's hypothesis survives for the champion.** Even M2 at 0.55
   accuracy has 27 % of failures in the "angle ok, IoU too low" bucket —
   exactly the same pattern as the Phase-1 baselines, just smaller. Phase 1
   correctly identified the lever; the lever is still pulling.

5. **The 119k-parameter GG-CNN failed to learn at 0 % accuracy.** The output
   resolution is 218×218 vs the 224 input; the per-pixel quality map collapsed
   to near-uniform values across training (median IoU = 0.0 at test). Likely
   causes: (a) target Gaussian blobs at σ=4 px in a 218² grid are *very*
   sparse signal — 99.5 % of pixels are 0; (b) no skip connections to
   preserve spatial detail. Phase 3 will either properly weight the target
   sparsity (per-pixel class-balanced BCE) or add a U-Net skip path.

### What didn't work

- **M3 hybrid head.** The 18-way classification cross-entropy dominated the
  joint loss; the (cx, cy, w, h) regression head received almost no usable
  signal. The Redmon 2015 paper trained these losses in separate stages — we
  trained jointly, and the lesson is "joint multi-task losses on tiny
  datasets need very careful weighting." Phase 3 will either (a) drop the
  classification head entirely, (b) train the heads sequentially with the
  regression head frozen for the first N epochs, or (c) re-weight CE by
  1/N_examples_per_bin so the regression branch keeps capacity.
- **M4 GGCNNTiny.** As above — the per-pixel quality map collapsed. A 119 k
  parameter encoder-decoder on 200 images with no skip connections does not
  recover spatial structure.
- **Augmentations limited to flips.** With 200 training images and 11.2 M
  parameters, M2 likely overfits even at 40 epochs (train loss plateaued
  near 0.0035, several orders of magnitude below the test MSE). Phase 3
  needs rotations, depth-channel input, and possibly random crops.

### Frontier reference (for context — Phase 5 will run head-to-head)

| Approach | Year | Object-wise acc | Notes |
|---|---:|---:|---|
| Lenz et al. (Cornell, deep) | 2014 | 73.9 % | first deep-learning result |
| Redmon & Angelova (single-shot CNN) | 2015 | 84.9 % | full Cornell 885 imgs |
| GG-CNN (Morrison) | 2018 | ~73 % | per-pixel quality map |
| Cao et al. (Bilateral fusion) | 2023 | **97.8 %** | current SOTA |
| **M2 ResNet-18 Regressor (this work, 200 imgs)** | **2026** | **55.0 %** | 7.5× less training data than Redmon |

### Files

- `notebooks/phase2_multi_model.ipynb` (28 cells, 0 errors, 0 fake display-only,
  all training in cells, 3.2 MB executed notebook)
- `src/dataset.py`, `src/torch_models.py`, `src/trainer.py` — new
- `src/baselines.py` — fixed cv2.minAreaRect 90° angle branch (was inverted in
  Phase 1, verified empirically across 9 rotated-rectangle test cases)
- `tests/test_dataset_and_models.py` — 20 new unit tests (encode/decode round-trip,
  doubled-angle wraparound, angle-bin quantisation, model output shapes,
  param-count regression bounds, GG-CNN target map peak location)
- `results/metrics.json` — `phase_2` block appended
- `results/phase2_model_comparison.csv`
- `results/phase2_target_sanity.png`
- `results/phase2_accuracy_bar.png`
- `results/phase2_training_curves.png`
- `results/phase2_failure_decomposition.png`
- `results/phase2_champion_qualitative.png`
- `models/M2_ResNet18Regressor_phase2.pt` (44.8 MB, 11.2 M params, state_dict)
- `data/raw/cornell/archives_remaining/data{04..10}.tar.gz` — 2.85 GB,
  ~580 additional images downloaded in background ready for Phase 3 extraction

---

## Phase 1 — 2026-05-25 — Three non-learning baselines

**Dataset:** Cornell Grasping Dataset, archives 01–03 (300 images / 1 540
positive grasps / 881 negative grasps).
**Split:** object-wise; folders 01–02 train (200 images, 998 positives),
folder 03 test (100 images, 542 positives).
**Metric (primary):** Jiang IoU > 0.25 AND angle err < 30° (image-wise correctness).
**Secondary metrics:** median IoU on best-matching GT, median angle err.

### Leaderboard

| Rank | Baseline | Accuracy | Median IoU | Median angle err |
|---:|---|---:|---:|---:|
| 1 | B3 depth + antipodal heuristic | **17.0 %** | 0.159 | 37.0° |
| 2 | B2 image-centre + median size, angle=0 | 4.0 % | 0.083 | 24.0° |
| 3 | B1 random rectangle | 1.0 % | 0.000 | 46.0° |

### Failure-mode decomposition (% of held-out test, n = 100)

| failure mode | B1 random | B2 centre | B3 depth |
|---|---:|---:|---:|
| both ok (correct) | 1.0 | 4.0 | 17.0 |
| **angle ok, IoU too low** | 35.0 | **49.0** | 31.0 |
| IoU ok, angle wrong | 0.0 | 9.0 | 2.0 |
| both wrong | 64.0 | 38.0 | 50.0 |

### Published reference points (literature, same metric)

| Paper | Year | Object-wise | Image-wise |
|---|---|---:|---:|
| Lenz, Lee, Saxena | 2014 | 73.9 % | 75.6 % |
| Redmon & Angelova | 2015 | 84.9 % | 84.4 % |
| Morrison (GG-CNN) | 2018 | — | 73 % |
| Cao (Bilateral fusion) — *SOTA* | 2023 | 97.8 % | 99.4 % |

### Key findings

1. **IoU localisation, not orientation, is the bottleneck.** Across all three
   baselines the dominant failure mode is "angle within 30°, IoU below 0.25".
   Even B2 (constant angle=0°) gets the angle right 58 % of the time.
2. Cornell's *centring prior* buys only 3 pp over chance — grasps are
   scattered across each object's surface, not concentrated at its centroid.
3. Classical antipodal physics (B3) clears the centring prior by 13 pp and
   chance by 16 pp — but is still 56 pp behind the 2014 deep model, before
   any consideration of the 2023 SOTA.

### Files

- `notebooks/phase1_baseline.ipynb`
- `results/metrics.json`
- `results/phase1_sample_grasps.png`
- `results/phase1_grasp_distributions.png`
- `results/phase1_baseline_comparison.png`
- `results/phase1_failure_decomposition.png`
- `results/phase1_depth_baseline_qualitative.png`
- `reports/day1_phase1_report.md`

## Phase 3 — full data + rotation aug + depth channel + multi-positive (2026-05-27)

Dataset: Cornell full 885 images (extracted archives 04–10). Object-wise: train=785, test=100 (folder 03 held out, same as Phase 2).

| Experiment | Accuracy | Δ vs P2 M2 (0.550) | Median IoU | Median angle err |
|---|---:|---:|---:|---:|
| E3.1_full_data | 0.730 | +0.180 | 0.405 | 10.5° |
| E3.2_rot_aug | 0.700 | +0.150 | 0.369 | 8.0° |
| E3.3_depth_blue | 0.710 | +0.160 | 0.420 | 9.9° |
| E3.4_multi_pos | 0.730 | +0.180 | 0.428 | 5.4° |
| E3.5_all_knobs | 0.770 | +0.220 | 0.404 | 4.9° |

**Object-wise champion:** E3.5_all_knobs at 0.770  (Δ +0.220 vs Phase-2 M2).
