# Experiment Log — Robotic Grasp Prediction (DL-3)

Running log of every experiment. Most-recent phase on top. Numbers here are
the *source of truth* — anything that ends up in a LinkedIn post or the
project README is cross-checked against this file.

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
