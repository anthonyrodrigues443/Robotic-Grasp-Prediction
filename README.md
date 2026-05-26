# Robotic Grasp Prediction (DL-3)

> Predict 2-D oriented grasp rectangles from RGB-D images of household objects
> on a table. Cornell Grasping Dataset; Jiang IoU > 0.25 / angle < 30° metric.

**Status:** Phase 2 / 7 — five CNN/ViT architectures trained head-to-head on the
same Phase-1 object-wise split. Champion is the ImageNet-pretrained ResNet-18
regressor at 55.0 % Jiang accuracy (+38 pp over the Phase-1 depth-antipodal
baseline). Numbers in [results/EXPERIMENT_LOG.md](results/EXPERIMENT_LOG.md).
Each phase builds on the research log below.

## Research log

| Phase | Day | Title | Notebook | Report |
|------:|:---:|:-----:|:--------:|:------:|
| 1 | Mon 2026-05-25 | Domain research + dataset + 3 baselines | [phase1_baseline.ipynb](notebooks/phase1_baseline.ipynb) | [day1_phase1_report.md](reports/day1_phase1_report.md) |
| 2 | Tue 2026-05-26 | Multi-model experiment (CNN regressor, ResNet, GG-CNN, ViT, …) | [phase2_multi_model.ipynb](notebooks/phase2_multi_model.ipynb) | [day2_phase2_report.md](reports/day2_phase2_report.md) |
| 3 | Wed 2026-05-27 | Feature engineering + deep dive on top models | — | — |
| 4 | Thu 2026-05-28 | Hyperparameter tuning + error analysis | — | — |
| 5 | Fri 2026-05-29 | Advanced techniques + ablation + LLM head-to-head | — | — |
| 6 | Sat 2026-05-30 | Production pipeline + Gradio UI | — | — |
| 7 | Sun 2026-05-31 | Tests + README polish + final report | — | — |

## Dataset

Cornell Grasping Dataset (Jiang, Moseson & Saxena, 2011). 885 RGB-D images of
240 distinct household objects, ~8 000 annotated oriented grasp rectangles
(positive + negative). The official server is dead as of 2026 —
`data/README.md` documents the Wayback mirror used.

`data/raw/` is gitignored; you re-create it locally with the wget commands in
`data/README.md`.

## Primary metric (locked)

Jiang et al. 2011 **rectangle metric**: a prediction is correct iff there
exists a ground-truth positive grasp with **IoU > 0.25** AND **angular
difference < 30°**. Field-standard since 2011; every Cornell paper reports it.

## Project layout

```
Robotic-Grasp-Prediction/
├── README.md                this file
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md            dataset provenance + download instructions
│   ├── raw/                 (gitignored) extracted Cornell archives
│   └── processed/           (gitignored) cached depth maps, splits
├── src/
│   ├── cornell.py           dataset loader, GraspRect, IoU + Jiang metric
│   └── baselines.py         RandomBaseline / CenterHeuristicBaseline / DepthAntipodalBaseline
├── notebooks/
│   └── phase1_baseline.ipynb
├── results/
│   ├── metrics.json
│   ├── EXPERIMENT_LOG.md
│   └── phase1_*.png         plots from Phase 1
├── reports/
│   └── day1_phase1_report.md
└── tests/
    └── test_cornell.py
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fetch the data (~1.4 GB for archives 01-03 used in Phase 1):
bash scripts/fetch_cornell.sh    # written in Phase 2; Phase 1 used the inline curl from the report
jupyter nbconvert --to notebook --execute --inplace notebooks/phase1_baseline.ipynb
```

## Iteration Summary

### Phase 1: Domain Research + Dataset + 3 Baselines — 2026-05-25

<table>
<tr>
<td valign="top" width="38%">

**What was tested:** Three non-learning baselines on Cornell (object-wise split, 100 test images, 542 positives) under the Jiang IoU>0.25 / angle<30° metric. Random = 1.0 %, image-centre + median size = 4.0 %, depth-edge antipodal = 17.0 %.<br><br>
**What worked best:** B3 depth-edge antipodal heuristic — depth-thresholded object mask → `cv2.minAreaRect` → grasp across the narrow axis. Wins because the prediction *follows the object* instead of sitting at a fixed point.

</td>
<td align="center" width="24%">

<img src="results/phase1_failure_decomposition.png" width="220">

</td>
<td valign="top" width="38%">

**Key Insight:** Orientation isn't the hard part of Cornell. Across all three baselines the dominant failure mode is "angle within 30° but IoU too low" — 31–49 % of predictions. The CNN's real job in Phase 2 is **picking a grasp point on the object's surface**, not orienting the gripper.<br><br>
**Surprise:** Random chance was 1.0 %, not the 5–15 % I pre-registered. The Jiang metric's two AND constraints are negatively correlated on random predictions — passing IoU on some GT tends to fail its angle.<br><br>
**Research:** Jiang 2011 — original Cornell + IoU/angle metric, adopted verbatim. Lenz 2014 — first deep model at 73.9 % object-wise, so Phase 2's CNN must clear B3 by ≥ 50 pp.<br><br>
**Best Model So Far:** B3 depth + antipodal — **17.0 %** accuracy on the object-wise held-out split.

</td>
</tr>
</table>

### Phase 2: Five CNN/ViT Architectures Head-to-Head — 2026-05-26

<table>
<tr>
<td valign="top" width="38%">

**What was tested:** Five deliberately different models on the same Phase-1 object-wise split (200 train / 100 test) and the same Jiang metric. M1 TinyRedmonCNN from-scratch (35.0 %), M2 ResNet-18 regressor ImageNet-pretrained (**55.0 %**), M3 ResNet-18 hybrid head with 18-way angle classifier (13.0 %), M4 GGCNNTiny per-pixel quality map (0.0 %), M5 TinyViT from-scratch (47.0 %).<br><br>
**What worked best:** M2 ResNet-18 regressor — ImageNet transfer + a unified 6-vec (cx, cy, w, h, sin(2θ), cos(2θ)) output. +38 pp over the Phase-1 B3 baseline; median angle err 6.1°, median IoU 0.293.

</td>
<td align="center" width="24%">

<img src="results/phase2_accuracy_bar.png" width="220">

</td>
<td valign="top" width="38%">

**Key Insight:** Redmon's 2015 18-way angle-classification head loses by **42 pp** to continuous sin/cos regression on the same backbone. The CE term dominates the AdamW gradient on 11 examples-per-bin, and the (cx, cy, w, h) regression branch collapses to noise (median IoU 0.004). The 13-year-old design choice everyone copies doesn't survive a fair head-to-head against the simpler unified representation.<br><br>
**Surprise:** ViTs and CNNs fail in opposite directions. M5 TinyViT has the **highest** median IoU (0.364) but the **worst** angle error (36.8°) — 38 % of its failures are "IoU ok, angle wrong" vs M2's 8 %. Patch16 tokens lose intra-patch gradients that local convs preserve for orientation.<br><br>
**Research:** Redmon & Angelova ICRA 2015 — head-to-head'd their angle-discretisation choice on the same backbone, the simpler regression head wins. Dosovitskiy ICLR 2021 — predicted ViTs would underperform on 200 images; M5 instead failed *differently* from CNNs, suggesting hybrid ViT-localise + CNN-orient for Phase 3.<br><br>
**Best Model So Far:** M2 ResNet-18 regressor — **55.0 %** Jiang accuracy on the object-wise split.

</td>
</tr>
</table>

## Current Status

Phase 2 complete. Current best: **M2 ResNet-18 regressor at 55.0 % Jiang accuracy** (object-wise split, n=100), median angle err 6.1°, median IoU 0.293. +38 pp over the Phase-1 B3 baseline; missed the "+50 pp" pre-registered target by 12 pp. Phase 3 (Wed 2026-05-27) retrains M2 on the full 885-image benchmark (archives 04–10 already on disk) with rotation augmentation, depth as a 4th channel, and a multi-positive closest-GT loss; floor is to clear 55 % by ≥ 10 pp.

## Key Findings

1. **Continuous sin/cos angle regression beats Redmon's 18-way angle classification by 42 pp on the same ResNet-18 backbone** — M2 = 55 %, M3 = 13 %. Joint CE+MSE loss on 200 images / 18 bins crushes the regression branch; the unified 6-vec representation sidesteps the entire failure mode.
2. **ImageNet pretraining is worth +20 pp on 200 grasp images** — M2 (pretrained) vs M1 (from scratch, same regression head) = 55 % vs 35 %.
3. **ViTs and CNNs have dual failure modes.** M5 TinyViT highest median IoU (0.364) + worst angle err (36.8°); M2 ResNet-18 the inverted profile. Phase-1's "localisation is the bottleneck" framing survives for CNNs but **inverts for transformers** at this scale.
4. Orientation is not the bottleneck on Cornell — 27 % of M2's failures are still "angle ok, IoU too low"; the CV→CNN gap closed the "both wrong" bucket (50 %→10 %), not the localisation one.
5. The Jiang IoU∧angle metric is brutal in expectation — random chance is 1 %, not the 5–15 % a naive marginals-multiply estimate predicts.

## Models Compared

8 total: 3 Phase-1 baselines (random, centre+median-size, depth-edge antipodal) + 5 Phase-2 learned models (M1 TinyRedmonCNN, M2 ResNet-18 regressor, M3 ResNet-18 hybrid head, M4 GGCNNTiny, M5 TinyViT). Champion is M2 at 55.0 %.
