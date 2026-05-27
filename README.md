# Robotic Grasp Prediction (DL-3)

> Predict 2-D oriented grasp rectangles from RGB-D images of household objects
> on a table. Cornell Grasping Dataset; Jiang IoU > 0.25 / angle < 30° metric.

**Status:** Phase 3 / 7 — four orthogonal levers tested on top of the Phase-2 M2
ceiling (4× training data, rotation aug, depth-as-blue, multi-positive loss).
Champion is the all-knobs union E3.5 at **77.0 % Jiang accuracy** (+22 pp over
Phase-2 M2; crosses Lenz 2014 obj-wise 73.9 %; 8 pp short of Redmon 2015).
Numbers in [results/EXPERIMENT_LOG.md](results/EXPERIMENT_LOG.md).
Each phase builds on the research log below.

## Research log

| Phase | Day | Title | Notebook | Report |
|------:|:---:|:-----:|:--------:|:------:|
| 1 | Mon 2026-05-25 | Domain research + dataset + 3 baselines | [phase1_baseline.ipynb](notebooks/phase1_baseline.ipynb) | [day1_phase1_report.md](reports/day1_phase1_report.md) |
| 2 | Tue 2026-05-26 | Multi-model experiment (CNN regressor, ResNet, GG-CNN, ViT, …) | [phase2_multi_model.ipynb](notebooks/phase2_multi_model.ipynb) | [day2_phase2_report.md](reports/day2_phase2_report.md) |
| 3 | Wed 2026-05-27 | Full data + rotation aug + depth channel + multi-positive loss | [phase3_full_data_and_aug.ipynb](notebooks/phase3_full_data_and_aug.ipynb) | [day3_phase3_report.md](reports/day3_phase3_report.md) |
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

### Phase 3: Full Data + Rotation Aug + Depth Channel + Multi-Positive Loss — 2026-05-27

<table>
<tr>
<td valign="top" width="38%">

**What was tested:** Four orthogonal levers on top of Phase-2 M2 — (a) 4× larger training set (200 → 785 imgs, Cornell archives 04–10), (b) rotation aug ±30° with grasp transform, (c) depth-as-blue input substitution (Lenz 2014), (d) multi-positive closest-GT loss. Five experiments: E3.1 control 0.730, E3.2 rot 0.700, E3.3 depth 0.710, E3.4 multi-pos 0.730, **E3.5 all-knobs union 0.770 (+22 pp vs Phase-2 M2)**.<br><br>
**What worked best:** E3.5 — the union of rot + depth + multi-pos. Each single knob *underperformed or tied* the E3.1 full-data control; combined they compound super-additively into +4 pp over E3.1 and beat Lenz 2014 obj-wise (0.739).

</td>
<td align="center" width="24%">

<img src="results/phase3_leaderboard.png" width="220">

</td>
<td valign="top" width="38%">

**Key Insight:** Phase 2 was **data-starved, not model-starved** — same arch / optimizer / loss, just 585 more images, takes accuracy 0.550 → 0.730 (82 % of the total Phase-3 lift). Hyperparameter and architecture changes weren't where the easy wins lived.<br><br>
**Surprise:** Super-additivity of weak regularisers. Rot aug alone HURTS (−3 pp), depth-as-blue alone HURTS (−2 pp), multi-pos alone is neutral — but all three together beat every single-knob run by 4 pp. Three orthogonal regularisation channels compound where each subtracts on its own.<br><br>
**Research:** Lenz 2014 — depth-as-blue substitution (adopted in E3.3, marginally hurt accuracy but lifted median IoU as predicted). Redmon 2015 — rotation/translation aug (adopted in E3.2, hurt on Cornell because Phase-1's "orientation isn't the hard part" finding survives). Morrison 2018 GG-CNN — multi-positive supervision (adopted via cheaper closest-GT loss in E3.4, accuracy-neutral but best single-knob median IoU).<br><br>
**Best Model So Far:** E3.5 all-knobs ResNet-18 — **77.0 %** Jiang accuracy on the object-wise split (median IoU 0.404, median angle err 4.9°).

</td>
</tr>
</table>

## Current Status

Phase 3 complete. Current best: **E3.5 all-knobs ResNet-18 at 77.0 % Jiang accuracy** (object-wise split, n=100), median angle err 4.9°, median IoU 0.404. +22 pp over Phase-2 M2 (target was +10 pp; delivered +22 pp), crosses Lenz 2014 obj-wise (73.9 %) and GG-CNN 2018 (73 %), 8 pp short of Redmon 2015 (84.9 %). Phase 4 (Thu 2026-05-28) runs an Optuna sweep on the E3.5 stack (`lr`, `weight_decay`, `rotation_deg`, `batch_size`, `epochs`) plus a 4-channel RGB+depth variant to test whether the depth signal is real but was masked by the blue-channel loss. Target: close the 8-pp gap to Redmon 2015.

## Key Findings

1. **Super-additivity of weak regularisers.** Rotation aug, depth-as-blue, and multi-positive loss each *underperform or tie* the E3.1 full-data control (0.730) in isolation, but their union (E3.5) hits 0.770 — +4 pp over every single-knob run and crosses Lenz 2014. Three orthogonal regularisation channels compound where each subtracts on its own.
2. **Phase 2 was data-starved, not model-starved.** Same architecture, same optimizer, same loss — just 585 more training images (200 → 785) takes accuracy 0.550 → 0.730, closing 82 % of the total Phase-3 gap with zero new ideas.
3. **Continuous sin/cos angle regression beats Redmon's 18-way angle classification by 42 pp on the same ResNet-18 backbone** — Phase-2 M2 = 55 %, M3 = 13 %. The unified 6-vec representation sidesteps the entire CE-on-11-examples-per-bin failure mode.
4. **ImageNet pretraining is worth +20 pp on 200 grasp images** — Phase-2 M2 (pretrained) vs M1 (from scratch, same regression head) = 55 % vs 35 %.
5. **Orientation is not the bottleneck on Cornell — confirmed triply.** Phase-1 baselines, Phase-2 CNN failure decomposition, and Phase-3 rotation-aug ablation all show "angle within 30°, IoU too low" as the dominant failure mode. Rotation aug on top of E3.1 actually HURTS (−3 pp) — the model spending capacity on rotation invariance it never needs at test trades against spatial localisation.

## Models Compared

13 total: 3 Phase-1 baselines (random, centre+median-size, depth-edge antipodal) + 5 Phase-2 learned models (M1 TinyRedmonCNN, M2 ResNet-18 regressor, M3 ResNet-18 hybrid head, M4 GGCNNTiny, M5 TinyViT) + 5 Phase-3 ablations on the M2 stack (E3.1 full-data control, E3.2 +rotation aug, E3.3 +depth-as-blue, E3.4 +multi-positive loss, E3.5 all-knobs union). Champion is **E3.5 at 77.0 %**.
