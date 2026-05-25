# Robotic Grasp Prediction (DL-3)

> Predict 2-D oriented grasp rectangles from RGB-D images of household objects
> on a table. Cornell Grasping Dataset; Jiang IoU > 0.25 / angle < 30° metric.

**Status:** Phase 1 / 7 — domain research, dataset onboarded, three baselines
established (random, image-centre heuristic, depth-edge antipodal). Numbers in
[results/EXPERIMENT_LOG.md](results/EXPERIMENT_LOG.md). Each phase builds on the
research log below.

## Research log

| Phase | Day | Title | Notebook | Report |
|------:|:---:|:-----:|:--------:|:------:|
| 1 | Mon 2026-05-25 | Domain research + dataset + 3 baselines | [phase1_baseline.ipynb](notebooks/phase1_baseline.ipynb) | [day1_phase1_report.md](reports/day1_phase1_report.md) |
| 2 | Tue 2026-05-26 | Multi-model experiment (CNN regressor, ResNet, GG-CNN, ViT, …) | — | — |
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

## Current Status

Phase 1 complete. Current best: **B3 depth + antipodal heuristic at 17.0 % Jiang accuracy** (object-wise split, n=100). Phase 2 (Tue 2026-05-26) replicates Redmon & Angelova plus 4 modern CNN/ViT alternatives; target is to clear B3 by ≥ 50 pp.

## Key Findings

1. Orientation is not the bottleneck on Cornell — 58 % of constant-angle=0 predictions already land within 30° of some GT.
2. Cornell's image-centre prior buys only +3 pp over random; grasps within an object scatter across its surface.
3. Classical antipodal physics (depth + minAreaRect) closes 16 pp of the ~99 pp gap to SOTA — the remaining 82 pp is what the CNN's RGB learning pays for.
4. The Jiang IoU∧angle metric is brutal in expectation — random chance is 1 %, not the 5–15 % a naive marginals-multiply estimate predicts.

## Models Compared

3 baselines so far (random, centre+median-size, depth-edge antipodal). Phase 2 will add 5 learned models (Redmon-style CNN, ResNet-18 regressor, ResNet-18 + rotation classifier, GG-CNN, small ViT).
