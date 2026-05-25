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
