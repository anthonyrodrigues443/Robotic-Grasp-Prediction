# Final Report — Robotic Grasp Prediction (DL-3)
**Project window:** 2026-05-25 → 2026-05-31 (Phases 1–7)
**Champion:** `P4_tuned_champion.pt` — ImageNet-pretrained ResNet-18 global regressor
**Headline number:** **0.770 Jiang accuracy** (object-wise Cornell folder-03, n=100), reproducible from the saved artifact on CPU and MPS.

---

## 1. Problem & metric

Predict a single 2-D oriented grasp rectangle `(cx, cy, w, h, θ)` for a parallel-jaw
gripper from an RGB image of one tabletop object. Dataset: **Cornell Grasping
Dataset** (Jiang, Moseson & Saxena 2011) — 885 RGB-D images, ~240 objects, ~8 000
annotated grasps.

**Primary metric (locked Phase 1):** the **Jiang rectangle metric** — a prediction
is correct iff some ground-truth positive grasp has **IoU > 0.25 AND angular
difference < 30°**. Chosen because it is the field standard reported by every
Cornell paper since 2011, so our numbers are directly comparable to a 13-year
published leaderboard. Split is **object-wise** (held-out objects never seen in
training) — the harder, more honest generalisation setting.

## 2. The seven-phase arc

| Phase | Day | Question | Champion that day | Acc |
|------:|:---:|----------|-------------------|----:|
| 1 | Mon 05-25 | What's the non-learning floor? | B3 depth-edge antipodal heuristic | 0.170 |
| 2 | Tue 05-26 | Which of 5 architectures wins? | M2 ResNet-18 global regressor | 0.550 |
| 3 | Wed 05-27 | Data, aug, depth, or loss? | E3.5 all-knobs union | 0.770 |
| 4 | Thu 05-28 | Can tuning close the gap to Redmon? | P4 Optuna-tuned (40 ep) | 0.780* |
| 5 | Fri 05-29 | Does a per-pixel paradigm beat global? | **falsified** — global stays champ | 0.420 (per-pixel) |
| 6 | Sat 05-30 | Does the *saved artifact* reproduce? | productionised, verified | 0.770 |
| 7 | Sun 05-31 | Tests + docs + consolidation | — | 0.770 (shipped) |

\* The Phase-4 *in-session* eval logged 0.780; the persisted checkpoint reproduces
**0.770** (a single test image) under two independent harnesses on two devices. We
ship and document 0.770 as the honest deployed number.

## 3. Consolidated leaderboard (object-wise, n=100 unless noted)

| Rank | Model | Phase | Acc | Median IoU | Median ∠err |
|-----:|-------|:-----:|----:|-----------:|------------:|
| 1 | **P4 tuned ResNet-18 global regressor** ⭐ | 4 | **0.770** | 0.396 | 5.5° |
| 2 | E3.5 all-knobs (rot+depth-blue+multipos) | 3 | 0.770 | 0.404 | 4.9° |
| 3 | E3.1 full-data control | 3 | 0.730 | 0.405 | 10.5° |
| 3 | E3.4 multi-positive loss | 3 | 0.730 | 0.428 | 5.4° |
| 5 | E3.3 depth-as-blue | 3 | 0.710 | 0.420 | 9.9° |
| 6 | P4 4-ch depth (clean) | 4 | 0.700 | — | — |
| 6 | E3.2 rotation aug | 3 | 0.700 | 0.369 | 8.0° |
| 8 | E5.0 RGB global control (matched) | 5 | 0.690 | 0.417 | 8.0° |
| 9 | P4 4-ch depth (all-knobs) | 4 | 0.610 | — | — |
| 10 | M2 ResNet-18 (200-img, data-starved) | 2 | 0.550 | 0.293 | 6.1° |
| 11 | M5 TinyViT (scratch) | 2 | 0.470 | 0.364 | 36.8° |
| 12 | E5.3 ResNet18-FCN per-pixel (pretrained) | 5 | 0.420 | 0.251 | 8.5° |
| 13 | M1 TinyRedmonCNN (scratch) | 2 | 0.350 | 0.156 | 11.9° |
| 14 | B3 depth-edge antipodal (heuristic) | 1 | 0.170 | 0.159 | 37.0° |
| 15 | M3 ResNet-18 hybrid (18-way angle head) | 2 | 0.130 | 0.004 | 8.6° |
| 16 | B2 centre + median-size | 1 | 0.040 | 0.083 | 24.0° |
| 17 | B1 random | 1 | 0.010 | 0.000 | 46.0° |
| 17 | M4 GGCNNTiny / E5.1–5.2 per-pixel (scratch) | 2/5 | 0.000 | 0.000 | 40°+ |

**Versus published Cornell baselines (object-wise):** Lenz 2014 = 0.739 (**beaten**),
Redmon & Angelova 2015 = 0.849 (−7.9 pp), Cao 2023 SOTA = 0.978 (image-wise).

## 4. Frontier-LLM head-to-head (Phase 5, n=40 stratified)

| Model | Acc | Median IoU | Latency/img | Cost/1k |
|-------|----:|-----------:|------------:|--------:|
| **Custom global ResNet-18** ⭐ | 0.75 | 0.395 | **0.017 s** | **$0** |
| codex / GPT-5.5 | 0.75 | 0.419 | 41.0 s | $50 |
| claude / opus | 0.25 | 0.000 | 16.6 s | $22.50 |
| claude / haiku | 0.15 | 0.000 | 18.1 s | $1.50 |

GPT-5.5 *ties* the custom CNN on accuracy from raw pixels with no training — but at
~2,400× the latency and ~50,000× the cost. Both Claude models fail the spatial task
(median IoU 0.000: right gripper angle, wrong grasp centre). The free 17 ms CNN is
the correct production choice.

## 5. The five genuine findings

1. **The per-pixel paradigm LOSES to global box regression on Cornell — Phase-4
   prediction falsified.** Same ResNet-18 backbone, same ImageNet weights, only the
   output head differs: per-pixel 0.420 vs global 0.690 (−27 pp). Both from-scratch
   per-pixel nets stay at 0.000 even on 4× data with proper multi-grasp targets, so
   Phase-2's GG-CNN 0.0 was the paradigm-without-pretraining, not data starvation.
2. **Pretraining is the dominant lever — landed there 4 separate times.** FCN
   ablation 0.420 → 0.190 from scratch (−0.23, the single biggest measured effect). A
   per-pixel *decoder* has no ImageNet-pretrained equivalent, so the paradigm
   forfeits this project's strongest lever in exactly the part that localises.
3. **A frontier LLM is competitive but economically absurd.** GPT-5.5 ties on
   accuracy; the custom CNN wins on latency (2,400×), cost (50,000×), and determinism.
4. **Depth doesn't help this metric — three experiments agree.** Clean 4-channel
   RGB+D scores 0.700, *worse* than no-depth (0.730) and depth-as-blue (0.710).
   Re-initialising conv1 perturbs the ImageNet stem more than noisy Kinect depth
   repays. All shipped models are RGB-only.
5. **Super-additivity of weak regularisers.** Rotation aug, depth-as-blue, and
   multi-positive loss each underperform or tie the full-data control in isolation,
   but their union (E3.5) hits 0.770 — +4 pp over every single-knob run.

## 6. Production system (Phase 6–7)

- **One preprocessing path** (`src/data_pipeline.preprocess_image`) shared by train,
  evaluate, predict, and UI → provably zero train/serve skew (6.1 reproduced the
  harness number byte-for-byte on CPU and MPS).
- `src/predict.py` — `GraspPredictor`, ~7 ms/img on MPS, JSON CLI.
- `src/evaluate.py` — one-command re-verification of 0.770 + latency benchmark.
- `src/train.py` — config-driven champion retrain (`--smoke` for CI).
- `app.py` — Streamlit demo: upload → grasp overlay + metrics + LLM head-to-head.
- `models/model_card.md` — HF/Google-format card with explicit limitations + safety.
- **Tests:** 45 passing — research modules (cornell/dataset/models), production smoke
  path (data→predict→evaluate, asserted == reference), and Phase-7 data-free unit
  coverage (config, split, renderer, prediction DTO).

## 7. Limitations & future work

- **One grasp, one object.** No clutter, collision, or reachability reasoning.
- **RGB-only** — transparent / low-texture objects (where depth would help most)
  are a known weak spot; depth injection was falsified on this metric.
- **Not hardware-validated** — any physical deployment needs force feedback + e-stop.
- **The 7.9-pp gap to Redmon 2015** is a *paradigm* ceiling for global regressors on
  Cornell (Phase 5), not a tuning gap. Closing it likely needs a pretrained per-pixel
  decoder (e.g. a segmentation-pretrained backbone) — the one lever this project
  could not supply.

## 8. Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fetch Cornell folders 01-10 — see data/README.md (Wayback mirror)
python src/evaluate.py            # reproduces 0.770 + latency benchmark
python src/predict.py --image <rgb.png>
streamlit run app.py             # interactive demo
pytest -q                        # 45 tests
```
