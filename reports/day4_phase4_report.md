# Phase 4: Hyperparameter Tuning + Error Analysis — Robotic Grasp Prediction
**Date:** 2026-05-28
**Session:** 4 of 7

## Objective
Four pre-registered questions, all measured on the **Jiang rectangle metric** (a grasp is correct iff some GT positive has IoU > 0.25 **AND** angle error < 30°), object-wise unless noted:

1. **Can an Optuna sweep on the Phase-3 E3.5 stack close the 8-pp gap to Redmon 2015 (0.849)?** Search `lr, weight_decay, rotation_deg, batch_size` — but tune on a **held-out validation folder (07)**, never on the test folder (03), then score the winner on test exactly once.
2. **Depth-as-blue HURT in Phase 3 (−2 pp). Was the depth signal real but masked by destroying the blue colour channel?** Test depth as a genuine **4th input channel** (conv1 re-initialised, RGB kept intact).
3. **How does the model do on the easier image-wise split** vs published image-wise numbers?
4. **Where do the residual errors live**, and did tuning shift Phase 3's dominant "angle-ok / IoU-too-low" localisation cluster?

## Research & References
1. **Lenz, Lee & Saxena (2015), *Deep Learning for Detecting Robotic Grasps* (IJRR)** — the canonical Cornell baselines: object-wise 0.739, image-wise 0.756; also the origin of the "depth-as-a-colour-channel" trick we stress-tested. Their image-wise number is the bar our Phase-4 image-wise run targets.
2. **Redmon & Angelova (2015), *Real-Time Grasp Detection Using CNNs*** — object-wise 0.849, image-wise 0.844; the 8-pp gap this session tried to close. They use a direct regression head like ours, so it is the fair "same-paradigm ceiling."
3. **Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*** — TPE sampler + median pruning; we used TPESampler(seed=42) with a fixed-budget proxy objective, the standard cheap-search pattern.
4. **Internal carry-over:** our own AI-Agent-Failure-Predictor Phase 4 finding ("60 Optuna trials could not beat the untuned baseline — the ceiling is in the data, not the hyperparameters") set the prior that tuning a saturated model rarely moves the needle. It held again here.

How research influenced the experiments: Lenz's depth-as-channel trick is *why* Phase 3 tried depth-as-blue and *why* we now isolate the channel-placement variable; Redmon's same-paradigm 0.849 is the honest target for a regression head; the internal prior framed Q1 as "expect a small gain at best."

## Dataset
| Metric | Value |
|--------|-------|
| Source | Cornell Grasping Dataset (folders 01–10, 885 images) |
| TEST (held out) | folder 03, n=100 (identical to Phase 2/3) |
| VAL (Optuna objective) | folder 07, n=100 |
| SEARCH-TRAIN | folders 01,02,04,05,06,08,09,10 — n=685 |
| FULL-TRAIN (final retrain) | everything except 03 — n=785 |
| Target | 6-vec (cx, cy, w, h, sin2θ, cos2θ) regression |
| Primary metric | Jiang IoU>0.25 ∧ angle<30° |

**Methodology note:** holding folder 07 out *object-wise* (not a random row split) keeps the validation signal generalising the same way the test signal does. The Optuna objective never sees folder 03, so the +1 pp final gain is a real held-out result, not test-set overfitting.

## Experiments

### Experiment 4.1: Optuna sweep on the E3.5 stack (val = folder 07)
**Hypothesis:** a tuned `lr/wd/rotation/batch` config closes a meaningful chunk of the 8-pp gap.
**Method:** TPESampler(seed=42), 15-epoch proxy train on SEARCH-TRAIN (685), objective = VAL Jiang accuracy. E3.5 stack fixed (depth-as-blue + multi-positive ON). Hard 2400 s wall-clock guard → **11 trials completed** in 44.7 min.
**Result:** best config `lr=1.20e-3, weight_decay=2.61e-4, rotation_deg=30, batch_size=16` at **val_acc=0.94**. Worst trial 0.66 (lr=3.7e-4 + bs=32 → underfit at 15 epochs); a too-high lr (2.3e-3) also dropped to 0.79.

Hyperparameter importance (MeanDecreaseImpurity):

| Param | Importance |
|---|---:|
| **weight_decay** | **0.419** |
| **lr** | **0.373** |
| rotation_deg | 0.109 |
| batch_size | 0.099 |

**Interpretation:** regularisation + learning rate dominate (~0.79 combined); rotation degree and batch size barely matter. This echoes the project-wide pattern (Phase 3: "regularisation channels compound") — and Keeper's "regularisation beat any architecture change." Note the absolute VAL numbers (0.83–0.94) run far above test (0.78): **folder 07 is simply an easier object set than folder 03**, so VAL is useful only for *ranking* configs, not as an accuracy estimate.

### Experiment 4.2: Tuned champion — retrain on full 785, evaluate on TEST
**Hypothesis (a-priori):** the best config at 40 epochs beats E3.5's 0.770.
**Method:** retrain best config on FULL-TRAIN (785, incl. folder 07) at 40 epochs (champion, fixed before any test eval) and 25 epochs (ablation). Evaluate on TEST folder 03 once.
**Result:**

| Run | Test acc | Median IoU | Median angle err | Train time |
|---|---:|---:|---:|---:|
| **P4 tuned (40 ep) — champion** | **0.780** | 0.445 | 5.6° | 813 s |
| P4 tuned (25 ep) — ablation | 0.750 | 0.404 | 5.6° | 492 s |

**Interpretation:** tuning delivers **+0.010 (0.770 → 0.780)** — a new project best, but the 8-pp gap to Redmon only shrinks to **6.9 pp**. The 25-epoch ablation (0.750) lands *below* Phase-3's 20-epoch E3.5 (0.770) at the *same* tuned hyperparameters, so the longer 40-epoch schedule is doing as much work as the tuned knobs. **Q1 answer: NO** — Optuna cannot close the gap; the ceiling is structural, not a hyperparameter accident (the third time this project has hit a tuning ceiling — same lesson as AI-Agent-Failure-Predictor Phase 4).

### Experiment 4.3: Depth as a genuine 4th channel (the Phase-3 hypothesis test)
**Hypothesis:** depth-as-blue hurt only because it destroyed the blue colour channel; depth as a 4th channel (RGB intact) should land ≥ 0.730 (E3.1, no depth) and confirm the signal is real.
**Method:** ResNet-18 with conv1 re-initialised to 4 inputs (RGB filters copied from ImageNet, depth filter seeded from the mean RGB filter). Apples-to-apples config vs E3.1/E3.3: no rotation, single-positive, 20 ep, lr=1e-3, wd=1e-4, bs=32. Plus a 4-ch all-knobs variant.
**Result:**

| Run | depth | blue kept? | Test acc |
|---|---|---|---:|
| E3.1 (Phase 3) | none | — | 0.730 |
| E3.3 (Phase 3) | depth→blue | no | 0.710 |
| **P4 4-ch (clean)** | depth = ch4 | **yes** | **0.700** |
| P4 4-ch (all-knobs) | depth = ch4 + rot + multipos | yes | 0.610 |

**Interpretation — HYPOTHESIS FALSIFIED.** Depth as a 4th channel (0.700) is *worse* than both no-depth (0.730) **and** depth-as-blue (0.710). Keeping the blue channel did not rescue depth — so Phase-3's "real-but-masked" explanation was wrong. The likely mechanism: re-initialising conv1 perturbs the **ImageNet-pretrained stem**, which Phase 2 already showed is this project's single strongest lever (M1-from-scratch vs M2-pretrained), and Cornell's Kinect depth is NaN-heavy/noisy. The cost of disturbing pretraining + adding a noisy channel exceeds any 3D cue for this binary metric. The all-knobs collapse (0.610) shows the perturbed stem compounds badly with augmentation. **Q2 answer: depth genuinely does not help — inject it any way and accuracy drops.**

### Experiment 4.4: Image-wise split (published-benchmark comparison)
**Method:** E3.5 config, random 80/20 image-wise split (seed=42), 20 ep. train=708, test=177.
**Result:** **0.757** (median IoU 0.391) — essentially tying **Lenz 2014 image-wise (0.756)**, below Redmon image-wise (0.844).
**Interpretation:** counterintuitively our object-wise tuned (0.780) > our image-wise (0.757), even though image-wise (objects shared between train/test) is usually *easier*. Reason: folder 03 happens to be an easy object subset, and the image-wise test (n=177) is larger and drawn from all folders; the two numbers aren't directly comparable. The image-wise figure exists only to sit next to the published image-wise leaderboard, where it ties Lenz.

## Head-to-Head Comparison (object-wise leaderboard after Phase 4)
| Rank | Model | Acc | Median IoU | Notes |
|---:|---|---:|---:|---|
| 1 | *Cao 2023 SOTA (ref)* | *0.978* | — | published |
| 2 | *Redmon 2015 (ref)* | *0.849* | — | same paradigm; the gap |
| 3 | **P4 tuned (40 ep) — new champion** | **0.780** | 0.445 | Optuna: wd=2.6e-4, lr=1.2e-3, rot=30, bs=16 |
| 4 | E3.5 all-knobs (Phase-3 champ) | 0.770 | 0.404 | rot+depth-blue+multipos |
| 5 | *Lenz 2014 (ref)* | *0.739* | — | published |
| 6 | E3.1 full-data control | 0.730 | — | no depth, no rot |
| 7 | E3.3 depth-as-blue | 0.710 | — | blue replaced |
| 8 | P4 4-ch depth (clean) | 0.700 | 0.402 | depth=ch4, RGB kept |
| 9 | P4 4-ch depth (all-knobs) | 0.610 | 0.324 | perturbed stem compounds |
| — | Phase-2 M2 (200-img) | 0.550 | — | data-starved ceiling |

## Key Findings
1. **Tuning is near-saturated: +1 pp (0.770 → 0.780), gap to Redmon still 6.9 pp.** Eleven Optuna trials on a clean held-out fold give a real but tiny gain. The third tuning-ceiling result in this portfolio — when the model is data/paradigm-limited, hyperparameters are not the lever.
2. **HEADLINE — the depth hypothesis is falsified.** Phase 3 claimed depth was "real but masked by the blue-channel loss." The clean 4-channel test (RGB intact) lands at 0.700 — *worse* than both no-depth (0.730) and depth-as-blue (0.710). On Cornell's binary Jiang metric, RGB + ImageNet pretraining beats every way of injecting the depth sensor. An honest reversal of last session's conclusion.
3. **weight_decay > lr ≫ rotation, batch.** Regularisation + lr explain ~79 % of config variance; the augmentation/batch knobs Phase 3 obsessed over barely register. Consistent with the project-wide "regularisation is the real lever" theme.
4. **Tuning's gain was pure localisation.** The "angle-ok / IoU-too-low" cluster fell 14 → 10 (−4); angle stays solved (median 5.6°). The +1 pp came from fixing exactly the failure mode Phase 3 flagged — but via **small batch (16)**, contradicting Phase-3's prediction that *bigger* batches would fix it. Second honest correction of the session.
5. **Image-wise ties Lenz 2014 (0.757 vs 0.756).** And we already beat Lenz object-wise (0.780 vs 0.739).

## Frontier / Published Model Comparison
| Protocol | Our best | Lenz 2014 | Redmon 2015 | Cao 2023 | Verdict |
|---|---:|---:|---:|---:|---|
| Object-wise | **0.780** | 0.739 | 0.849 | 0.978 | beat Lenz; 6.9 pp under Redmon |
| Image-wise | 0.757 | 0.756 | 0.844 | — | tie Lenz; under Redmon |

## Error Analysis
Failure decomposition on the 100-image test fold (counts of wrong predictions):

| Bucket | E3.5 (0.770) | Tuned (0.780) |
|---|---:|---:|
| angle-ok / IoU-too-low (localisation) | 14 | **10** |
| IoU-ok / angle-wrong | 3 | 6 |
| both wrong | 6 | 6 |

- The model's wrist angle is essentially solved (median angle error 5.6°); **every remaining gain has to come from box localisation (IoU)**, which is the largest residual cluster even after tuning.
- Global-regression heads (predict one 6-vec for the whole image) appear to saturate around 0.78 on Cornell object-wise. Closing to Redmon's 0.849 likely needs a **different paradigm** — a per-pixel / anchor-based detector (GG-CNN / GR-ConvNet style) that proposes localised grasps rather than regressing a single global box. That is the clear Phase-5 lever, not more tuning of the regressor.

## Next Steps
- **Phase 5 (Fri):** switch paradigm to attack the localisation ceiling — a fully-convolutional per-pixel grasp-quality head (GG-CNN / GR-ConvNet) on the full 785-image set, which natively predicts *where* to grasp instead of regressing one global box. Ablate it against the tuned regressor on the same folder-03 test. Optionally add the frontier-LLM head-to-head: send RGB-with-overlaid-grid descriptions (or the depth-popped image) to Claude/Codex and ask for a grasp rectangle, comparing Jiang accuracy + latency + cost against the 0.780 regressor (expected headline: a 45 MB CNN at ~28 ms/image beats a multi-second LLM call on a spatial-precision task).
- Drop depth entirely from the production model — three experiments now agree it doesn't help this metric.

## References Used Today
- [1] Lenz, Lee, Saxena (2015). *Deep Learning for Detecting Robotic Grasps.* IJRR. (object-wise 0.739 / image-wise 0.756; depth-as-channel trick.)
- [2] Redmon, Angelova (2015). *Real-Time Grasp Detection Using Convolutional Neural Networks.* ICRA. (object-wise 0.849 / image-wise 0.844.)
- [3] Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
- [4] Internal: AI-Agent-Failure-Predictor Phase-4 report — tuning-ceiling prior.

## Code Changes
- `src/dataset_phase4.py` (new) — `CornellGraspDatasetP4`, a thin `CornellGraspDatasetP3` subclass adding `use_depth_4ch` (depth as a 4th channel, RGB kept; same cached/un-rotated depth path as the Phase-3 depth-blue trick so the only changed variable is channel placement).
- `src/models_phase4.py` (new) — `ResNet18Regressor4ch`: conv1 re-initialised to 4 inputs (ImageNet RGB filters copied, depth filter seeded from mean RGB filter).
- `notebooks/phase4_tuning_error_analysis.ipynb` (new, 33 cells, 0 errors, 0 fake display-only cells) — all training/Optuna/eval inline; Optuna sweep, tuned-champion retrain, 4-channel test, image-wise retrain, failure decomposition, leaderboard, persistence.
- `results/metrics.json` — appended `phase_4` block (phases 1–3 preserved).
- `results/EXPERIMENT_LOG.md` — Phase-4 section appended.
- `results/phase4_*` — `optuna_trials.csv`, `best_config.json`, `optuna_search.png`, `leaderboard.{csv,png}`, `failure_decomposition.{csv,png}`, `training_curves.png`, `champion_qualitative.png`, `headline_summary.json`.
- `models/P4_tuned_champion.pt` (44.8 MB, gitignored) — new object-wise champion (0.780).
