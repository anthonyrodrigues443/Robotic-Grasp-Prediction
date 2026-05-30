# Phase 6: Production Pipeline + Streamlit UI — Robotic Grasp Prediction
**Date:** 2026-05-30
**Session:** 6 of 7

## Objective
Turn five phases of research notebooks into a clean, re-runnable production
system: a config-driven train/evaluate/predict pipeline around the champion, a
polished Streamlit demo, and an honest model card — and, in doing so, verify
that the *saved* champion artifact actually reproduces the headline number from
a cold start (not just inside the training session that produced it).

## Research & References
1. **Hugging Face / Google "Model Cards" (Mitchell et al., 2019)** — the
   intended-use / training-data / metrics / limitations structure adopted for
   `models/model_card.md`, including an explicit out-of-scope and safety section
   (relevant for a robotics model that could drive hardware).
2. **Jiang, Moseson & Saxena (2011), rectangle metric** — kept as the single
   evaluation contract end-to-end so the production number is directly
   comparable to Phases 1–5 and the published Cornell leaderboard.
3. **Train/serve skew guidance (Google ML "Rules of ML", #29 — "make it easy to
   serve the exact features used at training")** — motivated routing *both*
   training and inference through one `data_pipeline.preprocess_image`, so the
   served preprocessing is provably identical to the trained one.

How this shaped the session: rather than write a fresh inference path (the usual
source of a silent serve-time accuracy drop), the production layer is a thin
wrapper that reuses the exact `dataset` encode/decode and Phase-3 augmentation
recipe. That decision is what let me prove the artifact reproduces the harness
number byte-for-byte across CPU and MPS.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 885 (Cornell folders 01–10) |
| Split | object-wise; folder 03 held out as TEST (n=100), 785 train |
| Target | one oriented grasp rect per image → 6-vec `(cx,cy,w,h,sin2θ,cos2θ)` |
| Champion | `models/P4_tuned_champion.pt` (ResNet-18 regressor, 11.18 M params) |

## What was built
| Component | File | Purpose |
|-----------|------|---------|
| Central config | `config/config.yaml` | paths, champion recipe, Jiang thresholds, reference numbers |
| Data pipeline | `src/data_pipeline.py` | config loader, Cornell loader, object-wise split, **single shared `preprocess_image`** |
| Inference | `src/predict.py` | `GraspPredictor` + `GraspPrediction`; CLI; orientation-norm confidence heuristic |
| Evaluation | `src/evaluate.py` | reproduces folder-03 Jiang accuracy + steady-state latency benchmark; CLI |
| Training | `src/train.py` | config-driven champion retrain (rotation aug + multi-positive loss); `--smoke` |
| UI | `app.py` | Streamlit: example/upload → grasp overlay + metrics + sidebar model card + LLM head-to-head |
| Render util | `src/viz.py` | shared grasp-rectangle drawing (UI + screenshot use the same path) |
| Model card | `models/model_card.md` | HF/Google-format card with explicit limitations + safety |
| Prod tests | `tests/test_production.py` | 3 smoke tests; full folder-03 eval asserted == reference |

## Experiments / Verification

### 6.1 — Does the saved artifact reproduce the reported number?
**Hypothesis:** the persisted `P4_tuned_champion.pt` scores the Phase-4 headline
0.780 on folder-03 from a cold load.
**Method:** load the `state_dict` into a fresh `ResNet18Regressor`, run the new
production `evaluate.py` *and* the original research `trainer.evaluate`, on both
MPS and CPU.
**Result:**

| Harness | Device | Jiang acc | Median IoU | Median angle err |
|---------|--------|----------:|-----------:|-----------------:|
| `src/evaluate.py` (new) | MPS | 0.770 | 0.396 | 5.5° |
| `trainer.evaluate` (research) | MPS | 0.770 | 0.396 | 5.5° |
| `trainer.evaluate` (research) | CPU | 0.770 | 0.396 | — |
| *Phase-4 report (in-session)* | MPS | *0.780* | *0.445* | *5.6°* |

**Interpretation:** the new production pipeline and the original harness agree
**exactly** (0.770) and are **device-independent** — so the serving path is
faithful, no train/serve skew. The 1-pp gap to the Phase-4 in-session 0.780 is a
single test image: the logged 0.780 was the best in-memory eval during the
session; the persisted checkpoint is 0.770. I took 0.770 as the honest deployed
number everywhere (config, model card, UI, tests) rather than quietly shipping
the rosier figure.

### 6.2 — Steady-state inference latency (warmup excluded)
**Method:** 50-run timed loop after 5 warmup forwards, per device.
**Result:** MPS **7.1 ms/img (141 img/s)**; CPU 327.7 ms/img (3 img/s).
**Interpretation:** at 7 ms the 45 MB CNN is ~**5,800× faster** than the
Phase-5 GPT-5.5 grasp call (≈41 s/img) at zero marginal cost — the production
framing of the Phase-5 headline.

## Key Findings
1. **The shipped artifact is 0.770, not 0.780 — and now it's *provably* 0.770.**
   The same number falls out of two independent harnesses on two devices, and a
   unit test pins it. "Reproducible from the saved file" is a stronger claim than
   any single notebook cell.
2. **One preprocessing function kills train/serve skew.** Reusing
   `dataset.encode_target`/`decode_prediction` and the Phase-3 augmentation in
   the production wrapper is *why* 6.1 matched exactly. The most common
   productionisation bug (a subtly different resize/normalise at serve time)
   can't occur when there's one code path.
3. **A regressor has no honest "probability," so don't fake one.** The confidence
   field is documented as the `(sin2θ, cos2θ)` norm heuristic, not a calibrated
   score — surfaced as such in both the UI caption and the model card.

## Frontier Model Comparison (carried from Phase 5, shown in the UI)
| Model | Jiang acc | Median IoU | Latency | Cost/1k |
|-------|----------:|-----------:|--------:|--------:|
| **Custom Global ResNet-18** | 0.75 | 0.395 | **0.017 s** | **$0** |
| codex/gpt-5.5 | 0.75 | 0.419 | 41.0 s | $50 |
| claude/opus | 0.25 | 0.000 | 16.6 s | $22.5 |
| claude/haiku | 0.15 | 0.000 | 18.1 s | $1.5 |
*(n=40 stratified subset; Claude models fail spatial localisation — right angle, wrong place.)*

## Error Analysis
Unchanged from Phase 4/5 — the residual failures are the "angle-ok / IoU-too-low"
localisation cluster (the global-regression ceiling). Production work did not
touch the model weights, so the failure profile is identical; it's documented in
the model card's limitations (RGB-only, single-object, no clutter/collision).

## Next Steps
- **Phase 7 (Sun 2026-05-31):** full pytest coverage (data_pipeline, viz,
  train smoke, predict/evaluate), README rewrite with architecture diagram +
  UI screenshot + the all-phases results table, and final polish. Optionally run
  the full `src/train.py` 40-epoch retrain once to land `champion_reproduced.pt`
  next to the shipped checkpoint and confirm the recipe regenerates ~0.77.

## References Used Today
- [1] Mitchell et al. (2019), *Model Cards for Model Reporting* — https://arxiv.org/abs/1810.03993
- [2] Jiang, Moseson, Saxena (2011), *Efficient Grasping from RGBD Images* — Cornell rectangle metric
- [3] Google, *Rules of Machine Learning* (Rule #29, training/serving skew) — https://developers.google.com/machine-learning/guides/rules-of-ml

## Code Changes
- **New:** `config/config.yaml`, `src/data_pipeline.py`, `src/predict.py`,
  `src/evaluate.py`, `src/train.py`, `src/viz.py`, `app.py`,
  `models/model_card.md`, `tests/test_production.py`,
  `results/phase6_evaluation.json`, `results/ui_screenshot.png`.
- **Modified:** `requirements.txt` (+streamlit, +PyYAML).
- **Verification:** full suite 36 passing (33 prior + 3 production); Streamlit app
  executes top-to-bottom with 0 exceptions via `AppTest`.
