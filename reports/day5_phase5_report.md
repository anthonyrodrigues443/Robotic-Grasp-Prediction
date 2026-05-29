# Phase 5: Per-Pixel Paradigm Switch + Frontier-LLM Head-to-Head — Robotic Grasp Prediction
**Date:** 2026-05-29
**Session:** 5 of 7

## Objective
Phase 4 ended with a hypothesis, not a result: *"Global-regression heads saturate ~0.78 on Cornell object-wise. Closing to Redmon's 0.849 likely needs a different paradigm — a per-pixel grasp-quality detector (GG-CNN / GR-ConvNet) that proposes localised grasps rather than regressing one global box."* Phase 5 builds that detector properly and tests the claim head-on.

Three pre-registered questions, all on the **Jiang rectangle metric** (correct iff some GT positive has IoU > 0.25 **and** angle error < 30°), object-wise, test = folder 03:

1. **Does the per-pixel paradigm beat the 0.78 global regressor?** Isolated cleanly: a `ResNet18-FCN` that shares the *exact same ImageNet-pretrained backbone* as the global regressor and changes **only the output head** (per-pixel maps vs a global 6-vec).
2. **Was Phase 2's GG-CNN 0.0 just data starvation?** Re-run the *identical* `GGCNNTiny` on 4× the data (785 imgs) with proper **multi-grasp** targets painted from *all* positives.
3. **Frontier-LLM head-to-head:** can Claude Opus/Haiku and Codex GPT-5.5 localise a grasp from the raw image, and how do accuracy / latency / cost compare to the CNN?

## Research & References
1. **Morrison, Corke & Leitner (2018), *Closing the Loop for Robotic Grasping* (RSS)** — GG-CNN: per-pixel quality + cos2θ/sin2θ + width maps, supervised from *every* labelled grasp. This is the multi-grasp target representation Phase 2 lacked (it painted a single blob). — https://arxiv.org/abs/1804.05172
2. **Kumra, Joshi & Sahin (2020), *Antipodal Robotic Grasping using GR-ConvNet* (IROS)** — residual generative grasp CNN, SOTA-adjacent on Cornell. `GRConvNetLite` is a shrunk version (conv-down → 5 residual blocks → transposed-conv up + skip). — https://arxiv.org/abs/1909.04810
3. **Redmon & Angelova (2015), *Real-Time Grasp Detection Using CNNs* (ICRA)** — the same-paradigm global-regression ceiling (object-wise 0.849) this project has chased since Phase 3.
4. **Internal carry-over (Phase 2/4):** ImageNet pretraining is this project's single strongest lever (M1-scratch ≪ M2-pretrained); depth *hurts* the binary Jiang metric (3 experiments agree). So all Phase-5 models are **RGB-only**, and the paradigm test puts a per-pixel head on the *pretrained* backbone.

How research shaped the experiments: Morrison's multi-grasp target is *why* we rebuild GG-CNN with all-positives maps; Kumra's residual net is the "better architecture" control; the internal prior is *why* the headline experiment isolates the paradigm by holding pretraining constant.

## Dataset
| Metric | Value |
|--------|-------|
| Source | Cornell Grasping Dataset (folders 01–10, 885 images) |
| TEST (held out) | folder 03, n=100 (identical to Phases 2–4) |
| TRAIN | folders 01,02,04–10 — n=785 |
| Positives/image | median 5, mean 5.8, max 25 |
| Per-pixel target | (quality, cos2θ, sin2θ, width) maps painted from **all** positives, 112² |
| Primary metric | Jiang IoU>0.25 ∧ angle<30° (object-wise) — same as Phases 1–4 |

**Engineering note:** target maps are painted once per dataset and cached; horizontal-flip augmentation is applied on the cached tensors (spatial flip + sin2θ sign negation, which is exact up to a 1-px registration offset on the smooth Gaussian blobs). This turned each epoch from painting-bound (~22 s) to compute-bound (~6 s), a ~3.7× speedup that made the 7-training session feasible in one notebook run.

## Experiments

### Experiment 5.0: Global-regression control (anchor)
**Method:** ResNet-18 (ImageNet-pretrained), global 6-vec head — the Phase-2/3/4 paradigm — trained on the *same* RGB + hflip data the per-pixel models see, Phase-4 tuned hyperparameters (lr=1.2e-3, wd=2.6e-4), 35 epochs.
**Result:** **0.690** acc, median IoU 0.417, median angle 8.0°. (The project record is the full Phase-4 stack at 0.780; 0.690 is the clean RGB-only anchor the per-pixel models are matched against.)

### Experiment 5.1: GG-CNN-v2 — rescue the Phase-2 0.0
**Hypothesis:** Phase 2's 0.0 was data starvation + single-blob targets; full data + multi-grasp maps should rescue it.
**Method:** identical `GGCNNTiny` (119K params), 785 imgs, multi-grasp targets, 35 epochs.
**Result:** **0.000** (median angle 40.6° ≈ random). **Hypothesis falsified** — 4× the data and proper targets did not move it off zero.

### Experiment 5.2: GR-ConvNet-lite (from scratch, residual + skip)
**Hypothesis:** a *better* per-pixel architecture closes the gap.
**Method:** `GRConvNetLite` (1.8M params, 5 residual blocks + skip), from scratch, 35 epochs.
**Result:** **0.000** acc (median angle 21.9° — it learned *some* orientation, but never localises). Architecture alone does not rescue the paradigm without pretraining.

### Experiment 5.3: ResNet18-FCN — the clean paradigm test
**Hypothesis (the Phase-4 prediction):** a per-pixel head on the pretrained backbone beats the global regressor and approaches Redmon.
**Method:** ResNet-18 pretrained encoder + FPN-style decoder → 112² per-pixel maps. **Same backbone + pretraining as E5.0; only the output head differs.** 30 epochs.
**Result:** **0.420** acc, median IoU 0.251 — **27 pp BELOW the global regressor (0.690)**. **Phase-4 hypothesis falsified.** With everything held equal, the per-pixel paradigm *loses*.

## Head-to-Head Comparison (object-wise, folder 03)
| Rank | Model | Paradigm | Pretrained | Accuracy | Median IoU | Params |
|---:|---|---|---|---:|---:|---:|
| — | *Redmon 2015 (ref)* | global-reg | — | *0.849* | — | — |
| — | *P4 tuned champion (ref)* | global-reg | yes | *0.780* | *0.445* | 11.2M |
| 1 | **GlobalReg_ResNet18 (control)** | global-reg | yes | **0.690** | 0.417 | 11.2M |
| 2 | ResNet18-FCN | per-pixel | yes | 0.420 | 0.251 | 11.3M |
| 3 | GRConvNet-lite | per-pixel | no | 0.000 | 0.000 | 1.8M |
| 3 | GGCNN-v2 | per-pixel | no | 0.000 | 0.000 | 119K |

## Key Findings
1. **HEADLINE — the per-pixel paradigm loses to global box regression on Cornell, with pretraining held constant.** Same ResNet-18 backbone, same ImageNet weights, only the output head changes: per-pixel maps 0.420 vs global 6-vec 0.690. The Phase-4 plan ("switch paradigm to break the localisation ceiling") is falsified — switching paradigm *lowered* the ceiling by 27 pp.
2. **The Phase-2 GG-CNN 0.0 was not (only) data starvation.** Full data + multi-grasp targets still scored 0.000. Both from-scratch per-pixel nets (GG-CNN-v2, GR-ConvNet-lite) are stuck at zero. The paradigm needs pretraining the from-scratch nets can't have.
3. **Pretraining is the dominant lever — the 4th time this project has landed there.** Ablation: ResNet18-FCN from scratch 0.420 → 0.190 (−0.23), the single biggest effect. The deeper problem: a per-pixel **decoder** has no ImageNet-pretrained equivalent, so the paradigm forfeits this project's strongest lever in exactly the part of the network that does the localisation.
4. **Counterintuitive: per-pixel localises *worse*, not better.** Failure decomposition — ResNet18-FCN has **39** "angle-ok / IoU-too-low" localisation failures vs the global head's **8**. The paradigm marketed as "predict *where* to grasp" is the one that can't localise here, because the global regressor offloads localisation onto the pretrained backbone + a 2-parameter head, while the FCN must learn a spatial quality map in an un-pretrained decoder from 785 images.
5. **Multi-grasp and skip help at the margin but can't rescue the paradigm.** Multi-grasp vs single-blob: 0.000 → 0.040; skip connection: +0.060. Real but tiny next to the −0.23 pretraining lever.

## Frontier Model Comparison (n=40 folder-03 sample, identical Jiang scorer)
| Model | Accuracy | Median IoU | Latency/img | Cost/1k | Verdict |
|---|---:|---:|---:|---:|---|
| Custom: Global ResNet18 | 0.75 | 0.395 | 0.017 s | $0.00 | fastest + free |
| codex / gpt-5.5 | 0.75 | 0.419 | 41.0 s | $50.00 | ties on acc, best IoU |
| Custom: ResNet18-FCN | 0.50 | 0.271 | 0.029 s | $0.00 | the losing paradigm |
| claude / opus | 0.25 | 0.000 | 16.6 s | $22.50 | can't localise |
| claude / haiku | 0.15 | 0.000 | 18.1 s | $1.50 | can't localise |

**GPT-5.5 genuinely ties the custom CNN on grasp accuracy (0.75) and edges it on median IoU (0.419 vs 0.395)** — a frontier model is competitive at grasp localisation from raw pixels. But it is ~2,400× slower (41 s vs 17 ms) and ~50,000× more expensive. Both Claude models fail the spatial task outright (median IoU 0.000): they reliably get the gripper *angle* but predict grasp centres in the wrong place. Latency includes CLI startup; the cost math reflects equivalent direct-API usage. The custom n=40 numbers (0.75 / 0.50) run above the full-folder-03 figures (0.69 / 0.42) due to subset sampling variance.

## Error Analysis
- The residual ceiling is **localisation** for *every* model, but the per-pixel models make it dramatically worse (39 vs 8 IoU-too-low failures). This is the mechanistic explanation for why the paradigm switch backfired.
- From-scratch per-pixel nets converge on orientation (GR-ConvNet median angle 21.9°) long before localisation — they paint roughly-right grasp *angles* across the image but never concentrate quality on the object, so argmax lands off-target → IoU 0.
- Output-resolution sanity: GT-roundtrip ceiling is 0.90 / 0.99 / 1.00 at 56² / 112² / 218², so the 112² map resolution is not the binding constraint — the model, not the decoder, is the limit.

## Next Steps
- **Phase 6 (Sat):** production pipeline + interactive UI around the **global regressor** (the paradigm that actually works here) — `src/train.py`, `src/predict.py`, `src/evaluate.py`, model card, and a Streamlit app that takes an image and overlays the predicted grasp + (optionally) the runner-up frontier-LLM grasp for contrast. Drop the per-pixel models from production; keep them documented as the negative result.
- A fair future test of the paradigm would need a *pretrained* per-pixel decoder (e.g., a segmentation-pretrained backbone+decoder) or a much larger grasp dataset (Jacquard, 50k+) where from-scratch per-pixel nets are normally trained — Cornell's 785 images structurally favour the pretrained-backbone + simple-head global regressor.

## References Used Today
- [1] Morrison, Corke, Leitner (2018). *Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach.* RSS. https://arxiv.org/abs/1804.05172
- [2] Kumra, Joshi, Sahin (2020). *Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network.* IROS. https://arxiv.org/abs/1909.04810
- [3] Redmon, Angelova (2015). *Real-Time Grasp Detection Using Convolutional Neural Networks.* ICRA.
- [4] Internal: Robotic-Grasp Phase 2/4 reports — ImageNet-pretraining lever; depth-hurts result.

## Code Changes
- `src/dataset_phase5.py` (new) — `GraspMapDataset` (precomputed, cached multi-grasp target maps + on-the-fly hflip), `paint_target_maps` (all-positives quality/cos2θ/sin2θ/width painter, with `multi_grasp` + `sigma_frac` ablation flags), `decode_grasp_map` (smoothed-argmax → GraspRect, scored by the same Jiang metric as the global head).
- `src/models_phase5.py` (new) — `GRConvNetLite` (residual + optional skip, from scratch, 112² out) and `ResNet18FCN` (pretrained encoder + FPN decoder, 112² out, `pretrained` ablation flag).
- `src/llm_grasp_eval.py` (new) — frontier-LLM grasp harness: Claude (image-path-in-prompt) + Codex (`-i` image) CLI calls, defensive parsing, append-only idempotent cache, 2026 cost model. Mirrors the Fraud-Detection `mark_phase5` pattern.
- `notebooks/phase5_per_pixel_and_llm.ipynb` (new, 38 cells, 0 errors, 0 fake display-only cells) — all training/eval/ablation/LLM-scoring inline.
- `results/metrics.json` — appended `phase_5` (phases 1–4 preserved).
- `results/EXPERIMENT_LOG.md` — Phase-5 section.
- `results/phase5_{target_maps, leaderboard, training_curves, ablation, failure_decomposition, champion_qualitative, llm_comparison}.png`, `results/phase5_{leaderboard, ablation, failure_decomposition}.csv`, `results/llm_vs_custom.csv`, `results/phase5_headline_summary.json`, `results/phase5_llm_cache/`.
- `models/P5_*.pt` (gitignored) — global control, GG-CNN-v2, GR-ConvNet-lite, ResNet18-FCN, and the best-per-pixel champion.
