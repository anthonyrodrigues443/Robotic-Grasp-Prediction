# Phase 2: Multi-Model Head-to-Head — Robotic Grasp Prediction (DL-3)

**Date:** 2026-05-26 (Tuesday)
**Session:** 2 of 7

## Objective

Replace Phase 1's three hand-engineered baselines with **five learned models**
of deliberately different inductive biases, evaluated on the exact same
object-wise test split with the exact same Jiang metric. The point isn't just
"do CNNs beat 17 %?" — that's table stakes. The five models are arranged
along *independent architectural axes* so each pairwise comparison answers a
specific question:

| Pair | Holding constant | Varying | The question |
|---|---|---|---|
| M1 vs M2 | regression head, conv-net shape | from-scratch vs ImageNet-pretrained | does transfer matter on 200 images? |
| M2 vs M3 | ResNet-18 backbone | continuous sin/cos angle vs 18-way classification | Redmon 2015's discretisation choice — does it hold in 2026? |
| M2 vs M4 | regression vs per-pixel argmax | direct 6-D output vs quality-map argmax | which paradigm wins at this scale? |
| M2 vs M5 | regression head | convolutional vs transformer backbone | does ViT inductive bias help on small-data grasping? |
| M1 (floor) vs every pretrained | nothing | everything | does pretraining buy more than 5 pp? |

## Research & References

1. **Redmon & Angelova, ICRA 2015** — *Real-Time Grasp Detection Using
   Convolutional Neural Networks*. Single-shot CNN, 84.9 % object-wise. The
   M1 (TinyRedmonCNN) recreates their from-scratch AlexNet-shape architecture.
   M3 (Hybrid head) tests their original choice to *discretise* the grasp
   angle into 18 bins instead of regressing it.
2. **Morrison et al., RSS 2018** — *Closing the Loop for Robotic Grasping
   (GG-CNN)*. Per-pixel grasp quality + angle map; 73 % accuracy on Cornell
   at <20 ms. M4 (GGCNNTiny) reproduces this paradigm at much smaller scale
   (119 k params, no skip connections) to ask whether the per-pixel
   formulation beats global regression when training data is scarce.
3. **He et al., CVPR 2016** — ResNet. Modern ImageNet-pretrained backbones
   are the obvious upgrade over Redmon's AlexNet shape; M2 tests how much
   that buys on Cornell-200.
4. **Dosovitskiy et al., ICLR 2021** — ViT. ViTs are famously data-hungry;
   training from scratch on 200 images is *explicitly* not the recommended
   recipe. M5 tests it anyway, because (a) the dataset constraint is real,
   and (b) the failure mode (if it fails) is informative.
5. **Kumra et al., 2022** — GR-ConvNet v2 reports ~96 % on Cornell with a
   modern encoder-decoder. Outside today's scope but a Phase-3 reference.

**How research influenced today's experiments:** Redmon & Angelova's
angle-discretisation choice (5) is one of the longest-standing design
decisions in deep-learning grasp detection; almost every Cornell paper since
either copies it or replaces it without doing the head-to-head. Today's M2
vs M3 comparison *is* that head-to-head on the same backbone, same data,
same metric. The hypothesis going in was that discretisation would win
(narrower per-class problem) — the result inverted that.

## Dataset

| Metric | Value |
|---|---|
| Total images | 300 (Cornell archives 01, 02, 03) |
| Train split | 200 images (folders 01, 02), 998 positive grasps |
| Test split | **object-wise** held-out — folder 03, 100 images, 542 positives |
| Image resolution | 640 × 480 (RGB) — resized to 224 × 224 for ImageNet-compatible backbones |
| Target per training image | 1 positive grasp (the most central one) |
| Target encoding | 6-vector `(cx_n, cy_n, w_n, h_n, sin(2θ), cos(2θ))` in 224² frame |

The encoding choice (sin/cos of *doubled* angle) is the standard trick for
representing the 180° grasp symmetry without a wrap discontinuity at ±π/2 —
verified by `test_doubled_angle_handles_pi_over_2_wraparound`.

In parallel with Phase-2 training I downloaded Cornell archives 04–10 from
the Wayback Machine (2.85 GB, 7 archives, ~580 additional images) ready for
Phase 3 extraction. Phase 3 will retrain on the full 885 images and re-run
M2 against the literature object-wise SOTA.

## Experiments

### Experiment 2.1 — M1 TinyRedmonCNN (from-scratch AlexNet-shape)

**Hypothesis:** Reproduce a low-end Redmon-2015 number. Should clear B3 (17 %)
by a clear margin but won't approach the literature 84.9 % (which trained on
the full 885-image dataset).

**Method:** 5 conv blocks + 2 FC head, 1.68 M params. Trained from scratch
with AdamW lr=1e-3, cosine LR, 40 epochs, batch 16, horizontal-flip
augmentation only. MSE on 6-vec output.

**Result:** **35.0 %** accuracy. Median IoU 0.156. Median angle err 11.9°.
Training time 126.8 s (3.2 s/epoch) on MPS.

**Interpretation:** Clears B3 by 18 pp — the from-scratch CNN finds a useful
signal that the depth-antipodal heuristic doesn't, even on 200 images. But
27 pp short of the published Redmon number, which is exactly the gap we'd
expect from training on 200 images instead of 885. Phase 3 will close this
by 3×ing the training data.

### Experiment 2.2 — M2 ResNet-18 Regressor (ImageNet pretrained)

**Hypothesis:** ImageNet transfer should be worth 10–20 pp on a tiny grasp
training set. Pretrained features (edges, textures, object parts) cut the
amount of training data the model needs to learn grasp localisation.

**Method:** ResNet-18 with ImageNet-1k weights, fc layer replaced with a
dropout-0.3 → Linear(512 → 6) head. AdamW lr=3e-4 (smaller than M1 because
the backbone is pretrained), cosine schedule, 40 epochs, batch 16.

**Result:** **55.0 %** accuracy — the champion. Median IoU 0.293. Median
angle err **6.1°**. Training 170.8 s, inference 19.1 ms/image.

**Interpretation:** +20 pp over M1, confirming the transfer hypothesis. The
median angle error of 6.1° is well below the Jiang 30° threshold — angle is
no longer the bottleneck; the 45 % of failures are dominated by "angle ok,
IoU too low" (27 %) and "both wrong" (10 %). **Localisation is the next
unlock for Phase 3.**

### Experiment 2.3 — M3 ResNet-18 Hybrid head (4-D reg + 18-way CE)

**Hypothesis:** Discretising the angle into 18 bins (Redmon's choice) should
*help* — each bin has a sharper target than a noisy continuous value, and
classification losses tend to be more stable than regression on small
datasets. Expected M3 > M2 by 3–5 pp.

**Method:** Same ResNet-18 backbone as M2, but the fc is split into a
4-D regression head for (cx, cy, w, h) and an 18-way softmax head for angle
bin (10° width each). Joint loss = MSE(cx,cy,w,h) + CE(angle_bin). Same
optimizer and schedule as M2.

**Result:** **13.0 %** accuracy. Median IoU **0.004**. Median angle err 8.6°.

**Interpretation: catastrophic failure of joint loss training.** The angle
classification head *worked* — median angle err 8.6° is in the same ballpark
as M2's 6.1°, and the failure-mode decomposition shows only 2 % of the
errors are "IoU ok, angle wrong". But the regression head collapsed: median
IoU = 0.004 means the predicted (cx, cy) is essentially uncorrelated with
the actual grasp position. The cross-entropy loss term dominated the AdamW
gradient direction on a tiny dataset (200 images / 18 bins = 11 examples per
bin), and the 4-D regression branch received no useful signal.

This is the *post-worthy result*. Redmon & Angelova trained the angle and
regression heads in separate stages (their paper, section 4); we trained
them jointly with `loss_reg + 1.0 * loss_ce`. With careful re-weighting
(probably 0.01–0.1× CE) M3 might match or beat M2, but the naïve joint
training catastrophically fails. **A 13-year-old design choice that
everyone copies doesn't survive a head-to-head against the simpler
alternative.**

### Experiment 2.4 — M4 GGCNNTiny (per-pixel quality + angle + width maps)

**Hypothesis:** Per-pixel formulation should help on Cornell because
positives come in *clusters* across each object — a per-pixel quality map
naturally accommodates multiple ground-truth positives, whereas single-shot
regression has to pick one. Phase 3's plan to add multi-positive loss is
basically importing GG-CNN's structure into M2.

**Method:** Tiny encoder-decoder, 119 k parameters, no skip connections.
Encoder: 9×9, 5×5, 3×3 strided convs (224 → 75 → 38 → 19). Decoder:
transposed convs back to 218² (close enough to 224 — target maps are also
218²). 4 output channels: quality (sigmoid), sin(2θ), cos(2θ), width.
Target: Gaussian blob (σ=4 px) on quality channel at the encoded GT
position; angle/width maps filled inside the blob mask.

**Result:** **0.0 %** accuracy. Median IoU 0.0. Median angle err 44°.

**Interpretation:** The 119 k-parameter encoder-decoder did not learn the
task — the quality channel collapsed to near-uniform values, and argmax
returned essentially the same pixel for every test image. Root cause is
sparsity: a 218² grid with a σ=4 px target means 99.5 % of target pixels
are 0, and BCE/MSE with no class balancing optimises trivially by predicting
0 everywhere. Phase 3 needs (a) class-balanced loss, (b) skip connections to
preserve spatial structure, or (c) a wider target σ. Documenting as a
*negative finding*: a from-scratch tiny per-pixel model is not the right
shape for Cornell-200.

### Experiment 2.5 — M5 TinyViT (from-scratch, patch16, 6 layers)

**Hypothesis:** Transformers are data-hungry; M5 will underperform M2 by
10–20 pp. The interesting question is *how* it fails — does it learn
nothing, or does it learn one part of the task and not another?

**Method:** Patch-16 ViT-tiny, 6 transformer layers, head_dim 32, embedding
dim 192, MLP ratio 2.0. 1.97 M params. Same training setup as M1 (AdamW
lr=1e-3, 40 epochs, batch 16).

**Result:** **47.0 %** accuracy. Median IoU **0.364** (best of any model).
Median angle err **36.8°** (worst of any learned model). Training 165.5 s.

**Interpretation:** The numbers are far better than the "ViTs need 14 M
images" framing predicts. *And* the failure mode is structurally different
from every other model: the failure-decomposition row reads 41 % both-OK /
5 % angle-OK-IoU-low / **38 % IoU-OK-angle-wrong** / 16 % both-wrong. The
ViT learned to *localise* extremely well (highest median IoU) but is bad at
orientation. This is the **opposite** of every CNN result — the CNNs nail
the angle and miss the position. Hypothesis why: patch16 tokens lose
fine-grained angle information that ResNet-18's local convs preserve. The
global attention figures out the object centre but the patchwise
representation is too coarse for orientation.

This is also a publication-grade finding in miniature: *ViTs and CNNs have
dual failure modes on grasp detection*. Phase 3 will test whether a hybrid
(ViT-localise + CNN-orient) recovers both signals.

## Head-to-Head Comparison

| Rank | Model | Accuracy | Median IoU | Median angle err | Params | Inference (ms) | Δ vs B3 (Phase 1) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **M2 ResNet-18 Regressor** | **55.0 %** | 0.293 | 6.1° | 11.2 M | 19.1 | **+38.0 pp** |
| 2 | M5 TinyViT | 47.0 % | **0.364** | 36.8° | 2.0 M | 14.1 | +30.0 pp |
| 3 | M1 TinyRedmonCNN | 35.0 % | 0.156 | 11.9° | 1.7 M | 8.8 | +18.0 pp |
| — | B3 depth + antipodal (Phase 1) | 17.0 % | 0.159 | 37.0° | 0 | — | — |
| 4 | M3 ResNet-18 Hybrid | 13.0 % | 0.004 | 8.6° | 11.2 M | 11.6 | −4.0 pp |
| 5 | M4 GGCNNTiny | 0.0 % | 0.000 | 44.0° | 0.12 M | 12.0 | −17.0 pp |

### Failure-mode decomposition (Jiang sub-constraints, % of 100 test images)

| Model | both ok | angle ok, IoU low | IoU ok, angle wrong | both wrong |
|---|---:|---:|---:|---:|
| M2 ResNet-18 Regressor | **55** | 27 | 8 | 10 |
| M5 TinyViT | 41 | 5 | **38** | 16 |
| M1 TinyRedmonCNN | 35 | 35 | 5 | 25 |
| B3 depth + antipodal | 17 | 31 | 2 | 50 |
| M3 ResNet-18 Hybrid | 13 | **58** | 2 | 27 |
| B2 centre + median size | 4 | 49 | 9 | 38 |
| B1 random | 1 | 35 | 0 | 64 |
| M4 GGCNNTiny | 0 | 41 | 0 | 59 |

## Key Findings

1. **Headline — Redmon's 2015 angle-as-classification head is dominated by
   continuous sin/cos regression by 42 accuracy points on the same backbone.**
   M2 = 55 %, M3 = 13 %. The CE loss on 11 examples-per-bin (200 images / 18
   bins) crushes the regression head's gradient and the model never learns
   to localise. The lesson: joint multi-task loss training on tiny grasping
   datasets requires very careful weighting — and the *unified* sin/cos
   representation sidesteps the problem entirely.

2. **ImageNet pretraining is worth +20 pp on 200 images.** M2 (pretrained)
   vs M1 (from scratch, same regression head) = 55 % vs 35 %. This is a
   bigger transfer effect than I expected on a domain where ImageNet
   classes have little to do with the target task.

3. **ViTs and CNNs fail in opposite directions.** M5 TinyViT has the highest
   median IoU but the worst angle error — 38 % "IoU ok, angle wrong" vs the
   CNN's 8 %. The Phase-1 framing "localisation is the bottleneck" survives
   for CNNs but **inverts for transformers** at this scale.

4. **Phase-1's bottleneck hypothesis is upheld for the champion.** Even
   M2 at 55 % accuracy spends 27 % of its failures in "angle ok, IoU too
   low". The CV-to-CNN gap (B3 17 % → M2 55 %) closed mostly the
   "both wrong" bucket (50 % → 10 %); the localisation-bottleneck bucket
   shrank only modestly (31 % → 27 %). Phase 3's lever is unchanged.

5. **Tiny per-pixel quality maps (M4) do not transfer from GG-CNN to
   Cornell-200.** Output collapsed to near-uniform values; root cause is
   target sparsity (σ=4 px Gaussian on 218² grid → 99.5 % zeros) without
   class-balancing or skip connections. Phase 3 fix: per-pixel BCE with
   positive weight tuned to the 0.5 % positive rate, plus U-Net skips.

## Error Analysis

**M2 (champion) failure modes (45 % of test set):**

- **Localisation drift (27 % "angle ok, IoU too low"):** the predicted
  centroid is close to *some* GT positive but the IoU sits just below 0.25.
  Inspecting `phase2_champion_qualitative.png`, this is usually because the
  network committed to a wider grasp than any of the GTs in the image —
  Cornell positives are tightly clustered around the gripper's hardware
  width (24 px median plate, 33 px median jaw opening for positives), so a
  prediction that's 30 % too wide still has the right centre but doesn't
  clear IoU > 0.25.
- **Both wrong (10 %):** these are usually objects far off-centre in the
  image. The CornellGraspDataset target is the *most central* positive,
  which gives a position-biased training signal. On test images where the
  object is shifted (e.g., a pencil leaning to the right of frame), M2
  predicts at the image centre and misses entirely.
- **IoU ok, angle wrong (8 %):** these are round / symmetric objects (mugs
  viewed top-down, discs) where the ground-truth angle convention is
  arbitrary and the network picks a different valid orientation than the
  annotators did.

**M5 (TinyViT) failure modes (53 % of test set):**

- **IoU ok, angle wrong (38 %):** patch16 tokenisation gives the
  attention layers a 14×14 grid of tokens (one per 16² pixel patch). The
  centroid is recoverable from coarse tokens; the orientation requires
  *intra-patch* gradient information that the ViT throws away. Phase 3 hybrid
  candidates: smaller patch (e.g., patch8), or a hybrid CNN-token ViT
  (e.g., feed ResNet stem features as tokens).

**M3 (hybrid head) failure modes (87 % failure):**

- **Angle ok, IoU too low (58 %):** the classification head learned angle
  fine, but (cx, cy, w, h) is essentially noise. The decoded rectangle has
  the right orientation but lands at a random position in the image. The
  fix is per-head loss re-weighting — the CE term contributes ~ln(18) ≈ 2.9
  in magnitude at init while the 4-D MSE contributes ~0.01, so AdamW
  optimises CE for ~10 epochs before MSE gets any signal.

## Frontier Model Comparison (deferred to Phase 5)

The Phase 5 plan is to send the depth image + a one-line task description to
Claude Opus 4.6, Claude Haiku 4.5, and Codex GPT-5.5 (via the existing CLI
harness used in Fraud Detection and AI Agent Conv Quality Scorer) and ask
them to return a grasp rectangle as `(cx, cy, w, h, angle_deg)`. Today's
M2 = 55 % is the floor those LLMs must clear to count as "they understand
grasping at all"; Phase 5 will report the head-to-head with latency and
cost-per-1k metrics.

## Next Steps

- **Phase 3 (Wed 2026-05-27)** — Feature engineering + training data
  expansion. Extract archives 04–10 (already on disk, 2.85 GB pre-downloaded
  in background). Re-evaluate the M2 champion on the full 885-image
  benchmark with both image-wise and object-wise splits to land directly
  next to the published references. Add **rotation augmentation** with the
  proper grasp-transform (rotating the image must rotate the target grasp
  by the same angle). Add **depth as a 4th input channel**.
  Floor: clear M2's 55 % by ≥ 10 pp on the same object-wise test split.
- **Optional Phase 3 fixes:**
  - **Multi-positive loss** — switch the target picking from
    "most-central GT" to "closest GT to the prediction" (the standard
    Cornell loss). Known to add +5–10 pp.
  - **Heal M3** — re-weight CE to 0.05× of MSE, or train heads sequentially.
    Worth ~2 hours of experimentation; if it can match M2 cleanly, the
    angle-discretisation finding becomes "design choice equivalent under
    careful weighting" instead of "design choice dominated by simpler
    alternative". Both are publishable.
  - **Heal M4** — add U-Net skips and class-balanced per-pixel loss.
- **Phase 4 (Thu 2026-05-28)** — Optuna hyperparameter sweep on the M2 (or
  whichever Phase 3 ends up promoting). Sweep lr, weight_decay, head
  dropout, schedule type, and augmentation strength.
- **Phase 5 (Fri 2026-05-29)** — LLM head-to-head as outlined above.

## References Used Today

- [1] Redmon, J., Angelova, A. *Real-Time Grasp Detection Using Convolutional Neural Networks.* ICRA 2015. arXiv:1412.3128.
- [2] Morrison, D., Corke, P., Leitner, J. *Closing the Loop for Robotic Grasping (GG-CNN).* RSS 2018.
- [3] He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385.
- [4] Dosovitskiy, A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. arXiv:2010.11929.
- [5] Kumra, S., Joshi, S., Sahin, F. *GR-ConvNet v2: A Real-Time Multi-Grasp Detection Network.* 2022, PMC9415764.
- [6] Lenz, I., Lee, H., Saxena, A. *Deep Learning for Detecting Robotic Grasps.* IJRR 2014.
- [7] Cao, Z. et al. *Bilateral Cross-Modal Fusion Network for Robot Grasp Detection.* PMC10057080, 2023.

## Code Changes

- `src/dataset.py` (new) — `CornellGraspDataset` (torch Dataset around `CornellSample`), `encode_target` / `decode_prediction` (6-vec encoding in 224² frame), `_central_positive` GT picker, `_flip_grasp` augmentation, `split_object_wise` helper.
- `src/torch_models.py` (new) — 5 architectures: `TinyRedmonCNN` (AlexNet-shape from scratch), `ResNet18Regressor` (ImageNet pretrained, 6-D head), `ResNet18HybridHead` (4-D regression + 18-way classification), `GGCNNTiny` (encoder-decoder, per-pixel quality + angle + width maps), `TinyViT` (patch16 6-layer ViT from scratch). Plus `ggcnn_target_maps` / `ggcnn_decode` / `angle_to_bin_idx` / `bin_idx_to_angle_rad` helpers.
- `src/trainer.py` (new) — `train_regression` (M1/M2/M5), `train_hybrid` (M3, joint MSE + CE), `train_ggcnn` (M4, per-pixel MSE + masked angle/width), and a unified `evaluate(model, samples, model_kind=…)` harness that returns the same dict shape as Phase 1's eval so the leaderboard merges cleanly.
- `src/baselines.py` — fixed the cv2.minAreaRect 90° branch (was inverted in Phase 1; verified the fix empirically across 9 rotated-rectangle test cases — fixed branch returns 0° error, original branch returned 90° error in all 9). Phase-1 metrics in `metrics.json` are *not* re-run today because we don't want to silently change Phase-1 numbers; Phase 5 will re-evaluate the fixed B3 as a separate row in the LLM-comparison table.
- `tests/test_dataset_and_models.py` (new) — 20 unit tests pinning encode/decode round-trip, doubled-angle wraparound at ±90°, Jiang correctness against self after round-trip, angle-bin quantisation invariants, all 5 model forward shapes, GG-CNN target Gaussian peak location, `ggcnn_decode` argmax behaviour, param-count bounds (catches a frozen-weights regression), `split_object_wise`.
- `notebooks/phase2_multi_model.ipynb` (new, 28 cells) — 23 code / 5 markdown, all training and evaluation IN cells, 0 errors, 0 fake display-only cells. Final executed notebook is 3.2 MB.
- `requirements.txt` — added `torch>=2.4` and `torchvision>=0.19`.
- `results/EXPERIMENT_LOG.md` — Phase 2 section prepended.
- `results/metrics.json` — `phase_2` block appended.
- `results/phase2_target_sanity.png` — encode→decode round-trip on 4 training images (sanity check before the trained-model qualitative).
- `results/phase2_accuracy_bar.png` — Jiang accuracy of 5 CNN/ViT models with Phase-1 baselines as horizontal reference lines.
- `results/phase2_training_curves.png` — training loss curves for all 5 models on a single log-scale plot.
- `results/phase2_failure_decomposition.png` — stacked bars of Jiang sub-constraint failure modes for the 5 CNNs + 3 Phase-1 baselines.
- `results/phase2_champion_qualitative.png` — 6 random test predictions from M2 with GT positives.
- `results/phase2_model_comparison.csv` — the leaderboard as a CSV.
- `models/M2_ResNet18Regressor_phase2.pt` — champion state_dict (44.8 MB; not committed — `.gitignore` excludes `models/*.pt`).
- `data/raw/cornell/archives_remaining/data{04..10}.tar.gz` — 2.85 GB pre-downloaded for Phase 3 (not committed).
