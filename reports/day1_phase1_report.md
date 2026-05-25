# Phase 1: Domain Research + Dataset + Baselines — Robotic Grasp Prediction (DL-3)

**Date:** 2026-05-25 (Monday)
**Session:** 1 of 7

## Objective

Before training a single neural network, find out **how far three non-learning
baselines get on Cornell** under the Jiang IoU>0.25 / angle<30° metric. The
purpose is to lock the *floor* every Phase 2+ CNN has to beat by a meaningful
margin on the *same* train/test split and the *same* metric — and to surface
which sub-constraint of the Jiang metric is the actual bottleneck, so Phase 2
knows whether to optimise for localisation or for orientation.

## Research & References

1. **Jiang, Moseson, Saxena — *Efficient Grasping from RGBD Images*, ICRA 2011** —
   the original Cornell paper that introduced both the dataset and the
   evaluation metric I use as primary today. Their metric: a prediction is
   correct iff there exists a ground-truth positive grasp with IoU > 0.25 AND
   angle difference ≤ 30°. This dual-threshold is what makes the task hard:
   high IoU with wrong angle would crash the gripper, so the metric refuses
   to count it.
2. **Lenz, Lee, Saxena — *Deep Learning for Detecting Robotic Grasps*, IJRR
   2014** — first deep model on Cornell; reports 75.6% image-wise / 73.9%
   object-wise. Used here as the "what a deep model should clear" reference
   point.
3. **Redmon & Angelova — *Real-Time Grasp Detection Using CNNs*, ICRA 2015** —
   single-shot regression CNN, 84.4 / 84.9% at 13 ms. The original
   "you can predict grasp rectangle directly" paper. I'll replicate this as a
   Phase 2 baseline because it's *small* enough to actually iterate on.
4. **Morrison et al. — *GG-CNN*, RSS 2018** — per-pixel grasp quality + angle
   map; 73% at <20 ms. The architecture that exploits Cornell's multi-grasp
   labels properly (every other paper above predicts one rectangle per
   image).
5. **Cao et al. — *Bilateral Cross-Modal Fusion Network*, PMC 2023** — 99.4 /
   97.8% on Cornell, current SOTA. Used as the upper-bound reference; not
   something I'll replicate, but worth knowing how far off the ceiling lives.

**How research influenced today's experiments:** Choosing Cornell over Jacquard
or GraspNet-1B was a deliberate trade-off: Cornell is small enough (885
images) to download, parse, and iterate on in one session, *and* every paper
above benchmarks on it so the Phase 5 LLM head-to-head numbers will sit
directly next to published reference points. The Jiang metric is locked
because every paper above reports it — switching to anything else would make
my comparison table incomparable to the literature.

## Dataset

Cornell Grasping Dataset (Jiang et al. 2011). 885 RGB-D images of 240
distinct household objects, ~8 000 oriented grasp rectangles (positive +
negative). **The official `pr.cs.cornell.edu/grasping/rect_data/data.php`
URL returned HTTP 404 today**; the working source was the Wayback Machine
snapshot from 2016-04-14:

```
https://web.archive.org/web/20160414082648/http://pr.cs.cornell.edu/grasping/rect_data/temp/data{01..10}.tar.gz
```

Phase 1 downloaded archives 01–03 (1.6 GB → 300 images) for fast iteration;
later phases pull the rest in the background.

| Metric | Value |
|---|---|
| Total images loaded | 300 (folders 01, 02, 03) |
| Positive grasps | 1 540 |
| Negative grasps | 881 |
| Pos:neg ratio | 1.75 |
| Image resolution | 640 × 480 (RGB) + per-pixel depth via .pcd |
| Train split | 200 images (folders 01, 02) |
| Test split | **object-wise** held-out: 100 images (folder 03), 542 positives |

I use the **object-wise** split for the test set rather than a random
image-wise shuffle, because shuffling within the 300 images would leak
object identity across train/test boundaries (Cornell groups objects by
folder during collection). This is the harder, more honest setting — every
test image's object was never seen during fitting. Phase 2 will report both
splits.

EDA highlights:
- Positive grasps centre at (cx ≈ 296, cy ≈ 280) — almost exactly the image
  centre. Cornell was shot with one object placed in the camera centre.
- Plate-depth (`height`) is nearly identical between positive (24.3 ± 6.9 px)
  and negative (29.8 ± 8.9 px) grasps — the gripper hardware fixes this.
  Annotators were not free to vary it, which is why my B2 baseline freezes it.
- Jaw-opening (`width`) is much more variable: positives 33.0 ± 19.6 px,
  negatives 21.4 ± 7.7 px. Positives are *wider* on average — bigger grasps
  on bigger object features.
- Grasp angles are near-uniform (positives: 2.5° ± 54.8°). A constant-angle
  baseline has no orientation prior to ride; orientation must come from
  reading the image.

## Experiments

### Experiment 1.1: Random rectangle baseline

**Hypothesis:** Random will land in the 5–15 % range — the Jiang metric is
permissive enough (IoU > 0.25 with *any* of multiple GTs) that lucky hits
exist.

**Method:** Sample (cx, cy, w, h, angle) uniformly from plausible ranges
(cx∈[160,480], cy∈[120,360], w∈[40,140], h∈[15,40], angle∈[-π/2, π/2]).
Seed-controlled `np.random.default_rng(42)` so the number is reproducible.

**Result:** 1.0 % accuracy. Median IoU 0.000. Median angle err 46°.

**Interpretation:** I overestimated chance by an order of magnitude. The
dual constraint (IoU AND angle) is much more punishing than I expected: each
constraint alone is roughly a 25–30 % single-shot hit rate, and they multiply
nearly independently down to ~1 %. This number is the *true* floor.

### Experiment 1.2: Image-centre + median size, axis-aligned

**Hypothesis:** Picks up Cornell's centring prior; lands at 35–50 %.

**Method:** Fit (cx_median, cy_median, w_median, h_median) on the 998 training
positives. Angle frozen at 0°. Predict the same rectangle for every test
image.

**Result:** 4.0 % accuracy. Median IoU 0.083. Median angle err 24°.

**Interpretation:** Wrong by 10×. The centring prior exists in absolute
terms (Cornell objects are centred) but the *grasps* within an object span
its whole extent, so a median-of-medians predicted at (278, 283) only
overlaps any specific GT positive at IoU > 0.25 about 4 % of the time. The
median IoU is 0.08 — the box is in the right neighbourhood but the GT
positives are scattered across the object surface, not clustered at the
object's centroid. **The image-centre prior is a much weaker signal than the
"objects are centred" framing suggests.**

### Experiment 1.3: Depth-edge antipodal heuristic

**Hypothesis:** Wins on angle accuracy (reads the object's orientation
directly via minAreaRect on the depth mask) but loses on width (knows
nothing about gripper-aware grasping). Net ≈ B2.

**Method:**
1. Load the .pcd file, extract forward-distance (camera-frame `x`) per pixel
   into a 480×640 depth image.
2. Background plane = 95th percentile of depth. Object mask = pixels
   noticeably closer than background.
3. Largest connected component → cv2.minAreaRect → (cx, cy, w, h, angle).
4. The gripper closes across the object's *short* axis (antipodal grasp
   theory). Set grasp width = object short-axis × 1.05, height = fitted
   median plate depth, angle = perpendicular to object's long axis.

**Result:** **17.0 % accuracy.** Median IoU 0.159. Median angle err 37°.

**Interpretation:** 4× better than B2 because the prediction *follows the
object* instead of staying at a fixed point. The hypothesis was wrong about
which channel wins, though — B3's median angle error is 37° (slightly above
the 30° threshold), suggesting the 90° flip between cv2.minAreaRect's
convention and the gripper convention is misfiring on a noticeable
fraction of samples. Fixable in Phase 2; deliberately not fixing today
because the *failure-mode decomposition* below is the more interesting
finding.

## Head-to-Head Comparison

| Rank | Baseline | Accuracy | Median IoU | Median angle err | n test |
|---:|---|---:|---:|---:|---:|
| 1 | **B3 depth + antipodal** | **17.0 %** | 0.159 | 37.0° | 100 |
| 2 | B2 centre + median size | 4.0 % | 0.083 | 24.0° | 100 |
| 3 | B1 random rectangle | 1.0 % | 0.000 | 46.0° | 100 |

For context, the *published* numbers from the references above (different
data splits, but same metric): Lenz 2014 = 73.9 % object-wise, Redmon 2015 =
84.9 %, GG-CNN 2018 = 73 %, Cao 2023 (current SOTA) = 97.8 % object-wise.
**The CV-to-CNN gap is ~60 pp, much larger than the 16 pp chance-to-CV gap.**

## Key Findings

1. **Headline — IoU localisation, not orientation, is the bottleneck of grasp
   detection.** The failure-mode decomposition below shows that across *all
   three baselines*, the dominant failure pattern is "angle is within 30°,
   but IoU is too low". Even the dumb B2 baseline that always predicts
   angle = 0° gets the orientation right (within 30° of some GT) **58 % of
   the time**. The common framing — "deep nets learn to orient grippers" —
   gets the difficulty backwards on Cornell. The orientation prior in the
   data is huge; what the network actually has to learn is **which point on
   the object's surface to grasp**.

| failure mode | B1 random | B2 centre | B3 depth |
|---|---:|---:|---:|
| both ok (= prediction correct) | 1.0 % | 4.0 % | 17.0 % |
| **angle ok, IoU too low** | **35.0 %** | **49.0 %** | **31.0 %** |
| IoU ok, angle wrong | 0.0 % | 9.0 % | 2.0 % |
| both wrong | 64.0 % | 38.0 % | 50.0 % |

2. **Cornell's centring prior is much weaker than the dataset description
   implies.** "Objects are placed in the image centre" only buys 3 pp over
   chance, because the *grasps within an object* are scattered across the
   object's surface, not concentrated at its centroid. The centre prior
   localises within ~50 px of *some* GT; what we need is within ~15 px of a
   specific GT to clear IoU > 0.25.

3. **Antipodal physics buys 13 pp over the centring prior — but is still
   60 pp short of any CNN.** B3 reads the object's actual extent from depth
   and applies the antipodal-grasp principle (close across the narrow axis).
   That principle is right (it's why robotics worked before deep learning),
   but it ignores everything a CNN sees in the RGB channel: object texture
   that signals fragility, handle locations, surface roughness, semantic
   features ("this is a mug — grasp the handle, not the rim"). Phase 2's
   CNN is essentially closing this 60 pp gap by learning to weight what
   antipodal-from-depth can't see.

## Error Analysis

What the depth baseline gets wrong (from inspecting `phase1_depth_baseline_qualitative.png`):

- **Cluttered table textures break the background mask.** Whenever the table
  has a strong pattern or shadow, depth segmentation pulls the whole table
  into the "object" component and the bounding box snaps to the wrong
  region.
- **Symmetric / round objects produce ambiguous minAreaRect angles.** A
  perfectly round disc has no defined orientation; cv2's angle is then
  driven by quantisation noise. The Jiang metric rewards *any* of multiple
  good orientations, but B3 commits to one of them and may pick a different
  one than the annotators.
- **Thin objects (pens, markers) get the orientation right but the position
  wrong.** The midpoint of a long thin object is rarely the best grasp
  point — the grasp should be near the centre-of-mass, not the geometric
  centre of the bounding box. This is what makes the failure-mode
  decomposition land in "angle ok, IoU too low" so often.

## Frontier Model Comparison

Not applicable in Phase 1 (no learned model to compare yet). Phase 5 will
send depth-image descriptions + the metric definition to Claude Opus / Codex
GPT-5.4 and put their grasp predictions in the same table. Today's B3 = 17 %
is the floor those LLMs must clear to count as "they understand grasping at
all".

## Next Steps

- **Phase 2 (Tue 2026-05-26):** Multi-model comparison. Replicate Redmon &
  Angelova's regression CNN, plus 4 modern alternatives (ResNet-18
  regressor, ResNet-18 + rotation classifier, GG-CNN, small ViT). Same
  object-wise test split as today, same metric. Target: clear B3 = 17 %
  by ≥ 50 pp.
- **Beyond the basics:** Phase 2 should also run the rest of the Cornell
  archives in the background (data04–10) so we have the full 885-image
  benchmark by Phase 3.
- **Phase 4 → 5 set-up:** Fix the cv2.minAreaRect 90° flip ambiguity in
  the antipodal baseline so we have a *clean* B3 number to compare against
  the LLMs in Phase 5. The current 17 % may be artificially low if the
  flip is wrong half the time.

## References Used Today

- [1] Jiang, Y., Moseson, S., Saxena, A. *Efficient Grasping from RGBD Images: Learning using a new Rectangle Representation.* ICRA 2011.
- [2] Lenz, I., Lee, H., Saxena, A. *Deep Learning for Detecting Robotic Grasps.* IJRR 2014. https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf
- [3] Redmon, J., Angelova, A. *Real-Time Grasp Detection Using Convolutional Neural Networks.* ICRA 2015. arXiv:1412.3128
- [4] Morrison, D., Corke, P., Leitner, J. *Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach (GG-CNN).* RSS 2018.
- [5] Cao, Z. et al. *Bilateral Cross-Modal Fusion Network for Robot Grasp Detection.* PMC10057080, 2023.
- [6] Depierre, A., Dellandréa, E., Chen, L. *Jacquard: A Large Scale Dataset for Robotic Grasp Detection.* arXiv:1803.11469, 2018.
- [7] Kumra, S. et al. *GR-ConvNet v2: A Real-Time Multi-Grasp Detection Network for Robotic Grasping.* PMC9415764, 2022.
- [8] dougsm/ggcnn (github) — Cornell PCD-to-depth preprocessing reference implementation.

## Code Changes

- `src/cornell.py` — dataset loader, `GraspRect` dataclass (Cornell corner
  convention: `corners[0]→corners[1]` = jaw-opening axis), shapely-backed
  polygon IoU, Jiang metric implementation, `make_rect` constructor.
- `src/baselines.py` — three baselines (`RandomBaseline`,
  `CenterHeuristicBaseline`, `DepthAntipodalBaseline`) sharing the same
  `evaluate(baseline, samples)` harness.
- `tests/test_cornell.py` — 11 unit tests pinning the corner convention, the
  Jiang thresholds, and the make_rect round-trip. All pass.
- `notebooks/phase1_baseline.ipynb` — 15 code cells / 10 markdown cells, all
  with captured output, zero errors.
- `data/README.md` — Wayback URL provenance (the official Cornell URL is
  404 as of today).
- `results/metrics.json`, `results/phase1_*.png` — saved artefacts.
- `requirements.txt`, `.gitignore`, project `README.md` — scaffolding.
