# Phase 7: Testing + README + Consolidation — Robotic Grasp Prediction
**Date:** 2026-05-31
**Session:** 7 of 7

## Objective
Close the project: full pytest coverage of the production path *and* its pure
logic, a README that reads like a mini paper (architecture diagram, consolidated
leaderboard, frontier head-to-head, UI screenshot), and a single consolidated
final report. No new modelling — Phase 6 already proved the artifact reproduces
0.770; today is about making the work verifiable and legible to a cold reader.

## Research & References
1. **pytest "skipif" pattern for artifact-dependent tests** — the Phase-6
   production tests skip without the (gitignored) checkpoint/data; Phase 7 adds a
   *data-free* layer so coverage survives a clean checkout. Guided splitting tests
   into "needs artifacts" vs "pure logic."
2. **Google "Rules of ML" #29 (train/serve skew)** — already the spine of Phase 6;
   Phase 7's `test_phase7_coverage` adds an explicit assertion that the PIL and
   ndarray code paths through `preprocess_image` produce *identical* tensors,
   pinning the single-preprocessing-path guarantee in a unit test.
3. **Mitchell et al. 2019 (Model Cards)** — the existing card was reviewed for
   consistency with the shipped 0.770 numbers during consolidation.

How this shaped the session: rather than chase the 7.9-pp gap to Redmon (a
paradigm ceiling, per Phase 5), I invested the final session in *trust* —
reproducibility, coverage, and documentation — which is what makes the negative
result and the LLM head-to-head credible to a reader.

## What was done

### 7.1 — Data-free unit coverage (`tests/test_phase7_coverage.py`, 9 tests)
**Method:** exercise the production helpers with no checkpoint/data —
`load_config` contract, `resolve_path` (abs + relative), `object_wise_split`
no-leakage with lightweight stubs, `preprocess_image` PIL≡ndarray equivalence,
`viz.render_grasp` returns a Figure (with/without GT), and the `GraspPrediction`
DTO (`to_dict` serialisable, `to_grasp_rect` round-trips the centre).
**Result:** 9/9 pass in 3.3 s on a clean path.
**Interpretation:** the suite now has real coverage of config, split, renderer,
and the prediction contract even where the artifacts are absent — the Phase-6
smoke tests skip there.

### 7.2 — Full suite
**Result:** **45 passing** (36 prior + 9 new), 13 s. No errors, no new skips on
this machine (data + checkpoint present, so the production smoke path runs and
asserts `accuracy == reference` exactly).

### 7.3 — Documentation consolidation
- `reports/final_report.md` — 8-section consolidated write-up: problem/metric, the
  7-phase arc, the 17-row consolidated leaderboard, the frontier table, the five
  findings, the production system, limitations, and a reproduce block.
- `README.md` — fixed stale Phase-5 status → Phase-7 complete; corrected the
  project layout (was still showing only Phase-1 files) and the Setup section
  (removed a reference to a `scripts/fetch_cornell.sh` that was never written →
  point at `data/README.md`); added a Mermaid architecture diagram, the
  consolidated leaderboard, the frontier-LLM table, a production-demo section
  embedding `results/ui_screenshot.png`, and Phase 6/7 iteration blocks.

## Consolidated leaderboard (object-wise, n=100)
| Rank | Model | Phase | Acc |
|-----:|-------|:-----:|----:|
| 1 | **P4 tuned ResNet-18 global regressor** ⭐ | 4 | **0.770** |
| 2 | E3.5 all-knobs union | 3 | 0.770 |
| 12 | ResNet18-FCN per-pixel (pretrained) | 5 | 0.420 |
| 17 | per-pixel from scratch (M4/E5.1–5.2) | 2/5 | 0.000 |

(Full 17-row table + the n=40 frontier head-to-head in `reports/final_report.md`.)

## Key Findings
1. **The shipped claim is now test-pinned end-to-end.** A unit test asserts the
   saved artifact scores `== reference` (0.770) on folder-03, and a separate test
   proves the two `preprocess_image` input paths are bit-identical — so "0.770,
   reproducible, no train/serve skew" is enforced by CI, not just prose.
2. **Coverage survives a clean checkout.** Splitting tests into artifact-dependent
   (skip) vs pure-logic (always run) means `pytest` stays green for a reviewer who
   clones without downloading 1.4 GB of Cornell data.
3. **Documentation debt was real.** The README still described a Phase-1-only repo
   and pointed at a script that never existed — exactly the kind of rot that makes
   a strong result look untrustworthy. Consolidation was the highest-value use of
   the final session, not more tuning against a known paradigm ceiling.

## Error Analysis
No model changes, so the failure profile is unchanged from Phase 4/5: the residual
errors are the "angle-ok / IoU-too-low" localisation cluster — the global-regression
ceiling documented in the model card's limitations.

## Next Steps
- Project complete. Optional future work: a *pretrained* per-pixel decoder
  (segmentation-pretrained backbone) is the one lever that could attack the 7.9-pp
  gap to Redmon without forfeiting ImageNet transfer — the structural cause Phase 5
  pinned. Out of scope for this 7-day build.

## References Used Today
- [1] pytest docs, *Skip and xfail* — https://docs.pytest.org/en/stable/how-to/skipping.html
- [2] Google, *Rules of ML* (Rule #29) — https://developers.google.com/machine-learning/guides/rules-of-ml
- [3] Mitchell et al. (2019), *Model Cards for Model Reporting* — https://arxiv.org/abs/1810.03993

## Code Changes
- **New:** `tests/test_phase7_coverage.py` (9 tests), `reports/final_report.md`,
  `reports/day7_phase7_report.md`.
- **Modified:** `README.md` (status, layout, setup, architecture diagram,
  consolidated leaderboard, frontier table, UI screenshot, Phase 6/7 blocks).
- **Verification:** `pytest -q` → 45 passing.
