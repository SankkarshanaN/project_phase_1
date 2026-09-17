# Phase 1-4 Results Summary

For the full, current, journal-ready writeup with all caveats and honest
limitations, see `docs/RESULTS.md` (quantitative results, measured against the
**15,270-frame `data/raw_v2` dataset**, recollected 2026-09-16) and
`docs/DASHBOARD.md` (live demo explained panel-by-panel). This file is a short
pointer, not a substitute.

> **The images in this folder are dated 27-28 July** -- early pipeline
> verification captured before the v2 rework (the lateral-sign fix, the
> occlusion-detector rewrite, the roaming-ego scenario). They still show the
> pipeline's components working -- a bounding box surviving partial occlusion,
> MiDaS depth alongside RGB, the 3-panel dashboard layout -- but **none of the
> numbers below, or in the images' captions, should be read as the current
> measured results.** A fresh capture against the fixed pipeline and final
> dataset has not been taken; do that before using these in a presentation
> where the numbers matter, or pull current figures from `results/figures/`
> instead, which are regenerated from `data/raw_v2` and are current as of
> `docs/RESULTS.md`.

## Phase 1 — Environment Setup
- CARLA 0.10.0 connected, sensor rig (RGB + depth + semseg + radar) frame-synced, 0 dropped frames over 100 ticks.

## Phase 2 — Dataset Collection (superseded -- see docs/RESULTS.md §1)
- Original run: 8,280 frames across 3 staged scenarios x 2 weather conditions.
- **Current: 15,270 frames across 4 scenarios** (adds the roaming-ego
  `urban_crossing` scenario), all clear-day -- CARLA 0.10.0's weather API does
  not function on this build (confirmed by measurement; see `docs/RESULTS.md`
  and `scenarios/scenario_d_adverse_weather.py`), so adverse conditions are
  evaluated by calibrated post-hoc degradation instead of simulated weather.
  This is the *second* v2 collection (2026-09-16), after a camera-projection
  bug in the occlusion ground truth was found and fixed -- see Phase 4 below
  and CLAUDE.md's "RESOLVED" note.

## Phase 3 — Evidential Confidence Scorer (superseded -- see docs/RESULTS.md §2)
- Original figure: 99.7% validation accuracy. **This number is invalid** -- it
  came from a per-crop random train/validation split that put near-duplicate
  crops of the same object on both sides of the boundary.
- **Current: 0.962 +/- 0.008**, on a split grouped by episode so no frame from
  a validation episode appears in training. This is the honest number.
- Radar confidence score (density + depth-cluster-tightness proxy) and
  uncertainty flag (threshold-based) are implemented and shown live in the
  dashboard; thresholds are calibrated from measured real CARLA frames (see
  `perception/sensor_health.py`'s header).

## Phase 4 — Three-State Occlusion Detector (superseded -- see docs/RESULTS.md §3)
- Original figure: precision 0.98, recall 0.55 (on the old, staged-only dataset).
- An intermediate figure (precision 0.888, recall 0.629, F1 0.736, cell
  agreement 0.913) was measured after fixing the shadow-marching defect
  described below, but **turned out to be invalid**: ground truth and the
  runtime detector shared an identical camera-projection sign bug, so
  comparing them agreed with itself regardless of what either was doing.
- On the fixed ground truth, recall first collapsed to precision 0.842 /
  recall 0.479 / F1 0.610 at a `shadow_tolerance` of 0.02 -- the max
  achievable recall at ANY threshold with `GROUND_QUANTILE` still at its old
  value of 0.30. That quantile turned out to be the real bottleneck: a ring
  that is 68-94% genuinely occluded contaminates even its bottom-30th-
  percentile disparity estimate with near-object readings. **Current, fixed
  2026-09-17 and validated on a disjoint held-out sample:
  `GROUND_QUANTILE = 0.01`, `DEFAULT_SHADOW_TOLERANCE = 0.001` --
  precision 0.727, recall 0.917, F1 0.811.** This is no longer the pipeline's
  weakest component. The march-vs-ground-profile algorithm fix (recall
  capped at 0.259 by marching, 0.629 after switching to a per-frame
  ground-disparity fit) remains a valid *relative* improvement from earlier
  in the project's history -- both were measured against the same ground
  truth at the time. Full account, including the projection-bug fix and both
  re-run sweeps, in `docs/RESULTS.md` §3.
- `4_occlusion_grid_validation.jpg` shows an OLD detector's output and should
  not be quoted as current.

## Live Dashboard
- `5_live_dashboard_3panel.jpg`: the 3-panel live demo layout -- camera feed
  with plain-language detection labels, the occlusion grid, and a raw radar
  bird's-eye scatter plot. The layout is current; the specific numbers visible
  in this particular screenshot predate later fixes (notably a health-monitor
  bug where the radar/camera "disagreement" penalty fired on ~100% of frames
  because the radar's 35 deg FOV is narrower than the camera's 90 deg --
  fixed in `perception/pipeline.py`).

## Image files in this folder
1. `1_scenario_A_occluded_pedestrian_bbox.jpg` — bus occluding a pedestrian; pedestrian still
   correctly boxed while mostly hidden.
2. `2_scenario_B_blindspot_cutin_bbox.jpg` — cut-in vehicle merged from blind spot, now visible ahead.
3. `3_midas_depth_comparison.jpg` — RGB frame next to its MiDaS depth estimate.
4. `4_occlusion_grid_validation.jpg` — RGB / predicted 20x20 occlusion grid / ground-truth grid, side by side (old detector -- see note above).
5. `5_live_dashboard_3panel.jpg` — the live 3-panel dashboard (camera / occlusion map / radar view).

## Known limitations (current -- see docs/RESULTS.md for the full list)
- MiDaS gives *relative*, not metric, depth (no fixed physical scale) — this is exactly why the
  occlusion detector fuses in radar rather than trusting camera depth alone.
- The occlusion detector recalls 0.917 at precision 0.727 as of 2026-09-17 (was 0.479 recall
  before a `GROUND_QUANTILE` fix -- see `docs/RESULTS.md` §3); no longer the weakest component.
- End-to-end pipeline FPS measures 2-9 in the live dashboard (5.5-6.0 after moving MiDaS off
  the per-frame path), against the spec's 15+ MiDaS-alone target on a different GPU. Dominated
  by CARLA 0.10.0's UE5.5 rendering overhead plus running 4 models concurrently on one GPU.
- Two radar faults are difficult for a self-referential health monitor to see: range bias when
  present from before the monitor starts observing, and clutter once sustained past its ~200-frame
  baseline window (a fix for the latter was tried and reverted -- it made ordinary healthy driving
  worse). See `docs/RESULTS.md` §6 for the full account, including a cross-sensor range check
  added to close part of the bias gap.
- CARLA 0.10.0's weather API remains unfixable from the Python API on this build -- confirmed
  again 2026-09-17 at Epic render quality with manually constructed weather parameters, and no
  accessible sun/sky/light actor exists as a workaround.
