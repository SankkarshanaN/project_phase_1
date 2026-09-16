# Phase 1-4 Results Summary

For the full, journal-ready writeup with all caveats and honest limitations, see
`docs/RESULTS.md` (quantitative results) and `docs/DASHBOARD.md` (live demo
explained panel-by-panel). This file is the short version.

## Phase 1 — Environment Setup
- CARLA 0.10.0 connected, sensor rig (RGB + depth + semseg + radar) frame-synced, 0 dropped frames over 100 ticks.

## Phase 2 — Dataset Collection
- **8,280 frames** (spec target: 8,000) across all 3 scenario types (occluded pedestrian
  crossing, multi-occlusion, blind-spot cut-in) x 2 weather conditions (clear + heavy
  rain/night/fog).
- Full YOLO-format bounding box labels + occlusion ground truth + radar data per frame.

## Phase 3 — Evidential Confidence Scorer
- Trained on 5,253 held-out labeled crops (vehicle / pedestrian / background).
- **Validation accuracy: 99.7%** (only 12 errors out of 5,253)
- Radar confidence score (rule-based SNR proxy) and uncertainty flag (threshold-based) added
  after an explicit spec audit -- both were gaps in the first pass, now implemented and shown
  live in the dashboard.

## Phase 4 — Three-State Occlusion Detector
- Built from MiDaS-small monocular depth + radar only (no CARLA ground-truth cheat).
- Validated over 200 sampled frames (80,000 cells): precision 0.98, recall 0.55, F1 0.71,
  89% overall cell agreement against ground truth. Precision is strong; recall is the honest
  weak point (see `docs/RESULTS.md` for why, and what it means).
- `4_occlusion_grid_validation.jpg`: predicted OCCLUDED region (red, middle panel) visually
  matches the ground-truth occlusion shape (red, right panel) for one example frame.

## Live Dashboard
- `5_live_dashboard_3panel.jpg`: the rebuilt 3-panel live demo -- camera feed with
  plain-language detection labels, the occlusion grid, and a raw radar bird's-eye scatter
  plot, all from the car's own point of view, running against live open-world CARLA traffic.

## Image files in this folder
1. `1_scenario_A_occluded_pedestrian_bbox.jpg` — bus occluding a pedestrian; pedestrian still
   correctly boxed while mostly hidden.
2. `2_scenario_B_blindspot_cutin_bbox.jpg` — cut-in vehicle merged from blind spot, now visible ahead.
3. `3_midas_depth_comparison.jpg` — RGB frame next to its MiDaS depth estimate.
4. `4_occlusion_grid_validation.jpg` — RGB / predicted 20x20 occlusion grid / ground-truth grid, side by side.
5. `5_live_dashboard_3panel.jpg` — the live 3-panel dashboard (camera / occlusion map / radar view).

## Known limitations (worth stating proactively)
- MiDaS gives *relative*, not metric, depth (no fixed physical scale) — this is exactly why the
  occlusion detector fuses in radar rather than trusting camera depth alone.
- End-to-end pipeline FPS measured 2-9 (not the spec's 15+ MiDaS-alone target on an RTX 3050) —
  dominated by CARLA 0.10.0's UE5.5 rendering overhead plus running 4 models concurrently on
  one GPU. See `docs/RESULTS.md` for the full discussion, including an observed CARLA
  server performance-degradation-over-uptime finding.
