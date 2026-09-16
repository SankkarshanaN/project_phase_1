# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CARLA-simulator research pipeline for a self-driving perception stack that
**quantifies and explains what it cannot see**: evidential (Dirichlet) detection
confidence, a three-state bird's-eye occlusion grid from monocular depth + radar, and
an EKF tracker feeding crossing-intent prediction for pedestrians emerging from behind
occluders. There is no package manifest, no test suite, and no linter config — it is
run as scripts from the project root.

Use the **`oaps-gpu`** conda env. It is the only one with `carla` + torch/CUDA +
ultralytics + sklearn together; `carla-sim` has no `carla` module at all.

```powershell
$PY = "$env:USERPROFILE\anaconda3\envs\oaps-gpu\python.exe"
```

Read `docs/RESULTS.md` and `docs/DASHBOARD.md` before changing `perception/` or
`models/` — they record why several non-obvious choices were made and carry every
measured number. Both describe the **v2 dataset** (15,420 frames, `data/raw_v2`)
and are current. Any figure quoted from the v1 era — the 99.7% validation accuracy
above all — is invalid; see "Dataset v1 was broken" below.

## Hard prerequisite: a running CARLA server

Every script under `carla_tools/`, `scenarios/`, and `scripts/` except
`generate_report_figures.py` and `validate_occlusion_grid.py` needs a CARLA **0.10.0**
server already running. The code never launches it. Start it separately:

```powershell
E:\Carla-0.10.0\Carla-0.10.0-Win64-Shipping\CarlaUnreal.exe -quality-level=Low -ResX=800 -ResY=600
```

`scripts/live_demo.py` needs the server started **without** `-RenderOffScreen`
(it drives the spectator camera).

Version specifics that are load-bearing, not incidental:
- 0.10.0 ships only `Town10HD_Opt` (+ a mine test map), so all four scenarios run in
  one town and `collect_dataset.py` uses a single connection with no town reloads.
- 0.10.0 replaced 0.9.x's entire vehicle catalog. Blueprint names in
  `configs/scenarios.yaml` were verified live against `world.get_blueprint_library()`.
  There is no Tesla/Audi/BMW; ego is `vehicle.lincoln.mkz`, bus is
  `vehicle.fuso.mitsubishi`. Verify any new blueprint name against the live catalog.
- `carla_tools.client.disconnect()` is **deliberately a no-op**. Restoring
  `synchronous_mode=False` hard-crashes the process (`STATUS_STACK_BUFFER_OVERRUN`)
  once any Traffic-Manager actor has been spawned and destroyed. Don't "fix" it —
  `connect()` re-applies synchronous mode every session.
- A long-lived CARLA server measurably degrades (one run fell from ~12 FPS to ~0.4 FPS
  with no error). If throughput collapses mid-run, restart the server before debugging
  the Python side.

Several docstrings still say "0.9.16" — the configs and blueprint names are the
authority; the code targets 0.10.0.

## Commands

All commands run from the project root (every script does its own
`sys.path.insert(0, <project root>)`, so `python scripts/foo.py` works directly).

Needs a CARLA server:

```powershell
# Verify server + sensor rig frame-sync over 100 ticks, report FPS
& $PY scripts/setup_check.py

# Roaming-ego scenario: drives the city, stages occlusion encounters en route
& $PY scenarios/scenario_e_urban_crossing.py --episodes 2 --max-steps 400

# Staged scenarios (each module is also a CLI)
& $PY scenarios/scenario_a_occluded_pedestrian.py --episodes 3 --max-steps 120
& $PY scenarios/scenario_b_blindspot_cutin.py --episodes 3
& $PY scenarios/scenario_c_multi_occlusion.py --episodes 3
& $PY scenarios/scenario_d_adverse_weather.py --episodes 2

# Full collection
& $PY scripts/collect_dataset.py --target-frames 15000 --out-dir data/raw_v2

# Live 3-panel dashboard (server must render on-screen; 'q' to quit)
& $PY scripts/live_demo.py
```

No server needed:

```powershell
# Gate a pilot collection BEFORE committing to a long run. Always do this first.
& $PY scripts/check_dataset.py --data-dir data/raw_v2

# Retrospective will_cross ground truth from recorded trajectories
& $PY scripts/label_crossing_intent.py --data-dir data/raw_v2

# The camera / radar / fused ablation
& $PY scripts/evaluate_prediction.py --data-dir data/raw_v2 --ablation all --evidential

# Component 9 -- inject sensor faults and measure what breaks, and whether the
# health monitor notices. The second question is the important one.
& $PY scripts/adversarial_test.py --data-dir data/raw_v2

# Quick training run
& $PY models/train_evidential.py --data-dir data/raw_v2 --epochs 15

# All reported figures + metrics.json (retrains the head)
& $PY scripts/generate_report_figures.py

# Spot-check the occlusion grid against saved ground truth for ONE frame
& $PY scripts/validate_occlusion_grid.py --frame-name urban_crossing_clear_day_ep00000_f0020
```

There is no `requirements.txt`. The environment needs: the `carla` 0.10.0 Python API,
`torch`, `ultralytics`, `opencv-python`, `numpy`, `pyyaml`, `matplotlib`, `seaborn`,
`scikit-learn`. MiDaS is pulled at runtime via `torch.hub.load("intel-isl/MiDaS", ...)`,
so the first occlusion-grid run needs network access.

### Which script actually trains the model of record

Both `models/train_evidential.py` and `scripts/generate_report_figures.py` write to
`models/evidential_detector.pt`, but they are **not** equivalent:

- `models/train_evidential.py` — plain Adam at a constant 1e-3, saves the **last**
  epoch. Useful for a fast iteration loop; its val accuracy oscillates.
- `scripts/generate_report_figures.py:train_and_evaluate` — adds `CosineAnnealingLR`
  and saves the **best-validation-epoch** checkpoint. This is the trainer of record and
  the checkpoint `live_demo.py` loads.

If you change the model or the loss, mirror the change in both, and regenerate figures
with `generate_report_figures.py` so `results/metrics.json` stays consistent with the
shipped `.pt`.

Both split train/validation with `train_evidential.episode_split`, **grouped by
episode, never per crop**. This matters more than it sounds: a per-crop random split
put all 90 episodes of the v1 dataset on both sides of the boundary, so the model was
scored on recognising the same vehicle it had trained on in a near-identical frame.
That is where the 99.7% came from. Do not revert it to `random_split`.

## Dataset v1 was broken — what changed and why

`data/raw` (v1, 8,280 frames) has four defects that invalidate results computed from
it. All are fixed in the code; the data itself must be re-collected into
`data/raw_v2`. Knowing these prevents re-introducing them:

- **Fully-occluded actors were deleted.** `data_collector` dropped every box scoring
  `visibility < 0.15`, so the hidden pedestrian — the subject of the project — never
  reached disk. This is why the uncertainty calibration curve came out flat.
- **The scenarios never produced occlusion anyway.** Walkers were spawned 3 m to the
  *side* of the occluder at the same longitudinal station, giving zero depth
  separation. Measured across the whole jitter range, the best case was 21% obscured
  and most were fully visible. `scenario_gen._shadow_position` now solves for a point
  on the camera-to-occluder ray, past the far face.
- **Weather never applied.** 2,160 frames tagged `heavy_rain_night_fog` are bright
  clear-day images. **Root cause confirmed on the live server: CARLA 0.10.0's weather
  API is non-functional on this build.** `get_weather()` returns all-zero fields for
  every preset, and an alternating A/B with a 40-tick settle gives ClearNoon vs
  HardRainNight deltas of −3.0, −3.1, +0.3 brightness points — noise, where a real
  night scene is 100+ darker. The collection code was never at fault; the simulator
  ignored every request silently. `_apply_weather` now raises `WeatherNotApplied` and
  `collect_dataset.py` skips adverse collection rather than writing mislabelled
  frames. Adverse-condition robustness is evaluated instead by applying calibrated
  degradations to captured frames (`scripts/adversarial_test.py`) — see
  `scenarios/scenario_d_adverse_weather.py` for the full evidence and reasoning.
- **85% of blind-spot frames were empty.** The cut-in spawned 12 m behind closing at
  2.78 m/s, so it was behind the camera for the first 40 of 60 recorded ticks, and it
  was the scenario's only actor. Retimed, and the two vehicles now share one waypoint
  chain so they cannot take different junction branches.

**Always gate a pilot collection with `scripts/check_dataset.py` before a long run.**
Every check in it corresponds to one of the above.

## Architecture

Four layers, strictly one-directional (`scripts` → `scenarios` → `carla_tools` →
`perception`/`models` → `common`):

- **`common/`** — `config.load_yaml(name)` resolves `configs/*.yaml` relative to the
  project root, so config loading is independent of the working directory. Every
  module takes an already-loaded `cfg` dict as an optional argument and falls back to
  `load_yaml` — pass the dict through in loops rather than reloading.
  `WallclockBudget` is the hard cutoff collection loops poll via `.expired()`.
- **`carla_tools/`** — everything that touches the simulator. `sensors.py` owns the
  rig (RGB + depth + semseg + radar + collision + lane-invasion, all sharing the RGB
  camera transform so per-pixel projections line up) and the `SensorBuffer`
  queue-per-sensor pattern required by synchronous mode. `scenario_gen.py` spawns
  scenarios procedurally from a seed (waypoint walk → occluder → actors), never
  hand-placed. `data_collector.record_episode` is the single recording loop every
  scenario shares.
- **`perception/`** — the *runtime, sensor-only* pathway. MiDaS monocular depth
  (`depth_midas.py`) + radar, deliberately **not** CARLA's depth buffer.
  `geometry.py` owns every camera/radar frame conversion — import from there rather
  than re-deriving a sign convention. See the pipeline map below.
- **`models/`** — `EvidentialHead` (Sensoy et al. 2018 Dirichlet head + annealed-KL
  loss) on a small 64×64 crop CNN. Attached as a *separate* classifier over YOLOv8n's
  box proposals rather than surgery on YOLO's DFL head: YOLO answers *where*, this
  answers *what and how sure*.
- **`scenarios/`** — each module exposes both `run_episode(...)` (library entry point
  used by `collect_dataset.py`) and a `main()` CLI. Adding a scenario means adding
  both plus a `configs/scenarios.yaml` block.

### The ground-truth / runtime split — the central design rule

Two parallel occlusion pipelines exist, and conflating them silently invalidates every
result:

| | Ground truth (labels + validation only) | Runtime (what the system may use) |
|---|---|---|
| Module | `carla_tools/occlusion_mask.py`, `carla_tools/true_occupancy.py` | `perception/occlusion_grid.py`, `perception/depth_midas.py` |
| Source | CARLA depth buffer, semseg, actor bounding boxes | MiDaS disparity + radar returns |
| Labels | `VISIBLE / OCCLUDED / UNKNOWN` (0/1/2) | `VISIBLE / OCCLUDED / EMPTY / UNKNOWN` (0/1/2/3) |

**The two label enums are different** — `occlusion_mask.UNKNOWN == 2` but
`occlusion_grid.EMPTY == 2`. Always import the constant from the right module; never
compare the grids cell-for-cell without mapping. `generate_report_figures.py` imports
them under distinct aliases (`OCCLUDED` vs `GT_OCCLUDED`) for exactly this reason.

Never let anything under `perception/` or `scripts/live_demo.py` reach for a CARLA
depth buffer, semseg image, or actor list. That is the whole claim of the project.

`perception/occlusion_grid.classify_grid` follows a fixed decision order — shadow
(OCCLUDED) → radar return (VISIBLE) → camera-clear (EMPTY) → UNKNOWN. Occlusion wins
over radar because a reflection off the occluder's own surface otherwise reads as
"visible". `shadow_tolerance` (default 0.12) is the one real tuning knob; measured on v2,
precision 0.89 / recall 0.63 / F1 0.74 at the default. Left at 0.12 rather than
the best-F1 0.08 deliberately: F1 differs by 0.002 while precision is 0.873
against 0.748, and false alarms cost more in something feeding a braking decision.

**Do not reintroduce a marching shadow propagation here.** `_shadow_grid_from_disparity`
used to walk outward cell by cell carrying a running ground expectation, which
capped recall at 0.259. Every marched track burns its first cell seeding the
trend, and a seed can never be flagged — going from 20 marched columns to 40
ray-correct angular bins *doubled* the seeds and dropped recall to 0.297, which
is how the cause was isolated. `_ground_profile` now fits bare-ground disparity
per range ring once per frame and tests each cell independently, mirroring how
the ground truth is computed (`BevProjector.compute_labels` is also per-cell with
no propagation). Recall went 0.259 -> 0.629, F1 0.408 -> 0.736. Full account in
`docs/RESULTS.md` §3.

`classify_grid` takes an optional `norm_disparity` so a real-time caller can
refresh MiDaS every few frames while folding in radar every frame. MiDaS dominates
its cost; everything else is arithmetic over 400 cells.

### The roaming ego stalls easily, and a stalled ego collects nothing

`scenario_e_urban_crossing` is the flagship and the only scenario where the ego
drives, so it carries the dataset's scene diversity. It is also the easiest one to
break silently, because a broken run still writes thousands of well-formed frames.

Two settings are load-bearing, both established by measurement rather than taste:

- **`EncounterManager._kerb_location` must clear the ego's own lane.** The occluder
  was originally parked a flat 1.6 m right of lane centre. Town10HD_Opt lanes are
  ~3.5 m wide and the occluder blueprints are 2.5-2.8 m across, so the parked bus
  straddled the ego's lane. The autopilot queued behind it and never moved again:
  measured across the first collection, the ego was below 0.1 m/s for **79%** of
  urban frames, every episode ended still in `triggered`, and 3,713 frames yielded
  **3** usable crossing observations. The offset is now
  `lane_width/2 + occluder_half_width + clearance`, applied a tick after spawn once
  the real bounding box is readable.
- **`_tune_ego_autopilot`** (`ignore_lights_percentage`, speed difference). With
  stock Traffic Manager settings the ego still idled at red lights for 54% of ticks.
  Town10HD_Opt is small and heavily signalled, and those ticks cost the same to
  simulate and record as any other while containing no occlusion and no closing
  speed. Measured over a 790-frame probe, tuning took the ego from 209 m to 522 m
  driven, stationary time from 56% to 15%, and staged encounters from 4 to 11.

`EncounterManager` runs inside `record_episode`'s tick loop and **must never call
`world.tick()`** -- an extra tick pushes an extra frame into every sensor queue and
desynchronises the rig. Anything needing a spawned actor's transform or bounding box
read back therefore has to be deferred a tick (see `_stage` / `_finish_staging`).

If a change to this scenario looks fine, check the ego actually drove before trusting
the frame count: mean speed, fraction of ticks below 0.1 m/s, and distinct encounters
per episode. All three were normal-looking-but-wrong in the first v2 collection.

### Dataset layout

Flat and split-free, keyed by a basename shared across three directories:

```
data/raw_v2/images/<scenario>_<weather>_ep<5d>_f<4d>.jpg   RGB 800x600
data/raw_v2/labels/<same>.txt                              YOLO: class cx cy w h
data/raw_v2/meta/<same>.npz                                everything else
```

**`labels/` and `meta/` deliberately disagree, and that is the design.** `labels/`
stays plain 5-column YOLO carrying only camera-observable actors (`tier != OCCLUDED`),
so ultralytics trains on it unmodified and never sees objects contributing zero pixels.
`meta/` carries the complete record including **amodal boxes for fully-hidden actors** —
where the actor truly is, though nothing can see it. That is ground truth only a
simulator can supply and the supervision signal for the whole project, so it is kept,
not discarded.

Per-object arrays in each `.npz`, all aligned and length M:

```
obj_actor_id  obj_class  obj_box_px(M,4)  obj_visibility  obj_truncation
obj_tier      obj_xy_ego(M,2)  obj_vel_ego(M,2)  obj_is_vru
```

plus `sim_time` (needed for exact velocity differencing), ego pose and velocity, the
BEV grids, and `radar_pts`. The roaming scenario also stamps `encounter_id` /
`encounter_phase` via `record_episode`'s `annotate` hook.

`obj_visibility` and `obj_truncation` are separate on purpose: an actor at the edge of
frame is truncated, not occluded, and conflating them (as the old 8-corner measure did)
made unobstructed actors look hidden and get dropped. Tiers are `OCCLUDED / PARTIAL /
VISIBLE` with boundaries in `configs/bev.yaml`; the continuous fraction is stored too,
so analysis can re-bin without re-collecting.

Classes are `0=vehicle`, `1=pedestrian`; training adds `2=background` from sampled
non-overlapping crops. The first 5 ticks of each episode (`WARMUP_STEPS`) are discarded
so sensors and actors settle.

Episodes are isolated in `collect_dataset._run_quota`: a failed episode is logged and
skipped (partial frames stay on disk), aborting only after
`MAX_CONSECUTIVE_EPISODE_FAILURES` (5). A single `queue.Empty` sensor timeout used to
take down a multi-thousand-frame run — keep that isolation in place for any new
long-running collection loop.

## Conventions worth matching

- **Explain the non-obvious in the docstring.** This codebase deliberately documents
  empirical findings, crash workarounds, and honest limitations inline (see
  `client.disconnect`, `depth_midas`, `radar_confidence`, `collect_dataset._run_quota`).
  Match that when you hit a simulator quirk — don't silence it.
- **Report measured numbers, not targets.** `docs/RESULTS.md` states measured FPS
  against a 15+ FPS spec target, the occlusion detector's remaining under-detection,
  and two honest null results: the evidential-to-prediction coupling, and the fact
  that a sensor fault present from startup cannot be detected by a self-referential
  health monitor at all. Keep results in that register.
- MiDaS returns *relative inverse disparity*, not metric depth — thresholds must be
  per-frame normalized (`normalize_disparity`), never treated as meters.
- CARLA's radar exposes no SNR; `perception/radar_confidence.py` is an explicitly
  documented density + depth-cluster-tightness proxy. Keep it labeled as a proxy.
- BEV grid convention is ego-local `(forward, lateral)`, `forward` in `0..extent_m`,
  **`lateral` positive to the LEFT**, shared by `BevProjector`,
  `compute_true_occupancy_grid`, `occlusion_grid`, `data_collector` and `tracking`.
  Radar azimuth is positive to the *right*, so the two differ by a sign — get it from
  `perception.geometry`, never re-derive it.

  **This sign is easy to get wrong and was wrong in two modules.** The natural
  formula `-dx*sin(yaw) + dy*cos(yaw)` projects onto CARLA's own
  `get_right_vector()`, so it yields the RIGHT component and must be negated.
  `data_collector._to_ego_frame` and `true_occupancy` both omitted that negation,
  which mirrored every recorded object position and the whole true-occupancy grid
  against the runtime frame. The symptom was subtle and misleading: the sensor
  ablation reported a *fused* filter doing worse on lateral velocity than a
  radar-only one that cannot measure lateral velocity at all.

  Settle it against the camera, not by algebra —
  `scratchpad/test_lateral_sign.py` places a probe along CARLA's right vector and
  checks all four of: which half of the image it appears in, what `_to_ego_frame`
  reports, which pixel `BevProjector` maps that cell to, and which cells
  `true_occupancy` marks. All four must agree.
- Ego speed and yaw rate are proprioception, not privileged simulator state: any real
  vehicle reads them off its own wheel encoders and IMU. `tracking` uses them to
  compensate for ego motion and stays on the runtime side of the split. Other actors'
  positions and velocities do not get the same pass.
- **`sensor_health.py`'s thresholds are calibrated from measured real CARLA frames**,
  not chosen by intuition. An earlier hand-guessed set was wrong by an order of
  magnitude — "good" sharpness at 300 when real frames run 1400–18500 — which scored a
  perfect camera at 0.38 and would have silently suppressed camera evidence throughout
  the fusion. The measured distributions are in that module's header. Re-measure before
  changing them, and re-measure if the camera resolution or JPEG quality changes.

## The nine-component pipeline

The methodology in the review deck numbers nine components; this is where each lives.

| # | Component | Module |
|---|---|---|
| 1 | Environment setup | `carla_tools/sensors.py`, `client.py` |
| 2 | Scenario generation | `carla_tools/scenario_gen.py`, `urban_world.py`, `scenarios/` |
| 3 | Evidential confidence scorer | `models/evidential_head.py`, `evidential_classifier.py` |
| 4 | Three-state occlusion detector | `perception/occlusion_grid.py` (BEV), `carla_tools/occlusion_tiers.py` (per-object) |
| 5 | Bayesian contradiction resolver | `perception/contradiction.py` |
| 6 | Particle-filter hidden-hazard tracker | `perception/particle_tracker.py` |
| 7 | Dynamic risk scoring engine | `perception/risk.py` |
| 8 | Sensor health monitor | `perception/sensor_health.py` |
| 9 | Adversarial robustness testing | `scripts/adversarial_test.py` |

`perception/tracking.py` (EKF) sits between 4 and 6: it tracks actors **while
visible**, and hands off to the particle filter when one goes fully occluded.

Two components are load-bearing in ways that aren't obvious from their names:

- **`contradiction.py` conditions its likelihoods on occlusion state.** That is the
  whole point. "Camera sees nothing" in an EMPTY cell is strong evidence of a clear
  road; the same silence in an OCCLUDED cell is almost no evidence at all. Measured:
  P(hazard) 0.011 vs 0.152 for identical sensor readings. Fixed-weight fusion cannot
  make that distinction.
- **`particle_tracker.py` is not interchangeable with the EKF.** Behind an occluder
  the distribution is genuinely multi-modal — emerge at the front, emerge at the rear,
  or stop — and a Gaussian puts its mean *inside the bus*, where the pedestrian
  certainly is not. Measured after 20 hidden frames: 46% of belief mass well left of
  the mean, 31% well right, only 23% near it.

## Crossing-intent prediction

`tracking.py` → `intent.py`. Two things about it are easy to get wrong:

- **Radar cannot see lateral velocity.** Doppler measures the radial component, so a
  pedestrian crossing your path has near-zero radial velocity — the component that
  decides the outcome is the one radar is blindest to. Measured on a crossing walker:
  radar-only lateral velocity error 1.40 m/s against a truth of 1.40 (it estimates
  zero), fused 0.13. The `_jacobian` range-rate row is where this lives.
- **Ablation modes must be compared on shared observations, not aggregate means.**
  Camera-only, radar-only and fused each hold tracks on a *different* subset of
  actors -- measured on the v2 dataset, 3,158 / 2,248 / 3,185 moving observations
  with only 578 tracked by all three. Their aggregate MAEs are therefore computed
  over different populations of different difficulty, and comparing them reverses
  the result: unpaired, fused looks worse than camera on lateral velocity (1.038
  vs 0.973); paired on the same objects and frames, fused is clearly better
  (0.711 vs 0.865, radar 1.599). `evaluate_prediction._print_paired` is the
  headline table for that reason. Keep the unpaired table too -- how many objects
  a configuration can track at all is a real property -- but never quote it as an
  estimation-accuracy comparison.

- **Inflating covariance by classifier uncertainty does NOT make a prediction more
  cautious.** Widening a distribution pushes its probability toward 0.5, so where the
  mean trajectory already enters the corridor, more uncertainty *lowers* p_cross —
  measured, 0.615 down to 0.570. `intent.py` therefore keeps `p_cross` honest and
  exposes `p_cross_cautious` as the worst case over the ambiguity set, which is
  monotone in uncertainty by construction. `is_risk` tests the cautious one.
