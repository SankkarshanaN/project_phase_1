# Live Dashboard — Panel-by-Panel Documentation

Produced by `scripts/live_demo.py`. Two windows are shown simultaneously,
both driven from the same ego vehicle, which drives itself (CARLA autopilot)
through an open world of live traffic — nothing is scripted per-frame except
the occlusion encounters staged ahead of the ego as it drives
(see "Why staged occluders?" below):

- **CARLA's own render window** — the raw 3D driver POV. The spectator
  camera is set to the ego's RGB sensor transform every tick
  (`spectator.set_transform(rig.rgb.get_transform())`), so this window shows
  exactly what the pipeline's camera sees, at full simulator fidelity.
- **The annotated dashboard window** (`Self-Driving Car That Explains What
  It Can't See`) — the same feed, run through the actual Phase 1–4 pipeline
  in real time. Nothing in this window is pre-recorded; every box,
  confidence value, and grid cell is computed from that tick's live sensor
  data. Labels are deliberately plain-language (percentages, "Not sure",
  "Hidden zones") so a non-technical viewer can follow it without
  narration — this document has the technical field names underneath.

## Dashboard layout

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  Title: "Self-Driving Car That Explains What It Can't See"                       │
│  Subtitle: what each panel below is                                              │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│  LEFT PANEL                │  MIDDLE PANEL             │  RIGHT PANEL               │
│  "What the car sees"       │  "What's hidden"          │  "Raw radar view"          │
│  camera + live boxes       │  20x20 occlusion grid     │  radar point cloud, BEV    │
│  (Week 2 data + Week 3     │  (Week 4)                 │  (raw sensor read, not     │
│   confidence/radar/flag)   │                           │   the fused decision)      │
├──────────────────────────┴──────────────────────────┴───────────────────────────┤
│  Status bar: time | speed | objects seen | not sure | radar strength |           │
│  hidden zones | (engine fps, tick)                                               │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Left panel — "What the car sees"

- Boxes are proposed by a pretrained YOLOv8n (COCO weights; only the
  `person`/`car`/`bus`/`truck` classes are kept and mapped onto our two
  classes, vehicle and pedestrian).
- Each box is then re-scored by **our own trained model**
  (`models/evidential_detector.pt`, from Phase 3): the crop is resized to
  64×64, passed through a small CNN encoder, and into the `EvidentialHead`.
  YOLOv8n answers *where*; our head answers *what, and how sure*.
- On-screen label: `<class>   Camera:<%>   Radar:<%>   [<- NOT SURE]`. Plain
  labels map to the technical values like this:
  - `Camera:<%>` — the Dirichlet mean probability for the predicted class
    (`EvidentialHead.expected_probability`) — the spec's "camera confidence
    score (0-1, calibrated)", shown as a percentage.
  - (not printed on screen, but drives the tag) **uncertainty** —
    `num_classes / sum(alpha)` — low when the head has accumulated a lot of
    consistent evidence for one class, high when it hasn't (occluded,
    blurry, ambiguous, or out-of-distribution crops).
  - `Radar:<%>` — the spec's "radar confidence score (rule-based from
    signal-to-noise ratio)" (`perception/radar_confidence.py`). CARLA's
    radar API doesn't expose a physical SNR value, so this is a documented
    rule-based proxy: 50% return density in the box's angular window (more
    detections = stronger aggregate signal) + 50% depth-cluster tightness
    (a real target reflects at a consistent depth; noise scatters). Scoped
    to the box via `pixel_range_to_azimuth_range`, an approximate
    pixel→azimuth mapping (not a full extrinsic calibration).
  - `[<- NOT SURE]` (amber outline + thicker box) appears when
    `uncertainty >= 0.5` (`models.evidential_head.uncertainty_flag`) — the
    spec's "uncertainty flag when below threshold", surfaced visually
    rather than only as a number.

### Middle panel — "What's hidden" (20×20 bird's-eye occlusion grid)

Built by `perception/occlusion_grid.classify_grid`, from **camera (MiDaS
monocular depth) + radar only** — no privileged simulator ground truth is
used here (that's reserved for offline validation, see below). The grid is
drawn with the car at the bottom, looking forward/up, matching a normal
bird's-eye-view map convention.

| Color | On-screen label | Technical meaning |
|---|---|---|
| 🔴 Red | **Blocked** | OCCLUDED — something nearer than the ground breaks the expected clear-road disparity trend along that ray, i.e. a real obstacle is blocking the view of that cell. |
| 🔵 Blue | **Clear** | EMPTY — MiDaS disparity at that cell is consistent with clear, unobstructed ground. |
| 🟢 Green | **Seen** | VISIBLE — a radar detection landed in that cell (independent of the camera). |
| ⬜ Gray | **Unknown** | Outside the camera's field of view / sensor range — no information either way. |

Decision order (from the spec): OCCLUDED is checked first, then VISIBLE
(radar), then EMPTY (camera), else UNKNOWN — a confirmed visual occlusion
overrides a radar return, since a reflection off an occluder's own surface
can otherwise look like a "visible" return.

### Right panel — "Raw radar view"

A separate, more literal view from the occlusion grid: every live radar
detection this tick, plotted at its actual position (not binned into the
20×20 grid or fused with anything). Distance rings every 20m, car marker at
the bottom. Dot color = the detection's radial velocity:

| Color | Meaning |
|---|---|
| 🔴 Red | Closing in on the car (negative radial velocity) |
| 🟡 Yellow | Roughly matching the car's own speed |
| 🔵 Blue | Pulling away from the car |

This is the sensor's raw read, shown alongside the fused occlusion decision
in the middle panel so a viewer can see *why* the middle panel says what it
says — e.g. a cluster of yellow dots explains a green (Seen) cell.

### Status bar

Plain-language summary, left to right: `Time` (sim seconds) / `Speed`
(km/h, from the vehicle's own velocity vector) / `Objects seen` (box count
this frame) / `Not sure about` (how many of those tripped the uncertainty
flag) / `Radar strength` (whole-frame radar confidence, same rule-based
score as the per-box one) / `Hidden zones` (how many of the 400 grid cells
are currently Blocked/red — a one-number "how much is currently hidden from
this car" summary). In parentheses: measured pipeline FPS (rolling window
of the last 30 frames' wall-clock timestamps, not claimed) and the raw tick
counter — see `docs/RESULTS.md` for what this measured FPS actually came out
to and how it compares to the spec's 15+ FPS target.

### Why staged occluders?

Free-roam autopilot traffic alone made real occlusion events (something
genuinely hidden behind another object) too rare to see live — moving traffic
rarely stays aligned between the car and another actor for more than an
instant. `carla_tools.urban_world.EncounterManager` therefore stages them: it
watches the ego's lane each tick, parks an oversized vehicle at the kerb on a
suitable stretch ahead, and places a pedestrian **on the ray from the camera
through the occluder's body, past its far face**, so the encounter opens with
the pedestrian genuinely invisible. The walker's behaviour state machine is
then triggered as the ego closes, so the crossing coincides with the approach.

The placement is verified rather than assumed: `_shadow_covers` checks the
occluder's true silhouette from the camera's eye point before committing, and
re-sites the walker if it does not cover. This matters — the earlier approach
of scattering parked vehicles and waiting for geometry to line up produced a
dataset with **no fully-occluded frames at all**.

Detections, confidence values, tracks and occlusion states are still computed
live from that tick's sensor data. Only the staging is scripted.

**Two settings here are load-bearing, and both were established by measurement
rather than taste** (see `docs/RESULTS.md` §7):

- The occluder must park **clear of the ego's own lane**. Parked at a flat 1.6 m
  from lane centre it straddled the lane, the autopilot queued behind it, and the
  ego stopped for the rest of the episode — 79% of frames stationary.
- The ego's autopilot needs `ignore_lights_percentage` tuning, or it idles at red
  lights for over half of all ticks on a map as small and heavily signalled as
  Town10HD_Opt.

The same `urban_world` module drives both this dashboard and dataset collection,
so what the demo shows and what the metrics are computed from cannot drift apart.

## Where the offline, quantitative numbers come from

The live dashboard is for demonstration — it shows the pipeline working,
but a single recording isn't evidence of accuracy. The actual measured
numbers (training curves, confusion matrix, uncertainty calibration,
occlusion-grid precision/recall against simulator ground truth over
hundreds of sampled frames) are generated separately and saved to
`results/figures/` + `results/metrics.json`:

| Script | Produces |
|---|---|
| `scripts/generate_report_figures.py` | training curves, confusion matrix, uncertainty calibration, occlusion-grid validation |
| `scripts/evaluate_prediction.py --ablation all` | the camera / radar / fused tracking and intent tables |
| `scripts/sweep_shadow_tolerance.py` | the occlusion detector's precision/recall curve over its one threshold |
| `scripts/adversarial_test.py` | injected sensor faults, and whether the health monitor notices |

**Read `docs/RESULTS.md` for the numbers themselves.** Headline figures on the
15,420-frame v2 dataset: evidential validation accuracy **0.948 ± 0.004** on a
split grouped by episode; fused lateral-velocity error **0.711 m/s** against
camera-only 0.865 and radar-only 1.599 on identical observations; crossing-intent
F1 **0.862** fused against 0.786 camera-only, warning 2.32 s before corridor
entry.

Note that the dashboard's own confidence numbers come from the same checkpoint
`generate_report_figures.py` trains, so the two are consistent by construction.
