# Second-pass experiment gap matrix

Built by inspecting `results/audit/*.json` (first-pass outputs), the dataset on
disk, and the implementation. Nothing was executed before this matrix existed.

**Status legend** — COMPLETE · NEEDS ADDITIONAL ANALYSIS · SUPPORTED — RUN ·
PARTIALLY SUPPORTED · NOT SUPPORTED · SHOULD NOT BE ADDED

---

## Enabling discoveries made during inspection

Three facts found while inspecting, which determine most rows below:

1. **`data/raw_v2_stale_projection` is a complete, 189-episode collection with
   ZERO episode-ID overlap with the 170 training/validation episodes**, and it
   carries its own `intent_labels.npz`. CLAUDE.md documents that the
   `BevProjector` sign error affected *only* the BEV occlusion grid. Verified
   empirically on a 150-frame probe of each collection:

   | | `occ_grid` OCCLUDED fraction | `obj_tier` dist (OCC/PART/VIS) | mean `obj_visibility` |
   |---|---|---|---|
   | `raw_v2` (fixed) | mean 0.511, median 0.534 | 0.320 / 0.112 / 0.568 | 0.588 |
   | `raw_v2_stale_projection` | mean **0.179**, median 0.176 | 0.341 / 0.098 / 0.562 | 0.582 |

   The BEV grid differs by ~3x while per-object tiers and visibility are
   near-identical — consistent with the bug being confined to `occ_grid` and
   the per-object pipeline (`bbox_projection`) being unaffected.
   **Consequence: this collection is usable as an independent test set for the
   classifier, tracking and intent, but MUST NOT be used for the occlusion
   detector, whose ground truth it corrupts.**

2. **`intent_labels.npz` already stores `time_to_entry_s`** — real
   ground-truth time-until-corridor-entry, computed by
   `label_crossing_intent.py` using the project's own existing corridor
   definition. Lead-time analysis therefore needs no new label definition.
   It also stores `forward_m`, `lateral_m`, `visibility`, `tier`, `scenario`.

3. **Every fault in `scripts/adversarial_test.py` exposes its severity as a
   keyword argument** (`darkness(factor=0.12)`, `fog(strength=0.75)`,
   `blur(k=9)`, `noise(sigma=45.0)`, `lens_blocked(fraction=0.4)`,
   `radar_dropout(keep=0.15)`, `radar_clutter(n_extra=40)`,
   `radar_bias(offset_m=2.5)`, `extrinsic_drift(yaw_deg=4.0)`). Sweeping an
   existing parameter is not a new fault model.

**Counter-discovery:** `PerceptionPipeline.__init__` takes only
`(bev_cfg, device, yolo, evidential, rng)`. There is **no flag to disable**
occlusion conditioning, the contradiction resolver, or the particle filter.
Those ablations would require rewriting the architecture, which the brief
forbids.

---

## Matrix

| # | Experiment | Already complete? | Current evidence | Additional evaluation needed? | Supported by current code/data? | Action |
|---|---|---|---|---|---|---|
| 1 | Dataset statistics | YES | `dataset_statistics.json` — 15,270 frames / 170 eps / 55,803 instances | No | — | **COMPLETE** |
| 2 | Episode-disjointness (train/val) | YES | `splits.json` — PASS, no overlap | Deeper: duplicate/near-identical crops, actor overlap | Yes | **NEEDS ADDITIONAL ANALYSIS** |
| 3 | Classifier core metrics (validation) | YES | 96.79% acc, macro-F1 0.965 | No — but must never be called "test" | — | **COMPLETE** |
| 4 | **Classifier on INDEPENDENT TEST set** | NO | none — first audit concluded no test set existed | Yes — major methodological gap | **Yes** (189 unseen episodes, 2D labels unaffected) | **SUPPORTED — RUN** |
| 5 | Uncertainty core (ECE/Brier/NLL/AUROC) | YES | ECE 0.0622, AUROC 0.942 | No | — | **COMPLETE** |
| 6 | Uncertainty vs distance / object size / occlusion + Spearman | NO | only VISIBLE-vs-PARTIAL split exists | Yes | Yes (`obj_xy_ego`, `obj_box_px` matchable to crops) | **SUPPORTED — RUN** |
| 7 | Occlusion detector P/R/F1 (held-out) | YES | 0.706 / 0.918 / 0.798, IoU 0.664 | No | — | **COMPLETE** |
| 8 | Occlusion spatial error analysis (FP/FN by range ring, boundary, radar density) | NO | only severity + scenario splits exist | Yes | Yes (GT grid + predicted grid are cell-wise) | **SUPPORTED — RUN** |
| 9 | Occlusion on independent test set | NO | — | Would be valuable | **No** — the only unused collection has corrupted `occ_grid` | **NOT SUPPORTED** |
| 10 | Tracking means + episode bootstrap CIs | YES | 845 paired obs, 44 eps, CIs present | No | — | **COMPLETE** |
| 11 | Tracking error **distributions** (median/IQR/p90/p95/worst) | NO | only mean/rmse/median/p95 summary | Yes — tail behaviour is the safety-relevant part | Yes (raw paired errors re-derivable) | **SUPPORTED — RUN** |
| 12 | Tracking on independent test set | NO | — | Yes | Yes (positions/velocities unaffected by the bug) | **SUPPORTED — RUN** |
| 13 | Effect sizes for fusion comparisons | NO | only p-values + CIs | Yes — p-values alone do not convey magnitude | Yes (paired errors) | **SUPPORTED — RUN** |
| 14 | Crossing-intent core metrics | YES | AUC 0.873 / F1 0.855 (fused) | No | — | **COMPLETE** |
| 15 | **Intent lead-time analysis** | NO | none | Yes — key safety question | **Yes** (`time_to_entry_s` is real GT) | **SUPPORTED — RUN** |
| 16 | Intent coverage analysis (why n differs per mode) | NO | counts noted but unexplained | Yes | Yes (track counts derivable during replay) | **SUPPORTED — RUN** |
| 17 | Intent on independent test set | NO | — | Yes | Yes (`intent_labels.npz` present in test collection) | **SUPPORTED — RUN** |
| 18 | Ablation A/B/C (camera / radar / camera+radar) | YES | first-pass ablation | No | — | **COMPLETE** |
| 19 | Ablation D (occlusion conditioning off) | NO | — | Would be valuable | **No** — no disable flag; needs architecture rewrite | **SHOULD NOT BE ADDED** |
| 20 | Ablation E (evidential uncertainty off) | PARTIAL | `--evidential` flag exists | Yes | Partially — flag affects only `p_cross_cautious`, not tracking | **PARTIALLY SUPPORTED** |
| 21 | Ablation F (contradiction resolver off) | NO | — | Would be valuable | **No** — no disable flag | **SHOULD NOT BE ADDED** |
| 22 | Ablation G (particle filter vs EKF coast) | YES | hidden-target experiment, 15 events | Covered by row 23 | — | **COMPLETE (sample-limited)** |
| 23 | Hidden-target: diagnose the 15-event limit | NO | first audit flagged it but did not diagnose | Yes — is it data, or an evaluation restriction? | Yes | **SUPPORTED — RUN** |
| 24 | Fault severity sweeps (bias / clutter / drift / camera faults) | NO | single severity per fault only | Yes — dose-response is the real result | Yes (severity are kwargs) | **SUPPORTED — RUN** |
| 25 | Fault detection delay / recovery | NO | binary caught/not-caught only | Yes | **Partially** — only `radar_clutter_onset` has an onset (frame 100); all others are on from frame 1 | **PARTIALLY SUPPORTED** |
| 26 | Silent-failure degradation quantification | PARTIAL | degradation deltas exist | Yes — pair degradation against health score across severity | Yes | **SUPPORTED — RUN** (merged into row 24) |
| 27 | Scenario-wise evaluation | PARTIAL | occlusion by scenario only | Extend to classifier / tracking / intent | Yes (`scenario` field present) | **SUPPORTED — RUN** |
| 28 | Distance / range analysis | NO | none | Yes | Yes (`forward_m`, `obj_xy_ego`) | **SUPPORTED — RUN** |
| 29 | Object-size analysis | NO | none | Yes | Yes (`obj_box_px`) | **SUPPORTED — RUN** |
| 30 | Risk engine / TTC | NO (skipped) | `risk_metrics.json` — no GT | Re-checked this pass: no `risk`/`ttc`/`hazard`/`collision` key in any meta record; no collision log persisted | **No** | **NOT SUPPORTED** |
| 31 | Denominator audit | NO | implicit only | Yes — manuscript wording risk | Yes | **SUPPORTED — RUN** |
| 32 | Runtime profiling | YES | 8 stages, 52.5 ms serial | No | — | **COMPLETE** |
| 33 | Multi-town / Town05 | NO | — | — | Excluded by instruction; CARLA 0.10.0 ships one town | **SHOULD NOT BE ADDED** |

---

## Execution plan (priority order)

**Tier 1 — resolves the largest methodological gaps**
- Row 4, 12, 17: independent test-set evaluation (classifier, tracking, intent)
- Row 15: intent lead-time
- Row 23: hidden-target diagnosis

**Tier 2 — strengthens existing claims**
- Row 24/26: fault severity dose-response
- Row 11, 13: error distributions and effect sizes
- Row 6: uncertainty vs difficulty

**Tier 3 — completeness and hygiene**
- Row 8: occlusion spatial errors
- Row 27, 28, 29: scenario / distance / size breakdowns
- Row 2: deeper leakage audit
- Row 31: denominator audit

**Not run, with reasons recorded**: rows 9, 19, 21, 30, 33.

---

# Execution record (appended after the plan above was carried out)

The matrix above is preserved exactly as it was written **before anything ran**.
This section records what actually happened to each planned row, so the two can
be compared without editing the original.

| Planned row(s) | Ran as | Outcome |
|---|---|---|
| 4, 12, 17 | SP-01 | **EXECUTED.** Independent test on 189 disjoint episodes / 62,528 crops. Classifier 0.9680 vs 0.9679 validation. Occlusion deliberately excluded — see row 9. |
| 15 | SP-02 | **EXECUTED.** Fused recall 0.909 / 0.746 / 0.696 at 0–0.5 / 0.5–1.0 / 1.0–1.5 s vs camera 0.693 / 0.556 / 0.306. Bands beyond 1.5 s have n<20 and are marked unreliable. |
| 23 | SP-03 | **EXECUTED (diagnostic).** Funnel 202 → 32 → 15. Verdict A + pipeline-inherent; the first audit had scored exactly the valid set. |
| 24, 26 | SP-04 | **EXECUTED.** Monitor never flags any of the three silent faults at any severity up to 8× shipped. Radar health *rises* with clutter (0.91 → 0.99). |
| 11, 13 | SP-05, SP-06 | **EXECUTED.** Fused v_lat median 0.341 (camera 0.690) but p95 2.476 (camera 2.125). dz = −0.145 vs camera (negligible), −1.115 vs radar (large). |
| 6, 27, 28 | SP-07 | **EXECUTED.** Spearman(uncertainty, error) = 0.297. Scenario and range breakdowns in `breakdowns.json`. |
| 8 | SP-08 | **EXECUTED.** Hypothesis overturned: recall is flat with range (0.938 → 0.897); precision collapses in the *near* field (0.127 at 0–10 m). Interior recall 0.921 vs boundary 0.913 — misses are not a boundary artefact. |
| 2 | SP-09 | **EXECUTED.** 0 episode overlap. 1.60% of val crops have a Hamming≤3 twin in train, concentrated in vehicles (2.36%); pedestrians 0.06%. |
| 31 | — | **EXECUTED.** `DENOMINATOR_AUDIT.md`. |
| 29 | SP-07 | **SKIPPED, recorded.** Saved raw predictions carry tier and visibility but not crop pixel dimensions; visibility was used as the difficulty proxy instead. Re-running the classifier purely to recover box sizes was not justified. |
| 9, 19, 21, 30, 33 | — | **NOT RUN, as planned.** Reasons unchanged from the matrix above. |

## Corrections made to this pass's own work

- **SP-09's within-episode similarity comparator is invalid and is labelled as
  such in `leakage_audit.json`.** It walks hashes in sorted *value* order, not
  frame order, so it does not measure consecutive-frame similarity and must not
  be quoted as the leakage a per-crop split would have caused.
- **SP-09 uses Hamming ≤ 3, not ≤ 5 as first drafted.** Exhaustive near-duplicate
  search at ≤5 is not tractable by 4-band LSH (the pigeonhole guarantee holds
  only to ≤3), and an all-pairs scan over ~60k hashes was not worth its cost for
  a hygiene check. The stricter threshold is also the more defensible one for a
  "near-duplicate" claim.
- **SP-02's per-band precision and F1 must not be quoted.** All bands share one
  negative pool while positives fall from ~1,600 to a handful, so precision is
  driven toward zero by construction. Recall and ROC-AUC are the interpretable
  columns.

## Figure audit (appended while preparing the manuscript)

Checked every figure on disk against the numbers the project currently claims.

- **`results/figures/occlusion_grid_validation.png` was STALE** — generated
  2026-09-16 16:17, before the 2026-09-17 `GROUND_QUANTILE` fix, so it showed
  the detector at recall 0.173 instead of 0.917. It shares a timestamp with the
  `results/metrics.json` block the first audit flagged as ERROR; both came from
  the same pre-fix run and both were regenerated together.

- **`results/figures/particle_modes.png` is NOT stale** — an earlier claim in
  this session that it was is **withdrawn**. Re-running
  `fig_particle_modes` with its own seed (`default_rng(1)`) reproduces the
  stored values exactly. The apparent disagreement was a time-point mismatch,
  not a stale figure:

  | | continues | slows | stops | turns back | spread |
  |---|---|---|---|---|---|
  | t = 2.0 s (quoted in CLAUDE.md) | 0.256 | 0.236 | 0.235 | 0.273 | 2.251 m |
  | t = 4.0 s (stored as `final_modes`) | 0.157 | 0.156 | 0.502 | 0.186 | 2.543 m |

  **Manuscript hazard, though:** the figure is a stackplot across the whole
  0-4 s window, so its right edge shows "stops" holding 50% of the mass. Placing
  the 26/24/24/27 figures beside it invites a reader to look at the endpoint and
  conclude one mode dominates. Either mark t = 2.0 s on the plot and quote the
  2 s numbers, or quote the endpoint — do not mix the two.

- **Verified current, no action:** `contradiction_matrix.png` (0.00882 / 0.15196,
  matches the claimed 0.009 / 0.152), `shadow_tolerance_sweep.png` (0.727 /
  0.917 / 0.811 at tol = 0.001), all 7 first-pass audit figures, all 7
  second-pass figures.

## Corrections to this pass's own reporting

- **"Only fused lateral velocity degrades materially" was wrong.** Drawing the
  SP-01 figure exposed it: on the independent test set *all three* tracking axes
  degrade, and position degrades most — lateral velocity +5.9%, forward velocity
  +9.4%, **position +12.0%** (2.032 → 2.276 m). The report now states all three.
- **The accuracy row mixed sign conventions.** `generalisation_gap_accuracy` is
  stored as validation − test, while every other row used test − validation, so
  the same result appeared as −0.0001 in one place and +0.0001 in another. All
  rows now use test − validation.
