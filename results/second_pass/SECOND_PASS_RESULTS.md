# Second-pass results

Generated 2026-09-23T06:27:37.754071+00:00 · commit `385309f8db`

This pass did not re-run experiments the first audit had already validated. It identified what was missing, checked what the code and data could actually support, and ran only those. Seven new measurements were executed — one of them a diagnostic that resolved an open question from the first pass — and six candidate experiments were refused, each with a stated reason.

## 1. The first audit's biggest limitation is largely resolved

The first audit concluded that no independent test split existed. That was true of `data/raw_v2` alone but incomplete: `data/raw_v2_stale_projection` is a complete 62,528-crop, 189-episode collection with **zero episode-ID overlap** with the 170 train+val episodes.

It was archived because of the `BevProjector` sign error. That defect is documented as confined to the BEV occlusion grid, and this pass verified that empirically before using the data:

- `occ_grid` OCCLUDED fraction differs 2.9x (0.511 vs 0.179)
- per-object tier distribution agrees to within 0.020
- mean visibility agrees to within 0.006

So the per-object pipeline is sound there and the collection is usable for the classifier, tracking and intent — **but not for the occlusion detector, which this pass explicitly refuses to score against corrupted labels.**

| Metric | Validation | Independent test | Gap |
|---|---|---|---|
| Classifier accuracy | 0.9679 (n=12,552) | **0.9680** (n=62,528) | +0.0001 |
| Macro-F1 | 0.9652 | 0.9621 | -0.0031 |
| ECE (lower better) | 0.0622 | 0.0626 | +0.0003 |
| Error-detection AUROC | 0.9424 | 0.9460 | +0.0036 |
| Fused v_lat MAE | 0.672 | 0.711 | +0.039 |
| Fused intent F1 | 0.855 | 0.862 | +0.006 |

Gap = test − validation; for ECE lower is better, so a positive gap there is a small degradation.

**Classification and calibration transfer with no measurable loss** — accuracy moves by 6e-5, ECE by 0.0003, and error-detection AUROC actually improves. Fused intent F1 replicates on an independent sample (+0.006).

**Tracking is where the cost shows, and it is on every axis, not just one:** lateral velocity +5.9%, forward velocity +9.4%, position +12.0%. Position is the largest, not lateral velocity: 2.032 → 2.276 m (+0.244). The test tracking sample is also far smaller than the classifier one (578 paired observations), so these are noisier than the crop-level figures above.

## 2. Crossing intent versus lead time (new)

`intent_labels.npz` already stores `time_to_entry_s`, so this needed no new label definition. Recall by how far ahead the crossing was:

| Lead time | camera | fused | positives (fused) |
|---|---|---|---|
| 0.0-0.5s | 0.693 | **0.909** | 1597 |
| 0.5-1.0s | 0.556 | **0.746** | 59 |
| 1.0-1.5s | 0.306 | **0.696** | 23 |
| 1.5-2.0s | 0.036 | **0.545** | 11 (n<20) |
| 2.0-2.5s | 0.000 | **0.600** | 5 (n<20) |
| 2.5-3.0s | 0.000 | **0.000** | 2 (n<20) |

Fusion roughly doubles detection at 1.0–1.5 s of lead time (0.696 vs 0.306) and is 15x better at 1.5–2.0 s, though that band has only 11 positives.

> **Do not quote per-band precision or F1.** Every band shares one large negative pool while positives fall from ~1,600 to a handful, so precision is driven to near zero by construction. Recall and ROC-AUC are the interpretable columns. Realised entry times cluster hard near zero, so the dataset supports lead-time claims only to about 1.5 s.

## 3. Why the hidden-target experiment had only 15 events (resolved)

- 202 ground-truth occlusion transitions
- 32 had a confirmed track beforehand (**170 lost here**)
- 15 stayed hidden ≥0.5 s with truth available
- 15 scored by the first audit — **identical to the valid set**

The evaluation was not defective: it scored exactly every legitimate event. The loss is upstream — 84% of actors that go behind an occluder were never confirmed-tracked first — and the median occlusion lasts only 4 frames (0.4 s). Cause **A (genuinely insufficient data)**, not an evaluation restriction. Loosening the tracker's confirmation threshold or the association gate would have raised n by changing the system under test, and was rejected.

## 4. Silent failures: dose-response (new)

Each fault is the shipped function called with a different value of its own existing severity keyword — no new fault model.

| Fault | Severity swept | Worst Δ position RMSE | Monitor ever flags? |
|---|---|---|---|
| `radar_bias` | offset_m 0.5–20.0 | +1.170 m | **NEVER** |
| `radar_clutter` | n_extra 10–320 | +1.148 m | **NEVER** |
| `extrinsic_drift` | yaw_deg 1.0–32.0 | +0.605 m | **NEVER** |

**The monitor never wakes up — at any severity tested, up to 8x the shipped magnitude.** Radar health stayed in 0.83–0.99 throughout. This upgrades the first audit's single-severity finding: it is not a threshold-tuning problem.

**The most important new negative result:** radar health *increases* monotonically with clutter — 10→0.91, 20→0.94, 40→0.97, 80→0.98, 160→0.98, 320→0.99. `perception.radar_confidence` is a density-and-tightness proxy because CARLA exposes no SNR, so injecting spurious returns makes the sensor score as *healthier* exactly as it becomes less trustworthy. No threshold on this statistic can fix that; it is structural.

## 5. Error distributions reframe the fusion claim (new)

| Mode | v_lat median | v_lat p95 | position median | position p95 |
|---|---|---|---|---|
| camera | 0.690 | 2.125 | 2.105 | 4.520 |
| radar | 1.853 | 2.000 | 2.263 | 4.410 |
| fused | 0.341 | 2.476 | 1.715 | 4.380 |

This is the clearest new insight in the pass. On **median** lateral velocity error fusion is dramatically better than camera (0.341 vs 0.690, a 51% reduction) — far more impressive than the mean comparison (0.672 vs 0.819) suggested. But on the **95th percentile** fusion is *worse* (2.476 vs 2.125). Fusion improves the typical case and slightly degrades the worst case, which is exactly why the mean difference failed to reach significance: the mean is dragged by a heavier tail.

Effect sizes (paired Cohen's dz, negative = fusion better):

| Comparison | dz | P(fused better) | Magnitude |
|---|---|---|---|
| v_lat fused vs camera | -0.145 | 0.633 | **negligible** |
| v_lat fused vs radar | -1.115 | 0.877 | large |
| v_fwd fused vs camera | -0.448 | 0.843 | small |
| v_fwd fused vs radar | +0.186 | 0.420 | **negligible** |
| position fused vs camera | -0.189 | 0.568 | **negligible** |
| position fused vs radar | -0.372 | 0.754 | small |

This quantifies what the first audit's p-values only implied: fused-vs-camera on lateral velocity is a **negligible** effect (dz=-0.145), while fused-vs-radar is **large** (dz=-1.115).

## 6. Where the occlusion detector fails (new)

The first audit reported one aggregate P/R/F1. This asks where in the grid the errors sit, at the same fixed operating point and on the same held-out frames (300, disjoint from the sweep that chose q and tau).

| Range | Precision | Recall | F1 | GT-occluded cells |
|---|---|---|---|---|
| 0-10m | 0.127 | 0.938 | 0.224 | 599 |
| 10-20m | 0.577 | 0.937 | 0.714 | 11,691 |
| 20-30m | 0.739 | 0.931 | 0.824 | 21,801 |
| 30-40m | 0.860 | 0.897 | 0.878 | 25,654 |

The natural hypothesis — that monocular disparity degrades with distance, so recall should fall with range — is **wrong here**. Recall is nearly flat (0.938 to 0.897). It is *precision* that is range-dependent, and it collapses at the **near** end, the opposite of what the physics predicts.

Part of that is a small-base effect: the first 4 m are never genuinely occluded, so the 0-10 m band has only 599 positive cells against which 3,853 false positives are scored. But the absolute count is the operationally relevant number: that is about 13 spurious OCCLUDED cells per frame within 10 m of the ego — the zone where a braking decision is actually made. This is the detector's concrete weakness and it is invisible in the aggregate 0.73 / 0.92 / 0.81.

**The misses are real, not a labelling artefact.** A plausible dismissal of the remaining ~8% false-negative rate is that it is discretisation disagreement at the edges of occluded regions. It is not:

- interior recall **0.921** (39,229 cells)
- boundary recall **0.913** (20,516 cells)
- boundary cells are 34.3% of all occluded cells and account for 36.5% of all false negatives — proportional

So the residual under-detection is distributed through the interior of occluded regions. It is genuine failure to see a shadow, and should be reported as such rather than explained away.

## 7. Is the episode-disjoint split leak-free at pixel level? (new)

`episode_split` guarantees no episode appears on both sides, and that holds: **0 overlapping episodes** across 136 train / 34 val. That guarantee does not by itself rule out a near-identical crop appearing on both sides, because scenarios are generated procedurally and CARLA 0.10.0 ships one town and a small vehicle catalog. A 64-bit dHash of all 47,638 train and 16,197 val crops:

- exact hash collisions: **42 val crops (0.259%)**
- Hamming<=3 near-duplicates: **259 val crops (1.599%)**

| Class | Near-dup val crops | Val crops | Rate |
|---|---|---|---|
| vehicle | 175 | 7,423 | 2.36% |
| background | 83 | 7,132 | 1.16% |
| pedestrian | 1 | 1,642 | 0.06% |

The residual leak is small but it is **not** the harmless background-patch story one would assume. It concentrates in **vehicles (2.36%)** — consistent with the same blueprint parked in a similar pose across different episodes, which episode splitting cannot remove because it is catalog reuse, not temporal adjacency. Pedestrians, the project's actual subject, are essentially untouched at **0.06%**.

Its effect is bounded two ways. Arithmetically, even if every one of the 259 were a free win, removing them could move validation accuracy by at most 1.6%. Empirically and more convincingly, the independent test set in section 1 is a **different collection entirely** and scored 0.9680 against validation's 0.9679 — if validation were materially inflated by this leak, the test score would have come in below it. It did not.

## 8. Not run, and why

- **Occlusion detector on independent test** — NOT SUPPORTED. the only unused collection has the corrupted occ_grid ground truth
- **Ablation D/F (occlusion conditioning, contradiction resolver)** — SHOULD NOT BE ADDED. PerceptionPipeline exposes no disable flags; would require rewriting the architecture
- **Risk engine / TTC** — NOT SUPPORTED. re-checked: no risk/ttc/hazard key in any meta record
- **Fault detection delay / recovery** — PARTIALLY SUPPORTED. only radar_clutter_onset has an onset (frame 100); all other faults are on from frame 1
- **Classifier object-size analysis** — SKIPPED. saved raw predictions carry tier/visibility but not crop pixel dimensions; visibility used instead
- **Multi-town / Town05** — SHOULD NOT BE ADDED. excluded by instruction; CARLA 0.10.0 ships one town

## 9. What changed for the manuscript

1. **Report the independent test result.** The "no test split" limitation can be retired for the classifier, tracking and intent — with the caveat about why that collection was archived stated plainly.
2. **Reframe fusion around the median and the effect size**, not the mean. Median lateral error halves; the mean does not move significantly because the tail widens. Say both.
3. **Add the lead-time result** — fusion doubles detection at 1–1.5 s before corridor entry. This is the strongest safety-relevant new claim.
4. **Strengthen the silent-failure section**: the monitor fails across the entire severity range, and the radar health statistic is actively inverted by clutter.
5. **State the hidden-target limitation as a perception-chain property**, not an evaluation artefact: 84% of occluded actors were never tracked first.
6. **Report the near-field precision collapse** (0.127 at 0-10 m) as the occlusion detector's remaining weakness, and state that the misses are interior, not boundary artefacts.
7. **The split survives a pixel-level audit** — worth one sentence, since the v1 per-crop-split failure is already in the paper. Note the residual 2.36% vehicle near-duplicate rate honestly.
8. Apply `DENOMINATOR_AUDIT.md` to every reported number.
