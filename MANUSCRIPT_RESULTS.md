# Manuscript-ready results
Every figure below was recomputed from raw data by the audit scripts in `scripts/audit/`. Source file and sample count are given for each block so each number can be traced. **Numbers not listed here were not computed and must not be claimed.**

## Dataset
The evaluation dataset comprises **15,270 frames** across **170 episodes** in a single CARLA town (`Town10HD_Opt`), containing **55,803 object instances** of which **18,096** are vulnerable road users. Per-object occlusion tiers: OCCLUDED 20,912, PARTIAL 5,308, VISIBLE 29,583. All frames are clear-daylight; CARLA 0.10.0's weather API is non-functional on this build, so adverse conditions are evaluated by photometric degradation of recorded frames rather than by simulated weather.

*Source: `results/audit/dataset_statistics.json`.*

## Splits
Crops are split **by episode, never by frame**: 144 training and 26 validation episodes (12,552 validation crops). Episode disjointness was verified programmatically (PASS -- no episode appears on both sides). **No independent test split exists**: the best-epoch checkpoint is selected on the same validation split on which accuracy is reported.

*Source: `results/audit/splits.json`.*

## Evidential classifier
On the 12,552-crop episode-disjoint validation split, the evidential classifier attains **96.79% accuracy** (macro-F1 0.965, weighted-F1 0.968). Per class: vehicle P 0.988 / R 0.949 / F1 0.968 (n=5,654), pedestrian P 0.935 / R 0.981 / F1 0.957 (n=1,326), background P 0.957 / R 0.984 / F1 0.970 (n=5,572).

Uncertainty quality, computed from raw Dirichlet outputs: **ECE 0.0622**, Brier 0.0613, NLL 0.1520, and — the operative claim — **error-detection AUROC 0.942**, i.e. predicted uncertainty separates misclassified from correctly classified crops far above chance. Uncertainty responds to occlusion as designed: partially-occluded crops carry mean uncertainty 0.216 at 88.6% accuracy (n=1,269), versus 0.133 at 97.1% for fully visible crops (n=5,711).

> **Caveat that must accompany this block.** Fully-occluded objects are absent from the classifier's data by construction (`labels/` carries only camera-observable actors), so this comparison spans VISIBLE vs PARTIAL only.

*Source: `results/audit/classification_metrics.json`, `uncertainty_metrics.json`.*

## Occlusion detector
At the fixed operating point (q=0.01, tau=0.001), evaluated on **300 frames disjoint from the sample used to select that operating point** (120,000 BEV cells), the detector attains precision **0.706**, recall **0.918**, F1 **0.798**, IoU 0.664, cell agreement 0.769. Per-frame F1 is 0.778 (95% CI [0.760, 0.795], frame-level bootstrap).

The previously reported figures (0.727/0.917/0.811) come from the tuning sample itself; the optimism gap is **+0.0129 F1**, with recall essentially unchanged (-0.0004).

Performance is strongly severity-dependent: at 0.20-0.50 scene occlusion, P 0.545 / R 0.929 (n=127 frames); at 0.50-0.65 scene occlusion, P 0.806 / R 0.916 (n=94 frames); at 0.65-1.01 scene occlusion, P 0.944 / R 0.909 (n=69 frames). Recall is near-constant across severity while precision collapses in mostly-clear scenes, where few cells are genuinely occluded.

*Source: `results/audit/occlusion_metrics.json`.*

## Sensor fusion: tracking
On **845 paired observations** (identical actors and frames tracked by all three configurations, drawn from 44 episodes):

| Configuration | v_lat MAE (m/s) | 95% CI | v_fwd MAE (m/s) | Position MAE (m) |
|---|---|---|---|---|
| camera | 0.819 | [0.691, 0.964] | 1.048 | 2.327 |
| radar | 1.683 | [1.567, 1.782] | 0.155 | 2.457 |
| fused | 0.672 | [0.489, 0.910] | 0.199 | 2.032 |

Confidence intervals are episode-level bootstrap (10,000 resamples over 44 episodes), because consecutive frames within an episode are correlated and frame-level intervals would be far too narrow.

**Paired significance (the defensible form of the fusion claim).** Fusion is significantly better than a single sensor precisely on the axis that sensor is structurally blind to: v_lat fused vs radar (-1.011, p=0.0000); v_fwd fused vs camera (-0.849, p=0.0000); position fused vs radar (-0.425, p=0.0062). The following differences are **not** distinguishable from zero at this sample size: v_lat fused vs camera (-0.147, p=0.1994); v_fwd fused vs radar (+0.043, p=0.0892); position fused vs camera (-0.295, p=0.1534).

*Source: `results/audit/tracking_metrics.json`, `statistics.json`.*

## Crossing-intent prediction
| Configuration | n | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced acc. |
|---|---|---|---|---|---|---|---|
| camera | 3,665 | 0.789 | 0.804 | 0.833 | 0.654 | 0.733 | 0.755 |
| radar | 2,491 | 0.743 | 0.829 | 1.000 | 0.018 | 0.035 | 0.509 |
| fused | 3,065 | 0.873 | 0.866 | 0.818 | 0.896 | 0.855 | 0.825 |

Radar alone achieves precision 1.000 at recall 0.018 — it almost never predicts a crossing, so its precision is vacuous; F1 is the honest summary. Note the three configurations score different numbers of observations (camera 3,665, radar 2,491, fused 3,065), so these rows are not paired; coverage itself differs by configuration.

*Source: `results/audit/intent_metrics.json`.*

## Fault injection
Across **10 injected fault conditions**, the self-referential health monitor detects **6**: `blur`, `darkness`, `fog`, `lens_blocked`, `noise`, `radar_dropout`. It fails to detect **4**: `extrinsic_drift`, `radar_bias`, `radar_clutter`, `radar_clutter_onset`. Of those, the script classifies `radar_clutter`, `radar_bias`, `radar_clutter_onset` as *silent failures* — they measurably degraded the pipeline while health remained nominal, which is the most dangerous category.

*Source: `results/audit/fault_metrics.json`.*

## Runtime
Perception-module latency measured on recorded frames with CARLA **out of the loop**, on NVIDIA GeForce RTX 5060 Laptop GPU: sensor_health_update 25.8 ms, midas_depth 16.3 ms, yolov8n_detect 8.9 ms, occlusion_grid_given_disparity 0.7 ms, evidential_head_per_crop 0.6 ms, ekf_tracking_step 0.1 ms, particle_filter_step_400p 0.1 ms, radar_confidence 0.0 ms. Serial sum is **52.5 ms** (19.0 FPS upper-bound cost).

> The live demo's 2–9 FPS is a *synchronous-mode simulation tick rate* that includes CARLA rendering and the server round-trip. It is not perception throughput and the two must not be reported as the same quantity.

*Source: `results/audit/runtime_metrics.json`.*

## Experiments that were NOT performed
- **G: risk engine** — SKIPPED: no risk/TTC/hazard ground truth exists in the dataset
- **M: multi-town generalisation** — SKIPPED: excluded by explicit instruction; CARLA 0.10.0 ships only Town10HD_Opt in any case
- **Hidden-target prediction** — executed but sample-limited: only 15 scorable occlusion events across 10 episodes; no headline claim is supportable.
