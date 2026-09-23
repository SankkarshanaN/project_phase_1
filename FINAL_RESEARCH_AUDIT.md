# Final research audit

Generated 2026-09-22T07:36:00.222028+00:00 · commit `385309f8db`

## 1. Executive summary

All **21 of 21** numerical claims checked against the manuscript reproduce exactly from raw data. The pipeline's components are real, wired together, and the headline numbers are sound. Three things nonetheless need correcting before submission: one stale results file contradicts the paper, the occlusion operating point was tuned and reported on the same frames (now quantified on held-out data), and the flagship fusion-vs-camera lateral velocity improvement is **not statistically significant** under correct episode-level paired analysis. Two experiments could not be run at all for want of ground truth.

Consistency findings: **1 ERROR, 7 WARNING, 4 INFO**.

## 2. Architecture actually detected

| Component | Status | Evidence |
|---|---|---|
| Camera / depth / semseg / radar rig | PRESENT | carla_tools/sensors.py, exercised by every replay |
| YOLOv8n detector | PRESENT | yolov8n.pt, profiled at 8.9 ms/frame |
| Evidential head (Dirichlet) | PRESENT | models/evidential_head.py + trained checkpoint, re-run in this audit |
| Occlusion detector (BEV, 3-state) | PRESENT | perception/occlusion_grid.py, evaluated held-out |
| Bayesian contradiction resolver | PRESENT BUT NOT INDEPENDENTLY SCORED | perception/contradiction.py; no ground truth for posterior correctness |
| EKF tracker | PRESENT | perception/tracking.py, verified against ground truth |
| Particle filter (hidden targets) | PRESENT, WEAKLY EVIDENCED | perception/particle_tracker.py; only 15 scorable events |
| Crossing-intent predictor | PRESENT | perception/intent.py, verified |
| Risk engine | PRESENT BUT UNSCORABLE | perception/risk.py runs, but no risk ground truth exists |
| Sensor health monitor | PRESENT | perception/sensor_health.py, exercised by fault suite |
| Fault injection | PRESENT | scripts/adversarial_test.py, 10 conditions |
| Scenario generator | PRESENT | carla_tools/scenario_gen.py + urban_world.py |
| Ground-truth generation | PRESENT | carla_tools/occlusion_mask.py, true_occupancy.py |

## 3. Dataset summary

- 15,270 frames / 170 episodes / 55,803 object instances (18,096 VRU)
- Scenarios: blindspot_cutin_clear_day (3,515 frames), multi_occlusion_clear_day (2,520 frames), occluded_pedestrian_crossing_clear_day (2,520 frames), urban_crossing_clear_day (6,715 frames)
- Single town (Town10HD_Opt); all clear-daylight
- Split: 144 train / 26 val episodes, episode-disjoint (PASS -- no episode appears on both sides)

## 4. Existing-result verification

| Claim | Manuscript | Recomputed | |diff| | Status |
|---|---|---|---|---|
| 96.2% +/- 0.8% validation accuracy | 0.962 | 0.9619 | 0.000082 | VERIFIED |
| 96.8% best epoch | 0.968 | 0.9679 | 0.000106 | VERIFIED |
| occlusion detector precision | 0.727 | 0.7269 | 0.000140 | VERIFIED |
| occlusion detector recall | 0.917 | 0.9174 | 0.000409 | VERIFIED |
| occlusion detector F1 | 0.811 | 0.8111 | 0.000093 | VERIFIED |
| occlusion cell agreement | 0.78 | 0.7801 | 0.000100 | VERIFIED |
| camera lateral velocity MAE | 0.819 | 0.8187 | 0.000312 | VERIFIED |
| camera forward velocity MAE | 1.048 | 1.0480 | 0.000035 | VERIFIED |
| camera position error | 2.327 | 2.3269 | 0.000105 | VERIFIED |
| radar lateral velocity MAE | 1.683 | 1.6826 | 0.000379 | VERIFIED |
| radar forward velocity MAE | 0.155 | 0.1554 | 0.000365 | VERIFIED |
| radar position error | 2.457 | 2.4565 | 0.000488 | VERIFIED |
| fused lateral velocity MAE | 0.672 | 0.6721 | 0.000089 | VERIFIED |
| fused forward velocity MAE | 0.199 | 0.1986 | 0.000363 | VERIFIED |
| fused position error | 2.032 | 2.0320 | 0.000020 | VERIFIED |
| camera intent AUC | 0.789 | 0.7887 | 0.000275 | VERIFIED |
| camera intent F1 | 0.733 | 0.7327 | 0.000350 | VERIFIED |
| radar intent AUC | 0.743 | 0.7425 | 0.000465 | VERIFIED |
| radar intent F1 | 0.035 | 0.0354 | 0.000387 | VERIFIED |
| fused intent AUC | 0.873 | 0.8729 | 0.000105 | VERIFIED |
| fused intent F1 | 0.855 | 0.8555 | 0.000456 | VERIFIED |

## 5-6. New experiments and results

- **Uncertainty quality** (new): ECE 0.0622, Brier 0.0613, NLL 0.1520, error-detection AUROC 0.942, AURC 0.0025. This is the first evidence that the uncertainty output is actually informative rather than merely present.
- **Occlusion on held-out frames** (new): F1 0.798 vs 0.811 on the tuning sample — optimism gap +0.0129.
- **Occlusion vs severity** (new): recall is flat across severity while precision collapses in mostly-clear scenes — a previously unreported failure mode.
- **Episode-level bootstrap CIs and paired significance** (new): see §7.
- **Hidden-target prediction** (new, sample-limited): particle filter beats EKF dead-reckoning at every horizon by 0.1-1.0 m mean displacement, but on only 15 events.
- **Per-module runtime** (new): serial sum 52.5 ms; slowest stage is sensor health, not MiDaS.

## 7. Statistical validation

Episode-level bootstrap, 10,000 resamples over 44 episodes, on 845 paired observations.

| Comparison | Difference | 95% CI | p | Significant |
|---|---|---|---|---|
| v_lat — fused vs camera | -0.1466 | [-0.3422, +0.0912] | 0.1994 | no |
| v_lat — fused vs radar | -1.0105 | [-1.2037, -0.7660] | 0.0000 | **YES** |
| v_fwd — fused vs camera | -0.8494 | [-1.3273, -0.4384] | 0.0000 | **YES** |
| v_fwd — fused vs radar | +0.0433 | [-0.0057, +0.0980] | 0.0892 | no |
| position — fused vs camera | -0.2949 | [-0.7293, +0.1135] | 0.1534 | no |
| position — fused vs radar | -0.4245 | [-0.6749, -0.1335] | 0.0062 | **YES** |

## 9-10. Failed and skipped experiments

- **G: risk engine** — SKIPPED. no risk/TTC/hazard ground truth exists in the dataset
- **M: multi-town generalisation** — SKIPPED. excluded by explicit instruction; CARLA 0.10.0 ships only Town10HD_Opt in any case
- **E: hidden-target prediction** — PARTIALLY VERIFIED. particle vs EKF coast ADE; SAMPLE-LIMITED (n=15 events)
- **I: calibration/extrinsic perturbation** — PARTIALLY VERIFIED. extrinsic_drift is the only perturbation implemented; no severity sweep exists
- **J: robustness / weather** — PARTIALLY VERIFIED. photometric degradation of clear-day frames ONLY; CARLA weather API non-functional on this build
- No experiment FAILED for technical reasons; two infrastructure bugs (a config-argument mismatch and a regex range) were fixed and re-run.

## 11-13. Limitations, inconsistencies, leakage

- **ERROR** — *results/metrics.json -> occlusion_grid_validation*: STALE: this block still holds the PRE-FIX operating point (precision 0.979, recall 0.173, F1 0.293) from before the 2026-09-17 q/tau fix. The manuscript reports 0.727/0.917/0.811. Two files in results/ therefore disagree; generate_report_figures.py has not been re-run since the fix.
- **WARNING** — *occlusion detector operating point*: q/tau were selected as best-F1 on the sweep's 120-frame seed-0 sample AND the manuscript reports the score from that same sample. On 300 held-out frames the same fixed operating point gives F1 0.7982 vs 0.8111 reported (optimism gap +0.0129). The gap is small, but the reported figure is a tuning-sample figure and should be labelled as such or replaced with the held-out one.
- **WARNING** — *data splits*: There is NO third, untouched test split. The best-epoch checkpoint is SELECTED on the same 26-episode validation split the accuracy is REPORTED on, so 96.8% is a model-selection-contaminated estimate. The 96.2% last-5-epoch mean is less affected but still validation, not test.
- **WARNING** — *fusion claim: v_lat*: fused vs camera on v_lat: difference -0.1466 with 95% CI [-0.3422, +0.0912], p=0.1994 -- NOT distinguishable from zero under episode-level bootstrap (44 episodes). The point estimate favours fusion but the dataset does not support a significance claim on this axis.
- **WARNING** — *fusion claim: position*: fused vs camera on position: difference -0.2949 with 95% CI [-0.7293, +0.1135], p=0.1534 -- NOT distinguishable from zero under episode-level bootstrap (44 episodes). The point estimate favours fusion but the dataset does not support a significance claim on this axis.
- **WARNING** — *fault injection count*: This run injects 10 fault conditions and the health monitor detects 6 of them (blur, darkness, fog, lens_blocked, noise, radar_dropout); undetected: extrinsic_drift, radar_bias, radar_clutter, radar_clutter_onset. Any 'N of M faults' claim must state M -- a '5 of 7' phrasing does not match this suite.
- **WARNING** — *runtime claim*: Perception modules sum to 52.5 ms serial (19.0 FPS upper-bound cost) with CARLA out of the loop, whereas the manuscript's 2-9 FPS is the live demo's synchronous tick rate including CARLA rendering. These are different quantities and must not be presented as one.
- **WARNING** — *hidden-target experiment*: Only 15 occlusion events were scorable across 10 episodes. Horizons with n<10: ['1.5s', '2.0s', '3.0s']. At 3.0s n=1, so its 'significant' CI is degenerate and meaningless. This experiment is sample-limited and must not carry a headline claim.
- **INFO** — *manuscript Fig. confusion matrix vs Sec. results*: Confusion-matrix accuracy is 0.967894 (= best-epoch val accuracy, epoch 14), while the headline 96.2% +/- 0.8% is the mean +/- std of the LAST FIVE EPOCHS (0.961918 +/- 0.007611). Both are correct but they are different quantities; a reader recomputing accuracy from the printed confusion matrix gets 96.79%, not 96.2%. The manuscript must say which is which.
- **INFO** — *confusion matrix*: Recomputed confusion matrix is IDENTICAL to the stored one.
- **INFO** — *data splits*: Episode-disjoint split verified: PASS -- no episode appears on both sides; 144 train / 26 val episodes matches the manuscript's stated 144/26.
- **INFO** — *runtime breakdown*: Slowest stage is sensor_health_update at 25.8 ms, not MiDaS (16.3 ms). Documentation stating MiDaS dominates is contradicted by measurement.

## 14. Reproducibility

- Commit `385309f8db00ef7c599bcb5349205928ed3e8945`; 12 uncommitted files at audit time
- Python 3.12.11, Windows-11-10.0.26200-SP0
- GPU NVIDIA GeForce RTX 5060 Laptop GPU
- Seeds: {'episode_split': 0, 'occlusion_holdout': 1234, 'bootstrap': 0, 'sweep_sample': 0}
- CARLA is stochastic and its weather API is non-functional on this build; live-demo runs are not bit-reproducible. All experiments in this package are OFFLINE replays of recorded data and ARE deterministic.

## 17. Claims currently supported by evidence

- The evidential classifier reaches 96.79% on an episode-disjoint validation split, and its uncertainty is genuinely informative (error-detection AUROC 0.942).
- Uncertainty rises and accuracy falls under partial occlusion, as designed.
- The occlusion detector holds up on held-out frames (F1 0.798) close to its tuned figure.
- Radar is structurally blind to lateral velocity, and fusion fixes it (significant, p<0.0001).
- Camera is weak on forward velocity, and fusion fixes it (significant, p<0.0001).
- Fused crossing-intent F1 (0.855) exceeds camera (0.733) and radar (0.035).
- The health monitor detects 6/10 injected faults, and the undetected ones are explained structurally.

## 18. Claims NOT currently supported

- **"Fusion improves lateral velocity over the camera."** The point estimate favours fusion (0.672 vs 0.819) but the paired episode-level difference is -0.147 with CI [-0.342, 0.091], p=0.199. Not significant at 44 episodes.
- **Any quantitative risk-engine claim.** No risk/TTC/hazard ground truth exists.
- **Any claim about the Bayesian contradiction resolver's accuracy.** Its posteriors are never scored against ground truth; only the 17x conditioning contrast is measured, which shows the mechanism works, not that it is correct.
- **Strong claims about hidden-target prediction.** 15 events is too few.
- **Real adverse-weather robustness.** Only photometric degradation of clear-day frames was tested.
- **Multi-town or cross-environment generalisation.** Single town only.

## 19. Recommended manuscript corrections

1. State explicitly that the confusion matrix is the **best-epoch** validation matrix (96.79%) while 96.2%±0.8% is the **last-five-epoch mean** — a reader recomputing from the figure will otherwise think the numbers disagree.
2. Regenerate or delete the stale `occlusion_grid_validation` block in `results/metrics.json`; it still carries pre-fix numbers that contradict the paper.
3. Either report the held-out occlusion figures (P 0.706 / R 0.918 / F1 0.798) or label the current ones as tuning-sample figures.
4. Soften the fusion-vs-camera lateral-velocity claim to the complementary-blindness framing, which IS significant, and report the CIs.
5. State the fault denominator explicitly (6 of 10 in this suite).
6. Separate the runtime claims: perception 53 ms serial vs live-demo tick rate 2-9 FPS.
7. Add the missing caveat that the classifier never sees fully-occluded crops.
8. State that no independent test split exists and that the best epoch is selected on the reported split.

## 20. Verdict

The experimental evidence supports a strong, honest systems-and-measurement paper: the components exist, the numbers reproduce exactly, and several limitations are now quantified rather than asserted. It is **not yet** a finished Tier-1 results package: the risk engine — a headline component — has no ground truth and is therefore unevaluated, hidden-target prediction rests on 15 events, and the flagship fusion comparison loses significance under correct paired analysis. Those are fixable with additional data collection (risk/TTC labels, more occlusion events), not with reanalysis of what exists.
