# Denominator audit

Every headline metric with the population it was actually computed over. The dataset contains 15,270 frames, but **no metric in this project is computed over 15,270 frames** — quoting that number next to a result would misstate the sample by more than an order of magnitude.

| Metric | Value | Computed over | Count | NOT |
|---|---|---|---|---|
| Classifier accuracy (validation) | 0.9679 | validation crops, 26 episodes | 12,552 crops | frames |
| Classifier accuracy (independent test) | 0.9680 | test crops, 189 unseen episodes | 62,528 crops | frames |
| Occlusion P/R/F1 (held-out) | 0.706/0.918/0.798 | BEV grid cells over 300 frames | 120,000 cells | frames or objects |
| Tracking v_lat/v_fwd/position MAE | fused 0.672 m/s | paired observations tracked by ALL modes AND moving laterally >=0.3 m/s | 845 observations from 44 episodes | frames or all VRUs |
| Crossing-intent AUC/F1 (fused) | 0.873/0.855 | scored observations for THAT configuration (unpaired across modes) | 3,065 observations | frames; and NOT paired |
| Fusion significance tests | p-values / CIs | episode-level bootstrap over paired observations | 845 observations, 44 episodes | frames |
| Hidden-target ADE | sample-limited | occlusion events (one event = one actor going hidden once) | 15 events from 10 episodes | frames or observations |
| Fault injection | 6 detected | fault conditions over 24 sampled episodes | 10 conditions | frames |
| Runtime latency | 52.5 ms serial | per-call timings on recorded frames, CARLA out of loop | 40 timed calls per stage | the live demo's 2-9 FPS tick rate |
| Intent lead-time recall | band-specific | positives whose TRUE time-to-entry falls in the band, vs a shared negative pool | 1,597 positives in 0.0-0.5s down to 2 in 2.5-3.0s (fused) | a per-band negative pool |

## Denominator rules for the manuscript

1. Never attach "15,270 frames" to a performance number. It describes the dataset, not any metric's sample.
2. Tracking numbers are **845 paired observations from 44 episodes** — say both, because 845 sounds large and 44 does not.
3. Intent numbers are **not paired** across camera/radar/fused (camera 3,665, radar 2,491, fused 3,065); never call that comparison paired.
4. Occlusion numbers are per **cell**, not per frame or object.
5. Classifier numbers are per **crop**, and the validation and test sets are different sizes (12,552 vs 62,528).
