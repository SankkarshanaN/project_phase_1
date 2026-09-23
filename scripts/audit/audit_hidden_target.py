"""AUDIT PHASE 7 -- hidden-target prediction: particle filter vs EKF coast.

This is the experiment the manuscript describes qualitatively (the particle
filter's multi-modality) but never scores against ground truth. It is possible
here only because the dataset deliberately keeps AMODAL positions for actors
that are fully occluded (`obj_xy_ego` is recorded regardless of visibility --
see CLAUDE.md on why those rows were kept rather than dropped). That is the
ground truth a hidden-target error can be measured against.

Protocol, mirroring `perception.pipeline`'s own handoff:
  * replay an episode through the same EKF tracker used everywhere else,
    with occluded actors withheld from the detections exactly as at runtime;
  * at the frame an actor's tier first becomes OCCLUDED, seed a
    `HiddenHazardTracker` from the LAST CONFIRMED estimate (pipeline._handoff
    does the same, and for the same stated reason);
  * from that moment propagate BOTH the particle filter and the EKF's own
    dead-reckoning, with no measurement of any kind reaching either;
  * score both against the recorded amodal truth at fixed horizons.

Neither predictor is re-implemented; both are the shipped modules.

Writes results/audit/hidden_prediction_metrics.json + figure.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carla_tools.occlusion_tiers import OCCLUDED
from common.config import load_yaml
from common.npz_io import load_npz
from perception.particle_tracker import HiddenHazardTracker
from perception.tracking import MultiObjectTracker
from scripts.evaluate_prediction import (_episode_key, _match_tracks_to_truth,
                                          build_detections)

DATA = Path("data/raw_v2")
OUT = Path("results/audit")
HORIZONS_S = [0.5, 1.0, 1.5, 2.0, 3.0]
MISS_THRESHOLD_M = 2.0      # a "miss" if predicted position is this far off


def main():
    bev_cfg = load_yaml("bev.yaml")
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])
    cam_cfg = bev_cfg["camera"]
    horizons_f = [int(round(h / dt)) for h in HORIZONS_S]

    by_ep = defaultdict(list)
    for p in sorted((DATA / "meta").glob("*.npz")):
        by_ep[_episode_key(p)].append(p)

    # err[method][horizon_frames] -> list of displacement errors (m)
    err = {m: defaultdict(list) for m in ("particle", "ekf_coast")}
    n_events = 0
    n_episodes_with_events = 0
    spread_at = defaultdict(list)      # particle position_spread, for reporting

    for key, paths in sorted(by_ep.items()):
        frames = [load_npz(p) for p in sorted(paths, key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
        tracker = MultiObjectTracker()
        # actor -> truth xy per frame index, for scoring during occlusion
        truth_xy = [{} for _ in frames]
        tier_of = [{} for _ in frames]
        for fi, fr in enumerate(frames):
            for j in range(len(fr["obj_actor_id"])):
                if not fr["obj_is_vru"][j]:
                    continue
                aid = int(fr["obj_actor_id"][j])
                truth_xy[fi][aid] = np.asarray(fr["obj_xy_ego"][j], float)
                tier_of[fi][aid] = int(fr["obj_tier"][j])

        last_seen = {}          # actor -> (xy, vel, P) at last CONFIRMED sighting
        prev_tier = {}
        ep_events = 0

        for fi, fr in enumerate(frames):
            ego_speed = float(np.hypot(fr["ego_vx"], fr["ego_vy"]))
            tracker.predict(dt, ego_speed=ego_speed)
            dets, truth = build_detections(fr, "fused", cam_cfg, {})
            tracker.update(dets, ego_vel=np.array([ego_speed, 0.0]))

            confirmed = tracker.confirmed_tracks()
            for aid, track, xy, vel, tier in _match_tracks_to_truth(confirmed, truth):
                if tier != OCCLUDED and not track.is_coasting:
                    last_seen[aid] = (track.position.copy(), track.velocity.copy(),
                                       track.P.copy())

            for aid, tier in tier_of[fi].items():
                became_hidden = (tier == OCCLUDED and prev_tier.get(aid) not in (OCCLUDED, None))
                if not became_hidden or aid not in last_seen:
                    continue
                xy0, vel0, P0 = last_seen[aid]
                pf = HiddenHazardTracker(init_xy=xy0, init_vel=vel0, init_cov=P0,
                                          rng=np.random.default_rng(0), is_vru=True)
                ekf_xy, ekf_v = xy0.copy(), vel0.copy()
                scored_any = False

                for step in range(1, max(horizons_f) + 1):
                    gi = fi + step
                    if gi >= len(frames):
                        break
                    es = float(np.hypot(frames[gi]["ego_vx"], frames[gi]["ego_vy"]))
                    # Both predictors advance with NO measurement, only ego motion.
                    pf.predict(dt, ego_speed=es, ego_yaw_rate=0.0)
                    ekf_xy = ekf_xy + ekf_v * dt - np.array([es * dt, 0.0])

                    if tier_of[gi].get(aid) != OCCLUDED:
                        break            # no longer hidden; stop scoring this event
                    if aid not in truth_xy[gi]:
                        break
                    if step not in horizons_f:
                        continue
                    gt = truth_xy[gi][aid]
                    err["particle"][step].append(float(np.linalg.norm(pf.mean_position - gt)))
                    err["ekf_coast"][step].append(float(np.linalg.norm(ekf_xy - gt)))
                    spread_at[step].append(float(pf.position_spread))
                    scored_any = True

                if scored_any:
                    ep_events += 1
                prev_tier_update = True  # noqa: F841

            prev_tier = dict(tier_of[fi])

        n_events += ep_events
        if ep_events:
            n_episodes_with_events += 1

    def stats(vals):
        a = np.asarray(vals, float)
        if a.size == 0:
            return None
        return {"n": int(a.size), "ade_mean_m": float(a.mean()),
                "rmse_m": float(np.sqrt(np.mean(a ** 2))),
                "median_m": float(np.median(a)), "p95_m": float(np.percentile(a, 95)),
                "miss_rate_at_2m": float((a > MISS_THRESHOLD_M).mean())}

    per_h = {}
    for h_s, h_f in zip(HORIZONS_S, horizons_f):
        row = {}
        for m in ("particle", "ekf_coast"):
            s = stats(err[m][h_f])
            if s:
                row[m] = s
        if row:
            sp = np.asarray(spread_at[h_f], float)
            row["particle_position_spread_m"] = (
                {"mean": float(sp.mean()), "median": float(np.median(sp))}
                if sp.size else None)
            # Paired difference on the SAME events.
            a = np.asarray(err["particle"][h_f], float)
            b = np.asarray(err["ekf_coast"][h_f], float)
            if a.size and a.size == b.size:
                d = a - b
                rng = np.random.default_rng(0)
                boots = [float(rng.choice(d, size=d.size, replace=True).mean())
                         for _ in range(2000)]
                lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
                row["paired_particle_minus_ekf"] = {
                    "mean_difference_m": float(d.mean()), "ci95": [lo, hi],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "better": "particle" if d.mean() < 0 else "ekf_coast",
                    "bootstrap_unit": "event (NOT episode -- see note)",
                }
            per_h[f"{h_s:.1f}s"] = row

    out = {
        "experiment_id": "AUDIT-E-HIDDEN-TARGET",
        "status": "EXECUTED" if per_h else "NO EVENTS FOUND",
        "ground_truth": ("recorded amodal obj_xy_ego for actors whose tier is OCCLUDED; "
                          "available only because fully-hidden actors are deliberately "
                          "retained in meta/ (they were dropped in dataset v1)"),
        "protocol": ("seed at first OCCLUDED frame from the last confirmed EKF estimate "
                      "(mirrors pipeline._handoff), then propagate both predictors with "
                      "no measurement; score against amodal truth while still hidden"),
        "methods_compared": {
            "particle": "perception.particle_tracker.HiddenHazardTracker (mean_position)",
            "ekf_coast": "constant-velocity dead reckoning with ego-motion compensation, "
                          "i.e. what perception.tracking does while coasting",
        },
        "n_occlusion_events_scored": n_events,
        "n_episodes_with_events": n_episodes_with_events,
        "miss_threshold_m": MISS_THRESHOLD_M,
        "denominator": "one sample = one (occlusion event, horizon) pair",
        "by_horizon": per_h,
        "caveats": [
            "Events, not episodes, are the bootstrap unit here -- several events can "
            "come from one episode, so these CIs are less conservative than the "
            "episode-level ones used for the tracking comparison.",
            "FDE is not reported separately: an event is scored only while the actor "
            "is still hidden, so the last scored horizon IS the final displacement "
            "for that event and reporting it twice under another name would be "
            "double-counting.",
            "The particle filter's own emergence prediction is not scored here; this "
            "measures position belief only.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hidden_prediction_metrics.json").write_text(json.dumps(out, indent=2))

    if per_h:
        figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=200)
        hs = [h for h in per_h if "particle" in per_h[h]]
        xs = [float(h[:-1]) for h in hs]
        for m, c, lbl in (("particle", "#1f9e6e", "particle filter"),
                          ("ekf_coast", "#c0392b", "EKF dead reckoning")):
            ys = [per_h[h][m]["ade_mean_m"] for h in hs]
            ns = [per_h[h][m]["n"] for h in hs]
            ax.plot(xs, ys, marker="o", color=c, label=lbl)
        ax.set_xlabel("time hidden (s)"); ax.set_ylabel("mean displacement error (m)")
        ax.set_title("Hidden-target position error while fully occluded",
                     fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(figdir / "audit_hidden_target_ade.png")
        plt.close(fig)

    print(json.dumps({"n_events": n_events, "n_episodes": n_episodes_with_events,
                      "by_horizon": per_h}, indent=2)[:2600])


if __name__ == "__main__":
    main()
