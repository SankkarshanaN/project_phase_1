"""An actual p_cross-over-time trace, replayed through the real tracker and
intent predictor -- not illustrative.

Why this exists
----------------
`generate_framework_figures.fig_crossing_timeline` produces
`figures/crossing_timeline.png`, and despite its name and an earlier version
of that script's own header comment claiming it plots "visibility, tier and
p_cross over time," it does not compute or plot p_cross at all -- only
visibility fraction and occlusion tier. A mentor-facing page built from it
inherited a caption describing a figure that does not exist. This script is
the fix: it actually runs `perception.tracking` + `perception.intent` over
recorded frames and plots the real output.

It plots two real encounters on purpose:
  - a genuine crossing (p_cross rises, the person actually enters the
    ego's corridor)
  - a genuine near-miss: the model's probability rises to a real, moderate
    value as the walker approaches the corridor edge, then falls back to
    zero as they veer off and never enter -- because a forecast that is
    never wrong is not a forecast, and showing only the easy case would
    misrepresent what "might cross" means. The fused model's measured
    precision is 0.892 at recall 0.834 (see docs/RESULTS.md SS5); cases
    like this are exactly what keeps precision below 1.0.

Selecting the near-miss example is NOT fully automatic. An actor is only a
candidate if the retrospective `will_cross` label (from
label_crossing_intent.py) is never True for them, they were ever within
sensor range (see NEAR_FORWARD_M/NEAR_LATERAL_M below -- otherwise the
longest-observed "non-crosser" is usually a background pedestrian 100+ m
away that the tracker never confirms a track for at all), and the resulting
p_cross trace was inspected by hand for a clean, artifact-free rise and
fall (several candidates hit p_cross=1.0 and never cross, which needs a
tracking-artifact audit -- an abrupt spike could be an ID switch rather than
a genuine ambiguous approach -- before being trusted as an honest example).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml
from common.npz_io import load_npz
from perception.tracking import MultiObjectTracker
from scripts.evaluate_prediction import _episode_key, _match_tracks_to_truth, build_detections, predict_crossing

# A candidate actor must have come within this range of the ego at some
# point, or `pick()` selects background pedestrians the tracker never
# confirms a track for -- found by direct inspection: an early candidate sat
# at (138 m forward, 35 m lateral), 100+ m from every confirmed track, for
# all 317 of its recorded observations.
NEAR_FORWARD_M, NEAR_LATERAL_M = 40.0, 15.0

# Hand-verified near-miss: a smooth rise 0.18 -> 0.38 as the walker approaches
# the corridor edge, then a smooth fall back to 0.00 as they veer off, never
# entering. No discontinuities, no ID-switch artifacts -- see the module
# docstring for why this was not picked automatically.
DEFAULT_NEAR_MISS = ("urban_crossing_clear_day_ep67000", 27)


def _load_reachable_actors(data_dir: Path):
    """Returns (by_actor, reachable) -- retrospective will_cross history per
    (episode, actor_id), and the subset ever within sensor range."""
    lab = load_npz(data_dir / "intent_labels.npz")
    by_actor = defaultdict(list)
    reachable = set()
    for i in range(len(lab["episode"])):
        key = (str(lab["episode"][i]), int(lab["actor_id"][i]))
        by_actor[key].append((int(lab["frame_idx"][i]), bool(lab["will_cross"][i])))
        if (abs(float(lab["forward_m"][i])) <= NEAR_FORWARD_M
                and abs(float(lab["lateral_m"][i])) <= NEAR_LATERAL_M):
            reachable.add(key)
    return by_actor, reachable


def pick_crosser(by_actor, reachable, min_len=25):
    """Longest-observed reachable actor whose `will_cross` is True somewhere
    in its history -- see the module docstring for why "somewhere," not
    "at the final frame": once someone HAS entered the corridor, later
    frames are labelled False (no longer "about to"), so no actor's final
    label is ever True."""
    best_key, best_n = None, 0
    for key, rows in by_actor.items():
        if key not in reachable or len(rows) < min_len:
            continue
        if not any(v for _, v in rows):
            continue
        if len(rows) > best_n:
            best_key, best_n = key, len(rows)
    return best_key


def trace_episode(by_ep, cam_cfg, dt, ep_key, target_actor):
    """Replays one episode through a fresh tracker, returns the real
    (time_s, p_cross, p_cross_cautious, tier, in_corridor_now) arrays for one
    actor."""
    frames = sorted(by_ep[ep_key], key=lambda q: int(q.stem.rsplit("_f", 1)[1]))
    tracker = MultiObjectTracker()
    ts, p_cross, p_cautious, in_corridor_now, tier_seen = [], [], [], [], []
    for fr_path in frames:
        frame = load_npz(fr_path)
        es = float(np.hypot(frame["ego_vx"], frame["ego_vy"]))
        tracker.predict(dt, ego_speed=es)
        dets, truth = build_detections(frame, "fused", cam_cfg, {})
        tracker.update(dets, ego_vel=np.array([es, 0.0]))

        target_row = next((r for r in truth if r[0] == target_actor), None)
        if target_row is None:
            continue
        matches = _match_tracks_to_truth(tracker.confirmed_tracks(), [target_row])
        if not matches:
            continue
        _aid, best, xy, vel, tier = matches[0]

        pred = predict_crossing(best, ego_speed=es)
        ts.append(int(frame["frame_idx"]))
        p_cross.append(pred.p_cross)
        p_cautious.append(pred.p_cross_cautious)
        tier_seen.append(int(tier))
        in_corridor_now.append(bool(abs(float(xy[1])) <= 1.75 and float(xy[0]) > 0))

    return (np.array(ts, dtype=float) * dt, np.array(p_cross), np.array(p_cautious),
            np.array(tier_seen), np.array(in_corridor_now))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw_v2")
    ap.add_argument("--out", default="results/figures/crossing_probability_trace.png")
    ap.add_argument("--near-miss-episode", default=DEFAULT_NEAR_MISS[0])
    ap.add_argument("--near-miss-actor", type=int, default=DEFAULT_NEAR_MISS[1])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cam_cfg = load_yaml("bev.yaml")["camera"]
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])

    by_actor, reachable = _load_reachable_actors(data_dir)
    by_ep = defaultdict(list)
    for p in sorted((data_dir / "meta").glob("*.npz")):
        by_ep[_episode_key(p)].append(p)

    crosser_key = pick_crosser(by_actor, reachable)
    near_miss_key = (args.near_miss_episode, args.near_miss_actor)
    print(f"crossing example:  {crosser_key}")
    print(f"near-miss example: {near_miss_key}")
    if crosser_key is None:
        raise SystemExit("No reachable actor with a positive will_cross label found.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    for ax, key, title in ((axes[0], crosser_key, "A genuine crossing"),
                            (axes[1], near_miss_key, "Approached, then did not cross")):
        ep, aid = key
        t, pc, pcc, tiers, corridor = trace_episode(by_ep, cam_cfg, dt, ep, aid)
        ax.plot(t, pc, color="#0f6e63", linewidth=2, label="p_cross (honest)", zorder=3)
        ax.plot(t, pcc, color="#b4560a", linewidth=1.4, linestyle="--",
                label="p_cross_cautious", zorder=2, alpha=0.85)
        ax.fill_between(t, 0, 1, where=corridor, color="#c0392b", alpha=0.12,
                        step="mid", label="actually in corridor")
        ax.axhline(0.5, color="#888", linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("time (s)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
        ax.grid(alpha=0.2)
        ax.tick_params(labelleft=True)  # sharey hides these on the right panel by default

    axes[0].set_ylabel("crossing probability")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Real p_cross traces, replayed through the actual tracker + intent "
                 "predictor -- not illustrative", fontsize=10, y=1.03)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
