"""Derives per-frame crossing-intent ground truth from a collected dataset.

For every VRU in every frame, answers: did this actor actually enter the ego's
path corridor within the next `horizon` seconds?

Runs entirely offline -- no CARLA server, no GPU. It reads only the per-object
arrays `data_collector` writes into each frame's `.npz`.

Why the label is retrospective rather than taken from the walker's scripted
behaviour: what a walker was *told* to do and what actually happened routinely
differ. They collide with street furniture, get blocked on a kerb, are still
mid-road when the episode ends, or -- in the `stop_midway` case -- deliberately
halt somewhere that may or may not be inside the corridor depending on the
geometry that episode happened to draw. Labelling from the recorded trajectory
measures the outcome; labelling from `BehaviourParams.kind` would measure the
script. The behaviour name is still recorded so results can be sliced by it,
but it is never the label.

The corridor test is done in the ego frame of each individual frame, using the
`obj_xy_ego` positions already stored there. No world-frame reconstruction is
needed, and none is attempted: re-deriving relative geometry from absolute ego
poses would reintroduce rounding error for no benefit.

Output: one consolidated `intent_labels.npz` at the dataset root, a flat table
with one row per (frame, actor).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml
from perception.intent import DEFAULT_CORRIDOR_HALF_WIDTH

DEFAULT_HORIZON_S = 3.0

# Envelope defining which VRUs a crossing predictor would plausibly be asked
# about, used only for reporting class balance honestly -- see below.
NEAR_LATERAL_M = 12.0
NEAR_FORWARD_M = 40.0


def _episode_key(path: Path) -> str:
    return path.stem.rsplit("_f", 1)[0]


def _in_corridor(xy: np.ndarray, half_width: float) -> np.ndarray:
    """Laterally inside the ego's path AND still ahead of it.

    Same test as `intent.predict_crossing`, deliberately -- a prediction is
    only meaningful if it is scored against the same definition of the event
    it was predicting.
    """
    if xy.size == 0:
        return np.zeros(0, dtype=bool)
    return (np.abs(xy[:, 1]) <= half_width) & (xy[:, 0] > 0.0)


def label_episode(frames: list[Path], horizon_s: float, half_width: float) -> list[dict]:
    """Labels one episode. `frames` must be its .npz paths, any order."""
    loaded = []
    for p in sorted(frames, key=lambda q: int(q.stem.rsplit("_f", 1)[1])):
        d = np.load(p, allow_pickle=True)
        if "obj_actor_id" not in d.files:
            # A frame written before the per-object record existed. Skip the
            # whole episode rather than labelling it half-way.
            return []
        loaded.append((p, d))

    if not loaded:
        return []

    times = np.array([float(d["sim_time"]) for _p, d in loaded])

    # Per-actor occupancy over the episode: {actor_id: {frame_index: in_corridor}}
    occupancy: dict[int, dict[int, bool]] = defaultdict(dict)
    for i, (_p, d) in enumerate(loaded):
        ids = d["obj_actor_id"]
        vru = d["obj_is_vru"]
        inside = _in_corridor(d["obj_xy_ego"], half_width)
        for aid, is_vru, ins in zip(ids, vru, inside):
            if is_vru:
                occupancy[int(aid)][i] = bool(ins)

    rows = []
    for i, (path, d) in enumerate(loaded):
        ids = d["obj_actor_id"]
        for j, aid in enumerate(ids):
            if not d["obj_is_vru"][j]:
                continue
            aid = int(aid)

            # Look forward over the horizon for the first frame this actor is
            # inside the corridor. Strictly future frames: "will cross" must
            # not be satisfied by the current frame, or a walker already in the
            # road trivially predicts themselves.
            will_cross, entry_time = False, np.nan
            for k in range(i + 1, len(loaded)):
                if times[k] - times[i] > horizon_s:
                    break
                if occupancy[aid].get(k, False):
                    will_cross, entry_time = True, float(times[k] - times[i])
                    break

            rows.append({
                "episode": _episode_key(path),
                "scenario": str(d["scenario_type"]),
                "weather": str(d["weather_name"]),
                "frame_idx": int(d["frame_idx"]),
                "actor_id": aid,
                "in_corridor_now": bool(occupancy[aid].get(i, False)),
                "will_cross": will_cross,
                "time_to_entry_s": entry_time,
                "visibility": float(d["obj_visibility"][j]),
                "tier": int(d["obj_tier"][j]),
                "forward_m": float(d["obj_xy_ego"][j, 0]),
                "lateral_m": float(d["obj_xy_ego"][j, 1]),
                "v_forward": float(d["obj_vel_ego"][j, 0]),
                "v_lateral": float(d["obj_vel_ego"][j, 1]),
                "ego_speed": float(np.hypot(d["ego_vx"], d["ego_vy"])),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw_v2")
    ap.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S,
                     help="How far ahead an entry counts as 'will cross'.")
    ap.add_argument("--corridor-half-width", type=float, default=None,
                     help="Defaults to perception.intent's corridor width, so labels "
                          "and predictions share one definition of the event.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    half_width = args.corridor_half_width or DEFAULT_CORRIDOR_HALF_WIDTH
    data_dir = Path(args.data_dir)
    meta_dir = data_dir / "meta"
    if not meta_dir.is_dir():
        raise SystemExit(f"No meta/ directory under {data_dir}. Collect a dataset first.")

    by_episode = defaultdict(list)
    for p in meta_dir.glob("*.npz"):
        by_episode[_episode_key(p)].append(p)
    if not by_episode:
        raise SystemExit(f"No .npz frames found in {meta_dir}")

    print(f"Labelling {len(by_episode)} episodes, horizon {args.horizon_s}s, "
          f"corridor half-width {half_width} m")

    rows, stale, no_vru = [], 0, 0
    for key in sorted(by_episode):
        got = label_episode(by_episode[key], args.horizon_s, half_width)
        if not got and by_episode[key]:
            # Distinguish the two reasons an episode yields nothing. A
            # blind-spot episode contains only vehicles and correctly produces
            # no VRU rows; an episode collected before the per-object record
            # existed is unusable and must be re-collected. Reporting both the
            # same way made the healthy case look like a defect.
            first = np.load(by_episode[key][0], allow_pickle=True)
            if "obj_actor_id" not in first.files:
                stale += 1
            else:
                no_vru += 1
        rows.extend(got)

    if stale:
        print(f"  {stale} episodes have no per-object record "
              f"(collected before that existed -- re-collect them)")
    if no_vru:
        print(f"  {no_vru} episodes contain no VRUs at all (expected for "
              f"vehicle-only scenarios such as blindspot_cutin)")
    if not rows:
        raise SystemExit("No VRU rows produced. Nothing to label.")

    out = Path(args.out) if args.out else data_dir / "intent_labels.npz"
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(out, **cols)

    n = len(rows)
    pos = int(cols["will_cross"].sum())
    print(f"\n{n} VRU observations across {len(by_episode)} episodes -> {out}")
    print(f"  will_cross positives: {pos} ({pos / n:.1%})")
    print(f"  already in corridor:  {int(cols['in_corridor_now'].sum())}")
    tiers = {0: "OCCLUDED", 1: "PARTIAL", 2: "VISIBLE"}
    for t, name in tiers.items():
        m = cols["tier"] == t
        if m.any():
            print(f"  tier {name:<9} {int(m.sum()):6d} rows, "
                  f"{cols['will_cross'][m].mean():.1%} positive")

    # Balance, judged on the right population.
    #
    # The raw rate over every VRU row is not the number to tune against. Once
    # the ego roams a populated city, most VRU observations are background
    # pedestrians strolling a pavement tens of metres to one side -- trivially
    # negative, and never a case a crossing predictor would be asked about.
    # Measured on a 790-frame probe: 12.3% positive over all 819 rows, 28.5%
    # over the 333 rows within reach, and 4 of the 10 VRUs that actually came
    # near the ego crossed. The first figure reads as a broken dataset and the
    # last two say the behaviour weights are doing their job, so reporting only
    # the first sends the reader off to retune settings that are already right.
    near = ((np.abs(cols["lateral_m"]) <= NEAR_LATERAL_M)
            & (cols["forward_m"] > 0.0) & (cols["forward_m"] <= NEAR_FORWARD_M))
    n_near = int(near.sum())
    frac = pos / n
    if n_near:
        frac_near = float(cols["will_cross"][near].mean())
        print(f"  within {NEAR_LATERAL_M:.0f} m laterally and {NEAR_FORWARD_M:.0f} m "
              f"ahead: {n_near} rows, {frac_near:.1%} positive")
    else:
        frac_near = frac

    # A genuinely degenerate balance makes every downstream metric meaningless,
    # so say so here rather than letting it surface as a suspiciously good
    # number -- but judge it on the reachable population.
    if frac_near < 0.15 or frac_near > 0.85:
        print(f"\n  WARNING: class balance among reachable VRUs is {frac_near:.1%} "
              f"positive. Prediction metrics on this will be dominated by the "
              f"majority class -- check the walker behaviour weights in "
              f"configs/scenarios.yaml.")


if __name__ == "__main__":
    main()
