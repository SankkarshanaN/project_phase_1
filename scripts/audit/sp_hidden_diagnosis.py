"""SECOND PASS -- why did the hidden-target experiment score only 15 events?

The first audit found 15 scorable occlusion events across 10 episodes and
correctly refused to build a claim on them. It did not establish WHY. This
script counts, stage by stage, how many candidate events the data contains and
where they are lost, so the limit can be attributed to one of:

  A genuinely insufficient data
  B an evaluation-script restriction
  C overly strict matching
  D incorrect ground-truth filtering
  E legitimate trajectories that were excluded in error

It only COUNTS. It does not loosen any criterion, and it does not re-run the
metric with relaxed gates to manufacture a larger n. If the loss turns out to
be a defensible property of the pipeline, the answer stays "sample-limited".

Writes results/second_pass/hidden_target_diagnosis.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carla_tools.occlusion_tiers import OCCLUDED
from common.config import load_yaml
from common.npz_io import load_npz
from perception.tracking import MultiObjectTracker
from scripts.evaluate_prediction import (_episode_key, _match_tracks_to_truth,
                                          build_detections)

DATA = Path("data/raw_v2")
OUT = Path("results/second_pass")


def main():
    bev = load_yaml("bev.yaml")
    cam_cfg = bev["camera"]
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])

    by_ep = defaultdict(list)
    for p in sorted((DATA / "meta").glob("*.npz")):
        by_ep[_episode_key(p)].append(p)

    # Funnel counters, from the widest possible population down to what the
    # first audit actually scored.
    c = Counter()
    loss_reason = Counter()
    hidden_run_lengths = []

    for key, paths in sorted(by_ep.items()):
        frames = [load_npz(p) for p in sorted(paths, key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
        tier_of, truth_xy = [], []
        for fr in frames:
            t, x = {}, {}
            for j in range(len(fr["obj_actor_id"])):
                if not fr["obj_is_vru"][j]:
                    continue
                aid = int(fr["obj_actor_id"][j])
                t[aid] = int(fr["obj_tier"][j])
                x[aid] = np.asarray(fr["obj_xy_ego"][j], float)
            tier_of.append(t); truth_xy.append(x)

        # --- stage 1: every VRU transition into OCCLUDED, ignoring tracking
        prev = {}
        transitions = []      # (frame_index, actor)
        for fi, t in enumerate(tier_of):
            for aid, tier in t.items():
                if tier == OCCLUDED and prev.get(aid) not in (OCCLUDED, None):
                    transitions.append((fi, aid))
                    # how long does it stay hidden with truth available?
                    run = 0
                    for gi in range(fi, len(frames)):
                        if tier_of[gi].get(aid) != OCCLUDED or aid not in truth_xy[gi]:
                            break
                        run += 1
                    hidden_run_lengths.append(run)
            prev = dict(t)
        c["s1_occlusion_transitions"] += len(transitions)

        # --- stage 2: of those, which had the actor CONFIRMED-tracked beforehand?
        tracker = MultiObjectTracker()
        last_seen = set()
        confirmed_before = set()
        prev = {}
        for fi, fr in enumerate(frames):
            es = float(np.hypot(fr["ego_vx"], fr["ego_vy"]))
            tracker.predict(dt, ego_speed=es)
            dets, truth = build_detections(fr, "fused", cam_cfg, {})
            tracker.update(dets, ego_vel=np.array([es, 0.0]))
            for aid, track, xy, vel, tier in _match_tracks_to_truth(
                    tracker.confirmed_tracks(), truth):
                if tier != OCCLUDED and not track.is_coasting:
                    last_seen.add(aid)
            for aid, tier in tier_of[fi].items():
                if tier == OCCLUDED and prev.get(aid) not in (OCCLUDED, None):
                    if aid in last_seen:
                        confirmed_before.add((fi, aid))
            prev = dict(tier_of[fi])
        c["s2_had_confirmed_track_before_occlusion"] += len(confirmed_before)

        # --- stage 3: of those, which stay hidden long enough to score the
        # shortest horizon (0.5 s = 5 frames) with truth present?
        for fi, aid in confirmed_before:
            run = 0
            for gi in range(fi + 1, len(frames)):
                if tier_of[gi].get(aid) != OCCLUDED or aid not in truth_xy[gi]:
                    break
                run += 1
            if run >= 5:
                c["s3_hidden_at_least_0p5s"] += 1
            else:
                loss_reason[f"re-emerged or truth gone within {run} frames"] += 1

    runs = np.asarray(hidden_run_lengths, float)
    dist = {
        "n_transitions": int(runs.size),
        "median_frames_hidden": float(np.median(runs)) if runs.size else None,
        "mean_frames_hidden": float(runs.mean()) if runs.size else None,
        "frac_hidden_>=5_frames_0p5s": float((runs >= 5).mean()) if runs.size else None,
        "frac_hidden_>=10_frames_1p0s": float((runs >= 10).mean()) if runs.size else None,
        "frac_hidden_>=20_frames_2p0s": float((runs >= 20).mean()) if runs.size else None,
        "frac_hidden_>=30_frames_3p0s": float((runs >= 30).mean()) if runs.size else None,
        "p90_frames_hidden": float(np.percentile(runs, 90)) if runs.size else None,
        "max_frames_hidden": float(runs.max()) if runs.size else None,
    }

    prior = json.loads(Path("results/audit/hidden_prediction_metrics.json").read_text())

    s1, s2, s3 = c["s1_occlusion_transitions"], c["s2_had_confirmed_track_before_occlusion"], \
        c["s3_hidden_at_least_0p5s"]
    out = {
        "experiment_id": "SP-03-HIDDEN-TARGET-DIAGNOSIS",
        "status": "EXECUTED",
        "question": "is the 15-event limit data, or an evaluation restriction?",
        "funnel": {
            "stage_1_occlusion_transitions_in_ground_truth": s1,
            "stage_2_actor_was_confirmed_tracked_before_going_hidden": s2,
            "stage_3_stays_hidden_at_least_0.5s_with_truth": s3,
            "stage_4_scored_by_first_audit": prior["n_occlusion_events_scored"],
        },
        "attrition": {
            "stage_1_to_2_lost": s1 - s2,
            "stage_1_to_2_cause": (
                "The actor was never CONFIRMED-tracked while visible, so there is no "
                "state to seed a hidden-target predictor from. MultiObjectTracker "
                "requires MIN_HITS_TO_CONFIRM=3 consecutive associations. This is a "
                "genuine property of the pipeline, not an evaluation artefact: with "
                "no prior track there is literally nothing to propagate."),
            "stage_2_to_3_lost": s2 - s3,
            "stage_2_to_3_cause": ("re-emerged (or amodal truth ended) before the "
                                    "shortest 0.5s horizon could be scored"),
        },
        "hidden_run_length_distribution": dist,
        "verdict": {
            "classification": None,     # filled below
            "explanation": None,
        },
        "actions_considered_and_rejected": [
            "Lowering MIN_HITS_TO_CONFIRM to admit more events -- rejected: it would "
            "change the tracker's behaviour, i.e. the system under test.",
            "Widening the 5 m association gate -- rejected: same reason, and it would "
            "flatter the result by admitting looser matches.",
            "Seeding the hidden predictor from ground truth instead of the last "
            "confirmed estimate -- rejected: that leaks privileged state and is not "
            "what pipeline._handoff does at runtime.",
        ],
    }

    # Decide the verdict from the numbers rather than asserting one.
    if s1 > 0 and s2 / max(s1, 1) < 0.25:
        out["verdict"]["classification"] = ("A + pipeline-inherent: genuinely few "
                                             "scorable events")
        out["verdict"]["explanation"] = (
            f"Of {s1} ground-truth occlusion transitions, only {s2} "
            f"({s2/max(s1,1):.1%}) involved an actor the tracker had already "
            f"confirmed, and only {s3} of those stayed hidden long enough to score. "
            f"The dominant loss is upstream of the hidden-target evaluation: most "
            f"actors that go behind an occluder were never solidly tracked in the "
            f"first place. That is a real limitation of the perception chain, not a "
            f"bug in the evaluation, and it cannot be fixed by changing the scoring.")
    else:
        out["verdict"]["classification"] = "needs manual review -- see funnel"
        out["verdict"]["explanation"] = (
            f"stage1={s1} stage2={s2} stage3={s3} scored={prior['n_occlusion_events_scored']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hidden_target_diagnosis.json").write_text(json.dumps(out, indent=2))

    print(json.dumps({"funnel": out["funnel"], "attrition_counts": {
        "1->2": out["attrition"]["stage_1_to_2_lost"],
        "2->3": out["attrition"]["stage_2_to_3_lost"]},
        "run_lengths": dist, "verdict": out["verdict"]["classification"]}, indent=2))


if __name__ == "__main__":
    main()
