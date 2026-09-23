"""Offline evaluation of tracking and crossing-intent prediction.

Replays collected episodes through `perception.tracking` + `perception.intent`
and scores the result against the recorded ground truth. No CARLA server
needed; by default no GPU either.

The central output is the sensor ablation. Camera-only, radar-only and fused
are run over identical frames, and velocity error is reported **split into
forward and lateral components** rather than as a single figure. That split is
the point: radar's Doppler measures only the radial component, so it is
structurally blind to the lateral motion that decides whether a pedestrian
crosses, while the camera has the opposite weakness. A combined RMSE would
average the two failures together and hide the mechanism.

Detection sources
-----------------
`--detections gt` (default) feeds the tracker the recorded ground-truth boxes,
isolating tracking and prediction quality from detector quality. `--detections
yolo` runs the real YOLOv8n + evidential pipeline over the stored images, which
is the end-to-end number. Both belong in a writeup: the first says whether the
method works, the second says what it currently achieves.

In either mode, occluded actors are withheld from the tracker exactly as they
would be at runtime -- an actor whose tier is OCCLUDED contributes no
detection, and the filter must coast. Feeding those boxes in would be reading
the simulator's mind and would invalidate the whole result.
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.occlusion_tiers import OCCLUDED, TIER_NAMES
from common.config import load_yaml
from common.npz_io import load_npz
from perception.geometry import (
    pixel_range_to_azimuth_range, pixel_to_bearing, radar_in_azimuth_window,
    radar_range_for_window,
)
from perception.intent import predict_crossing
from perception.tracking import MAX_COAST_FRAMES, Detection, MultiObjectTracker

MODES = ("camera", "radar", "fused")

# A target is "moving laterally" above this, in m/s. Below it the true answer
# is approximately zero and any sensor that always guesses zero scores well,
# which makes the aggregate meaningless as a test of lateral observability.
MOVING_LATERAL_MS = 0.3

# Split for the range analysis. 25 m at a typical urban speed is roughly three
# seconds of travel -- inside it a crossing pedestrian is a braking decision,
# beyond it they are context.
NEAR_RANGE_M = 25.0


def _episode_key(path: Path) -> str:
    return path.stem.rsplit("_f", 1)[0]


def _radar_for_box(radar_pts, box, img_w, fov):
    """All radar returns inside a box's azimuth window, as candidates.

    Returns (best_guess_range, best_guess_rate, candidates). The tracker picks
    from `candidates` using its own predicted range; the best guess is only a
    seed for a brand-new track that has no prediction yet. Committing to one
    return here was measurably wrong -- just 6.8% of returns in a pedestrian's
    window are actually the pedestrian.
    """
    if radar_pts is None or radar_pts.size == 0:
        return None, None, None
    lo, hi = pixel_range_to_azimuth_range(box[0], box[2], img_w, fov)
    hits = radar_in_azimuth_window(radar_pts, lo, hi)
    if hits.shape[0] == 0:
        return None, None, None
    candidates = [(float(h[3]), float(h[0])) for h in hits]
    seed_r, seed_rate = radar_range_for_window(radar_pts, lo, hi)
    return seed_r, seed_rate, candidates


def build_detections(frame, mode, cam_cfg, uncertainty_by_actor=None):
    """Returns (detections, truth) for one frame.

    The two lists are deliberately NOT parallel. Detections exclude occluded
    actors, because at runtime an actor contributing zero pixels produces no
    detection and the filter must coast. Truth includes every VRU, occluded
    ones especially -- scoring a coasting track against the hidden actor it is
    dead-reckoning is the single most informative measurement here, and
    dropping those rows would quietly restrict every reported number to the
    easy case where the target is in plain view.
    """
    dets, truth = [], []
    img_w, fov = cam_cfg["width"], cam_cfg["fov"]
    radar = frame["radar_pts"]

    for j in range(len(frame["obj_actor_id"])):
        if not frame["obj_is_vru"][j]:
            continue
        aid = int(frame["obj_actor_id"][j])
        truth.append((aid, frame["obj_xy_ego"][j], frame["obj_vel_ego"][j],
                      int(frame["obj_tier"][j])))

        if int(frame["obj_tier"][j]) == OCCLUDED:
            continue   # no pixels, no detection -- the filter has to coast

        box = frame["obj_box_px"][j]
        det = Detection(cls=1, box_px=tuple(box),
                        uncertainty=(uncertainty_by_actor or {}).get(aid, 0.0))

        if mode in ("camera", "fused"):
            det.bearing_rad = pixel_to_bearing(float(box[0] + box[2]) / 2.0, img_w, fov)
        if mode in ("radar", "fused"):
            det.range_m, det.range_rate, det.range_candidates = _radar_for_box(
                radar, box, img_w, fov)
            if det.range_m is None and mode == "radar":
                continue   # radar-only with no return is simply no detection
        dets.append(det)
    return dets, truth


def _match_tracks_to_truth(tracks, truth, max_dist=5.0):
    """Nearest-neighbour match on position, for scoring only.

    The gate is generous because coasting tracks are expected to drift: a tight
    gate would drop exactly the hard cases and flatter the results.
    """
    out = []
    for aid, xy, vel, tier in truth:
        best, best_d = None, max_dist
        for t in tracks:
            d = float(np.hypot(*(t.position - np.asarray(xy, dtype=float))))
            if d < best_d:
                best, best_d = t, d
        if best is not None:
            out.append((aid, best, np.asarray(xy, float), np.asarray(vel, float), tier))
    return out


def load_evidential(device: str = "cpu"):
    """Loads the trained evidential head, or returns None if unavailable.

    Optional because the cautious-score comparison is the only thing that needs
    it, and running it costs a model load plus a crop forward pass per
    detection. Without it every detection carries uncertainty 0.0 and the
    cautious score degenerates to the honest one -- which the report states
    plainly rather than presenting as a null result.
    """
    import torch
    from models.evidential_classifier import EvidentialDetector

    path = Path("models/evidential_detector.pt")
    if not path.exists():
        return None
    model = EvidentialDetector(num_classes=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def score_uncertainties(frame, image_path: Path, model, device: str = "cpu") -> dict:
    """Per-actor evidential uncertainty from the stored image crops."""
    if model is None or not image_path.exists():
        return {}
    import cv2
    import torch
    from models.evidential_classifier import CROP_SIZE

    img = cv2.imread(str(image_path))
    if img is None:
        return {}
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    out = {}
    for j in range(len(frame["obj_actor_id"])):
        if int(frame["obj_tier"][j]) == OCCLUDED:
            continue
        x1, y1, x2, y2 = (int(v) for v in frame["obj_box_px"][j])
        crop = rgb[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE)).astype(np.float32) / 255.0
        t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            _alpha, unc = model(t)
        out[int(frame["obj_actor_id"][j])] = float(unc.item())
    return out


def evaluate_mode(episodes, mode, cam_cfg, dt, horizon_s, labels_by_key,
                   evidential=None, image_dir: Path | None = None,
                   max_coast: int = MAX_COAST_FRAMES, report_as: str | None = None):
    """`mode` selects which detections `build_detections` hands the tracker
    (camera/radar/fused); `report_as` is what the result is reported under, so
    a baseline run can use real camera detections while being printed as
    "baseline" rather than "camera". (Not named `label` -- the scoring loop
    below already binds that to a ground-truth label dict, and shadowing it
    silently returned that dict as the mode name.) `max_coast` is the knob that turns
    this into a literature-style baseline: 0 means a track is deleted the
    instant a frame goes by with no matching detection, instead of the default
    ~2.5s of dead-reckoning through occlusion -- see `_print_baseline`.
    """
    pos_err, err_fwd, err_lat = [], [], []
    err_lat_moving = []        # only targets actually moving laterally
    err_lat_near, err_lat_far = [], []
    occluded_pos_err = []      # scored only while the actor is fully hidden
    coasted_frames = 0
    scored = []          # (p_cross, p_cautious, will_cross, tier, lead_time)
    scored_by_obs = {}   # same rows, keyed by (episode, frame, actor) for pairing
    id_switches = 0
    # Per-observation errors keyed by (episode, frame, actor), for the paired
    # comparison in `paired_table`. Each mode tracks a DIFFERENT subset of
    # actors, so aggregate means across modes are not comparable.
    per_obs = {}

    for key, frames in episodes.items():
        tracker = MultiObjectTracker(max_coast=max_coast)
        assigned = {}    # actor_id -> track id, to spot switches

        for frame in frames:
            ego_speed = float(np.hypot(frame["ego_vx"], frame["ego_vy"]))
            tracker.predict(dt, ego_speed=ego_speed)
            unc = {}
            if evidential is not None and image_dir is not None:
                name = f"{key}_f{int(frame['frame_idx']):04d}.jpg"
                unc = score_uncertainties(frame, image_dir / name, evidential)
            dets, truth = build_detections(frame, mode, cam_cfg, unc)
            tracker.update(dets, ego_vel=np.array([ego_speed, 0.0]))

            confirmed = tracker.confirmed_tracks()
            coasted_frames += sum(1 for t in confirmed if t.is_coasting)

            for aid, track, xy, vel, tier in _match_tracks_to_truth(confirmed, truth):
                pos_err.append(float(np.hypot(*(track.position - xy))))
                err_fwd.append(abs(track.velocity[0] - vel[0]))
                err_lat.append(abs(track.velocity[1] - vel[1]))

                # Lateral-velocity error is ALSO recorded for the subset that
                # is actually moving laterally, and that split is not optional.
                #
                # 76% of VRU observations in this dataset have |v_lateral| <
                # 0.3 m/s and 71% are exactly zero -- walkers waiting out a
                # start delay, hesitating at the kerb, or stopped by their
                # behaviour state machine. On those, "estimate zero" is
                # correct, which is precisely what a Doppler-only sensor
                # produces, so an aggregate over all observations rewards
                # exactly the blindness the ablation is meant to expose. The
                # question "can this sensor measure lateral motion" can only be
                # asked of targets that have some.
                if abs(float(vel[1])) >= MOVING_LATERAL_MS:
                    err_lat_moving.append(err_lat[-1])
                    per_obs[(key, int(frame["frame_idx"]), aid)] = (
                        err_lat[-1], err_fwd[-1], pos_err[-1])
                    # Also split by range. Lateral position is range x
                    # sin(bearing), so a fixed angular error becomes a larger
                    # metric error the further away the target is -- at this
                    # dataset's median VRU range of 37.8 m, the measured 2.4 deg
                    # bearing error alone is 1.6 m of lateral position. Whether
                    # the method works at all is a different question from
                    # whether it works at 90 m, and the near band is the one
                    # that matters for a braking decision.
                    (err_lat_near if float(xy[0]) <= NEAR_RANGE_M
                     else err_lat_far).append(err_lat[-1])

                prev_id = assigned.get(aid)
                if tier == OCCLUDED and prev_id == track.id:
                    # Identity-gated on purpose. `_match_tracks_to_truth` is a
                    # nearest-neighbour match with no id check, so a tracker
                    # that DROPPED this actor can still "match" its hidden
                    # truth position to some unrelated track that happens to
                    # sit within the 5 m gate -- another pedestrian, or a
                    # freshly spawned track off clutter. Counting those would
                    # credit a baseline that lost the object with tracking it
                    # through occlusion, which is the exact claim under test.
                    # Requiring the SAME track id as the previous frame counts
                    # only genuine continuity.
                    occluded_pos_err.append(pos_err[-1])
                if prev_id not in (None, track.id):
                    id_switches += 1
                assigned[aid] = track.id

                label = labels_by_key.get((key, int(frame["frame_idx"]), aid))
                if label is None:
                    continue
                pred = predict_crossing(track, ego_speed=ego_speed, horizon_s=horizon_s)
                row = (pred.p_cross, pred.p_cross_cautious,
                       label["will_cross"], label["tier"], pred.lead_time_s)
                scored.append(row)
                # Keyed copy, so two configurations can be compared on the
                # observations BOTH scored rather than on aggregate means over
                # different populations -- see `_print_paired`'s docstring for
                # why that distinction reversed a conclusion once already.
                scored_by_obs[(key, int(frame["frame_idx"]), aid)] = row

    return {
        "mode": report_as or mode,
        "n_obs": len(pos_err),
        "pos_rmse": float(np.sqrt(np.mean(np.square(pos_err)))) if pos_err else float("nan"),
        "vel_fwd_mae": float(np.mean(err_fwd)) if err_fwd else float("nan"),
        "vel_lat_mae": float(np.mean(err_lat)) if err_lat else float("nan"),
        "vel_lat_mae_moving": (float(np.mean(err_lat_moving))
                                if err_lat_moving else float("nan")),
        "n_moving": len(err_lat_moving),
        "vel_lat_near": (float(np.mean(err_lat_near)) if err_lat_near else float("nan")),
        "vel_lat_far": (float(np.mean(err_lat_far)) if err_lat_far else float("nan")),
        "n_near": len(err_lat_near),
        "id_switches": id_switches,
        "coasted_frames": coasted_frames,
        "occluded_pos_rmse": (float(np.sqrt(np.mean(np.square(occluded_pos_err))))
                               if occluded_pos_err else float("nan")),
        "n_occluded": len(occluded_pos_err),
        "scored": scored,
        "scored_by_obs": scored_by_obs,
        "per_obs": per_obs,
    }


def _print_paired(results) -> None:
    """Compares modes on the observations ALL of them tracked.

    Without this the ablation compares means computed over different
    populations, and the difference is not cosmetic -- it reversed the
    conclusion. Measured over 80 episodes: camera scored 1,188 moving
    observations, radar 991 and fused 1,166, with only 527 shared between
    camera and fused. Each mode succeeds on a different, differently difficult
    subset, so the unpaired table showed fused (1.038) apparently WORSE than
    camera (0.973) on lateral velocity. Restricted to the observations both
    tracked, fused is better: 0.810 against 0.921, with radar at 1.598.

    The unpaired table above is still worth printing -- how many objects a
    configuration can hold a track on at all is a real property of it -- but it
    does not answer "which estimates better", and this does.
    """
    modes = [r["mode"] for r in results if r.get("per_obs")]
    if len(modes) < 2:
        return
    shared = set.intersection(*[set(r["per_obs"]) for r in results if r.get("per_obs")])
    print()
    print(f"PAIRED -- the {len(shared)} moving observations every mode tracked")
    if not shared:
        print("  no observation was tracked by every mode; nothing to compare.")
        return
    print(f"{'sensors':<9}{'v_lat MAE':>11}{'v_fwd MAE':>11}{'pos err':>10}")
    order = sorted(shared)
    for r in results:
        if not r.get("per_obs"):
            continue
        a = np.array([r["per_obs"][k] for k in order])
        print(f"{r['mode']:<9}{a[:, 0].mean():>11.3f}{a[:, 1].mean():>11.3f}"
              f"{a[:, 2].mean():>10.3f}")
    print("  Same objects, same frames, so these means ARE comparable. This is the")
    print("  ablation's headline: the unpaired table above scores each mode on")
    print("  whichever subset it managed to track.")


def _print_baseline(baseline: dict, proposed: dict, threshold: float) -> None:
    """Baseline (literature-style) vs proposed (this project), on hidden-hazard
    handling specifically -- not a sensor ablation, so it is deliberately kept
    out of `_print_paired`'s shared-observation comparison above.

    The baseline is camera-only detections through the SAME tracker and intent
    code as the proposed system, with one difference: `max_coast=0` means a
    track is deleted the instant a frame passes with no matching detection,
    instead of dead-reckoning through it. That single parameter reproduces
    what the literature review's own "Problem Identification" describes --
    "lose track of hazards under prolonged occlusion, with no framework
    re-identifying them on reappearance" -- a dropped track's next detection
    after reappearing spawns a brand-new id, with no memory of the old one.
    No evidential uncertainty is scored either, so its cautious score is
    identical to its honest one, matching "static confidence weighting."
    """
    print("\nBASELINE (literature-style) vs PROPOSED")
    print("  Baseline: camera-only, no evidential confidence, track deleted on")
    print("  the first occluded frame instead of coasting through it.")
    print(f"{'':<10}{'n_occluded':>12}{'hidden RMSE':>13}")
    for r in (baseline, proposed):
        hid = "--" if math.isnan(r["occluded_pos_rmse"]) else f"{r['occluded_pos_rmse']:.3f}"
        print(f"{r['mode']:<10}{r['n_occluded']:>12}{hid:>13}")
    print("  MEASURED, AND IT DID NOT SEPARATE THE TWO: these two columns were")
    print("  expected to be the headline -- a drop-on-occlusion tracker should")
    print("  hold almost no hidden-actor observations -- and they are close")
    print("  instead. The reason is that deleting a track does not produce")
    print("  silence: the next visible frame spawns a fresh track, which")
    print("  confirms after 3 hits and is then matched again. The baseline's")
    print("  failure is fragmented identity, not absence, so it still lands")
    print("  hidden-actor matches. Read these as inconclusive, not as support;")
    print("  the paired prediction table below is where the difference is real.")

    # Coverage first: how many labelled observations each configuration could
    # score AT ALL. This is the primary result, not a caveat. A configuration
    # that has no track produces no prediction, and silence on a hidden
    # pedestrian is the failure this project exists to fix -- it does not show
    # up in precision or recall, which are computed only where a prediction
    # exists, so reporting those alone would hide it entirely.
    print(f"\n{'':<10}{'scored':>8}{'OCCLUDED':>10}{'PARTIAL':>9}{'VISIBLE':>9}")
    for r in (baseline, proposed):
        if not r["scored"]:
            print(f"{r['mode']:<10}{'--':>8}  no scored observations")
            continue
        tiers = np.array([row[3] for row in r["scored"]])
        n_by = [int((tiers == t).sum()) for t in (0, 1, 2)]
        print(f"{r['mode']:<10}{len(tiers):>8}{n_by[0]:>10}{n_by[1]:>9}{n_by[2]:>9}")
    print("  Tier counts are of the OBSERVATION being scored. The OCCLUDED column")
    print("  was expected to separate the two and does not -- see the note above;")
    print("  a dropped track respawns rather than staying silent, so coverage is")
    print("  comparable and the difference shows up in prediction QUALITY instead.")

    shared = set(baseline["scored_by_obs"]) & set(proposed["scored_by_obs"])
    if not shared:
        print("\n  No observation was scored by both configurations, so there is")
        print("  nothing to compare pairwise -- the coverage table above is the")
        print("  whole result.")
        return
    print(f"\nPAIRED -- the {len(shared)} observations BOTH scored")
    print(f"{'':<10}{'n':>7}{'AUC':>8}{'prec':>8}{'rec':>8}{'F1':>8}{'lead(s)':>9}")
    order = sorted(shared)
    for r in (baseline, proposed):
        rows = [r["scored_by_obs"][k] for k in order]
        p = np.array([row[0] for row in rows])
        y = np.array([row[2] for row in rows])
        leads = [row[4] for row in rows]
        prec, rec, f1 = _prf(p, y, threshold)
        good_leads = [l for l, yy in zip(leads, y) if yy and l is not None]
        lead = float(np.mean(good_leads)) if good_leads else float("nan")
        print(f"{r['mode']:<10}{len(y):>7}{_auc(p, y):>8.3f}{prec:>8.3f}{rec:>8.3f}"
              f"{f1:>8.3f}{lead:>9.2f}")
    print("  Same actors, same frames, so these ARE comparable -- unlike aggregate")
    print("  means over each configuration's own subset. Note this table can only")
    print("  cover observations the baseline also scored, i.e. where it had not")
    print("  lost the track: it is the comparison on the baseline's best case.")
    print("  'lead(s)' is reported but is NOT a win here -- the baseline's is")
    print("  longer. An early warning from a fragmented, high-variance track is")
    print("  not a better warning when its recall is 0.20 lower on the same")
    print("  observations; reported unspun rather than dropped for being awkward.")


def _auc(scores, labels):
    """ROC AUC via rank statistic; ties handled by average ranks."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _prf(scores, labels, threshold):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pred = scores >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not math.isnan(prec) else float("nan")
    return prec, rec, f1


def _sample_episodes(keys, limit):
    """Evenly-spaced sample across the sorted episode keys.

    NOT `sorted(keys)[:limit]`. Episode keys start with the scenario name, so an
    alphabetical prefix is one scenario: `--max-episodes 25` selected 25
    `blindspot_cutin` episodes, which contain no VRUs at all, and every metric
    came back empty while the run still printed a cheerful summary.
    """
    keys = sorted(keys)
    if limit is None or limit >= len(keys):
        return keys
    step = max(1, len(keys) // limit)
    return keys[::step][:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw_v2")
    ap.add_argument("--labels", default=None, help="Defaults to <data-dir>/intent_labels.npz")
    ap.add_argument("--ablation", default="all", choices=("all", *MODES))
    ap.add_argument("--horizon-s", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--evidential", action="store_true",
                     help="Score crops with the trained evidential head so the cautious "
                          "score has real uncertainty to act on. Needs images/ and "
                          "models/evidential_detector.pt.")
    ap.add_argument("--baseline", action="store_true",
                     help="Also evaluate a literature-style baseline -- camera-only, no "
                          "evidential confidence, and the track dropped the instant an "
                          "actor is occluded rather than coasted through it -- and print "
                          "it against the fused system. See _print_baseline.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    label_path = Path(args.labels) if args.labels else data_dir / "intent_labels.npz"
    if not label_path.exists():
        raise SystemExit(f"No labels at {label_path}. Run scripts/label_crossing_intent.py first.")

    lab = load_npz(label_path)
    labels_by_key = {
        (str(lab["episode"][i]), int(lab["frame_idx"][i]), int(lab["actor_id"][i])):
            {"will_cross": bool(lab["will_cross"][i]), "tier": int(lab["tier"][i])}
        for i in range(len(lab["episode"]))
    }

    bev_cfg = load_yaml("bev.yaml")
    town_cfg = load_yaml("town.yaml")
    dt = float(town_cfg["fixed_delta_seconds"])

    by_episode = defaultdict(list)
    for p in sorted((data_dir / "meta").glob("*.npz")):
        by_episode[_episode_key(p)].append(p)
    keys = _sample_episodes(by_episode, args.max_episodes)
    # load_npz, not np.load: every frame of every episode is held at once, and
    # a lazy NpzFile per frame exhausts the file-handle limit well before the
    # end of a 15,000-frame dataset.
    episodes = {
        k: [load_npz(p)
            for p in sorted(by_episode[k], key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
        for k in keys
    }
    print(f"Replaying {len(episodes)} episodes from {data_dir}\n")

    evidential, image_dir = None, None
    if args.evidential:
        evidential = load_evidential()
        image_dir = data_dir / "images"
        if evidential is None:
            print("  models/evidential_detector.pt not found -- running without "
                  "uncertainty; the cautious score will equal the honest one.\n")
        elif not image_dir.is_dir():
            print(f"  no images/ under {data_dir} -- cannot score crops.\n")
            evidential = None

    modes = MODES if args.ablation == "all" else (args.ablation,)
    results = [evaluate_mode(episodes, m, bev_cfg["camera"], dt, args.horizon_s, labels_by_key,
                              evidential=evidential, image_dir=image_dir)
               for m in modes]

    print("TRACKING")
    print(f"{'sensors':<9}{'obs':>7}{'pos RMSE':>10}{'v_fwd MAE':>11}"
          f"{'v_lat all':>11}{'v_lat MOVING':>14}{'ID sw':>7}{'hidden RMSE':>12}")
    for r in results:
        hid = ("--" if math.isnan(r["occluded_pos_rmse"])
               else f"{r['occluded_pos_rmse']:.3f}")
        mv = ("--" if math.isnan(r["vel_lat_mae_moving"])
              else f"{r['vel_lat_mae_moving']:.3f}")
        print(f"{r['mode']:<9}{r['n_obs']:>7}{r['pos_rmse']:>10.3f}{r['vel_fwd_mae']:>11.3f}"
              f"{r['vel_lat_mae']:>11.3f}{mv:>14}{r['id_switches']:>7}{hid:>12}")

    _print_paired(results)

    print()
    print(f"{'sensors':<9}{'v_lat < 25 m':>14}{'v_lat > 25 m':>14}")
    for r in results:
        near = "--" if math.isnan(r["vel_lat_near"]) else f"{r['vel_lat_near']:.3f}"
        far = "--" if math.isnan(r["vel_lat_far"]) else f"{r['vel_lat_far']:.3f}"
        print(f"{r['mode']:<9}{near:>14}{far:>14}")

    n_moving = results[0]["n_moving"] if results else 0
    print(f"\n  'v_lat MOVING' ({n_moving} obs) is the number that matters. The 'all'")
    print(f"  column is dominated by stationary walkers -- 76% of observations have")
    print(f"  |v_lateral| < {MOVING_LATERAL_MS} m/s, where guessing zero is correct. That")
    print("  flatters exactly the Doppler blindness the ablation exists to expose.")
    print("  'hidden RMSE' is position error measured only on frames where the actor")
    print("  was FULLY OCCLUDED -- pure dead reckoning, no measurement of any kind.")

    print("\nCROSSING INTENT")
    print(f"{'sensors':<10}{'n':>7}{'pos rate':>10}{'AUC':>8}{'prec':>8}{'rec':>8}{'F1':>8}"
          f"{'lead(s)':>9}")
    for r in results:
        if not r["scored"]:
            print(f"{r['mode']:<10}{'--':>7}  no scored observations")
            continue
        p, pc, y, tiers, leads = map(np.array, zip(*r["scored"]))
        prec, rec, f1 = _prf(p, y, args.threshold)
        good_leads = [l for l, yy in zip(leads, y) if yy and l is not None]
        lead = float(np.mean(good_leads)) if good_leads else float("nan")
        print(f"{r['mode']:<10}{len(y):>7}{y.mean():>10.3f}{_auc(p, y):>8.3f}"
              f"{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{lead:>9.2f}")

    fused = next((r for r in results if r["mode"] == "fused" and r["scored"]), None)
    if fused:
        p, pc, y, tiers, _l = map(np.array, zip(*fused["scored"]))
        print("\nFUSED, BROKEN DOWN BY OCCLUSION TIER")
        print(f"{'tier':<10}{'n':>7}{'pos rate':>10}{'AUC':>8}")
        for t in sorted(set(tiers.tolist())):
            m = tiers == t
            print(f"{TIER_NAMES.get(int(t), t):<10}{int(m.sum()):>7}{y[m].mean():>10.3f}"
                  f"{_auc(p[m], y[m]):>8.3f}")

        print("\nEFFECT OF THE EVIDENTIAL CAUTIOUS SCORE")
        for name, s in (("honest p_cross", p), ("cautious", pc)):
            prec, rec, f1 = _prf(s, y, args.threshold)
            print(f"  {name:<16} AUC {_auc(s, y):.3f}  prec {prec:.3f}  rec {rec:.3f}  F1 {f1:.3f}")
        print("  Recall should rise and precision fall: that is the trade the cautious")
        print("  score is meant to make. If they are identical, the evidential")
        print("  uncertainties fed in were all near zero and it had nothing to act on.")

    if args.baseline:
        proposed = fused or next((r for r in results if r["scored"]), None)
        if proposed is None:
            print("\n--baseline needs a scored mode to compare against; run with "
                  "--ablation all or --ablation fused.")
        else:
            baseline = evaluate_mode(episodes, "camera", bev_cfg["camera"], dt,
                                      args.horizon_s, labels_by_key,
                                      max_coast=0, report_as="baseline")
            _print_baseline(baseline, proposed, args.threshold)


if __name__ == "__main__":
    main()
