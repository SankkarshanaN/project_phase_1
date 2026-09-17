"""Component 9 -- Adversarial Robustness Testing.

Injects sensor faults into a recorded dataset, replays the full pipeline, and
measures what breaks.

Runs offline against collected frames -- no CARLA server -- so a fault suite can
be re-run after any code change without re-collecting.

Two questions per fault, and the second is the one that matters
---------------------------------------------------------------
  1. **How much does performance degrade?** Standard robustness reporting.
  2. **Does the system NOTICE?** A pipeline that degrades gracefully while
     believing itself healthy is more dangerous than one that fails loudly,
     because a confident wrong answer gets acted on. Every fault here is
     therefore also scored on whether `perception.sensor_health` detected it
     and lowered the sensor's weight.

That second question is why the health monitor infers degradation from the
sensor streams rather than reading simulator state: a fault it could look up
would be trivially detectable and the test would prove nothing.

Fault families
--------------
  darkness, fog, blur, noise, lens_blocked   -- camera
  radar_dropout, radar_clutter, radar_bias   -- radar
  extrinsic_drift                            -- calibration between them
  prolonged_occlusion                        -- the scenario stress, not a fault

`extrinsic_drift` deserves note: it degrades neither sensor individually, so
no single-sensor health check can see it. It shows up only as the two sensors
persistently disagreeing, which is precisely the cross-sensor signal the health
monitor watches for.
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.occlusion_tiers import OCCLUDED
from common.config import load_yaml
from common.npz_io import load_npz
from perception.geometry import (pixel_range_to_azimuth_range, pixel_to_bearing,
                                  radar_in_azimuth_window, range_from_pixel_height)
from perception.intent import predict_crossing
from perception.sensor_health import SensorHealthMonitor
from perception.tracking import Detection, MultiObjectTracker


# --------------------------------------------------------------------- faults

def _identity(rgb, radar, rng):
    return rgb, radar


def darkness(rgb, radar, rng, factor=0.12):
    """Night, or a failed exposure. Scales luminance hard."""
    return (None if rgb is None else np.clip(rgb.astype(np.float32) * factor, 0, 255)), radar


def fog(rgb, radar, rng, strength=0.75):
    """Fog / spray: contrast collapses toward a uniform grey veil."""
    if rgb is None:
        return rgb, radar
    veil = np.full_like(rgb, 190.0, dtype=np.float32)
    return np.clip(rgb.astype(np.float32) * (1 - strength) + veil * strength, 0, 255), radar


def blur(rgb, radar, rng, k=9):
    """Defocus or rain smear -- a box blur, no cv2 dependency."""
    if rgb is None:
        return rgb, radar
    img = rgb.astype(np.float32)
    pad = k // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    cum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    out = (cum[k:, k:] - cum[:-k, k:] - cum[k:, :-k] + cum[:-k, :-k]) / (k * k)
    return np.clip(out, 0, 255), radar


def noise(rgb, radar, rng, sigma=45.0):
    """Sensor noise at high ISO / gain."""
    if rgb is None:
        return rgb, radar
    return np.clip(rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape), 0, 255), radar


def lens_blocked(rgb, radar, rng, fraction=0.4):
    """Mud, ice or a sticker over part of the lens -- an opaque patch."""
    if rgb is None:
        return rgb, radar
    img = rgb.astype(np.float32).copy()
    h, w = img.shape[:2]
    img[:, : int(w * fraction)] = 12.0
    return img, radar


def radar_dropout(rgb, radar, rng, keep=0.15):
    """Blockage or partial failure: most returns lost."""
    if radar is None or radar.size == 0:
        return rgb, radar
    n = max(1, int(radar.shape[0] * keep))
    idx = rng.choice(radar.shape[0], size=n, replace=False)
    return rgb, radar[idx]


def radar_clutter(rgb, radar, rng, n_extra=40):
    """Multipath / interference: many spurious returns at random ranges."""
    base = np.empty((0, 4), np.float32) if radar is None else radar
    fake = np.stack([
        rng.normal(0, 3.0, n_extra),          # velocity
        rng.uniform(-0.30, 0.30, n_extra),    # azimuth
        np.zeros(n_extra),                    # altitude
        rng.uniform(3.0, 60.0, n_extra),      # depth
    ], axis=1).astype(np.float32)
    return rgb, np.vstack([base, fake])


def radar_bias(rgb, radar, rng, offset_m=2.5):
    """Range calibration drift -- every return reports too far away."""
    if radar is None or radar.size == 0:
        return rgb, radar
    out = radar.copy()
    out[:, 3] += offset_m
    return rgb, out


def extrinsic_drift(rgb, radar, rng, yaw_deg=4.0):
    """Camera/radar mounting misalignment.

    Neither sensor is individually degraded, so no single-sensor health check
    can detect it. It surfaces only as persistent disagreement about where
    things are -- the cross-sensor signal.
    """
    if radar is None or radar.size == 0:
        return rgb, radar
    out = radar.copy()
    out[:, 1] += math.radians(yaw_deg)
    return rgb, out


FAULTS = {
    "clean": _identity,
    "darkness": darkness,
    "fog": fog,
    "blur": blur,
    "noise": noise,
    "lens_blocked": lens_blocked,
    "radar_dropout": radar_dropout,
    "radar_clutter": radar_clutter,
    "radar_bias": radar_bias,
    "extrinsic_drift": extrinsic_drift,
    # Same clutter, but held off for a baseline period first. Every other
    # condition here injects its fault from frame one, which is also the one
    # case sensor_health.py's own relative check is structurally blind to --
    # there is nothing clean to compare against yet (see that module's
    # "Known limitations" docstring). This condition instead tests the
    # realistic case: a fault that starts partway through an already-running
    # drive, which is what the relative INCOHERENCE_RISE_RATIO check was
    # actually built for and had never been exercised by this harness.
    "radar_clutter_onset": radar_clutter,
}

CAMERA_FAULTS = {"darkness", "fog", "blur", "noise", "lens_blocked"}
RADAR_FAULTS = {"radar_dropout", "radar_clutter", "radar_bias", "radar_clutter_onset"}

# Frames of clean baseline before radar_clutter_onset switches the fault on.
# COHERENCE_BASELINE_MIN (perception.sensor_health) is 40 -- this clears it
# with margin so the relative check has a real norm to compare against.
CLUTTER_ONSET_DELAY_FRAMES = 100


# ------------------------------------------------------------------ pipeline

def _episode_key(path: Path) -> str:
    return path.stem.rsplit("_f", 1)[0]


def _radar_for_box(radar_pts, box, img_w, fov):
    if radar_pts is None or radar_pts.size == 0:
        return None, None
    lo, hi = pixel_range_to_azimuth_range(box[0], box[2], img_w, fov)
    hits = radar_in_azimuth_window(radar_pts, lo, hi)
    if hits.shape[0] == 0:
        return None, None
    return float(np.median(hits[:, 3])), float(np.median(hits[:, 0]))


def run_pipeline(episodes, fault_name, cam_cfg, dt, labels, rng, image_dir=None):
    """Replays every episode under one fault and returns aggregate metrics."""
    fault = FAULTS[fault_name]
    onset_delay = CLUTTER_ONSET_DELAY_FRAMES if fault_name == "radar_clutter_onset" else 0
    img_w, fov = cam_cfg["width"], cam_cfg["fov"]

    pos_err, lat_err, occ_err = [], [], []
    scores, truths = [], []
    health_cam, health_rad = [], []
    # Health AFTER the fault has actually switched on, separate from the
    # whole-run average -- averaging in the clean baseline period would
    # dilute exactly the number this condition exists to report.
    health_rad_post_onset = []
    n_detections = 0
    frame_counter = 0

    # ONE monitor across the whole run, not one per episode. A vehicle's health
    # monitor runs continuously through a drive; resetting it every ~60-frame
    # episode is a test artifact, and it starves the checks that need a long
    # window. The cross-sensor range test needs 200 observations and episodes
    # yield a median of 85, so it almost never ran. The tracker still resets per
    # episode -- tracks genuinely do not continue across a cut.
    monitor = SensorHealthMonitor()
    for key, frames in episodes.items():
        tracker = MultiObjectTracker()

        for frame in frames:
            frame_counter += 1
            active_fault = _identity if frame_counter <= onset_delay else fault
            radar = np.asarray(frame["radar_pts"])
            rgb = None
            if image_dir is not None:
                p = image_dir / f"{key}_f{int(frame['frame_idx']):04d}.jpg"
                if p.exists():
                    import cv2
                    img = cv2.imread(str(p))
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

            rgb_f, radar_f = active_fault(rgb, radar, rng)

            ego_speed = float(np.hypot(frame["ego_vx"], frame["ego_vy"]))
            tracker.predict(dt, ego_speed=ego_speed)

            dets, truth, disagreed = [], [], False
            range_pairs = []
            for j in range(len(frame["obj_actor_id"])):
                if not frame["obj_is_vru"][j]:
                    continue
                tier = int(frame["obj_tier"][j])
                truth.append((int(frame["obj_actor_id"][j]), frame["obj_xy_ego"][j],
                               frame["obj_vel_ego"][j], tier))
                if tier == OCCLUDED:
                    continue

                box = frame["obj_box_px"][j]
                # A camera fault suppresses detections in proportion to how
                # badly it degrades the image -- a blinded camera reports
                # nothing, which is the behaviour being tested.
                if rgb_f is not None and fault_name in CAMERA_FAULTS:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    crop = rgb_f[max(0, y1):max(y2, y1 + 1), max(0, x1):max(x2, x1 + 1)]
                    if crop.size and (crop.mean() < 20 or crop.std() < 6):
                        continue

                d = Detection(cls=1, box_px=tuple(box),
                               bearing_rad=pixel_to_bearing(float(box[0] + box[2]) / 2.0,
                                                             img_w, fov))
                d.range_m, d.range_rate = _radar_for_box(radar_f, box, img_w, fov)
                if d.range_m is None:
                    disagreed = True          # camera says here, radar says nothing
                else:
                    # Independent range from object size, for the health
                    # monitor's cross-sensor calibration check. This is what
                    # lets a from-startup range bias be seen at all.
                    cam_rng = range_from_pixel_height(box, 1, cam_cfg)
                    if cam_rng is not None:
                        range_pairs.append((d.range_m, cam_rng))
                dets.append(d)
                n_detections += 1

            tracker.last_range_innovation = None
            tracker.update(dets, ego_vel=np.array([ego_speed, 0.0]))
            monitor.update(rgb=rgb_f, radar_pts=radar_f, sensors_disagreed=disagreed,
                            range_innovation=tracker.last_range_innovation,
                            range_pairs=range_pairs)
            if onset_delay and frame_counter > onset_delay:
                health_rad_post_onset.append(monitor.report().radar)

            for aid, xy, vel, tier in truth:
                best, bd = None, 5.0
                for t in tracker.confirmed_tracks():
                    dd = float(np.hypot(*(t.position - np.asarray(xy, float))))
                    if dd < bd:
                        best, bd = t, dd
                if best is None:
                    continue
                pos_err.append(bd)
                lat_err.append(abs(best.velocity[1] - float(vel[1])))
                if tier == OCCLUDED:
                    occ_err.append(bd)

                lab = labels.get((key, int(frame["frame_idx"]), aid))
                if lab is not None:
                    pred = predict_crossing(best, ego_speed=ego_speed)
                    scores.append(pred.p_cross)
                    truths.append(lab)

        rep = monitor.report()
        health_cam.append(rep.camera)
        health_rad.append(rep.radar)

    return {
        "fault": fault_name,
        "n_obs": len(pos_err),
        "n_detections": n_detections,
        "pos_rmse": float(np.sqrt(np.mean(np.square(pos_err)))) if pos_err else float("nan"),
        "lat_vel_mae": float(np.mean(lat_err)) if lat_err else float("nan"),
        "occluded_rmse": float(np.sqrt(np.mean(np.square(occ_err)))) if occ_err else float("nan"),
        "auc": _auc(scores, truths),
        "health_camera": float(np.mean(health_cam)) if health_cam else float("nan"),
        "health_radar": float(np.mean(health_rad)) if health_rad else float("nan"),
        "health_radar_post_onset": (float(np.mean(health_rad_post_onset))
                                      if health_rad_post_onset else float("nan")),
    }


def _auc(scores, labels):
    if not scores:
        return float("nan")
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


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
    ap.add_argument("--labels", default=None)
    ap.add_argument("--faults", default="all")
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--no-images", action="store_true",
                     help="Skip image loading. Camera faults then affect only the "
                          "health monitor, not detections.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    label_path = Path(args.labels) if args.labels else data_dir / "intent_labels.npz"
    labels = {}
    if label_path.exists():
        lab = load_npz(label_path)
        labels = {(str(lab["episode"][i]), int(lab["frame_idx"][i]),
                   int(lab["actor_id"][i])): bool(lab["will_cross"][i])
                  for i in range(len(lab["episode"]))}
    else:
        print(f"  no labels at {label_path} -- intent AUC will be omitted\n")

    bev_cfg = load_yaml("bev.yaml")
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])

    by_ep = defaultdict(list)
    for p in sorted((data_dir / "meta").glob("*.npz")):
        by_ep[_episode_key(p)].append(p)
    keys = _sample_episodes(by_ep, args.max_episodes)
    # load_npz, not np.load -- see the note in evaluate_prediction: holding a
    # lazy NpzFile per frame runs the process out of file handles.
    episodes = {k: [load_npz(p)
                    for p in sorted(by_ep[k], key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
                for k in keys}

    image_dir = None if args.no_images else (data_dir / "images")
    if image_dir is not None and not image_dir.is_dir():
        image_dir = None

    names = list(FAULTS) if args.faults == "all" else ["clean"] + args.faults.split(",")
    print(f"Replaying {len(episodes)} episodes under {len(names)} conditions"
          f"{' (no images)' if image_dir is None else ''}\n")

    rng = np.random.default_rng(0)
    results = [run_pipeline(episodes, n, bev_cfg["camera"], dt, labels, rng, image_dir)
               for n in names]
    baseline = results[0]

    print(f"{'condition':<18}{'dets':>7}{'pos RMSE':>10}{'v_lat MAE':>11}"
          f"{'hidden RMSE':>13}{'AUC':>7}{'cam hp':>8}{'rad hp':>13}{'noticed':>9}")
    print("-" * 97)
    have_images = image_dir is not None
    for r in results:
        detected = "--"
        # The onset condition's whole-run health average blends in the clean
        # baseline period by construction, which would understate detection --
        # score it on health AFTER the fault actually switched on instead.
        rad_health_for_detection = (r["health_radar_post_onset"]
                                      if r["fault"] == "radar_clutter_onset"
                                      else r["health_radar"])
        if r["fault"] != "clean":
            if r["fault"] in CAMERA_FAULTS:
                # Without images there is nothing to corrupt and nothing for the
                # monitor to inspect, so the fault was never actually applied.
                # Reporting it as undetected would be a false accusation.
                detected = ("n/a" if not have_images
                            else "YES" if r["health_camera"] < 0.6 else "NO")
            elif r["fault"] in RADAR_FAULTS:
                detected = "YES" if rad_health_for_detection < 0.6 else "NO"
            else:
                detected = ("YES" if (r["health_camera"] < 0.8 or r["health_radar"] < 0.8)
                            else "NO")
        rad_col = (f"{r['health_radar']:.2f}->{rad_health_for_detection:.2f}"
                   if r["fault"] == "radar_clutter_onset" else f"{r['health_radar']:.2f}")
        print(f"{r['fault']:<18}{r['n_detections']:>7}{r['pos_rmse']:>10.3f}"
              f"{r['lat_vel_mae']:>11.3f}{r['occluded_rmse']:>13.3f}{r['auc']:>7.3f}"
              f"{r['health_camera']:>8.2f}{rad_col:>13}{detected:>9}")

    print("\nDEGRADATION VS CLEAN")
    for r in results[1:]:
        d_pos = r["pos_rmse"] - baseline["pos_rmse"]
        d_auc = r["auc"] - baseline["auc"]
        lost = baseline["n_detections"] - r["n_detections"]
        print(f"  {r['fault']:<18} pos +{d_pos:+.3f} m   AUC {d_auc:+.3f}   "
              f"detections lost {lost}")

    # Only faults that were actually applied AND measurably degraded something
    # count as silent failures. A fault that changed nothing was not survived,
    # it was never injected.
    def _degraded(r):
        return (r["pos_rmse"] - baseline["pos_rmse"] > 0.05
                or baseline["n_detections"] - r["n_detections"] > 0
                or (not math.isnan(r["auc"]) and baseline["auc"] - r["auc"] > 0.02))

    undetected = [
        r["fault"] for r in results[1:]
        if _degraded(r) and (
            (r["fault"] in CAMERA_FAULTS and have_images and r["health_camera"] >= 0.6)
            or (r["fault"] in RADAR_FAULTS and r["health_radar"] >= 0.6))
    ]
    print()
    # If the CLEAN baseline itself reads as unhealthy, the thresholds in
    # `sensor_health` do not suit this data and every row above is measured
    # against a broken reference. They are calibrated on real CARLA frames
    # (see that module's header), so this usually means synthetic or
    # otherwise atypical imagery rather than a genuine fault.
    if have_images and baseline["health_camera"] < 0.5:
        print(f"  WARNING: clean-baseline camera health is {baseline['health_camera']:.2f}.")
        print("  The health thresholds are calibrated for real CARLA frames; this input")
        print("  does not match them, so camera rows below are not meaningful.\n")
    if baseline["health_radar"] < 0.5:
        print(f"  WARNING: clean-baseline radar health is {baseline['health_radar']:.2f} -- "
              f"same caveat for the radar rows.\n")

    if not have_images:
        print("  NOTE: run without --no-images to exercise the camera faults; they were\n"
              "  not applied here and are reported as n/a rather than as passes.")

    # A run that tracked nothing proves nothing. `_degraded` requires a
    # measurable change from the baseline, so when every metric is empty NO
    # fault qualifies as a silent failure and the summary below would announce
    # a clean sweep on the strength of zero evidence. Say so instead.
    if baseline["n_detections"] == 0:
        print("  INVALID RUN: the clean baseline produced no detections, so every")
        print("  metric above is empty and no conclusion can be drawn about which")
        print("  faults were noticed. Check that the episode selection includes")
        print("  scenarios containing actors.")
        return

    if undetected:
        print(f"  SILENT FAILURES: {', '.join(undetected)}")
        print("  These degraded the pipeline without the health monitor noticing -- the "
              "most\n  dangerous category, since the system stays confident while wrong.")
    else:
        print("  Every injected sensor fault was detected by the health monitor.")


if __name__ == "__main__":
    main()
