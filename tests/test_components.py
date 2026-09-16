"""Components 5, 7 and 8: contradiction resolver, risk engine, sensor health."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

import scripts.adversarial_test as adv

from perception.contradiction import SensorEvidence, resolve
from perception.occlusion_grid import EMPTY, OCCLUDED, UNKNOWN, VISIBLE
from perception.particle_tracker import EmergencePrediction
from perception.risk import BRAKE, INFORM, NONE, WARN, assess
from perception.sensor_health import SensorHealthMonitor

ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{extra}]" if extra else ""))
    ok &= bool(cond)


print("=" * 70)
print("COMPONENT 5 -- Bayesian Contradiction Resolver")
print("=" * 70)

print("\nThe decisive case: nothing detected, but the view is blocked")
clear = resolve(SensorEvidence(camera_detected=False, radar_detected=False,
                                occlusion_state=EMPTY, is_vru=True))
blocked = resolve(SensorEvidence(camera_detected=False, radar_detected=False,
                                  occlusion_state=OCCLUDED, is_vru=True))
print(f"  view CLEAR,   nothing seen -> {clear}")
print(f"  view BLOCKED, nothing seen -> {blocked}")
check("blocked-view silence keeps hazard probability far higher",
      blocked.posterior > 3 * clear.posterior,
      f"{blocked.posterior:.3f} vs {clear.posterior:.3f}")
check("clear-view silence is strong evidence of absence", clear.posterior < 0.05,
      f"{clear.posterior:.3f}")
print(f"  explanation: {blocked.explanation}")

print("\nRadar-only return, by occlusion state")
r_occ = resolve(SensorEvidence(camera_detected=False, radar_detected=True,
                                radar_confidence=0.7, occlusion_state=OCCLUDED))
r_vis = resolve(SensorEvidence(camera_detected=False, radar_detected=True,
                                radar_confidence=0.7, occlusion_state=VISIBLE))
print(f"  behind an occluder -> {r_occ}")
print(f"  in plain view      -> {r_vis}")
check("radar-only is believed more when the camera COULDN'T have seen it",
      r_occ.posterior > r_vis.posterior,
      f"{r_occ.posterior:.3f} vs {r_vis.posterior:.3f}")

print("\nA blinded camera should not be able to argue either way")
healthy = resolve(SensorEvidence(camera_detected=False, radar_detected=True,
                                  radar_confidence=0.7, occlusion_state=VISIBLE,
                                  camera_health=1.0))
blind = resolve(SensorEvidence(camera_detected=False, radar_detected=True,
                                radar_confidence=0.7, occlusion_state=VISIBLE,
                                camera_health=0.1))
print(f"  healthy camera silent -> {healthy}")
print(f"  blinded camera silent -> {blind}")
check("a blinded camera's silence stops suppressing the hazard",
      blind.posterior > healthy.posterior,
      f"{blind.posterior:.3f} vs {healthy.posterior:.3f}")
check("its likelihood ratio is pulled toward 1", abs(blind.camera_likelihood_ratio - 1.0)
      < abs(healthy.camera_likelihood_ratio - 1.0),
      f"LR {blind.camera_likelihood_ratio:.2f} vs {healthy.camera_likelihood_ratio:.2f}")

print("\nBoth sensors agree")
both = resolve(SensorEvidence(camera_detected=True, camera_confidence=0.9,
                               camera_uncertainty=0.05, radar_detected=True,
                               radar_confidence=0.8, occlusion_state=VISIBLE))
print(f"  {both}")
check("agreement gives high confidence", both.posterior > 0.85, f"{both.posterior:.3f}")

print("\n" + "=" * 70)
print("COMPONENT 8 -- Sensor Health Monitor")
print("=" * 70)

rng = np.random.default_rng(0)

# Real CARLA frames, not synthetic ones. `sensor_health`'s thresholds are
# calibrated against this exact distribution (see that module's header), so a
# hand-made scene tests the fixture rather than the monitor -- three successive
# synthetic fixtures each scored a "clean" camera as failing for a different
# reason before this was switched to real data.
import random as _random
_random.seed(0)
# Any collected dataset will do -- the thresholds are calibrated against real
# CARLA output and v1/v2 are both that. Checked in order so the test keeps
# working after data/raw is archived.
FRAMES = []
for _cand in ("data/raw_v2/images", "data/raw_v1/images", "data/raw/images"):
    FRAMES = sorted((Path(r"E:\Project Phase 1") / _cand).glob("*.jpg"))
    if FRAMES:
        print(f"  using {len(FRAMES)} real frames from {_cand}")
        break

if not FRAMES:
    print("  no collected dataset found -- skipping camera-health checks")
    REAL = []
else:
    REAL = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32)
            for p in _random.sample(FRAMES, 30)]


# Real radar frames, for the same reason the images are real: the radar
# thresholds are calibrated against CARLA's actual output, which saturates at
# ~143 returns per frame. A hand-made 21-return fixture reads as a blocked
# sensor against those thresholds and tests nothing but the fixture.
_META = []
for _cand in ("data/raw_v2/meta", "data/raw_v1/meta", "data/raw/meta"):
    _META = sorted((Path(r"E:\Project Phase 1") / _cand).glob("*.npz"))
    if _META:
        break
REAL_RADAR = [np.load(p, allow_pickle=True)["radar_pts"]
              for p in _random.sample(_META, min(30, len(_META)))] if _META else []
_radar_i = [0]


def clustered_radar(n_targets=3, per_target=7):
    """One real radar frame, cycled. Falls back to a synthetic cluster only if
    no dataset is present."""
    if REAL_RADAR:
        f = REAL_RADAR[_radar_i[0] % len(REAL_RADAR)]
        _radar_i[0] += 1
        return f
    out = []
    for az, dp in ((0.10, 20.0), (-0.08, 12.0), (0.02, 33.0))[:n_targets]:
        out.append(np.stack([rng.normal(-8, 0.3, per_target),
                              rng.normal(az, 0.006, per_target),
                              np.zeros(per_target),
                              rng.normal(dp, 0.3, per_target)], axis=1))
    return np.vstack(out).astype(np.float32)


print("\nClear day vs heavy rain at night (real CARLA frames + injected fault)")
good = SensorHealthMonitor()
bad = SensorHealthMonitor()
for img in REAL:
    dark, _ = adv.darkness(img, None, rng)
    night, _ = adv.fog(dark, None, rng)
    good.update(rgb=img, radar_pts=clustered_radar())
    bad.update(rgb=night, radar_pts=clustered_radar())
g, b = good.report(), bad.report()
print(f"  clear day        -> {g}")
print(f"  rain + night     -> {b}")
for r in b.camera_reasons:
    print(f"      camera: {r}")
check("clear-day camera reads healthy", g.camera > 0.7, f"{g.camera:.2f}")
check("night/rain camera reads degraded", b.camera < 0.4, f"{b.camera:.2f}")
check("degradation is flagged", b.degraded)
check("radar is unaffected by darkness", b.radar > 0.6, f"{b.radar:.2f}")

print("\nRadar blockage")
blocked_radar = SensorHealthMonitor()
for img in REAL:
    blocked_radar.update(rgb=img, radar_pts=rng.normal(size=(1, 4)).astype(np.float32))
br = blocked_radar.report()
print(f"  {br}")
for r in br.radar_reasons:
    print(f"      radar: {r}")
check("sparse returns detected as radar degradation", br.radar < 0.5, f"{br.radar:.2f}")
check("camera still trusted", br.camera > 0.7, f"{br.camera:.2f}")

print("\nSustained disagreement lowers BOTH")
arguing = SensorHealthMonitor()
for img in REAL:
    arguing.update(rgb=img, radar_pts=clustered_radar(), sensors_disagreed=True)
a = arguing.report()
print(f"  {a}")
check("both sensors discounted when they persistently conflict",
      a.camera < g.camera and a.radar < g.radar,
      f"cam {a.camera:.2f}<{g.camera:.2f}, rad {a.radar:.2f}<{g.radar:.2f}")

print("\n" + "=" * 70)
print("COMPONENT 7 -- Dynamic Risk Scoring Engine")
print("=" * 70)


class T:
    def __init__(self, id, fwd, lat, vf=0.0, vl=0.0, cls=1):
        self.id, self.cls = id, cls
        self.position = np.array([fwd, lat])
        self.velocity = np.array([vf, vl])


class P:
    def __init__(self, p):
        self.p_cross, self.p_cross_cautious = p, p


clear_grid = np.full((20, 20), EMPTY, dtype=np.uint8)
blocked_grid = np.full((20, 20), EMPTY, dtype=np.uint8)
blocked_grid[:10, 8:16] = OCCLUDED

print("\nEmpty road, clear view")
r = assess(occlusion_grid=clear_grid, ego_speed=10.0)
print(f"  {r}")
check("no risk on an empty clear road", r.action == NONE, r.action_name)

print("\nSame road, but half the near view is blocked -- NO detections at all")
r = assess(occlusion_grid=blocked_grid, ego_speed=12.0)
print(f"  {r}")
print(r.explain())
check("occlusion alone produces non-zero risk", r.score > 0.0, f"{r.score:.2f}")
check("and a speed advisory", r.advisory_speed_ms is not None,
      f"{(r.advisory_speed_ms or 0) * 3.6:.0f} km/h")
print("     -> this is the capability a detector-driven system cannot have:")
print("        elevated risk with nothing detected, and a reason for it.")

print("\nVisible pedestrian, very likely to cross, close")
r = assess(tracks=[T(1, 12.0, -2.0, vl=1.4)], predictions={1: P(0.95)},
            occlusion_grid=clear_grid, ego_speed=12.0)
print(f"  {r}")
print(r.explain())
check("escalates to WARN or BRAKE", r.action >= WARN, r.action_name)

print("\nHIDDEN hazard only -- nothing visible anywhere")
emergence = EmergencePrediction(will_emerge=True, time_to_emerge_s=1.4,
                                 location_xy=(18.0, -1.0), probability=0.8,
                                 modes={"continues": 0.6, "turns back": 0.2})
r = assess(hidden_hazards=[emergence], occlusion_grid=blocked_grid, ego_speed=13.0)
print(f"  {r}")
print(r.explain())
check("a hazard nobody can see still raises risk", r.score > 0.25, f"{r.score:.2f}")
check("and triggers a driver-facing action", r.action >= INFORM, r.action_name)

print("\nSame hidden hazard, but it emerges well off to the side")
side = EmergencePrediction(will_emerge=True, time_to_emerge_s=1.4,
                           location_xy=(18.0, -9.0), probability=0.8,
                           modes={"continues": 0.6})
r_side = assess(hidden_hazards=[side], occlusion_grid=blocked_grid, ego_speed=13.0)
print(f"  {r_side}")
check("off-path emergence is scored lower", r_side.score < r.score,
      f"{r_side.score:.2f} < {r.score:.2f}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
