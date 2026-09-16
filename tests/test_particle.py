"""Does the particle filter earn its place over the EKF?

Three claims to check:
  1. It holds SEPARATE motion hypotheses instead of one averaged blob.
  2. It reasons from ABSENCE of detection -- belief concentrates into the
     occluded region with no measurement ever arriving.
  3. It predicts WHERE and WHEN the hazard reappears, while still invisible.
"""
import sys
import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

from perception.occlusion_grid import EMPTY, OCCLUDED, UNKNOWN, VISIBLE
from perception.particle_tracker import HiddenHazardTracker
from perception.tracking import MultiObjectTracker, Detection

GRID_CFG = {"size_cells": 20, "cell_size_m": 2.0, "extent_m": 40.0}
DT = 0.1
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{extra}]" if extra else ""))
    ok &= bool(cond)


def make_grid():
    """A bus parked ahead-right: cells behind it are OCCLUDED, rest VISIBLE."""
    g = np.full((20, 20), VISIBLE, dtype=np.uint8)
    # forward index 7-10, lateral index 10-13 -> the bus's shadow
    g[7:11, 10:14] = OCCLUDED
    return g


grid = make_grid()

print("1. The particle cloud holds distinct motion hypotheses")
# Hazard last seen at forward 16 m, lateral +2 m, walking right at 1.4 m/s
pf = HiddenHazardTracker(init_xy=(16.0, 2.0), init_vel=(0.0, -1.4),
                          rng=np.random.default_rng(1))
modes = pf.mode_masses()
print(f"     initial modes: { {k: round(v,3) for k,v in modes.items()} }")
check("all four hypotheses represented", len(modes) == 4)
check("no single hypothesis dominates at birth", max(modes.values()) < 0.6,
      f"max {max(modes.values()):.2f}")

print("\n2. Negative information: belief concentrates without any detection")
# Ego held stationary here. With a moving ego the occlusion grid must be
# recomputed each frame -- the bus's shadow shifts to nearer cells as you
# approach it -- and holding a static grid while driving 12 m would compare the
# hazard against a stale map. Ego motion is exercised in the pipeline test.
from perception.geometry import ego_xy_to_bev_cell


def mass_by_state(tracker, g, state):
    fi, li = ego_xy_to_bev_cell(tracker.particles[:, 0], tracker.particles[:, 1], GRID_CFG)
    ins = (fi >= 0) & (fi < 20) & (li >= 0) & (li < 20)
    m = np.zeros(len(tracker.particles), bool)
    m[ins] = g[fi[ins], li[ins]] == state
    return float(tracker.weights[m].sum())


occ, vis = [mass_by_state(pf, grid, OCCLUDED)], [mass_by_state(pf, grid, VISIBLE)]
for step in range(20):
    pf.predict(DT, ego_speed=0.0)
    pf.update_from_occlusion(grid, GRID_CFG, detections_xy=None)
    occ.append(mass_by_state(pf, grid, OCCLUDED))
    vis.append(mass_by_state(pf, grid, VISIBLE))

print(f"     mass in OCCLUDED cells: {occ[0]:.2f} -> {occ[-1]:.2f}")
print(f"     mass in VISIBLE  cells: {vis[0]:.2f} -> {vis[-1]:.2f}")
check("belief stays concentrated in the hidden region", occ[-1] > 0.5,
      f"{occ[-1]:.2f}")
check("particles drifting into open view are suppressed", vis[-1] < 0.25,
      f"{vis[-1]:.2f}")
check("filter did not degenerate", not pf.is_degenerate,
      f"ESS {pf.effective_sample_size:.0f}/{pf.n}")

print(f"\n     modes after 20 hidden frames: "
      f"{ {k: round(v,3) for k,v in pf.mode_masses().items()} }")

print("\n3. Emergence prediction while still completely invisible")
pred = pf.predict_emergence(grid, GRID_CFG, horizon_s=3.0)
print(f"     {pred}")
check("produces an emergence prediction", pred.probability > 0.0)
if pred.will_emerge:
    check("predicted location is ahead of the ego", pred.location_xy[0] > 0,
          f"forward {pred.location_xy[0]:.1f} m")
    check("emergence time within horizon", 0 < pred.time_to_emerge_s <= 3.0,
          f"{pred.time_to_emerge_s:.2f}s")

print("\n4. THE POINT: a Gaussian cannot represent this, a particle set can")
# Force a genuinely bimodal situation: half continue right, half reverse left.
pf2 = HiddenHazardTracker(init_xy=(16.0, 0.0), init_vel=(0.0, -1.4),
                          rng=np.random.default_rng(7))
for _ in range(15):
    pf2.predict(DT)
    pf2.update_from_occlusion(grid, GRID_CFG)

lat = pf2.particles[:, 1]
w = pf2.weights
mean_lat = float(np.average(lat, weights=w))
left_mass = float(w[lat < mean_lat - 0.5].sum())
right_mass = float(w[lat > mean_lat + 0.5].sum())
print(f"     weighted mean lateral position: {mean_lat:+.2f} m")
print(f"     mass well LEFT of the mean:  {left_mass:.2f}")
print(f"     mass well RIGHT of the mean: {right_mass:.2f}")
print(f"     mass NEAR the mean:          {1 - left_mass - right_mass:.2f}")
check("substantial mass on BOTH sides of the mean",
      left_mass > 0.15 and right_mass > 0.15,
      f"L={left_mass:.2f} R={right_mass:.2f}")
print("     -> a single Gaussian would place its peak where little mass actually is;")
print("        the particle set keeps the separated possibilities explicit.")

print("\n5. Re-identification on reappearance")
true_spot = pf2.mean_position
same = pf2.reidentification_score(true_spot)
elsewhere = pf2.reidentification_score(true_spot + np.array([0.0, 12.0]))
print(f"     detection at the tracked location:  score {same:.2f}")
print(f"     detection 12 m away:                score {elsewhere:.2f}")
check("matches the hazard it was tracking", same > 0.3, f"{same:.2f}")
check("rejects an unrelated detection", elsewhere < 0.05, f"{elsewhere:.2f}")

print("\n6. Contrast: the EKF collapses to one mode")
ekf = MultiObjectTracker()
# Confirm the track first (needs MIN_HITS_TO_CONFIRM detections), then occlude.
for k in range(5):
    ekf.predict(DT)
    ekf.update([Detection(bearing_rad=-0.01 * k, range_m=16.0 - 0.1 * k,
                           range_rate=-1.0, cls=1)])
for _ in range(15):
    ekf.predict(DT)
    ekf.update([])                     # occluded: no measurements at all
t = ekf.tracks[0]
print(f"     EKF state: forward {t.x[0]:.1f} lateral {t.x[1]:+.1f}, "
      f"lateral std {t.lateral_speed_std:.2f}")
print("     -> one mean, one covariance. It cannot say 'either front or rear of")
print("        the bus, but not the middle', which is the actual situation.")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
