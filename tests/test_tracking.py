"""Does the EKF recover a crossing pedestrian's lateral speed, and does the
camera/radar ablation show the fusion is actually necessary?

Simulates a walker crossing left-to-right in front of a stationary ego, feeds
the tracker noisy measurements, and compares recovered velocity to truth under
three sensor configurations.
"""
import sys
import math
import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

from perception.tracking import Detection, MultiObjectTracker

rng = np.random.default_rng(0)
DT = 0.1
BEARING_STD = math.radians(1.0)
RANGE_STD = 0.5
RATE_STD = 0.5


def truth_track(n=40, fwd=15.0, lat0=-3.0, v_lat=1.4, v_fwd=0.0):
    """A pedestrian crossing at constant velocity, in ego-frame coords."""
    for k in range(n):
        yield np.array([fwd + v_fwd * k * DT, lat0 + v_lat * k * DT, v_fwd, v_lat])


def measure(state, mode):
    fx, ly, vx, vy = state
    r = math.hypot(fx, ly)
    bearing = math.atan2(-ly, fx)
    rate = (vx * fx + vy * ly) / r
    det = Detection(cls=1)
    if mode in ("camera", "fused"):
        det.bearing_rad = bearing + rng.normal(0, BEARING_STD)
    if mode in ("radar", "fused"):
        det.range_m = r + rng.normal(0, RANGE_STD)
        det.range_rate = rate + rng.normal(0, RATE_STD)
    return det


def run(mode, v_lat=1.4):
    tk = MultiObjectTracker()
    err_lat, err_fwd, final = [], [], None
    for state in truth_track(v_lat=v_lat):
        tk.predict(DT, ego_speed=0.0, ego_yaw_rate=0.0)
        tk.update([measure(state, mode)], ego_vel=np.zeros(2))
        t = tk.tracks[0] if tk.tracks else None
        if t is not None and t.age > 12:      # let the filter converge
            err_lat.append(abs(t.x[3] - state[3]))
            err_fwd.append(abs(t.x[2] - state[2]))
            final = t
    return (float(np.mean(err_lat)) if err_lat else float("nan"),
            float(np.mean(err_fwd)) if err_fwd else float("nan"),
            final)


print("Truth: pedestrian crossing at v_lateral = 1.40 m/s, 15 m ahead, ego stationary")
print(f"{'sensors':<10} {'lat vel err':>12} {'fwd vel err':>12} {'lat vel std':>12}  verdict")
results = {}
for mode in ("radar", "camera", "fused"):
    el, ef, t = run(mode)
    results[mode] = el
    std = t.lateral_speed_std if t else float("nan")
    print(f"{mode:<10} {el:>10.3f}   {ef:>10.3f}   {std:>10.3f}")

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)
    return cond

print("\nAblation claims:")
check("fused beats radar-only on lateral velocity",
      results["fused"] < results["radar"],
      f"{results['fused']:.3f} vs {results['radar']:.3f}")
check("fused recovers lateral speed to better than 0.5 m/s",
      results["fused"] < 0.5, f"{results['fused']:.3f} m/s")

print("\nThe physics claim: Doppler alone cannot see lateral motion.")
print("Same crossing speed, but the walker placed dead ahead (lateral ~ 0),")
print("where the line of sight is exactly perpendicular to their motion:")
tk = MultiObjectTracker()
for k in range(40):
    st = np.array([15.0, -0.05 + 1.4 * k * DT, 0.0, 1.4])
    tk.predict(DT)
    tk.update([measure(st, "radar")], ego_vel=np.zeros(2))
t = tk.tracks[0]
print(f"  radar-only, dead ahead -> estimated v_lateral = {t.x[3]:+.3f} m/s (truth 1.400)")
print(f"                            lateral velocity std = {t.lateral_speed_std:.3f} m/s")
check("radar-only leaves lateral velocity badly wrong", abs(t.x[3] - 1.4) > 0.4)
check("and the filter reports that uncertainty honestly", t.lateral_speed_std > 0.5)

print("\nCoasting through an occlusion (no measurements at all for 15 frames):")
tk = MultiObjectTracker()
states = list(truth_track(n=45))
for k, st in enumerate(states):
    tk.predict(DT)
    if 20 <= k < 35:
        tk.update([], ego_vel=np.zeros(2))          # fully occluded
    else:
        tk.update([measure(st, "fused")], ego_vel=np.zeros(2))
    if k == 19:
        std_before = tk.tracks[0].lateral_speed_std
    if k == 34:
        t = tk.tracks[0]
        std_after, pos_err = t.lateral_speed_std, abs(t.x[1] - st[1])
print(f"  lateral vel std before occlusion: {std_before:.3f} m/s")
print(f"  lateral vel std after 15 hidden frames: {std_after:.3f} m/s")
print(f"  position error after coasting: {pos_err:.3f} m")
check("track survived the occlusion", len(tk.tracks) > 0)
check("it was coasting while hidden", std_after > std_before)
check("dead reckoning stayed within 1 m", pos_err < 1.0, f"{pos_err:.3f} m")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
