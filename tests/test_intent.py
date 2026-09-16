"""Does the crossing prediction separate crossers from non-crossers, warn early
enough to matter, and respond to evidential uncertainty?
"""
import sys
import math
import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

from perception.intent import predict_crossing
from perception.tracking import Detection, MultiObjectTracker

DT = 0.1
rng = np.random.default_rng(1)
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    ok &= bool(cond)


def observe(state, ego_vel):
    fx, ly, vx, vy = state
    r = math.hypot(fx, ly)
    rel = np.array([vx - ego_vel[0], vy - ego_vel[1]])
    return Detection(
        bearing_rad=math.atan2(-ly, fx) + rng.normal(0, math.radians(1.0)),
        range_m=r + rng.normal(0, 0.5),
        range_rate=(rel[0] * fx + rel[1] * ly) / r + rng.normal(0, 0.5),
        cls=1, uncertainty=0.05,
    )


def build_track(v_lat, lat0=-4.0, fwd=18.0, frames=15, ego_speed=0.0, uncertainty=0.05):
    """Runs the tracker over a few frames of a walker, returns the track."""
    tk = MultiObjectTracker()
    ego_vel = np.array([ego_speed, 0.0])
    for k in range(frames):
        st = np.array([fwd - ego_speed * k * DT, lat0 + v_lat * k * DT, 0.0, v_lat])
        tk.predict(DT, ego_speed=ego_speed)
        tk.update([observe(st, ego_vel)], ego_vel=ego_vel)
    t = tk.tracks[0]
    t.uncertainty = uncertainty
    return t


print("1. A walker crossing into the lane vs one walking parallel to it")
crosser = build_track(v_lat=1.4, lat0=-4.0)
parallel = build_track(v_lat=0.0, lat0=-4.0)
pc = predict_crossing(crosser, ego_speed=8.0)
pp = predict_crossing(parallel, ego_speed=8.0)
print(f"  crossing  walker: {pc}")
print(f"  parallel  walker: {pp}")
check("crosser gets a high probability", pc.p_cross > 0.8, f"{pc.p_cross:.2f}")
check("parallel walker gets a low one", pp.p_cross < 0.2, f"{pp.p_cross:.2f}")
check("crosser flagged as risk", pc.is_risk)
check("parallel walker not flagged", not pp.is_risk)

print("\n2. Someone who has already crossed and left the lane")
# 15 frames at 1.4 m/s puts them 2.1 m out, past the 1.75 m corridor edge,
# still moving away -- so a LOW probability is the correct answer here.
gone = build_track(v_lat=1.4, lat0=0.0)
pg = predict_crossing(gone, ego_speed=8.0)
print(f"  out of the lane and receding:   {pg}")
# Threshold 0.3, not 0.2. Measurement noise was recalibrated against real
# CARLA data (bearing 1.0 -> 2.4 deg, range 0.5 -> 2.0 m), so the filter's
# covariance is now honestly wider and a walker 2.1 m outside the lane
# genuinely retains some chance of re-entering within the horizon. The old
# 0.02 came from an over-confident filter, not from better reasoning.
check("correctly no longer a risk", pg.p_cross < 0.3, f"{pg.p_cross:.2f}")
# Someone mid-crossing, still inside the corridor, is.
mid = build_track(v_lat=1.4, lat0=-1.6)
pm = predict_crossing(mid, ego_speed=8.0)
print(f"  mid-crossing, inside the lane:  {pm}")
check("mid-crossing walker is in-path", pm.p_cross > 0.8, f"{pm.p_cross:.2f}")

print("\n3. Warning lead time: does it fire before the ego arrives?")
for fwd in (25.0, 18.0, 12.0):
    t = build_track(v_lat=1.4, lat0=-4.0, fwd=fwd, ego_speed=8.0)
    p = predict_crossing(t, ego_speed=8.0)
    print(f"  walker {fwd:4.1f} m ahead -> p={p.p_cross:.2f} "
          f"entry={p.time_to_entry_s}s arrival={p.time_to_ego_arrival_s:.1f}s "
          f"lead={p.lead_time_s:+.2f}s")
t = build_track(v_lat=1.4, lat0=-4.0, fwd=25.0, ego_speed=8.0)
p = predict_crossing(t, ego_speed=8.0)
check("entry predicted before the ego gets there", p.lead_time_s > 0, f"{p.lead_time_s:+.2f}s")

print("\n4. Evidential uncertainty makes the DECISION more cautious")
print("   (p_cross stays honest; the cautious worst case is what rises)")
# Marginal AND below 0.5 -- the regime where caution can add something.
# Above 0.5 the cautious figure correctly equals the honest one, since
# widening only moves probability toward a coin flip. See intent.py.
base = build_track(v_lat=0.5, lat0=-4.2)
honest, cautious = [], []
for u in (0.0, 0.25, 0.5, 0.75, 1.0):
    base.uncertainty = u
    p = predict_crossing(base, ego_speed=8.0)
    honest.append(p.p_cross)
    cautious.append(p.p_cross_cautious)
    print(f"  uncertainty {u:.2f} -> p_cross {p.p_cross:.3f}   cautious {p.p_cross_cautious:.3f}")
check("honest probability is not distorted by uncertainty",
      max(honest) - min(honest) < 1e-9, f"spread {max(honest) - min(honest):.2e}")
check("cautious figure rises with uncertainty", cautious[-1] > cautious[0],
      f"{cautious[0]:.3f} -> {cautious[-1]:.3f}")
check("cautious figure is monotonic",
      all(b >= a - 1e-9 for a, b in zip(cautious, cautious[1:])))
check("cautious never below honest", all(c >= h - 1e-9 for c, h in zip(cautious, honest)))

print("\n4b. A poorly-seen walker trips the warning that a well-seen one does not")
marg = build_track(v_lat=0.5, lat0=-4.2)
marg.uncertainty = 0.0
clear_call = predict_crossing(marg, ego_speed=8.0)
marg.uncertainty = 0.9
murky_call = predict_crossing(marg, ego_speed=8.0)
print(f"  clearly seen (u=0.0): p={clear_call.p_cross:.3f} "
      f"cautious={clear_call.p_cross_cautious:.3f} risk={clear_call.is_risk}")
print(f"  partial glimpse (u=0.9): p={murky_call.p_cross:.3f} "
      f"cautious={murky_call.p_cross_cautious:.3f} risk={murky_call.is_risk}")
check("the uncertain case is escalated", murky_call.p_cross_cautious > clear_call.p_cross_cautious)

print("\n5. A coasting (occluded) track predicts with wider spread")
tk = MultiObjectTracker()
ego_vel = np.array([8.0, 0.0])
for k in range(15):
    st = np.array([20.0 - 0.8 * k * DT, -4.0 + 1.4 * k * DT, 0.0, 1.4])
    tk.predict(DT, ego_speed=8.0)
    tk.update([observe(st, ego_vel)], ego_vel=ego_vel)
visible_p = predict_crossing(tk.tracks[0], ego_speed=8.0)
for _ in range(10):                       # walker disappears behind an occluder
    tk.predict(DT, ego_speed=8.0)
    tk.update([], ego_vel=ego_vel)
coast_p = predict_crossing(tk.tracks[0], ego_speed=8.0)
print(f"  while visible:  {visible_p}")
print(f"  after 10 hidden frames: {coast_p}")
check("track is marked coasting", tk.tracks[0].is_coasting)
check("still produces a prediction while hidden", coast_p.p_cross > 0.0)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
