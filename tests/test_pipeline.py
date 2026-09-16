"""End-to-end: does the integrated pipeline carry a hazard through a full
occlusion event, and do the component couplings actually fire?

Simulates a pedestrian walking behind a parked bus and out again, feeding the
pipeline detections + radar + an occlusion grid, and watches what it concludes
at each stage.
"""
import sys
import math

import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

from perception.occlusion_grid import EMPTY, OCCLUDED, VISIBLE
from perception.pipeline import PerceptionPipeline

DT = 0.1
BEV = {"camera": {"width": 800, "height": 600, "fov": 90, "x": 1.5, "z": 1.7},
       "bev_grid": {"size_cells": 20, "cell_size_m": 2.0, "extent_m": 40.0}}
FOCAL = 800 / (2 * math.tan(math.radians(90) / 2))
rng = np.random.default_rng(0)
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{extra}]" if extra else ""))
    ok &= bool(cond)


def grid_with_shadow():
    g = np.full((20, 20), EMPTY, dtype=np.uint8)
    g[6:11, 11:15] = OCCLUDED          # the bus's shadow
    return g


def box_for(fwd, lat):
    r = max(fwd, 1.0)
    cx = 400 - FOCAL * lat / r
    hw, hh = FOCAL * 0.35 / r, FOCAL * 0.9 / r
    return (cx - hw, 300 - hh, cx + hw, 300 + hh)


def radar_for(fwd, lat, ego_speed, n=8):
    r = math.hypot(fwd, lat)
    az = math.atan2(-lat, fwd)
    rate = (-ego_speed * fwd) / max(r, 1e-3)
    pts = [[rate + rng.normal(0, .2), az + rng.normal(0, .005), 0.0, r + rng.normal(0, .2)]
           for _ in range(n)]
    # roadside structure, so the frame looks like a real radar return set
    for az0, d0 in ((0.20, 15.0), (-0.22, 24.0)):
        pts += [[-ego_speed + rng.normal(0, .2), az0 + rng.normal(0, .006), 0.0,
                 d0 + rng.normal(0, .3)] for _ in range(5)]
    return np.array(pts, np.float32)


grid = grid_with_shadow()
pipe = PerceptionPipeline(bev_cfg=BEV)
EGO = 8.0

# Walker starts visible at lateral -5, walks right, passes behind the bus
# (roughly lateral +2 to +6 at forward ~16), then emerges.
print("Walking a pedestrian behind a parked bus and out again:\n")
print(f"  {'t':>4} {'state':>9} {'objs':>5} {'trks':>5} {'hidden':>7} "
      f"{'risk':>6} {'action':>7}  note")

phases, hidden_seen, risk_while_hidden = [], 0, []
for k in range(60):
    lat = -5.0 + 0.22 * k
    # The ego is CLOSING at EGO m/s, so the target's forward range must shrink.
    # Holding it constant made the filter's ego-motion compensation predict a
    # range 0.8 m short every frame, which broke association and spawned a new
    # track every few frames -- six tracks and seven particle filters for one
    # pedestrian.
    fwd = 34.0 - EGO * k * DT
    behind_bus = 2.0 <= lat <= 6.0

    if behind_bus:
        dets, radar = [], radar_for(fwd, lat, EGO, n=0)      # nothing observable
        state = "OCCLUDED"
    else:
        dets = [(box_for(fwd, lat), 1, 0.88, 0.12)]
        radar = radar_for(fwd, lat, EGO)
        state = "visible"

    res = pipe.process(radar_pts=radar, ego_speed=EGO, dt=DT,
                       occlusion_grid=grid, detections=dets)
    phases.append((state, len(res.tracks), len(res.hidden)))
    if behind_bus:
        hidden_seen = max(hidden_seen, len(res.hidden))
        risk_while_hidden.append(res.risk.score)

    if k % 6 == 0 or (behind_bus and len(res.hidden)):
        note = ""
        if res.hidden:
            _tid, em = res.hidden[0]
            note = str(em)[:52]
        print(f"  {k*DT:>4.1f} {state:>9} {len(res.objects):>5} {len(res.tracks):>5} "
              f"{len(res.hidden):>7} {res.risk.score:>6.2f} "
              f"{res.risk.action_name:>7}  {note}")

print("\n1. The hazard survives the occlusion")
check("a particle filter was spawned while hidden", hidden_seen > 0,
      f"max {hidden_seen} hidden hazards")
check("risk stayed non-zero while nothing was detectable",
      any(r > 0.0 for r in risk_while_hidden),
      f"max risk {max(risk_while_hidden):.2f}" if risk_while_hidden else "none")

print("\n2. Component couplings fire")
pipe2 = PerceptionPipeline(bev_cfg=BEV)
for _ in range(12):
    pipe2.process(radar_pts=radar_for(18, -5, EGO), ego_speed=EGO, dt=DT,
                  occlusion_grid=grid, detections=[(box_for(18, -5), 1, 0.9, 0.1)])
r = pipe2.process(radar_pts=radar_for(18, -5, EGO), ego_speed=EGO, dt=DT,
                  occlusion_grid=grid, detections=[(box_for(18, -5), 1, 0.9, 0.1)])
o = r.objects[0]
print(f"  detected object -> P(hazard)={o.hazard_posterior:.2f} trust={o.trusted}")
print(f"                     p_cross={o.p_cross}  track={o.track_id}")
check("contradiction resolver produced a posterior", 0.0 < o.hazard_posterior <= 1.0)
check("agreeing sensors give high confidence", o.hazard_posterior > 0.7,
      f"{o.hazard_posterior:.2f}")
check("the detection was bound to a track", o.track_id is not None)
check("and carries a crossing probability", o.p_cross is not None)

print("\n3. Health feeds the fusion: a blinded camera changes the conclusion")
dark = PerceptionPipeline(bev_cfg=BEV)
for _ in range(12):
    dark.process(rgb=np.full((120, 160, 3), 6.0), radar_pts=radar_for(18, -5, EGO),
                 ego_speed=EGO, dt=DT, occlusion_grid=grid,
                 detections=[(box_for(18, -5), 1, 0.9, 0.1)])
rd = dark.process(rgb=np.full((120, 160, 3), 6.0), radar_pts=radar_for(18, -5, EGO),
                  ego_speed=EGO, dt=DT, occlusion_grid=grid,
                  detections=[(box_for(18, -5), 1, 0.9, 0.1)])
print(f"  healthy camera -> health {r.health.camera:.2f}, P(hazard) {o.hazard_posterior:.2f}")
print(f"  blinded camera -> health {rd.health.camera:.2f}, "
      f"P(hazard) {rd.objects[0].hazard_posterior:.2f}")
check("the health monitor noticed the dark camera", rd.health.camera < 0.5,
      f"{rd.health.camera:.2f}")
check("and the fusion weighted its evidence down",
      rd.objects[0].hazard_posterior < o.hazard_posterior,
      f"{rd.objects[0].hazard_posterior:.2f} < {o.hazard_posterior:.2f}")

print("\n4. Risk with NOTHING detected, purely from blocked view")
empty = PerceptionPipeline(bev_cfg=BEV)
blocked = np.full((20, 20), EMPTY, dtype=np.uint8)
blocked[:10, 6:16] = OCCLUDED
re = empty.process(radar_pts=np.empty((0, 4), np.float32), ego_speed=13.0, dt=DT,
                   occlusion_grid=blocked, detections=[])
print(f"  {re.risk}")
print(re.risk.explain())
check("elevated risk with zero detections", re.risk.score > 0.0, f"{re.risk.score:.2f}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
