"""Live integration demo: ego drives autonomously (autopilot) along a fixed,
deterministic course -- no background traffic, no random pedestrians -- past
`N_COURSE_OCCLUDERS` (default 6) parked occluders spaced `COURSE_SPACING_M`
apart, each paired with a hidden pedestrian placed truly inside its shadow
and triggered to cross as the ego closes. `build_fixed_course` lays this out
once, before the drive starts, by calling
`carla_tools.urban_world.EncounterManager.stage_at_waypoint` at a caller-
chosen list of waypoints -- the same placement geometry the dynamic,
ego-relative staging used for dataset collection relies on
(scenario_e_urban_crossing.py), just driven from a fixed plan instead of
"~35m ahead of wherever the ego currently is". That switch was deliberate:
random background traffic and reactive staging made the demo's braking look
arbitrary (sometimes braking for no visible reason, camera timing varying
run to run) -- a fixed course of clearly separated, individually inspectable
encounters is far easier to read and to debug. The full Phase 1-4 pipeline
runs continuously on the live camera + radar feed. Labels are plain-language
on purpose (percentages, "Not sure", "Hidden zones") so a non-technical
viewer can follow it without narration; see docs/DASHBOARD.md for the
technical field names underneath.

  - CARLA's own window: the raw 3D driver POV (spectator locked to the RGB
    camera's transform every tick).
  - A second window, a three-panel dashboard:
      left   = camera feed with live bounding boxes (pretrained YOLOv8n
               proposals), each annotated with the trained evidential
               camera confidence, a rule-based radar confidence score, and
               a "NOT SURE" tag when uncertainty crosses the threshold
               (Week 2 scenario data / Week 3 confidence + radar + flag)
      middle = the live 20x20 occlusion grid, camera(MiDaS)+radar only
               (Week 4)
      right  = raw radar detections plotted in bird's-eye view (distinct
               from the occlusion grid -- this is the sensor's own read,
               not the fused decision), colored by closing speed
      status bar = sim time, speed, object/flag counts, radar strength,
              hidden-zone count, measured pipeline FPS

Confidence and detections are whatever the live traffic actually produces --
not scripted per-frame -- which is the honest thing to show for something
meant to demonstrate general validity rather than one hand-picked vignette.
Only the encounter staging (occluder placement, walker trigger point) is
deliberately constructed, and that mirrors how the dataset itself was built:
occlusions are encountered by a driving ego, not posed in front of a parked
one.

Run with CARLA already started WITHOUT -RenderOffScreen, e.g.:
    E:\\Carla-0.10.0\\Carla-0.10.0-Win64-Shipping\\CarlaUnreal.exe -quality-level=Low -ResX=1280 -ResY=800

Press 'q' in the dashboard window to stop and clean up.
"""
import math
import queue
import random
import sys
import time
from collections import deque
from pathlib import Path

# Consecutive tick failures (dropped sensor frame, an actor destroyed mid-tick)
# before giving up rather than retrying forever against a dead server.
MAX_TICK_FAILURES = 15

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import carla
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from carla_tools.client import connect, disconnect
from carla_tools.sensors import (
    radar_to_array, rgb_to_array, set_walker_velocity, spawn_sensor_rig,
)
# The exact camera degradation scripts/adversarial_test.py validates offline
# against recorded frames (component 9) -- reused here unmodified rather than
# reimplemented, so a live "fog" demo is provably the same fault the paper's
# adversarial numbers were measured against, not a fresh, unvalidated effect.
# Radar is untouched by design, in the function's own signature: fog scatters
# light, not radio, so this is the one fault where camera and radar are
# supposed to disagree, and the health monitor is expected to notice on the
# camera side (contrast collapse) and lower its fusion weight accordingly.
from scripts.adversarial_test import fog as inject_fog
# The real, tested implementation -- not this file's own duplicate, which
# independently reinvented ego spawning and never got the kerb-clearance fix
# applied to the dataset-collection path (see spawn_static_occluders'
# history for what that cost). EncounterManager is what actually pairs an
# occluder with a hidden pedestrian and triggers a genuine crossing as the
# ego closes; build_fixed_course drives it from a deterministic waypoint
# list instead of its own dynamic "35m ahead of the ego" staging.
from carla_tools.urban_world import EncounterManager
from carla_tools.urban_world import spawn_ego_with_autopilot as uw_spawn_ego_with_autopilot
from common.config import load_yaml
from models.evidential_classifier import CROP_SIZE, EvidentialDetector
from models.evidential_head import DEFAULT_UNCERTAINTY_THRESHOLD, EvidentialHead, uncertainty_flag
from perception.depth_midas import estimate_disparity, normalize_disparity
from perception.occlusion_grid import OCCLUDED, classify_grid, labels_to_rgb
from perception.pipeline import PerceptionPipeline
from perception.radar_confidence import radar_confidence, radar_confidence_in_region
from perception.risk import BRAKE
from perception.risk import NONE as NONE_ACTION

# Consecutive frames the risk action must stay below WARN before control is
# handed back to the Traffic Manager. Without this, a score sitting right at
# a tier boundary flickers the vehicle between manual control and autopilot
# every frame or two, which looks like a control fault rather than a decision.
BRAKE_RELEASE_FRAMES = 15

# Light brake held during the release frames after a stop, while the risk is
# clearing but control has not yet gone back to the Traffic Manager.
RELEASE_BRAKE_AMOUNT = 0.35

# Occlusion-aware cruising speed, applied through the Traffic Manager while
# the risk engine is still below WARN -- i.e. BEFORE any hazard is detected
# or even believed in.
#
# `perception.risk` already computes exactly this number: its third risk
# source ("occlusion burden") derives an advisory speed from how much of the
# near road is blocked, on the doctrine that you should only drive as fast as
# you can stop within the distance you can actually see. That advisory was
# computed and then thrown away here -- nothing in this demo ever read
# `RiskAssessment.advisory_speed_ms`. The ego therefore drove past parked
# occluders at full autopilot speed with ~62-67% of the near grid OCCLUDED
# (measured live), which left no margin at all when a pedestrian stepped out
# from behind one: the hidden-hazard prediction reached WARN 1.2s before the
# raw-proximity layer fired at 0.2m, far too late to stop from that speed.
#
# TM takes a percentage relative to the posted limit, so the advisory (m/s)
# is converted against `ego.get_speed_limit()` each time it moves materially.
# Re-applied only past this much change to avoid a TM call every frame.
ADVISORY_SPEED_MIN_DELTA_PCT = 5.0
# Never crawl: below this the demo stops being a driving demo. The advisory
# bottoms out here however much is occluded.
ADVISORY_SPEED_FLOOR_PCT = -60.0

# Cruising speed while the crossing watch is armed, i.e. while a pedestrian --
# visible or still hidden -- could be in the path before the ego arrives.
# Applied through the Traffic Manager, NOT the brake pedal: TM keeps driving,
# just slower, so there is no control handover and no way for a permanently
# armed alert to bring the car to a standstill (which is exactly what an
# earlier brake-pedal version of this did).
#
# This is the answer to braking late rather than a bigger emergency radius.
# Measured live: the raw-proximity layer fired at 3.9 m with the ego at
# 27 km/h (7.5 m/s), which needs ~4 m to stop plus a tick of latency -- not
# recoverable. The layer could not have fired sooner: these walkers start from
# REST at the occluder's front corner, so there is no motion to extrapolate
# until the ego is already alongside. At 14 km/h the same stop needs ~1.1 m
# and 3.9 m is comfortable. The speed is the variable that was wrong, not the
# detection distance.
WATCH_SPEED_MS = 4.0

# The ego's own baseline speed for this demo, as a TM percentage difference
# from the posted limit. `urban_world.EGO_SPEED_DIFF_PCT` is -15 (15% ABOVE
# the limit), tuned for dataset collection where a slow ego wastes ticks and
# collects nothing -- see `_tune_ego_autopilot`. That tuning is wrong for a
# safety demo: it spends the margin the braking layers need. Positive here
# means below the limit.
DEMO_EGO_SPEED_DIFF_PCT = 10.0

# How far ahead along the current lane the manual-control steering aims,
# and the largest steer value it will ever command.
STEER_LOOKAHEAD_M = 6.0
MAX_MANUAL_STEER = 0.6


def _lane_follow_steer(carla_map, ego, lookahead_m: float = STEER_LOOKAHEAD_M,
                         max_steer: float = MAX_MANUAL_STEER) -> float:
    """Steering toward a point on the current lane's centreline a short
    distance ahead, for use while manual control (WARN/BRAKE) has taken the
    wheel from the Traffic Manager.

    `steer=0.0` was tried first and is wrong on anything but a straight
    road: it drives the ego dead straight through whatever curve it happens
    to be on. That was fine as an edge case when only BRAKE (0.75) used
    manual control, briefly and rarely -- it stopped being fine once WARN
    (0.50) also does, for longer and more often, and directly caused a real
    collision live (the ego ran off the road during a 60-frame WARN episode
    that spanned a bend). This does not need to be a sophisticated
    controller -- CARLA's own waypoint graph already defines where the lane
    goes; simple proportional steering toward a point on it is enough to
    track a normal street curve at the reduced speeds WARN/BRAKE produce.
    """
    wp = carla_map.get_waypoint(ego.get_location())
    if wp is None:
        return 0.0
    ahead = wp.next(lookahead_m)
    if not ahead:
        return 0.0
    target = ahead[0].transform.location
    t = ego.get_transform()
    fwd, right = t.get_forward_vector(), t.get_right_vector()
    to_target = target - t.location
    lateral = to_target.x * right.x + to_target.y * right.y
    forward = to_target.x * fwd.x + to_target.y * fwd.y
    if forward <= 0.1:
        return 0.0
    return max(-max_steer, min(max_steer, math.atan2(lateral, forward) * 2.0))


# How close a staged walker may get to the ego's body before it is made to
# stop walking. See _hold_walkers_short_of_ego.
WALKER_YIELD_DISTANCE_M = 2.2


def _hold_walkers_short_of_ego(manager, ego) -> None:
    """Stops a staged walker rather than letting it walk into the ego.

    `WalkerDriver.step` commands a velocity along a fixed crossing direction
    and knows nothing about the ego -- see walker_behavior.py. That is correct
    for dataset collection, where the walker's trajectory is the label and must
    not be quietly altered by whatever the ego happens to do. It is wrong on
    screen: once the ego has correctly stopped for a pedestrian, the pedestrian
    keeps walking and strolls into the stationary bumper, which reads as the
    car having hit them when in fact it stopped in time.

    So the fix is applied here, in the demo, per tick after
    `EncounterManager.step` has set the velocities -- not in `WalkerDriver`,
    which `data_collector.record_episode` shares and whose recorded
    trajectories would change.

    Only ever removes motion, and only within WALKER_YIELD_DISTANCE_M, so the
    crossing itself -- the thing being demonstrated -- plays out untouched
    right up to the point of contact.
    """
    ego_loc = ego.get_location()
    for enc in manager.active:
        walker = enc.walker
        if walker is None or not walker.is_alive:
            continue
        loc = walker.get_location()
        if math.hypot(loc.x - ego_loc.x, loc.y - ego_loc.y) > WALKER_YIELD_DISTANCE_M:
            continue
        set_walker_velocity(walker, carla.Vector3D(1.0, 0.0, 0.0), 0.0)


# --------------------------------------------------------------------------
# Worst-case crossing watch (the "be wary, and remember" layer)
#
# The risk engine scores the EXPECTED case: `risk._hidden_risk` multiplies the
# emergence probability by how well the predicted emergence time lines up with
# the ego's arrival, so a pedestrian who MIGHT dart out scores lower than one
# confidently predicted to. That is the right way to rank hazards and the wrong
# way to decide whether to slow down, and the difference is what put the ego
# into a pedestrian: the hidden-hazard belief peaked at risk 0.56 -- WARN, a
# 0.35 brake -- 1.2 s before contact, because the belief was split four ways
# across motion hypotheses and no single one dominated. Every one of those
# hypotheses was survivable; the combination was scored as "probably fine".
#
# This layer asks a different question, the one a cautious human driver asks:
# not "will they cross?" but "COULD they be in front of me by the time I get
# there, if they move as fast as a person can?" That has no probability in it
# at all -- it is a reachability test against a worst-case speed -- so a belief
# split evenly across four hypotheses arms it just as firmly as a confident one.
#
# Being armed does NOT touch the controls. It is an alert -- shown on the
# dashboard, and remembered -- and the brake comes only from `in_path`, once
# someone actually is in the way. Wiring the armed state to a gentle brake was
# tried and reverted: along a road lined with parked occluders, every one of
# which has a pedestrian beside it permanently within sprinting distance of the
# path, the alert is almost always armed, and the ego braked itself to a
# standstill in the road and stayed there.
#
# It is also deliberately STATEFUL, which `risk.assess` is not (that function is
# shared with the offline evaluation and must stay a pure function of one frame,
# or the reported metrics stop meaning what they say). Once a pedestrian has
# been seen near the path, the alert latches for WATCH_MEMORY_S whether or not
# the belief survives -- a pedestrian stepping back behind the bus is not
# evidence that the danger has passed, and without memory the guard drops the
# instant the tracker loses them, which is precisely when it is needed most.
PEDESTRIAN_SPRINT_MS = 3.0           # worst-case dash, not a walking pace
WATCH_CORRIDOR_HALF_WIDTH_M = 1.75   # the ego's own path half-width
WATCH_HORIZON_S = 4.0                # only be wary about ground we will reach soon
WATCH_ARM_MARGIN_S = 1.0             # arm if they could make it this much before us
WATCH_MEMORY_S = 4.0                 # how long an alert survives without refresh
# A newly-seen pedestrian has to be this much more urgent (seconds of slack)
# than the one currently being reported before the display switches to it.
# Without it the banner picked whichever alert happened to come first out of
# the dict and flickered between six of them frame to frame, which reads as a
# broken readout rather than a system tracking something.
WATCH_SWITCH_MARGIN_S = 0.75


class CrossingWatch:
    """Latching alert for pedestrians who could reach the ego's path.

    Consumes only runtime perception output -- confirmed tracks and particle-
    filter emergence beliefs, both already in the ego frame -- so it stays on
    the correct side of the ground-truth/runtime split. The raw-actor check in
    `_emergency_stop_needed` is a separate, deliberately dumber backstop and is
    not part of this.
    """

    def __init__(self, memory_s: float = WATCH_MEMORY_S):
        self.memory_s = memory_s
        self._expires_at = {}        # key -> sim time the alert lapses
        self._alerts = {}            # key -> (slack_s, reason text)
        self._reported = None        # key currently being displayed -- see reason()
        self.in_path = None          # description of a pedestrian actually in the path

    def _consider(self, key, forward, lateral, ego_speed, now, label):
        """Arm on `key` if a worst-case dash puts them in the path in time."""
        if forward <= 0.0:
            return
        t_ego = forward / max(ego_speed, 0.5)
        if t_ego > WATCH_HORIZON_S:
            return
        gap = max(abs(lateral) - WATCH_CORRIDOR_HALF_WIDTH_M, 0.0)
        t_ped = gap / PEDESTRIAN_SPRINT_MS
        if t_ped > t_ego + WATCH_ARM_MARGIN_S:
            return                   # even flat out they cannot get there in time
        self._expires_at[key] = now + self.memory_s
        # Urgency ordering is time until WE reach them, ascending. Every alert
        # in this set has already passed the reachability test, so what
        # separates them is which one we run into first. Ordering by the
        # reachability slack (t_ped - t_ego) instead is wrong and was caught by
        # the test: it ranks a pedestrian 30 m away who needs only 0.8s to
        # reach the road ABOVE one already 5 m in front of us, because having
        # more spare time reads as more urgent under that metric.
        self._alerts[key] = (t_ego,
                             f"{label} {forward:.0f} m ahead, {gap:.1f} m from my "
                             f"path -- could be in it in {t_ped:.1f}s, "
                             f"I arrive in {t_ego:.1f}s")

    def update(self, now: float, tracks, hidden, ego_speed: float) -> None:
        self.in_path = None
        for t in tracks:
            if getattr(t, "cls", None) != 1:      # pedestrians only
                continue
            forward, lateral = float(t.position[0]), float(t.position[1])
            key = ("track", getattr(t, "id", None))
            self._consider(key, forward, lateral, ego_speed, now, "pedestrian")
            if (0.0 < forward <= WATCH_HORIZON_S * max(ego_speed, 0.5)
                    and abs(lateral) <= WATCH_CORRIDOR_HALF_WIDTH_M):
                if self.in_path is None or forward < self.in_path[0]:
                    self.in_path = (forward, f"pedestrian in path at {forward:.1f} m")
        for tid, emergence in (hidden or []):
            if emergence is None or emergence.location_xy is None:
                continue
            forward, lateral = emergence.location_xy
            self._consider(("hidden", tid), float(forward), float(lateral),
                            ego_speed, now, "hidden pedestrian")
        for key, expiry in list(self._expires_at.items()):
            if now >= expiry:
                del self._expires_at[key]
                self._alerts.pop(key, None)

        # Pick which alert the display speaks for. Sticky: the one already
        # being reported keeps the banner until it lapses or something
        # materially more urgent turns up, so the readout follows one
        # pedestrian through an encounter instead of flickering between all of
        # them. The alert set itself is unaffected -- this is presentation.
        if not self._alerts:
            self._reported = None
            return
        best = min(self._alerts, key=lambda k: self._alerts[k][0])
        if self._reported not in self._alerts:
            self._reported = best
        elif (self._alerts[best][0]
                < self._alerts[self._reported][0] - WATCH_SWITCH_MARGIN_S):
            self._reported = best

    @property
    def armed(self) -> bool:
        return bool(self._expires_at)

    def reason(self) -> str:
        """The reported alert, or empty if nothing is armed."""
        if self._reported is None:
            return ""
        return self._alerts[self._reported][1]


# Unconditional last-line-of-defense braking, independent of the perception
# pipeline entirely. The WARN/BRAKE logic above only reacts to a hazard this
# system's own YOLO+evidential+tracking chain has actually classified and
# scored -- it is silent if that chain misses something, and so is CARLA's
# own Traffic Manager collision avoidance, which was observed live driving
# into both pedestrians and other vehicles in this densely-populated scene.
# A real ADAS stack has exactly this kind of redundant, dumber safety layer
# underneath the smart one for precisely this reason: something has to stop
# the car even when the classifier is wrong, slow, or never ran at all.
EMERGENCY_STOP_RADIUS_M = 8.0
# Half-width of the ego's own path, not an angular cone. An angular cone was
# tried first and is wrong at close range: a bus safely kerb-parked 3.5 m to
# the side and 7 m ahead -- exactly what EncounterManager stages on purpose,
# and exactly the case the ego is supposed to just drive past -- sits at
# roughly 27 degrees, comfortably inside a 35-degree cone, so every staged
# encounter would have triggered a false emergency stop on approach. Lateral
# distance from the ego's own centreline is what actually answers "is this
# in my way", independent of how far ahead it is.
EMERGENCY_STOP_HALF_WIDTH_M = 1.6

# How far ahead an actor's own motion is extrapolated when deciding whether it
# is "in the way". Checking only where an actor is RIGHT NOW is what let the
# ego hit a pedestrian: measured live, the emergency layer's first ever
# trigger for that encounter was "pedestrian at 0.2 m" -- it had never fired
# at 8, 5 or 2 m, because the walker was outside the 1.6 m corridor until the
# single frame it stepped into it, already under the bumper. A pedestrian
# walking at 1.4 m/s crosses the whole corridor in about a second, so
# position alone gives this layer essentially no notice for exactly the
# dart-out case it exists to catch. One second of the actor's own velocity is
# enough to see it coming while still being far too short a horizon to
# manufacture hazards out of people walking along the pavement.
EMERGENCY_LOOKAHEAD_S = 1.0


def _emergency_stop_needed(world, ego, radius_m: float = EMERGENCY_STOP_RADIUS_M,
                             half_width_m: float = EMERGENCY_STOP_HALF_WIDTH_M,
                             lookahead_s: float = EMERGENCY_LOOKAHEAD_S) -> str | None:
    """Checks raw actor positions against the ego's own transform directly --
    no camera, no detector, no tracker in this path at all, so it cannot be
    fooled by anything that fools those. Returns a short description of the
    nearest imminent hazard in, or about to enter, the ego's path, or None.

    Deliberately crude: a real emergency-braking system does exactly this,
    a simple geometric check as the final layer, not a smarter one -- the
    whole point is redundancy against the smart layer failing. The one piece
    of prediction here is `lookahead_s`, and it is the minimum needed for the
    layer to fire before contact rather than at it -- see the constant.
    """
    t = ego.get_transform()
    ego_loc, fwd, right = t.location, t.get_forward_vector(), t.get_right_vector()
    best_forward, best_desc = radius_m, None

    def _consider(actor, label):
        nonlocal best_forward, best_desc
        d = actor.get_location() - ego_loc
        forward = d.x * fwd.x + d.y * fwd.y
        if forward <= 0.2 or forward >= best_forward:
            return
        lateral = d.x * right.x + d.y * right.y
        if abs(lateral) > half_width_m:
            # Off to the side right now -- but heading in? Project the actor's
            # own velocity into the ego frame and re-test. Only an actor
            # actually closing on the corridor is caught; one walking parallel
            # to it, or away, keeps the same lateral distance or grows it and
            # is still ignored, which is what keeps a kerb-parked occluder or
            # a pavement pedestrian from tripping this.
            v = actor.get_velocity()
            lateral_rate = v.x * right.x + v.y * right.y
            if abs(lateral + lateral_rate * lookahead_s) > half_width_m:
                return
            best_forward = forward
            best_desc = f"{label} entering path at {forward:.1f} m"
            return
        best_forward, best_desc = forward, f"{label} at {forward:.1f} m"

    for actor in world.get_actors().filter("walker.pedestrian.*"):
        _consider(actor, "pedestrian")
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id != ego.id:
            _consider(actor, "vehicle")
    return best_desc


# Frames between MiDaS refreshes in the live loop. See classify_grid.
DEPTH_EVERY = 3

# COCO class ids YOLOv8n was pretrained on that matter here.
COCO_PERSON, COCO_CAR, COCO_BUS, COCO_TRUCK = 0, 2, 5, 7
COCO_TO_OURS = {COCO_PERSON: 1, COCO_CAR: 0, COCO_BUS: 0, COCO_TRUCK: 0}  # -> vehicle=0, pedestrian=1
CLASS_NAMES = {0: "vehicle", 1: "pedestrian"}

# Must match models.evidential_classifier: 0=vehicle, 1=pedestrian, 2=background.
CLASS_BACKGROUND = 2

# Smaller than this (either side, in pixels) and a box gets dropped before
# scoring at all -- see the comment at the filter site. Small enough not to
# cut a genuinely distant pedestrian who is merely slender in frame, big
# enough to catch the fire-hydrant/bollard/road-sign class of false positive.
MIN_BOX_SIDE_PX = 14

# Fixed course: number of occluders placed at predetermined waypoints, and
# the distance between them along the road. Replaces EncounterManager's own
# dynamic "stage ~35m ahead as the ego drives" behaviour for this demo --
# see build_fixed_course's docstring for why.
N_COURSE_OCCLUDERS = 6
COURSE_SPACING_M = 45.0
TM_PORT = 8000


def load_models(device: str):
    yolo = YOLO("yolov8n.pt")  # auto-downloads pretrained COCO weights on first run
    evidential = EvidentialDetector(num_classes=3).to(device)
    evidential.load_state_dict(torch.load("models/evidential_detector.pt", map_location=device))
    evidential.eval()
    return yolo, evidential


def score_crop(evidential, device, rgb: np.ndarray, box) -> tuple[float, float]:
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = rgb[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0, 1.0
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        alpha, uncertainty = evidential(tensor)
        probs = EvidentialHead.expected_probability(alpha)
    return probs, uncertainty.item()


def pixel_range_to_azimuth_range(x1: float, x2: float, img_width: int, fov_deg: float,
                                   margin_deg: float = 3.0) -> tuple[float, float]:
    """Approximate a bounding box's horizontal pixel range as a radar
    azimuth window (radians), assuming the radar and camera are both
    forward-facing at ~0 yaw offset (true for `carla_tools.sensors.
    spawn_sensor_rig`'s mounting). This is a documented approximation -- not
    a full extrinsic calibration -- used only to scope per-detection radar
    confidence to roughly the right angular region."""
    fov_rad = math.radians(fov_deg)
    focal = img_width / (2.0 * math.tan(fov_rad / 2.0))
    cx = img_width / 2.0
    az1 = math.atan((x1 - cx) / focal)
    az2 = math.atan((x2 - cx) / focal)
    lo, hi = min(az1, az2), max(az1, az2)
    margin = math.radians(margin_deg)
    return (lo - margin, hi + margin)




GRID_LEGEND = [
    ((208, 59, 59), "RED = Blocked -- something may be hiding here"),
    ((66, 135, 245), "BLUE = Clear -- camera confirms open road"),
    ((12, 163, 12), "GREEN = Seen -- radar detects something here"),
    ((137, 135, 129), "GRAY = Unknown -- no data either way"),
]


def _label_with_background(img, text, org, color, font_scale=0.6, thickness=2):
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x - 2, y - th - baseline - 2), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _banner(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


MIN_GRID_PANEL_WIDTH = 480  # wide enough for the longest legend line at LEGEND_FONT_SCALE


def _render_grid_panel(grid: np.ndarray, panel_size: int) -> np.ndarray:
    vis = labels_to_rgb(grid)
    vis = np.flipud(vis)  # forward-index 0 (near ego) -> bottom of image
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    legend_h = 34 * len(GRID_LEGEND) + 20
    grid_area = panel_size - 34 - legend_h
    cell_px = max(4, grid_area // vis.shape[0])
    grid_big = cv2.resize(vis_bgr, (vis.shape[1] * cell_px, vis.shape[0] * cell_px),
                           interpolation=cv2.INTER_NEAREST)

    panel_w = max(grid_big.shape[1], MIN_GRID_PANEL_WIDTH)
    panel = np.full((panel_size, panel_w, 3), 30, dtype=np.uint8)
    grid_x_offset = (panel_w - grid_big.shape[1]) // 2
    panel[34:34 + grid_big.shape[0], grid_x_offset:grid_x_offset + grid_big.shape[1]] = grid_big

    legend_y = 34 + grid_big.shape[0] + 24
    for color_rgb, text in GRID_LEGEND:
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(panel, (14, legend_y - 14), (34, legend_y + 4), color_bgr, -1)
        cv2.putText(panel, text, (44, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 30

    _banner(panel, "WHAT'S HIDDEN (road map)")
    return panel


RADAR_DISPLAY_RANGE_M = 60.0
RADAR_RING_STEP_M = 20


def _render_radar_panel(radar_pts: np.ndarray, panel_size: int) -> np.ndarray:
    """Raw radar detections plotted in bird's-eye view (distinct from the
    occlusion grid, which is a derived VISIBLE/OCCLUDED/EMPTY/UNKNOWN
    summary) -- ego at the bottom-center, forward is up, distance rings
    every 20m, dots colored by closing speed."""
    panel_w = MIN_GRID_PANEL_WIDTH
    plot_top, caption_h = 34, 34
    plot_h = panel_size - plot_top - caption_h
    panel = np.full((panel_size, panel_w, 3), 15, dtype=np.uint8)

    cx = panel_w // 2
    cy = plot_top + plot_h - 12
    scale = (plot_h - 24) / RADAR_DISPLAY_RANGE_M

    for d in range(RADAR_RING_STEP_M, int(RADAR_DISPLAY_RANGE_M) + 1, RADAR_RING_STEP_M):
        r = int(d * scale)
        cv2.circle(panel, (cx, cy), r, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{d}m", (cx + 4, max(plot_top + 10, cy - r + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(panel, (cx, cy), (cx, plot_top + 8), (55, 55, 55), 1, cv2.LINE_AA)
    cv2.drawMarker(panel, (cx, cy), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)

    if radar_pts.shape[0] > 0:
        velocity, azimuth, depth = radar_pts[:, 0], radar_pts[:, 1], radar_pts[:, 3]
        forward = depth * np.cos(azimuth)
        lateral = -depth * np.sin(azimuth)
        px = (cx + lateral * scale).astype(int)
        py = (cy - forward * scale).astype(int)
        for x, y, v in zip(px, py, velocity):
            if plot_top <= y < plot_top + plot_h and 0 <= x < panel_w:
                if v < -1.0:
                    color = (0, 60, 255)     # closing in on the ego -- red
                elif v > 1.0:
                    color = (255, 160, 0)    # pulling away -- blue
                else:
                    color = (0, 230, 230)    # roughly matching ego's speed -- yellow
                cv2.circle(panel, (x, y), 3, color, -1, cv2.LINE_AA)

    cv2.putText(panel, "red = closing in    yellow = matching speed    blue = pulling away",
                (10, panel_size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    _banner(panel, "RAW RADAR VIEW")
    return panel


def _title_bar(width: int) -> np.ndarray:
    bar = np.zeros((58, width, 3), dtype=np.uint8)
    cv2.putText(bar, "Self-Driving Car That Explains What It Can't See",
                (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, "Left: what the camera sees  |  Middle: hidden vs. clear road  |  Right: raw radar",
                (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
    return bar


def _status_bar(width: int, tick: int, sim_time_s: float, ego_speed_kmh: float,
                 n_detections: int, n_occluded_cells: int, frame_radar_conf: float,
                 n_flagged: int, measured_fps: float, camera_health: float | None = None
                 ) -> np.ndarray:
    bar = np.zeros((36, width, 3), dtype=np.uint8)
    # Camera health is the sensor_health monitor's rolling 0-1 multiplier
    # (perception.sensor_health.HealthReport.camera) -- what actually drives
    # how much weight the fused decision gives the camera, not a per-box
    # detection confidence. Shown beside Radar strength on purpose: fog
    # collapses one and leaves the other alone, and that pairing is the
    # whole point of the fog toggle ('f').
    cam_txt = (f"{camera_health * 100:3.0f}%" if camera_health is not None else "--")
    cam_colour = (0, 255, 120)
    if camera_health is not None and camera_health < 0.6:
        cam_colour = (0, 80, 255)          # flag degraded health in orange/red
    text = (f"Time: {sim_time_s:5.0f}s   Speed: {ego_speed_kmh:4.0f} km/h   "
            f"Objects seen: {n_detections}   Not sure about: {n_flagged}   "
            f"Radar strength: {frame_radar_conf * 100:3.0f}%   Hidden zones: {n_occluded_cells}/400   "
            f"(engine speed: {measured_fps:.1f} frames/sec, tick {tick})")
    cv2.putText(bar, text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 1, cv2.LINE_AA)
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cam_label = f"   Camera strength: {cam_txt}"
    cv2.putText(bar, cam_label, (16 + text_w, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                cam_colour, 1, cv2.LINE_AA)
    return bar


# Distance the probe walks forward between placement attempts, vs. the
# minimum distance required between two successful placements. These are
# deliberately different (8m vs COURSE_SPACING_M=45m): the kerb is not
# uniformly clear along the road, so retrying every 8m gives a locally
# blocked spot several nearby chances, while still keeping successful
# occluders properly spread out rather than clustered wherever the first
# few attempts happened to land.
COURSE_PROBE_STEP_M = 8.0


def build_fixed_course(world, manager: EncounterManager, carla_map,
                        start_loc: carla.Location,
                        n_occluders: int = N_COURSE_OCCLUDERS,
                        spacing_m: float = COURSE_SPACING_M) -> int:
    """Places up to `n_occluders` occluder+pedestrian pairs, each at least
    `spacing_m` apart, along the road ahead of `start_loc`, once, before the
    drive starts -- replacing EncounterManager's own dynamic "stage one
    ~35m ahead as the ego drives" behaviour, which produced inconsistent,
    hard-to-predict encounter timing for a demo meant to show a simple,
    repeatable scenario.

    Walks forward in `COURSE_PROBE_STEP_M` steps and attempts a placement at
    every step once `spacing_m` has elapsed since the last success, rather
    than picking `n_occluders` exact waypoints up front and placing one at
    each. That first approach was tried and measured to fail badly: at 5 of
    6 evenly-spaced candidate spots on a real route, every occluder but the
    narrowest (sprinter.mercedes, 0.99m half-width) collided with sidewalk-
    adjacent geometry (buildings, trees, poles) that Town10HD_Opt is not
    uniformly clear of near the kerb -- one spot failed for all four
    blueprints -- so a single failed waypoint was simply lost, at one point
    placing only 2 of 6. This mirrors EncounterManager's own dynamic
    staging, which the same way expects placement to "legitimately fail
    often" and just retries at the next ahead-position (`_stage`'s
    docstring); a fixed course needs the same tolerance, just walked in
    finer steps so one blocked kerb spot costs a short retry, not an entire
    course slot.

    Reuses `EncounterManager.stage_at_waypoint` / `_finish_staging` for the
    actual placement -- the same kerb geometry and shadow-placement logic the
    dynamic path already relies on.

    `manager.max_active` must already be >= `n_occluders`, or
    `EncounterManager.step`'s own `_should_stage` will still try to stage
    additional dynamic encounters once the ego starts moving and gaps open
    up between the fixed ones. The caller is expected to construct `manager`
    with that in mind.

    Needs one `world.tick()` per successful placement attempt -- the real
    bounding box used to re-seat the occluder and site the walker is not
    readable until a tick after spawning (see
    `EncounterManager._finish_staging`) -- so this must run before the main
    per-frame loop starts, not inside it.
    """
    wp = carla_map.get_waypoint(start_loc)
    if wp is None:
        return 0
    since_last_placed = spacing_m   # allow an attempt right from the start
    seen_ids = set()
    # Generous cap on total road length probed -- a real road network can
    # dead-end into junctions repeatedly; give up rather than loop forever.
    max_total_m = spacing_m * n_occluders * 8
    traveled = 0.0
    while len(manager.active) < n_occluders and traveled < max_total_m:
        nxt = wp.next(COURSE_PROBE_STEP_M)
        if not nxt:
            break
        wp = nxt[0]
        if wp.id in seen_ids:
            break
        seen_ids.add(wp.id)
        traveled += COURSE_PROBE_STEP_M
        since_last_placed += COURSE_PROBE_STEP_M
        if wp.is_junction or since_last_placed < spacing_m:
            continue
        if not manager.stage_at_waypoint(wp):
            continue
        world.tick()
        before = len(manager.active)
        manager._finish_staging()
        if len(manager.active) > before:
            since_last_placed = 0.0
    return len(manager.active)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    yolo, evidential = load_models(device)

    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")
    scenarios_cfg = load_yaml("scenarios.yaml")

    print("Connecting to CARLA...")
    client, world = connect(town_cfg, town_override="Town10HD_Opt")
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    rng = random.Random(7)

    print("Spawning ego (autopilot)...")
    ego = uw_spawn_ego_with_autopilot(world, rng, TM_PORT, tm=tm)
    # Overrides the dataset-collection speed tuning applied inside
    # spawn_ego_with_autopilot -- see DEMO_EGO_SPEED_DIFF_PCT.
    tm.vehicle_percentage_speed_difference(ego, DEMO_EGO_SPEED_DIFF_PCT)
    # Fetched once, not per tick, for _lane_follow_steer -- world.get_map()
    # serialises the whole OpenDRIVE map from the server on every call, and
    # EncounterManager's own header documents a 50x slowdown from doing that
    # per tick. get_waypoint() on an already-fetched map is a cheap local
    # lookup, unlike the fetch itself.
    carla_map = world.get_map()

    # This is the actual "hidden pedestrian behind an occluder, triggered to
    # cross as the ego closes" scenario -- staged occluders + paired walkers
    # from this file's own spawn_static_occluders never produced that; they
    # were an unrelated vehicle and a set of aimlessly-wandering pedestrians
    # with no connection to each other.
    #
    # max_active is set to the fixed course size so EncounterManager.step's
    # own dynamic staging (`_should_stage`) never fires once the course is
    # built -- it only stages when len(active) < max_active, and the fixed
    # course fills that immediately. No background traffic either: a
    # deterministic, repeatable demo (6 occluders, spaced out, each paired
    # with one hidden pedestrian) reads far more clearly than one buried in
    # random vehicles and aimlessly-wandering background walkers.
    manager = EncounterManager(world, ego, rng, scenarios_cfg, bev_cfg,
                                max_active=N_COURSE_OCCLUDERS)
    print(f"Building fixed course: {N_COURSE_OCCLUDERS} occluders spaced "
          f"{COURSE_SPACING_M:.0f}m apart...")
    placed = build_fixed_course(world, manager, carla_map, ego.get_location())
    print(f"Course ready: {placed}/{N_COURSE_OCCLUDERS} occluder+pedestrian pairs placed.")

    rig = spawn_sensor_rig(world, ego, bev_cfg)

    WIN = ("Self-Driving Car That Explains What It Can't See  "
           "(press q to quit, f to toggle synthetic fog)")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 2150, 840)

    spectator = world.get_spectator()
    dt = world.get_settings().fixed_delta_seconds or 0.1

    # The full nine-component framework. Same object the offline evaluation
    # replays recorded frames through, so what the dashboard shows and what the
    # reported metrics measure cannot drift apart.
    pipeline = PerceptionPipeline(bev_cfg=bev_cfg, device=device)

    print("Running live -- click the window and press 'q' to stop.")
    step = 0
    fps_counter = deque(maxlen=30)
    cached_disparity = None
    consecutive_failures = 0
    # Whether this pipeline currently holds the wheel instead of the Traffic
    # Manager. The risk engine (component 7) was always designed to name an
    # action a real vehicle could take (perception/risk.py: "the output maps
    # onto something a vehicle could actually do") but nothing in this demo
    # ever executed BRAKE -- it only printed it. This closes that loop.
    manual_braking = False
    manual_hard_brake = False   # True = full stop, False = light release hold
    manual_is_emergency = False   # True = the raw-proximity layer, not the risk score
    clear_frames = 0
    # Last TM speed percentage actually applied, so the occlusion-aware
    # advisory is only pushed to TM when it moves materially.
    applied_speed_pct = DEMO_EGO_SPEED_DIFF_PCT
    watch = CrossingWatch()
    # Press 'f' to toggle synthetic fog live -- see inject_fog import above.
    # `fog()`'s signature takes an rng argument other faults in that module
    # use for randomness; fog itself is deterministic, so any generator does.
    fog_enabled = False
    fog_rng = np.random.default_rng(0)
    try:
        while True:
            try:
                frame = world.tick()
                # Stages/triggers/advances the actual hidden-pedestrian
                # encounters. Never calls world.tick() itself -- see the
                # class docstring -- so it is safe to drive from here right
                # after this tick, exactly like record_episode's on_tick
                # does for dataset collection.
                manager.step(dt)
                _hold_walkers_short_of_ego(manager, ego)
                spectator.set_transform(rig.rgb.get_transform())
                rgb_img = rig.rgb_buf.get(frame)
                radar_data = rig.radar_buf.get(frame)
                rgb = rgb_to_array(rgb_img)
                radar_pts = radar_to_array(radar_data)
                if fog_enabled:
                    # Radar deliberately not passed through -- see the import
                    # comment. Applied here, before YOLO/evidential/MiDaS all
                    # run on `rgb`, so every downstream stage sees the same
                    # degraded image a real fogged camera would produce, not
                    # a cosmetic overlay on an otherwise-clean detection.
                    # inject_fog returns float32 (the offline harness feeds it
                    # straight to numeric scoring); cv2.imshow and YOLO both
                    # expect the uint8 rgb_to_array normally returns, and
                    # cv2 silently treats an unconverted float array as
                    # already in [0,1] and blows the display out white.
                    rgb, _ = inject_fog(rgb, radar_pts, fog_rng)
                    rgb = rgb.astype(np.uint8)

                if step >= 10:  # let traffic settle
                    # Checked before anything else in this block, and does not
                    # depend on any of it succeeding -- see EMERGENCY_STOP_RADIUS_M.
                    emergency = _emergency_stop_needed(world, ego)

                    results = yolo.predict(rgb, conf=0.35, classes=list(COCO_TO_OURS.keys()), verbose=False)[0]
                    cam_panel = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

                    _banner(cam_panel, "WHAT THE CAR SEES (live camera)")
                    if fog_enabled:
                        # Labelled "SYNTHETIC" deliberately -- CARLA 0.10.0's
                        # own weather API does not work on this build (see
                        # CLAUDE.md / docs/RESULTS.md SS3), so this must never
                        # be allowed to read as real simulated weather.
                        _label_with_background(
                            cam_panel, "SYNTHETIC FOG -- camera degraded, radar unaffected",
                            (12, cam_panel.shape[0] - 14), (0, 220, 220), font_scale=0.55)

                    cam_cfg = bev_cfg["camera"]
                    boxes = results.boxes.xyxy.cpu().numpy()

                    # Run the full framework on this frame. YOLO has already been
                    # called above for the panel, so its boxes are handed in rather
                    # than letting the pipeline detect again -- one inference per
                    # frame, not two.
                    #
                    # YOLO's own COCO class (person/car/bus/truck, already
                    # filtered at predict() time above) is used only to decide
                    # which boxes are worth scoring at all -- the evidential
                    # head's own argmax then decides vehicle/pedestrian/
                    # background, not YOLO's forced label. A box the head
                    # believes is background is dropped rather than kept under
                    # YOLO's class at a low, unflagged confidence. See
                    # perception.pipeline.PerceptionPipeline._detect's
                    # docstring for the false positive (a van window read as
                    # a pedestrian) this closes.
                    dets = []
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        # A tiny box (a fire hydrant, a bollard, a road sign at
                        # a distance) is exactly where YOLO's COCO-trained
                        # "person" class is least reliable -- there is barely
                        # enough silhouette to tell a slender static prop from
                        # a distant pedestrian, and the evidential head, on a
                        # near-featureless upsampled crop, does not reliably
                        # reject it either. RESULTS.md already documents boxes
                        # under ~10-20px as "near-unrecognizable even to a
                        # human at that resolution" for vehicles; the same
                        # applies here. Cheaper to filter before the crop is
                        # even scored than to hope the classifier catches it.
                        if min(x2 - x1, y2 - y1) < MIN_BOX_SIDE_PX:
                            continue
                        probs, uncertainty = score_crop(evidential, device, rgb, box)
                        best = int(probs[0].argmax())
                        if best == CLASS_BACKGROUND:
                            continue
                        dets.append((tuple(float(v) for v in box), best,
                                      float(probs[0, best]), uncertainty))

                    # MiDaS is the single most expensive call in this loop, and
                    # depth changes far more slowly than the radar returns do. Refresh
                    # it every DEPTH_EVERY frames and keep folding in fresh radar
                    # every frame -- at city speed the scene shifts well under one 2 m
                    # grid cell between refreshes.
                    if step % DEPTH_EVERY == 0 or cached_disparity is None:
                        cached_disparity = normalize_disparity(estimate_disparity(rgb))
                    grid = classify_grid(rgb, radar_pts, bev_cfg,
                                          norm_disparity=cached_disparity)
                    result = pipeline.process(
                        rgb=rgb, radar_pts=radar_pts, ego_speed=ego.get_velocity().length(),
                        dt=dt, occlusion_grid=grid, detections=dets)

                    risk_action = result.risk.action if result.risk is not None else NONE_ACTION
                    # Worst-case reachability over the same perception output
                    # the risk engine scored, kept as a latching alert. See
                    # CrossingWatch -- this is what notices "they could be in
                    # front of me before I get there" while the expected-case
                    # score still reads as survivable.
                    watch.update(step * dt, result.tracks, result.hidden,
                                  ego.get_velocity().length())
                    if emergency is not None:
                        # Overrides everything above, including a BRAKE the
                        # risk score itself might not have reached -- that is
                        # the point of a layer that owes the classifier
                        # nothing. Same hysteresis on release as BRAKE/WARN.
                        clear_frames = 0
                        if not manual_braking or not manual_hard_brake:
                            manual_braking = True
                            ego.set_autopilot(False, TM_PORT)
                            print(f"[frame {step}] EMERGENCY STOP -- {emergency} "
                                  f"(ego {ego.get_velocity().length() * 3.6:.0f} km/h)", flush=True)
                        manual_hard_brake = True
                        manual_is_emergency = True
                        ego.apply_control(carla.VehicleControl(
                            throttle=0.0, brake=1.0, hand_brake=False,
                            steer=_lane_follow_steer(carla_map, ego)))
                    elif risk_action >= BRAKE or watch.in_path is not None:
                        # A pedestrian actually in the path is a stop, full
                        # stop -- it does not have to also win an argument with
                        # the risk score first. This is the committed end of the
                        # watch: it spends the approach remembering that someone
                        # could step out, and brakes once they have.
                        #
                        # There is deliberately no intermediate "ease off the
                        # throttle" tier between this and normal driving. One
                        # was tried and removed: braking gently on a *maybe*
                        # means braking almost continuously along a road lined
                        # with parked occluders, each with a pedestrian beside
                        # it permanently within sprinting distance of the path,
                        # and the ego simply stopped in the road and stayed
                        # there. The alert still exists and is still shown --
                        # it is just not wired to the brake pedal until the
                        # hazard is real.
                        clear_frames = 0
                        why = (f"risk={result.risk.score:.2f} ({result.risk.dominant})"
                               if risk_action >= BRAKE else watch.in_path[1])
                        if not manual_braking or not manual_hard_brake:
                            manual_braking = True
                            ego.set_autopilot(False, TM_PORT)
                            print(f"[frame {step}] BRAKING -- {why} "
                                  f"(ego {ego.get_velocity().length() * 3.6:.0f} km/h)", flush=True)
                        manual_hard_brake = True
                        manual_is_emergency = False
                        ego.apply_control(carla.VehicleControl(
                            throttle=0.0, brake=1.0, hand_brake=False,
                            steer=_lane_follow_steer(carla_map, ego)))
                    elif manual_braking:
                        # Hysteresis: only hand the wheel back after the hazard
                        # has stayed clear for a stretch, not the instant it
                        # dips, or a borderline case would flap the vehicle
                        # between manual control and autopilot every frame.
                        clear_frames += 1
                        if clear_frames >= BRAKE_RELEASE_FRAMES:
                            manual_braking = False
                            clear_frames = 0
                            ego.set_autopilot(True, TM_PORT)
                            print(f"[frame {step}] resuming autopilot", flush=True)
                        else:
                            # Still releasing: keep a light hold rather than
                            # coasting on stale control while it is borderline.
                            manual_hard_brake = False
                            manual_is_emergency = False
                            ego.apply_control(carla.VehicleControl(
                                throttle=0.0, brake=RELEASE_BRAKE_AMOUNT,
                                hand_brake=False,
                                steer=_lane_follow_steer(carla_map, ego)))
                    else:
                        # Normal driving, with the Traffic Manager at the wheel.
                        # Two proactive speed limits apply here, both through
                        # TM's own speed setting rather than the brake pedal, so
                        # neither can cause a control handover or stop the car:
                        #   - cruise no faster than the visible distance allows
                        #     (the occlusion burden advisory), and
                        #   - slow to WATCH_SPEED_MS while a pedestrian could
                        #     reach the path before we arrive.
                        # The second is what makes the difference between the
                        # emergency layer having 4 m to stop in and having
                        # enough. Whichever is slower wins.
                        limit_ms = max(ego.get_speed_limit() / 3.6, 1.0)
                        advisory = (result.risk.advisory_speed_ms
                                    if result.risk is not None else None)
                        if watch.armed:
                            advisory = min(advisory or WATCH_SPEED_MS, WATCH_SPEED_MS)
                        target_pct = DEMO_EGO_SPEED_DIFF_PCT
                        if advisory is not None:
                            target_pct = max((1.0 - advisory / limit_ms) * 100.0,
                                             ADVISORY_SPEED_FLOOR_PCT)
                            target_pct = max(target_pct, DEMO_EGO_SPEED_DIFF_PCT)
                        if abs(target_pct - applied_speed_pct) >= ADVISORY_SPEED_MIN_DELTA_PCT:
                            tm.vehicle_percentage_speed_difference(ego, target_pct)
                            if advisory is not None and target_pct > applied_speed_pct:
                                print(f"[frame {step}] SLOWING to {advisory * 3.6:.0f} km/h -- "
                                      + ("pedestrian could enter my path"
                                         if watch.armed else "road ahead is hidden"),
                                      flush=True)
                            applied_speed_pct = target_pct

                    n_flagged = 0
                    for obj in result.objects:
                        flagged = uncertainty_flag(obj.uncertainty, DEFAULT_UNCERTAINTY_THRESHOLD)
                        n_flagged += int(flagged)

                        x1, y1, x2, y2 = [int(v) for v in obj.box_px]
                        color = (0, 200, 0) if obj.cls == 0 else (0, 0, 220)
                        cv2.rectangle(cam_panel, (x1, y1), (x2, y2), color, 4 if flagged else 3)
                        if flagged:
                            cv2.rectangle(cam_panel, (x1, y1), (x2, y2), (0, 165, 255), 1)

                        class_label = "Vehicle" if obj.cls == 0 else "Pedestrian"
                        label = (f"{class_label}   Camera:{obj.confidence * 100:.0f}%   "
                                 f"Radar:{obj.radar_confidence * 100:.0f}%"
                                 + ("  <- NOT SURE" if flagged else ""))
                        _label_with_background(cam_panel, label, (x1, max(50, y1 - 12)), color)

                        # Crossing probability, only where it means something: a
                        # confirmed pedestrian track. Printing it for a parked car
                        # would be noise.
                        if obj.p_cross is not None and obj.cls == 1:
                            risky = obj.p_cross_cautious >= 0.5
                            _label_with_background(
                                cam_panel, f"May step out: {obj.p_cross_cautious * 100:.0f}%",
                                (x1, min(cam_panel.shape[0] - 8, y2 + 22)),
                                (0, 80, 255) if risky else (200, 200, 200))

                    # Hidden hazards -- tracked while completely invisible. This is
                    # the part no detector-driven display can show.
                    for _tid, em in result.hidden:
                        if em.will_emerge and em.location_xy is not None:
                            _label_with_background(
                                cam_panel,
                                f"HIDDEN: someone may step out in {em.time_to_emerge_s:.1f}s "
                                f"({em.probability * 100:.0f}%)",
                                (20, 92), (0, 165, 255), font_scale=0.7)
                    grid_panel = _render_grid_panel(grid, panel_size=cam_panel.shape[0])
                    radar_panel = _render_radar_panel(radar_pts, panel_size=cam_panel.shape[0])

                    total_width = cam_panel.shape[1] + grid_panel.shape[1] + radar_panel.shape[1] + 6  # +2 3px dividers
                    title = _title_bar(total_width)
                    ego_speed_kmh = 3.6 * (ego.get_velocity().length())
                    fps_counter.append(time.time())
                    measured_fps = (len(fps_counter) - 1) / max(fps_counter[-1] - fps_counter[0], 1e-6) if len(fps_counter) > 1 else 0.0
                    status = _status_bar(title.shape[1], step, step * dt, ego_speed_kmh,
                                          len(boxes), int((grid == OCCLUDED).sum()),
                                          radar_confidence(radar_pts), n_flagged, measured_fps,
                                          camera_health=(result.health.camera
                                                          if result.health is not None else None))

                    # Risk banner. Drawn only from WARN upward, so the display stays
                    # quiet when nothing is happening and a coloured bar actually
                    # means something when it appears.
                    if result.risk is not None and result.risk.action >= 2:
                        top = result.risk.factors and max(result.risk.factors,
                                                           key=lambda f: f.score)
                        bar_colour = (0, 0, 200) if result.risk.action >= 3 else (0, 140, 230)
                        cv2.rectangle(cam_panel, (0, 0), (cam_panel.shape[1], 34), bar_colour, -1)
                        _label_with_background(
                            cam_panel,
                            f"{result.risk.action_name}  risk {result.risk.score:.2f}"
                            + (f"  --  {top.detail}" if top else ""),
                            (12, 24), (255, 255, 255), font_scale=0.62)

                    # Unmistakable, separate from the informational risk banner
                    # above: this one only appears while the pipeline actually
                    # holds the wheel, including the few release frames after
                    # risk has already dipped (still braking, not yet handed
                    # back -- see BRAKE_RELEASE_FRAMES).
                    if manual_braking:
                        if manual_is_emergency:
                            bar_colour, text = (128, 0, 255), "EMERGENCY STOP -- RAW PROXIMITY"
                        elif manual_hard_brake:
                            bar_colour, text = (0, 0, 255), "AUTOPILOT OVERRIDDEN -- BRAKING"
                        else:
                            bar_colour, text = (0, 120, 255), "AUTOPILOT OVERRIDDEN -- RELEASING"
                        cv2.rectangle(cam_panel, (0, 40), (cam_panel.shape[1], 82), bar_colour, -1)
                        cv2.putText(cam_panel, text, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                    if watch.armed:
                        # The memory, made visible: this stays up while the alert
                        # is latched, including after the pedestrian has gone back
                        # out of sight, which is the whole reason it exists.
                        _label_with_background(
                            cam_panel, f"WATCHING: {watch.reason()}",
                            (12, min(cam_panel.shape[0] - 40, 110)),
                            (0, 200, 255), font_scale=0.55, thickness=1)

                    divider = np.full((cam_panel.shape[0], 3, 3), 90, dtype=np.uint8)
                    combined = np.vstack([title, np.hstack([cam_panel, divider, grid_panel, divider, radar_panel]), status])
                    cv2.imshow(WIN, combined)
                    if step % 30 == 0:
                        print(f"[frame {step}] detections={len(boxes)} flagged={n_flagged} "
                              f"radar_conf={radar_confidence(radar_pts):.2f} "
                              f"occluded_cells={int((grid == OCCLUDED).sum())}/400 fps={measured_fps:.1f} "
                              f"| tracks={len(result.tracks)} hidden={len(result.hidden)} "
                              f"risk={result.risk.score:.2f} {result.risk.action_name} "
                              f"| health {result.health}", flush=True)
                        import os
                        if os.environ.get("LIVE_DEMO_SNAPSHOT_DIR"):
                            cv2.imwrite(f"{os.environ['LIVE_DEMO_SNAPSHOT_DIR']}/snapshot_step{step:04d}.jpg", combined)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("f"):
                        fog_enabled = not fog_enabled
                        print(f"[frame {step}] synthetic fog "
                              f"{'ON' if fog_enabled else 'OFF'}", flush=True)

                step += 1
            except (queue.Empty, RuntimeError) as exc:
                # A dropped sensor frame or an actor CARLA's Traffic Manager
                # destroyed mid-tick used to crash the whole demo -- the entire
                # ~140-line tick body had no exception handling at all. Skip
                # the bad tick and keep going; only give up if failures are
                # sustained, which means the server died rather than one bad tick.
                consecutive_failures += 1
                print(f"  tick {step} failed: {exc} "
                      f"({consecutive_failures}/{MAX_TICK_FAILURES} consecutive)", flush=True)
                step += 1
                if consecutive_failures >= MAX_TICK_FAILURES:
                    print("  too many consecutive tick failures -- stopping", flush=True)
                    break
                continue
            consecutive_failures = 0
    finally:
        cv2.destroyAllWindows()
        rig.destroy()
        manager.destroy_all()
        ego.destroy()
        disconnect(client, world)
        print("Cleaned up.")


if __name__ == "__main__":
    main()
