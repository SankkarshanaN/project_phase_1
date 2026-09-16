"""Live integration demo: ego drives autonomously (autopilot) through an open
world populated with background traffic, pedestrians, and a handful of
static parked "occluder" vehicles (same bus/van/truck blueprints as
Scenarios A/C -- free-roam traffic alone made real occlusion events too rare
to actually see), while the full Phase 1-4 pipeline runs continuously on its
camera + radar feed. Labels are plain-language on purpose (percentages,
"Not sure", "Hidden zones") so a non-technical viewer can follow it without
narration; see docs/DASHBOARD.md for the technical field names underneath.

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

Confidence, detections, and (aside from the static occluders) occlusion are
whatever the live traffic actually produces -- not scripted per-frame --
which is the honest thing to show for something meant to demonstrate
general validity rather than one hand-picked vignette.

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
from carla_tools.sensors import radar_to_array, rgb_to_array, spawn_sensor_rig
from common.config import load_yaml
from models.evidential_classifier import CROP_SIZE, EvidentialDetector
from models.evidential_head import DEFAULT_UNCERTAINTY_THRESHOLD, EvidentialHead, uncertainty_flag
from perception.depth_midas import estimate_disparity, normalize_disparity
from perception.occlusion_grid import OCCLUDED, classify_grid, labels_to_rgb
from perception.pipeline import PerceptionPipeline
from perception.radar_confidence import radar_confidence, radar_confidence_in_region

# Frames between MiDaS refreshes in the live loop. See classify_grid.
DEPTH_EVERY = 3

# COCO class ids YOLOv8n was pretrained on that matter here.
COCO_PERSON, COCO_CAR, COCO_BUS, COCO_TRUCK = 0, 2, 5, 7
COCO_TO_OURS = {COCO_PERSON: 1, COCO_CAR: 0, COCO_BUS: 0, COCO_TRUCK: 0}  # -> vehicle=0, pedestrian=1
CLASS_NAMES = {0: "vehicle", 1: "pedestrian"}

N_BACKGROUND_VEHICLES = 15
N_BACKGROUND_WALKERS = 15
N_STATIC_OCCLUDERS = 6
OCCLUDER_BLUEPRINTS = ["vehicle.fuso.mitsubishi", "vehicle.sprinter.mercedes", "vehicle.firetruck.actors"]
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


def spawn_ego_with_autopilot(world, tm, rng):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    ego_bp = bp_lib.filter("vehicle.lincoln.mkz")[0]
    for _ in range(20):
        sp = rng.choice(spawn_points)
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego is not None:
            ego.set_autopilot(True, TM_PORT)
            return ego
    raise RuntimeError("Failed to spawn ego after 20 attempts")


def spawn_background_traffic(client, world, tm, rng):
    """Standard CARLA traffic-population pattern (vehicles on autopilot,
    walkers on AI controllers with random destinations) -- same approach as
    CARLA's own PythonAPI/examples/generate_traffic.py.

    Walkers are spawned BEFORE vehicles (not after): spawning both at once
    showed only ~2/15 walkers succeeding, since by the time walkers were
    attempted every vehicle spawn point was already occupied and blocking
    nearby nav-mesh locations. Also over-sample nav locations (3x target)
    since many random draws land somewhere a `try_spawn_actor` still
    rejects (steep terrain, inside geometry, etc.)."""
    bp_lib = world.get_blueprint_library()

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walkers = []
    attempts = 0
    while len(walkers) < N_BACKGROUND_WALKERS and attempts < N_BACKGROUND_WALKERS * 4:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(rng.choice(walker_bps), carla.Transform(loc))
        if w is not None:
            walkers.append(w)
    world.tick()

    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicles = []
    for sp in spawn_points[:N_BACKGROUND_VEHICLES]:
        v = world.try_spawn_actor(rng.choice(vehicle_bps), sp)
        if v is not None:
            v.set_autopilot(True, TM_PORT)
            vehicles.append(v)

    controller_bp = bp_lib.find("controller.ai.walker")
    controllers = []
    for w in walkers:
        c = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=w)
        if c is not None:
            controllers.append(c)
    world.tick()

    for c in controllers:
        c.start()
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            c.go_to_location(dest)
        c.set_max_speed(0.9 + rng.random() * 0.8)

    print(f"Background traffic: {len(vehicles)} vehicles, {len(walkers)} walkers")
    return vehicles, walkers, controllers


def spawn_static_occluders(world, ego, rng):
    """Free-roam autopilot traffic alone made real occlusion events (radar
    or camera-visible objects hidden behind something) rare -- moving
    traffic rarely lines up between the ego and another actor for more than
    an instant. A handful of stationary, oversized parked vehicles (the same
    bus/van/truck blueprints used as occluders in Scenarios A/C) placed
    directly along the ego's starting lane guarantees the first occlusion
    is visible almost immediately, plus a few more scattered around town so
    it keeps happening over a longer drive -- same principle as the staged
    scenarios, just distributed through open-world traffic instead of one
    hand-placed vignette."""
    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    occluders = []

    ego_wp = carla_map.get_waypoint(ego.get_location())
    for dist in (15.0, 30.0, 50.0):
        ahead = ego_wp.next(dist)
        if not ahead:
            continue
        wp = ahead[0]
        right = wp.transform.get_right_vector()
        loc = carla.Location(
            wp.transform.location.x + right.x * 1.5,
            wp.transform.location.y + right.y * 1.5,
            wp.transform.location.z + 0.3,
        )
        bp = bp_lib.filter(rng.choice(OCCLUDER_BLUEPRINTS))
        if not bp:
            continue
        v = world.try_spawn_actor(bp[0], carla.Transform(loc, wp.transform.rotation))
        if v is not None:
            v.set_simulate_physics(False)
            occluders.append(v)

    spawn_points = carla_map.get_spawn_points()
    rng.shuffle(spawn_points)
    for sp in spawn_points:
        if len(occluders) >= N_STATIC_OCCLUDERS:
            break
        bp = bp_lib.filter(rng.choice(OCCLUDER_BLUEPRINTS))
        if not bp:
            continue
        v = world.try_spawn_actor(bp[0], sp)
        if v is not None:
            v.set_simulate_physics(False)
            occluders.append(v)

    print(f"Static occluders (parked buses/vans/trucks): {len(occluders)}")
    return occluders


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
                 n_flagged: int, measured_fps: float) -> np.ndarray:
    bar = np.zeros((36, width, 3), dtype=np.uint8)
    text = (f"Time: {sim_time_s:5.0f}s   Speed: {ego_speed_kmh:4.0f} km/h   "
            f"Objects seen: {n_detections}   Not sure about: {n_flagged}   "
            f"Radar strength: {frame_radar_conf * 100:3.0f}%   Hidden zones: {n_occluded_cells}/400   "
            f"(engine speed: {measured_fps:.1f} frames/sec, tick {tick})")
    cv2.putText(bar, text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 1, cv2.LINE_AA)
    return bar


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    yolo, evidential = load_models(device)

    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")

    print("Connecting to CARLA...")
    client, world = connect(town_cfg, town_override="Town10HD_Opt")
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    rng = random.Random(7)

    print("Spawning ego (autopilot) and background traffic...")
    ego = spawn_ego_with_autopilot(world, tm, rng)
    vehicles, walkers, controllers = spawn_background_traffic(client, world, tm, rng)
    occluders = spawn_static_occluders(world, ego, rng)
    rig = spawn_sensor_rig(world, ego, bev_cfg)

    WIN = "Self-Driving Car That Explains What It Can't See  (press q to quit)"
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
    try:
        while True:
            try:
                frame = world.tick()
                spectator.set_transform(rig.rgb.get_transform())
                rgb_img = rig.rgb_buf.get(frame)
                radar_data = rig.radar_buf.get(frame)
                rgb = rgb_to_array(rgb_img)
                radar_pts = radar_to_array(radar_data)

                if step >= 10:  # let traffic settle
                    results = yolo.predict(rgb, conf=0.35, classes=list(COCO_TO_OURS.keys()), verbose=False)[0]
                    cam_panel = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

                    _banner(cam_panel, "WHAT THE CAR SEES (live camera)")

                    cam_cfg = bev_cfg["camera"]
                    boxes = results.boxes.xyxy.cpu().numpy()
                    coco_classes = results.boxes.cls.cpu().numpy().astype(int)

                    # Run the full framework on this frame. YOLO has already been
                    # called above for the panel, so its boxes are handed in rather
                    # than letting the pipeline detect again -- one inference per
                    # frame, not two.
                    dets = []
                    for box, coco_cls in zip(boxes, coco_classes):
                        our_cls = COCO_TO_OURS.get(coco_cls, 0)
                        probs, uncertainty = score_crop(evidential, device, rgb, box)
                        dets.append((tuple(float(v) for v in box), our_cls,
                                      float(probs[0, our_cls]), uncertainty))

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
                                          radar_confidence(radar_pts), n_flagged, measured_fps)

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

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

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
        for c in controllers:
            c.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in controllers + walkers + vehicles + occluders])
        ego.destroy()
        disconnect(client, world)
        print("Cleaned up.")


if __name__ == "__main__":
    main()
