"""Shared per-episode recording loop used by every scenario script and by
scripts/collect_dataset.py.

Each frame saves:
  images/<name>.jpg   -- RGB frame
  labels/<name>.txt   -- YOLO-format boxes (class cx cy w h, normalized).
                         **Camera-observable actors only**, i.e. tier != OCCLUDED.
  meta/<name>.npz     -- everything else: BEV grids, radar, ego state, and the
                         complete per-object record including fully-occluded
                         actors, their continuous visibility, and their true
                         position/velocity in the ego frame.

Why the labels file and the meta record differ
----------------------------------------------
`labels/` has to stay plain 5-column YOLO so ultralytics can train on it
unmodified, and training a detector on actors that contribute zero pixels is
actively harmful. But an amodal box for a *fully hidden* actor is the single
most valuable ground truth a simulator can provide -- it is the supervision
signal for "predict where the thing you cannot see actually is", and no real
dataset can supply it. So it goes to the npz instead of being thrown away.

This is a correction of earlier behaviour, and worth stating plainly because
it invalidated a chunk of the previous dataset: this loop used to drop any box
scoring `visibility < 0.15` outright. The fully-occluded pedestrian -- the
subject of the entire project -- was therefore deleted from every label file
before it ever reached disk, which is why the evidential head's uncertainty
calibration came out flat (see docs/RESULTS.md). Nothing is dropped now; the
tier decides which file a box lands in.

Boxes are emitted for every live vehicle/walker actor, not just the scenario's
"reveal" actor, so the detector also sees the occluder itself and background
traffic.
"""
import math
from pathlib import Path

import carla
import cv2
import numpy as np

from carla_tools.bbox_projection import build_projection_matrix, project_actor_bbox, to_yolo_format
from carla_tools.occlusion_mask import BevProjector
from carla_tools.occlusion_tiers import OCCLUDED, tier_for
from carla_tools.sensors import (
    depth_to_meters,
    radar_to_array,
    rgb_to_array,
    semseg_to_labels,
    spawn_sensor_rig,
)
from carla_tools.true_occupancy import compute_true_occupancy_grid

CLASS_VEHICLE = 0
CLASS_PEDESTRIAN = 1
WARMUP_STEPS = 5


def _class_id_for(actor: carla.Actor) -> int | None:
    if actor.type_id.startswith("vehicle"):
        return CLASS_VEHICLE
    if actor.type_id.startswith("walker.pedestrian"):
        return CLASS_PEDESTRIAN
    return None


def _to_ego_frame(x: float, y: float, ego_x: float, ego_y: float, ego_yaw: float):
    """World (x, y) -> ego-local (forward, lateral).

    Same convention as `occlusion_mask.BevProjector` and
    `true_occupancy.compute_true_occupancy_grid`: forward is ahead of the ego,
    lateral is **positive to the left**. Every consumer of this dataset assumes
    it, so it must not be redefined locally anywhere.
    """
    dx, dy = x - ego_x, y - ego_y
    cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)
    forward = dx * cos_y + dy * sin_y
    # NEGATED deliberately. `-dx*sin + dy*cos` is the projection onto CARLA's
    # own get_right_vector(), so without the sign flip this returns the RIGHT
    # component while the project's convention -- BevProjector, occlusion_grid,
    # geometry, tracking, intent -- defines +lateral as LEFT.
    #
    # Verified against the camera rather than by algebra: a probe placed 6 m
    # along CARLA's right vector appears at image column 541 of 800 (right
    # half), the unnegated formula reported lateral +6.0, and BevProjector maps
    # (forward 18, lateral +6) to pixel x=219 (left half). Same object, opposite
    # sides. Every recorded obj_xy_ego/obj_vel_ego was mirrored against the
    # runtime frame, which is why the tracking ablation showed a fused filter
    # doing worse on lateral velocity than a sensor that cannot measure it.
    lateral = -(-dx * sin_y + dy * cos_y)
    return forward, lateral


def _rotate_to_ego(vx: float, vy: float, ego_yaw: float):
    """World velocity -> ego-local (forward, lateral) velocity. Rotation only:
    this is the actor's velocity over ground, not relative to a moving ego.
    Keeping it absolute means the recorded truth stays valid regardless of what
    the ego does, and a consumer that wants relative motion can subtract the
    stored ego velocity.

    Same lateral sign convention as `_to_ego_frame`: positive is LEFT.
    """
    cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)
    return vx * cos_y + vy * sin_y, -(-vx * sin_y + vy * cos_y)


def record_episode(out_dir: str, world: carla.World, ego: carla.Actor, bev_cfg: dict,
                    scenario_type: str, episode_id: int, max_steps: int,
                    weather_name: str | None = None, on_tick=None, annotate=None) -> int:
    """`on_tick(step, frame)`, if given, is called once per tick before any
    recording happens -- used to drive scripted actors (walker behaviour state
    machines, Scenario B's cut-in manoeuvre).

    Note it fires during the `WARMUP_STEPS` warm-up ticks too, whose frames are
    discarded so sensors and actors can settle. Anything stateful hooked in
    here must tolerate being stepped for frames that are never recorded.

    `annotate()`, if given, is called once per recorded frame and must return a
    dict of extra keys to merge into that frame's `.npz`. Used by the roaming
    scenario to stamp which staged encounter a frame belongs to, so evaluation
    can slice to the frames where something was happening instead of averaging
    over minutes of empty street. Keys must not collide with the ones written
    below; colliding keys are dropped with a warning rather than silently
    overwriting the core record.
    """
    out_path = Path(out_dir)
    (out_path / "images").mkdir(parents=True, exist_ok=True)
    (out_path / "labels").mkdir(parents=True, exist_ok=True)
    (out_path / "meta").mkdir(parents=True, exist_ok=True)

    rig = spawn_sensor_rig(world, ego, bev_cfg)
    projector = BevProjector(bev_cfg)
    cam_cfg = bev_cfg["camera"]
    K = build_projection_matrix(cam_cfg["width"], cam_cfg["height"], cam_cfg["fov"])

    frame_count = 0
    try:
        for step in range(max_steps):
            frame = world.tick()
            if on_tick is not None:
                on_tick(step, frame)
            rgb_img = rig.rgb_buf.get(frame)
            depth_img = rig.depth_buf.get(frame)
            semseg_img = rig.semseg_buf.get(frame)
            radar_data = rig.radar_buf.get(frame)

            if step < WARMUP_STEPS:
                continue  # let sensors/actors settle

            rgb = rgb_to_array(rgb_img)
            depth_m = depth_to_meters(depth_img)
            sem = semseg_to_labels(semseg_img)
            radar_pts = radar_to_array(radar_data)

            occ_grid, sem_grid = projector.compute_labels(depth_m, sem)
            true_occ_grid = compute_true_occupancy_grid(world, ego, bev_cfg)

            ego_t = ego.get_transform()
            ego_v = ego.get_velocity()
            ego_x, ego_y = ego_t.location.x, ego_t.location.y
            ego_yaw = math.radians(ego_t.rotation.yaw)

            yolo_lines = []
            obj = {k: [] for k in ("actor_id", "cls", "box", "visibility", "truncation",
                                    "tier", "xy", "vel", "is_vru")}
            others = [a for a in world.get_actors().filter("vehicle.*")] + \
                     [a for a in world.get_actors().filter("walker.pedestrian.*")]
            for actor in others:
                if actor.id == ego.id:
                    continue
                class_id = _class_id_for(actor)
                if class_id is None:
                    continue
                bbox = project_actor_bbox(actor, rig.rgb, K, depth_m=depth_m,
                                           width=cam_cfg["width"], height=cam_cfg["height"])
                if bbox is None:
                    continue  # no geometry in front of the camera at all -- nothing to record

                tier = tier_for(bbox["visibility"], bev_cfg)
                # Only camera-observable actors go into the YOLO labels; the
                # fully-occluded ones are still recorded below, amodally.
                if tier != OCCLUDED:
                    yolo_lines.append(
                        to_yolo_format(bbox, cam_cfg["width"], cam_cfg["height"], class_id))

                loc, vel = actor.get_location(), actor.get_velocity()
                obj["actor_id"].append(actor.id)
                obj["cls"].append(class_id)
                obj["box"].append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
                obj["visibility"].append(bbox["visibility"])
                obj["truncation"].append(bbox["truncation"])
                obj["tier"].append(tier)
                obj["xy"].append(_to_ego_frame(loc.x, loc.y, ego_x, ego_y, ego_yaw))
                obj["vel"].append(_rotate_to_ego(vel.x, vel.y, ego_yaw))
                obj["is_vru"].append(class_id == CLASS_PEDESTRIAN)

            weather_tag = weather_name or "clear_day"
            name = f"{scenario_type}_{weather_tag}_ep{episode_id:05d}_f{step:04d}"

            n_obj = len(obj["actor_id"])
            cv2.imwrite(str(out_path / "images" / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            with open(out_path / "labels" / f"{name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

            core = {
                "occ_grid", "sem_grid", "true_occ_grid", "radar_pts", "ego_x", "ego_y",
                "ego_z", "ego_yaw_rad", "ego_vx", "ego_vy", "sim_time", "scenario_type",
                "weather_name", "episode_id", "frame_idx",
            }
            extra = {}
            if annotate is not None:
                for k, v in (annotate() or {}).items():
                    if k in core or k.startswith("obj_"):
                        print(f"  [record_episode] annotate() key {k!r} collides with the "
                              f"core record; dropped.", flush=True)
                        continue
                    extra[k] = v

            np.savez_compressed(
                out_path / "meta" / f"{name}.npz",
                **extra,
                occ_grid=occ_grid,
                sem_grid=sem_grid,
                true_occ_grid=true_occ_grid,
                radar_pts=radar_pts,
                ego_x=ego_x,
                ego_y=ego_y,
                ego_z=ego_t.location.z,
                ego_yaw_rad=ego_yaw,
                ego_vx=ego_v.x,
                ego_vy=ego_v.y,
                # Simulation clock, not wall clock. Velocity differencing over
                # frames needs an exact dt, and the tick counter alone cannot
                # supply one if fixed_delta_seconds is ever changed.
                sim_time=world.get_snapshot().timestamp.elapsed_seconds,
                scenario_type=scenario_type,
                weather_name=weather_tag,
                episode_id=episode_id,
                frame_idx=step,
                # Per-object record, all arrays aligned and of length n_obj.
                # Includes fully-occluded actors, which carry an amodal box and
                # visibility 0.0 -- absent from labels/ by design.
                obj_actor_id=np.array(obj["actor_id"], dtype=np.int64),
                obj_class=np.array(obj["cls"], dtype=np.int64),
                obj_box_px=np.array(obj["box"], dtype=np.float32).reshape(n_obj, 4),
                obj_visibility=np.array(obj["visibility"], dtype=np.float32),
                obj_truncation=np.array(obj["truncation"], dtype=np.float32),
                obj_tier=np.array(obj["tier"], dtype=np.uint8),
                obj_xy_ego=np.array(obj["xy"], dtype=np.float32).reshape(n_obj, 2),
                obj_vel_ego=np.array(obj["vel"], dtype=np.float32).reshape(n_obj, 2),
                obj_is_vru=np.array(obj["is_vru"], dtype=bool),
            )
            frame_count += 1
    finally:
        rig.destroy()

    return frame_count
