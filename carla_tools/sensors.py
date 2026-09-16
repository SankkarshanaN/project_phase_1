"""Sensor rigs: RGB, depth, semantic-segmentation cameras + radar + collision/lane
sensors, plus pedestrian (walker) spawn/control helpers.

Each sensor pushes into a small queue (`SensorBuffer`) written from the CARLA
callback thread and read from the main sync loop after `world.tick()` --
standard pattern for CARLA synchronous mode with `listen()` callbacks.
"""
import queue
from dataclasses import dataclass, field

import carla
import numpy as np

from common.config import load_yaml


class SensorBuffer:
    def __init__(self):
        self._q: "queue.Queue" = queue.Queue()

    def callback(self, data):
        self._q.put(data)

    def get(self, frame: int, timeout: float = 15.0):
        while True:
            data = self._q.get(timeout=timeout)
            if data.frame == frame:
                return data
            # Drop stale frames (can happen right after spawn).
            if data.frame > frame:
                return data


@dataclass
class SensorRig:
    rgb: carla.Actor
    depth: carla.Actor
    semseg: carla.Actor
    radar: carla.Actor
    collision: carla.Actor
    lane_invasion: carla.Actor
    rgb_buf: SensorBuffer = field(default_factory=SensorBuffer)
    depth_buf: SensorBuffer = field(default_factory=SensorBuffer)
    semseg_buf: SensorBuffer = field(default_factory=SensorBuffer)
    radar_buf: SensorBuffer = field(default_factory=SensorBuffer)
    collision_events: list = field(default_factory=list)
    lane_invasion_events: list = field(default_factory=list)

    def destroy(self):
        for actor in (self.rgb, self.depth, self.semseg, self.radar, self.collision, self.lane_invasion):
            if actor is not None and actor.is_alive:
                actor.stop() if hasattr(actor, "stop") else None
                actor.destroy()


def spawn_sensor_rig(world: carla.World, ego: carla.Actor, bev_cfg: dict | None = None) -> SensorRig:
    cfg = bev_cfg or load_yaml("bev.yaml")
    bp_lib = world.get_blueprint_library()

    cam_cfg = cfg["camera"]
    transform = carla.Transform(carla.Location(x=cam_cfg["x"], z=cam_cfg["z"]))

    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(cam_cfg["width"]))
    rgb_bp.set_attribute("image_size_y", str(cam_cfg["height"]))
    rgb_bp.set_attribute("fov", str(cam_cfg["fov"]))
    rgb = world.spawn_actor(rgb_bp, transform, attach_to=ego)

    depth_cfg = cfg["depth_camera"]
    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(depth_cfg["width"]))
    depth_bp.set_attribute("image_size_y", str(depth_cfg["height"]))
    depth_bp.set_attribute("fov", str(depth_cfg["fov"]))
    depth = world.spawn_actor(depth_bp, transform, attach_to=ego)

    semseg_cfg = cfg["semseg_camera"]
    semseg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    semseg_bp.set_attribute("image_size_x", str(semseg_cfg["width"]))
    semseg_bp.set_attribute("image_size_y", str(semseg_cfg["height"]))
    semseg_bp.set_attribute("fov", str(semseg_cfg["fov"]))
    semseg = world.spawn_actor(semseg_bp, transform, attach_to=ego)

    radar_cfg = cfg["radar"]
    radar_bp = bp_lib.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", str(radar_cfg["horizontal_fov"]))
    radar_bp.set_attribute("vertical_fov", str(radar_cfg["vertical_fov"]))
    radar_bp.set_attribute("points_per_second", str(radar_cfg["points_per_second"]))
    radar_bp.set_attribute("range", str(radar_cfg["range"]))
    radar = world.spawn_actor(
        radar_bp, carla.Transform(carla.Location(x=cam_cfg["x"], z=1.0)), attach_to=ego)

    collision_bp = bp_lib.find("sensor.other.collision")
    collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)

    lane_bp = bp_lib.find("sensor.other.lane_invasion")
    lane_invasion = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)

    rig = SensorRig(rgb=rgb, depth=depth, semseg=semseg, radar=radar,
                     collision=collision, lane_invasion=lane_invasion)
    rig.rgb.listen(rig.rgb_buf.callback)
    rig.depth.listen(rig.depth_buf.callback)
    rig.semseg.listen(rig.semseg_buf.callback)
    rig.radar.listen(rig.radar_buf.callback)
    rig.collision.listen(lambda e: rig.collision_events.append(e))
    rig.lane_invasion.listen(lambda e: rig.lane_invasion_events.append(e))
    return rig


def rgb_to_array(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB


def depth_to_meters(image: carla.Image) -> np.ndarray:
    """CARLA depth encoding: normalized [0,1] across BGRA -> meters, per docs."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4).astype(np.float32)
    b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    return normalized * 1000.0  # far plane = 1000m


def semseg_to_labels(image: carla.Image) -> np.ndarray:
    """Returns the CARLA semantic tag per pixel (stored in the red channel)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, 2].copy()  # red channel (BGRA order) holds the tag id


def radar_to_array(radar_data: carla.RadarMeasurement) -> np.ndarray:
    """Returns an (N, 4) array of [velocity_ms, azimuth_rad, altitude_rad, depth_m]
    per radar detection this tick -- this is CARLA's actual RadarDetection
    struct field order (verified empirically: column 0 clusters near ego's
    closing speed, column 3 spans the configured sensor range in meters)."""
    return np.frombuffer(radar_data.raw_data, dtype=np.float32).reshape(-1, 4)


def spawn_walker(world: carla.World, blueprint_name: str, transform: carla.Transform) -> carla.Actor | None:
    """Spawns a pedestrian without an AI controller -- motion is driven
    manually via `set_walker_velocity` for scripted (non-autopilot) scenarios."""
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter(blueprint_name)
    if len(walker_bps) == 0:
        raise RuntimeError(f"No walker blueprint found matching '{blueprint_name}'")
    walker_bp = walker_bps[0]
    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "true")
    return world.try_spawn_actor(walker_bp, transform)


# CARLA 0.10.0 does not move walkers at the speed `WalkerControl.speed` asks
# for. Measured on this build: a walker commanded at 1.0 m/s covers ground at
# 0.0488 m/s, and the ratio is EXACTLY constant -- 0.0488 at commanded speeds
# of 1.0, 1.5, 3.0, 10.0 and 30.0 m/s, and identical across five walker
# blueprints including the child models. So it is a fixed scale factor, not
# saturation or a per-blueprint difference, and it can be compensated exactly.
#
# This matters far more than a cosmetic speed error. Uncompensated, a
# pedestrian scripted to cross at 1.2 m/s actually crawls at 0.06 m/s and
# travels 4 cm in a 65-tick episode -- they never leave the occluder's shadow,
# so every frame is OCCLUDED, no crossing ever happens, and the entire
# crossing-prediction dataset would be unusable while looking superficially
# fine. Re-measure this constant if the CARLA version changes.
WALKER_SPEED_SCALE = 1.0 / 0.0488


def set_walker_velocity(walker: carla.Actor, direction: carla.Vector3D, speed_ms: float) -> None:
    """Drives a walker at `speed_ms` OVER GROUND along `direction`.

    `direction` is normalized here, so callers may pass any non-zero vector.
    The commanded speed is scaled by `WALKER_SPEED_SCALE` to compensate for
    CARLA 0.10.0's walker-control scaling -- see the constant above.
    """
    norm = (direction.x ** 2 + direction.y ** 2 + direction.z ** 2) ** 0.5
    if norm < 1e-6:
        return
    control = carla.WalkerControl()
    control.direction = carla.Vector3D(direction.x / norm, direction.y / norm, direction.z / norm)
    control.speed = speed_ms * WALKER_SPEED_SCALE
    walker.apply_control(control)
