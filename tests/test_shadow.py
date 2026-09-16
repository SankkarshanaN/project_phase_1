"""Does the new shadow placement actually hide the walker, where the old
side-offset placement did not? Uses the real sample_visibility measurement
against a synthetic depth buffer built from the occluder's own geometry.
"""
import sys
import numpy as np
import carla

sys.path.insert(0, r"E:\Project Phase 1")

from carla_tools.bbox_projection import build_projection_matrix, sample_visibility, _box_frame, _project_points
from carla_tools.scenario_gen import _shadow_position, _shadow_covers, _box_half_width_along

W, H, FOV = 800, 600, 90
K = build_projection_matrix(W, H, FOV)
CAM = carla.Location(0.0, 0.0, 1.7)          # camera eye point
CAM_TF = carla.Transform(CAM, carla.Rotation(0, 0, 0))


class Obj:
    """Actor stand-in: a box centred at `c` (world) with half-extents `e`."""
    def __init__(self, c, e, yaw=0.0):
        self.bounding_box = carla.BoundingBox(carla.Location(0, 0, 0), carla.Vector3D(*e))
        self.bounding_box.rotation = carla.Rotation(0, 0, 0)
        self._tf = carla.Transform(carla.Location(*c), carla.Rotation(yaw=yaw))

    def get_transform(self): return self._tf
    def get_location(self): return self._tf.location


class Cam:
    def get_transform(self):
        return carla.Transform(CAM, carla.Rotation(0, 0, 0))


def depth_buffer_with(occluder):
    """Rasterises just the occluder into a depth buffer, everything else far."""
    buf = np.full((H, W), 1000.0, np.float32)
    origin, ex, ey, ez = _box_frame(occluder)
    # Dense point cloud over the whole box, splatted with a small footprint --
    # enough to fill the silhouette for this check.
    t = np.linspace(0, 1, 40)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    pts = origin + g[:, :1] * ex + g[:, 1:2] * ey + g[:, 2:3] * ez
    w2c = np.array(CAM_TF.get_inverse_matrix())
    px, py, depth, in_front = _project_points(pts, K, w2c)
    m = in_front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    ix, iy, dd = px[m].astype(int), py[m].astype(int), depth[m]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            jx, jy = np.clip(ix + dx, 0, W - 1), np.clip(iy + dy, 0, H - 1)
            np.minimum.at(buf, (jy, jx), dd)
    return buf


cam = Cam()
# A CARLA bus (vehicle.fuso.mitsubishi) is roughly 6 m long, 2.5 m wide, 3 m
# tall -- half-extents (3.0, 1.25, 1.5). Parked 11 m ahead, 1 m right of lane.
bus = Obj(c=(11.0, 1.0, 1.5), e=(3.0, 1.25, 1.5))
depth = depth_buffer_with(bus)

print("Occluder: 6.0 x 2.5 x 3.0 m bus, 11 m ahead, 1 m right of the lane")
print(f"Camera eye at ({CAM.x}, {CAM.y}, {CAM.z})\n")

print("OLD placement -- faithful: occluder at waypoint + right*jitter,")
print("walker at waypoint + right*3.0, both at the same longitudinal station:")
for jitter in (-1.0, -0.5, 0.0, 0.5, 1.0):
    bus_j = Obj(c=(11.0, jitter, 1.5), e=(3.0, 1.25, 1.5))   # occluder_wp + right*jitter
    d_j = depth_buffer_with(bus_j)
    walker = Obj(c=(11.0, 3.0, 0.9), e=(0.3, 0.35, 0.9))     # occluder_wp + right*3.0
    vis, _ = sample_visibility(walker, cam, K, d_j, W, H)
    print(f"  lateral jitter {jitter:+.1f} m (separation {3.0 - jitter:.1f} m) "
          f"-> visibility {vis:.3f}  "
          f"{'HIDDEN' if vis < 0.05 else 'VISIBLE - occlusion failed'}")

print("\nNEW placement -- walker on the line of sight, past the occluder:")
for clearance in (1.0, 1.5, 2.0, 2.5, 3.0):
    loc, dist = _shadow_position(CAM, bus, clearance)
    walker = Obj(c=(loc.x, loc.y, loc.z + 0.9), e=(0.3, 0.35, 0.9))
    covers = _shadow_covers(CAM, bus, carla.Location(loc.x, loc.y, loc.z))
    vis, _ = sample_visibility(walker, cam, K, depth, W, H)
    print(f"  clearance {clearance:.1f} m -> pos ({loc.x:5.2f}, {loc.y:5.2f}) "
          f"range {dist:5.2f} m  visibility {vis:.3f}  covers={covers}  "
          f"{'HIDDEN' if vis < 0.05 else 'VISIBLE — occlusion failed'}")

print("\nEmergence sweep from the new start position (walker steps left, toward the road):")
loc, _ = _shadow_position(CAM, bus, 1.5)
seen_tiers = []
from carla_tools.occlusion_tiers import TIER_NAMES, tier_for
cfg = {"occlusion_tiers": {"occluded_max": 0.05, "partial_max": 0.65}}
for step in np.arange(0.0, 4.01, 0.25):        # metres walked toward the lane
    walker = Obj(c=(loc.x, loc.y - step, loc.z + 0.9), e=(0.3, 0.35, 0.9))
    vis, _ = sample_visibility(walker, cam, K, depth, W, H)
    t = tier_for(float(vis), cfg)
    seen_tiers.append(t)
    bar = "#" * int(vis * 40)
    print(f"  walked {step:4.2f} m  vis={vis:.3f}  {TIER_NAMES[t]:<9s} |{bar}")

names = [TIER_NAMES[t] for t in seen_tiers]
print(f"\ntiers visited: {' -> '.join(dict.fromkeys(names))}")
ok = len(set(seen_tiers)) == 3 and seen_tiers == sorted(seen_tiers)
print("RESULT:", "PASS - full sweep, monotonic" if ok else "FAIL")
sys.exit(0 if ok else 1)
