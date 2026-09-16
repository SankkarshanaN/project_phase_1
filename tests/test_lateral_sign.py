"""Which way is +lateral? Settle it against the camera image, which cannot lie.

Places a probe vehicle a known distance along CARLA's own get_right_vector()
and checks, for that same probe:
  - which side of the RGB image it appears on
  - what `data_collector._to_ego_frame` reports
  - which BevProjector cell it projects to
  - which cell `compute_true_occupancy_grid` marks occupied
"""
import math
import sys

import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

import carla

from carla_tools.client import connect
from carla_tools.data_collector import _to_ego_frame
from carla_tools.occlusion_mask import BevProjector
from carla_tools.sensors import rgb_to_array, spawn_sensor_rig
from carla_tools.true_occupancy import compute_true_occupancy_grid
from common.config import load_yaml
from perception.geometry import ego_xy_to_bev_cell

town, bev = load_yaml("town.yaml"), load_yaml("bev.yaml")
client, world = connect(town)
bl = world.get_blueprint_library()
sp = world.get_map().get_spawn_points()

ego = None
for t in sp[:25]:
    ego = world.try_spawn_actor(bl.filter("vehicle.lincoln.mkz")[0], t)
    if ego:
        break
world.tick()
rig = spawn_sensor_rig(world, ego, bev)
for _ in range(5):
    world.tick()

tf = ego.get_transform()
right = tf.get_right_vector()
fwd = tf.get_forward_vector()
yaw = math.radians(tf.rotation.yaw)
print(f"ego yaw {tf.rotation.yaw:.1f}   forward ({fwd.x:+.2f},{fwd.y:+.2f})   "
      f"right ({right.x:+.2f},{right.y:+.2f})")

# A probe 18 m ahead and 6 m along CARLA's RIGHT vector.
probe_loc = carla.Location(
    tf.location.x + fwd.x * 18.0 + right.x * 6.0,
    tf.location.y + fwd.y * 18.0 + right.y * 6.0,
    tf.location.z + 0.3)
probe = world.try_spawn_actor(bl.filter("vehicle.fuso.mitsubishi")[0],
                               carla.Transform(probe_loc, tf.rotation))
if probe is None:
    print("probe spawn failed")
    rig.destroy(); ego.destroy(); raise SystemExit(1)
world.tick()
for _ in range(4):
    f = world.tick()

rgb = rgb_to_array(rig.rgb_buf.get(f))
cam = bev["camera"]

# 1. Which side of the image? Find the column with the most non-road pixels in
#    the lower half -- the bus is large and dark against pale tarmac.
lower = rgb[cam["height"] // 2:, :, :].mean(axis=2)
col_dark = (lower < lower.mean() - 25).sum(axis=0)
img_side = "RIGHT" if int(np.argmax(col_dark)) > cam["width"] / 2 else "LEFT"
print(f"\nprobe placed 6 m along CARLA right vector, 18 m ahead")
print(f"  1. appears on the {img_side} of the image "
      f"(darkest column x={int(np.argmax(col_dark))} of {cam['width']})")

# 2. What does _to_ego_frame say?
pl = probe.get_location()
f_m, l_m = _to_ego_frame(pl.x, pl.y, tf.location.x, tf.location.y, yaw)
print(f"  2. _to_ego_frame -> forward {f_m:+.1f}  lateral {l_m:+.1f}")

# 3. Where does BevProjector put a cell at that lateral?
proj = BevProjector(bev)
px, py, dist, valid = proj._project()
n = bev["bev_grid"]["size_cells"]
fi, li = ego_xy_to_bev_cell(np.array([f_m]), np.array([l_m]), bev["bev_grid"])
if 0 <= fi[0] < n and 0 <= li[0] < n:
    print(f"  3. that (forward,lateral) -> BEV cell [{fi[0]},{li[0]}], which "
          f"BevProjector projects to pixel x={px[fi[0], li[0]]:.0f} "
          f"({'RIGHT' if px[fi[0], li[0]] > cam['width'] / 2 else 'LEFT'} of image)")

# 4. Which cell does true_occupancy mark?
grid = compute_true_occupancy_grid(world, ego, bev)
occ = np.argwhere(grid > 0)
if len(occ):
    print(f"  4. true_occupancy marks cells at lateral indices "
          f"{sorted(set(occ[:, 1].tolist()))} (centre index is {n // 2})")

print()
if img_side == "RIGHT" and l_m > 0:
    print("  VERDICT: an object on the RIGHT of the image is recorded with POSITIVE")
    print("  lateral, but BevProjector/occlusion_grid define +lateral = LEFT.")
    print("  _to_ego_frame's lateral sign is INVERTED relative to the convention.")
elif img_side == "RIGHT" and l_m < 0:
    print("  VERDICT: right-of-image -> negative lateral. Consistent with "
          "+lateral = LEFT. No bug.")

probe.destroy()
rig.destroy()
ego.destroy()
world.tick()
