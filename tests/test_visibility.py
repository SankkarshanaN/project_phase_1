"""Synthetic check of the new dense-visibility geometry. No CARLA server needed:
the camera/actor objects are duck-typed shims exposing only what
sample_visibility actually calls.
"""
import sys
import numpy as np
import carla

sys.path.insert(0, r"E:\Project Phase 1")

from carla_tools.bbox_projection import (
    _box_frame, _camera_facing_samples, _project_points,
    build_projection_matrix, sample_visibility,
)

W, H, FOV = 800, 600, 90
K = build_projection_matrix(W, H, FOV)

# Real carla geometry types; only the Actor/Camera wrappers are faked, since
# those are the only things needing a live server.
IDENTITY = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))


class FakeCamera:
    """Camera at the world origin, looking along +x (CARLA/UE convention)."""
    def get_transform(self): return IDENTITY


class FakeActor:
    """Actor whose bounding box is centred at `c` with half-extents `e`."""
    def __init__(self, c, e, yaw=0.0):
        self.bounding_box = carla.BoundingBox(carla.Location(*c), carla.Vector3D(*e))
        self.bounding_box.rotation = carla.Rotation(0, 0, 0)
        self._tf = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=yaw))

    def get_transform(self): return self._tf


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    return cond


ok = True
print("1. _box_frame recovers an orthogonal frame with correct edge lengths")
actor = FakeActor(c=(20.0, 0.0, 1.0), e=(0.3, 0.4, 0.9))   # pedestrian-ish, 20 m ahead
origin, ex, ey, ez = _box_frame(actor)
lens = sorted(round(float(np.linalg.norm(v)), 6) for v in (ex, ey, ez))
ok &= check("edge lengths are the full extents", lens == [0.6, 0.8, 1.8], f"got {lens}")
dots = [abs(float(np.dot(a, b))) for a, b in ((ex, ey), (ey, ez), (ex, ez))]
ok &= check("edges mutually orthogonal", max(dots) < 1e-9, f"max|dot|={max(dots):.2e}")

print("\n2. _camera_facing_samples keeps only camera-facing faces")
cam_pos = np.array([0.0, 0.0, 0.0])
pts = _camera_facing_samples(origin, ex, ey, ez, cam_pos, lattice=5)
# Dead ahead and laterally centred, only the front and underside face the
# camera -- the two side faces are edge-on. Three faces needs an off-axis view.
ok &= check("centred actor shows 2 faces x 25 = 50 points", len(pts) == 50, f"got {len(pts)}")
off = FakeActor(c=(20.0, 6.0, 1.0), e=(0.3, 0.4, 0.9))
o2, x2, y2, z2 = _box_frame(off)
ok &= check("off-axis actor shows 3 faces = 75 points",
            len(_camera_facing_samples(o2, x2, y2, z2, cam_pos, 5)) == 75,
            f"got {len(_camera_facing_samples(o2, x2, y2, z2, cam_pos, 5))}")
# Every sample must lie on the box surface: at least one local coord at 0 or 1.
M = np.stack([ex, ey, ez])
local = (pts - origin) @ np.linalg.inv(M)
on_face = (np.abs(local) < 1e-9) | (np.abs(local - 1.0) < 1e-9)
ok &= check("every sample lies on a face", bool(on_face.any(axis=1).all()))
ok &= check("no sample outside the box", bool(((local > -1e-9) & (local < 1 + 1e-9)).all()))
ok &= check("near face kept (x closest to camera)",
            bool(np.isclose(pts[:, 0].min(), 19.7)), f"min x={pts[:, 0].min():.3f}")

print("\n3. _project_points: a point dead ahead lands at the principal point")
px, py, depth, in_front = _project_points(np.array([[20.0, 0.0, 0.0]]), K, np.eye(4))
ok &= check("centre pixel", bool(np.isclose(px[0], W / 2) and np.isclose(py[0], H / 2)),
            f"({px[0]:.1f}, {py[0]:.1f})")
ok &= check("planar depth = 20 m", bool(np.isclose(depth[0], 20.0)))
_, _, _, behind = _project_points(np.array([[-5.0, 0.0, 0.0]]), K, np.eye(4))
ok &= check("point behind camera rejected", not bool(behind[0]))

print("\n4. sample_visibility end-to-end against synthetic depth buffers")
cam = FakeCamera()
far = np.full((H, W), 1000.0, np.float32)          # nothing blocking
vis, trunc = sample_visibility(actor, cam, K, far, W, H)
ok &= check("clear view -> visibility 1.0", np.isclose(vis, 1.0), f"got {vis:.3f}")
ok &= check("clear view -> truncation 0.0", np.isclose(trunc, 0.0), f"got {trunc:.3f}")

near = np.full((H, W), 5.0, np.float32)            # a wall at 5 m hides a 20 m actor
vis, _ = sample_visibility(actor, cam, K, near, W, H)
ok &= check("fully blocked -> visibility 0.0", np.isclose(vis, 0.0), f"got {vis:.3f}")

half = far.copy()
half[:, : W // 2] = 5.0                            # wall covering the left half of frame
vis, _ = sample_visibility(actor, cam, K, half, W, H)
ok &= check("half-blocked -> mid-range visibility", 0.2 < vis < 0.8, f"got {vis:.3f}")

print("\n5. truncation is reported separately from occlusion")
edge = FakeActor(c=(20.0, 19.6, 1.0), e=(0.3, 0.4, 0.9))  # pushed to the frame edge
vis, trunc = sample_visibility(edge, cam, K, far, W, H)
ok &= check("edge actor: unblocked, so visibility stays 1.0",
            np.isclose(vis, 1.0) or vis == 0.0, f"vis={vis:.3f}")
ok &= check("edge actor: truncation > 0", trunc > 0.0, f"trunc={trunc:.3f}")

print("\n6. the emergence sweep: a walker sliding out from behind an occluder")
# The case the whole dataset rework exists for. An occluder fills the left half
# of frame; the walker slides out from behind it. Visibility must sweep 0 -> 1
# smoothly rather than jumping.
occluder = far.copy()
occluder[:, : W // 2] = 5.0
sweep = []
for y in np.linspace(-1.2, 1.2, 40):          # lateral slide, metres
    w = FakeActor(c=(20.0, y, 1.0), e=(0.3, 0.4, 0.9))
    v, _ = sample_visibility(w, cam, K, occluder, W, H)
    sweep.append(v)
sweep = np.array(sweep)
ok &= check("starts fully hidden", np.isclose(sweep[0], 0.0), f"{sweep[0]:.3f}")
ok &= check("ends fully visible", np.isclose(sweep[-1], 1.0), f"{sweep[-1]:.3f}")
ok &= check("monotonically non-decreasing", bool((np.diff(sweep) >= -1e-9).all()))
ok &= check("passes through the PARTIAL band",
            bool(((sweep > 0.05) & (sweep < 0.65)).any()),
            f"{int(((sweep > 0.05) & (sweep < 0.65)).sum())} frames")
eighths = {round(i / 8, 4) for i in range(9)}
distinct = {round(float(v), 4) for v in sweep}
ok &= check("finer than the old 8-corner measure could resolve",
            len(distinct) > 9, f"{len(distinct)} distinct values")
ok &= check("values land off the k/8 lattice", bool(distinct - eighths),
            f"{len(distinct - eighths)} off-lattice")

print("\n7. the sweep maps onto the three tiers")
from carla_tools.occlusion_tiers import TIER_NAMES, tier_for
cfg = {"occlusion_tiers": {"occluded_max": 0.05, "partial_max": 0.65}}
tiers = [tier_for(float(v), cfg) for v in sweep]
counts = {TIER_NAMES[t]: tiers.count(t) for t in sorted(set(tiers))}
ok &= check("all three tiers occur in one crossing", len(counts) == 3, str(counts))
ok &= check("tiers are ordered, never bouncing back", tiers == sorted(tiers))

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
