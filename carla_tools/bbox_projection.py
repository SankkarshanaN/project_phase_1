"""Project actor 3D bounding boxes into the RGB camera's 2D image plane, for
YOLO-format ground-truth labels.

Standard CARLA pinhole-camera projection (same approach as CARLA's own
`PythonAPI/examples` bounding-box tutorials): build an intrinsic matrix from
image size + FOV, transform each bbox world-space vertex into camera space
via the camera's inverse transform matrix, then project with the intrinsic
matrix.

The 2D box extent is the axis-aligned envelope of the 8 projected bbox
corners, and is **amodal**: it describes where the actor truly is, whether or
not any of it is visible. A fully-hidden pedestrian still gets a geometrically
valid box. Whether that box is worth keeping is the caller's decision, not
this module's.

Occlusion is measured separately, by dense sampling (`sample_visibility`).
Two earlier problems motivated the current implementation:

  - The old measure was the fraction of the **8 bbox corners** passing a depth
    test, so only nine distinct values existed. That is far too coarse to
    grade occlusion: a pedestrian with head and one shoulder showing scored
    identically to one half out from behind a bus, and an actor 60% obscured
    could still score 1.0 if all eight corners happened to land on unoccluded
    pixels.
  - Corners projecting **outside** the image incremented the denominator but
    could never increment the numerator, so a large, close, partly-out-of-frame
    actor was scored as heavily occluded despite nothing blocking it -- and was
    then dropped by the caller's threshold. Truncation is not occlusion; the
    two are now counted separately and returned separately.
"""
import numpy as np
import carla

from common.constants import DEPTH_TOLERANCE_M

# Per-face sampling resolution for `sample_visibility`: an N x N grid on each
# camera-facing box face, so roughly 3 * N**2 = 432 points per actor.
#
# N is not chosen for the *number* of points but for resolution across the
# occlusion boundary. An occluder edge sweeping over the actor can only hide
# whole lattice columns, so N bounds how finely the emergence from behind an
# occluder can be graded -- N distinct steps, no matter how many total samples
# there are. That sweep is the measurement this project cares about most, so
# it is the one that must not be coarse: N=5 resolved a full crossing into
# only 6 visibility levels.
#
# Measured on this machine, the per-actor cost is dominated by the fixed CARLA
# transform-matrix calls rather than the point count -- 177 us at N=5 against
# 214 us at N=12 -- so the extra resolution is nearly free against a pipeline
# already measured at 2-9 FPS.
DEFAULT_LATTICE = 12


def build_projection_matrix(width: int, height: int, fov: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def _project_point(loc: carla.Location, K: np.ndarray, w2c: np.ndarray):
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    point_camera = w2c @ point
    # UE4 sensor axes (x-fwd, y-right, z-up) -> pinhole convention (x-right, y-down, z-fwd)
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    if point_camera[2] <= 1e-3:
        return None, None
    point_img = K @ point_camera
    px = point_img[0] / point_img[2]
    py = point_img[1] / point_img[2]
    return (px, py), point_camera[2]


def _project_points(pts: np.ndarray, K: np.ndarray, w2c: np.ndarray):
    """Vectorised `_project_point` over an (N, 3) array of world points.

    Returns (px, py, depth, in_front). `depth` is camera-space forward Z --
    planar depth, matching what CARLA's depth camera stores, not Euclidean
    range. Entries where `in_front` is False carry meaningless px/py.
    """
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = homo @ w2c.T
    pc = np.stack([cam[:, 1], -cam[:, 2], cam[:, 0]], axis=1)
    depth = pc[:, 2]
    in_front = depth > 1e-3
    safe = np.where(in_front, depth, 1.0)  # avoid divide-by-zero on rejected points
    px = K[0, 0] * pc[:, 0] / safe + K[0, 2]
    py = K[1, 1] * pc[:, 1] / safe + K[1, 2]
    return px, py, depth, in_front


def _box_frame(actor: carla.Actor):
    """Returns (origin_corner, ex, ey, ez) spanning the actor's box in world
    coords, where the edge vectors are full extents, not half-extents.

    Built from the bounding box's own rotation and extent rather than from its
    8 projected corners. Two tempting corner-based shortcuts are both wrong on
    elongated boxes, and a pedestrian is exactly that -- tall and thin:

      - "the three nearest corners are the edge-neighbours" fails because a
        face diagonal across the two short axes is shorter than the long edge
        (for a 0.6 x 0.8 x 1.8 m walker: diagonal 1.0 m < edge 1.8 m).
      - ranking corners by dot product about the centre fails for the same
        reason -- the ordering depends on which extent dominates.

    Composing the transforms has neither failure mode and does not depend on
    CARLA's undocumented `get_world_vertices` corner ordering.
    """
    bb = actor.bounding_box
    box_to_world = (np.array(actor.get_transform().get_matrix())
                    @ np.array(carla.Transform(bb.location, bb.rotation).get_matrix()))
    axes = box_to_world[:3, :3]   # columns are the box's unit axes in world space
    centre = box_to_world[:3, 3]
    ex = 2.0 * bb.extent.x * axes[:, 0]
    ey = 2.0 * bb.extent.y * axes[:, 1]
    ez = 2.0 * bb.extent.z * axes[:, 2]
    return centre - 0.5 * (ex + ey + ez), ex, ey, ez


def _camera_facing_samples(origin, ex, ey, ez, cam_pos: np.ndarray, lattice: int):
    """Samples the box's camera-facing surface, returning an (N, 3) array.

    Surface, not volume, and camera-facing only -- both matter:

      - Sampling the box *interior* would be meaningless. Every interior point
        sits behind the actor's own front surface, so the depth buffer reports
        something nearer and the point reads as "occluded" by the actor
        itself. On a long vehicle that drives visibility toward zero even in
        clear view.
      - Sampling *all six* faces has the same defect at half strength: the
        rear faces are always self-occluded, capping a fully-visible actor
        near 0.5 instead of 1.0.

    Keeping only faces whose outward normal points toward the camera makes
    visibility mean "the fraction of the silhouette a camera could see if
    nothing were in the way", so an unobstructed actor scores ~1.0 and a
    fully-hidden one scores 0.0.
    """
    t = (np.arange(lattice) + 0.5) / lattice  # cell centres; never samples an edge exactly
    uu, vv = np.meshgrid(t, t, indexing="ij")
    uu = uu.ravel()[:, None]
    vv = vv.ravel()[:, None]

    chunks = []
    for normal_axis, a1, a2 in ((ex, ey, ez), (ey, ez, ex), (ez, ex, ey)):
        face = origin + uu * a1 + vv * a2
        for offset, outward in ((0.0, -normal_axis), (1.0, normal_axis)):
            pts = face + offset * normal_axis
            # Outward normal vs. the view direction to this face's centre.
            if np.dot(outward, pts.mean(axis=0) - cam_pos) < 0.0:
                chunks.append(pts)

    if not chunks:
        return np.empty((0, 3))
    return np.concatenate(chunks, axis=0)


def sample_visibility(actor: carla.Actor, camera: carla.Actor, K: np.ndarray,
                       depth_m: np.ndarray, img_w: int, img_h: int,
                       lattice: int = DEFAULT_LATTICE) -> tuple[float, float]:
    """Returns (visibility, truncation), both in [0, 1].

    `visibility` is the fraction of camera-facing surface samples that land
    **inside the image** and are not blocked by a nearer surface. `truncation`
    is the fraction of in-front samples that fall **outside** the image.

    These are deliberately separate numbers. Conflating them -- as the earlier
    8-corner implementation did, by counting out-of-frame corners in the
    denominator of visibility -- makes an unobstructed actor at the edge of
    frame look heavily occluded, which then trips any visibility threshold
    downstream. An actor can be fully visible and heavily truncated at once.
    """
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    cam_loc = camera.get_transform().location
    cam_pos = np.array([cam_loc.x, cam_loc.y, cam_loc.z])

    origin, ex, ey, ez = _box_frame(actor)
    pts = _camera_facing_samples(origin, ex, ey, ez, cam_pos, lattice)
    if len(pts) == 0:
        return 0.0, 0.0

    px, py, depth, in_front = _project_points(pts, K, w2c)
    if not in_front.any():
        return 0.0, 0.0

    inside = in_front & (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    truncation = float(1.0 - inside.sum() / in_front.sum())

    if not inside.any():
        # Entirely off-frame: nothing of it is visible in this image, but that
        # is truncation, not something blocking it.
        return 0.0, truncation

    ix = px[inside].astype(np.int32)
    iy = py[inside].astype(np.int32)
    unblocked = depth_m[iy, ix] >= (depth[inside] - DEPTH_TOLERANCE_M)
    return float(unblocked.sum() / inside.sum()), truncation


def project_actor_bbox(actor: carla.Actor, camera: carla.Actor, K: np.ndarray,
                        depth_m: np.ndarray | None = None, width: int = None, height: int = None,
                        lattice: int = DEFAULT_LATTICE):
    """Returns dict(x1, y1, x2, y2, visibility, truncation) in pixel coords,
    or None if the actor's bbox has no vertex projecting in front of the
    camera (or collapses to zero area once clamped to the image).

    The box is **amodal**: it is the envelope of the actor's true 3D extent,
    so a fully-occluded actor still gets a valid box with `visibility == 0.0`.
    That is intentional -- the box says where the actor *is*, and only the
    simulator can supply that once the camera can no longer see it. Callers
    that need a camera-observable box must filter on `visibility` or on
    `occlusion_tiers.tier_for(...)`; they must not assume a returned box means
    a visible actor.

    `visibility` and `truncation` come from `sample_visibility` (dense surface
    sampling). Without `depth_m` there is nothing to test occlusion against,
    so visibility is reported as 1.0 -- "no evidence of blocking" rather than
    a measurement.
    """
    w2c = np.array(camera.get_transform().get_inverse_matrix())
    verts = actor.bounding_box.get_world_vertices(actor.get_transform())

    xs, ys = [], []
    img_w = width if width is not None else (depth_m.shape[1] if depth_m is not None else None)
    img_h = height if height is not None else (depth_m.shape[0] if depth_m is not None else None)

    for v in verts:
        img_pt, _cam_depth = _project_point(v, K, w2c)
        if img_pt is None:
            continue
        xs.append(img_pt[0])
        ys.append(img_pt[1])

    if not xs:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if img_w is not None:
        x1, x2 = max(0.0, x1), min(float(img_w), x2)
    if img_h is not None:
        y1, y2 = max(0.0, y1), min(float(img_h), y2)
    if x2 <= x1 or y2 <= y1:
        return None

    if depth_m is not None and img_w is not None and img_h is not None:
        visibility, truncation = sample_visibility(actor, camera, K, depth_m, img_w, img_h, lattice)
    else:
        visibility, truncation = 1.0, 0.0

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "visibility": visibility, "truncation": truncation}


def to_yolo_format(bbox: dict, img_w: int, img_h: int, class_id: int) -> str:
    """Converts a pixel-space bbox dict to a YOLO label line:
    `class_id x_center y_center width height` normalized to [0, 1]."""
    x_c = (bbox["x1"] + bbox["x2"]) / 2.0 / img_w
    y_c = (bbox["y1"] + bbox["y2"]) / 2.0 / img_h
    w = (bbox["x2"] - bbox["x1"]) / img_w
    h = (bbox["y2"] - bbox["y1"]) / img_h
    return f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
