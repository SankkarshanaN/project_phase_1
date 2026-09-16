"""Camera/radar geometry shared by the runtime perception path.

These conversions were previously duplicated or trapped where nothing else
could reach them: `pixel_range_to_azimuth_range` lived inside
`scripts/live_demo.py`, and the radar polar-to-Cartesian conversion was inlined
in `occlusion_grid._radar_hit_grid`. Tracking needs both, and a second copy of
a sign convention is a silent correctness bug waiting to happen -- so they live
here and every caller imports them.

Frame conventions, used consistently across the whole project
------------------------------------------------------------
Ego-local Cartesian:  `forward` is ahead of the ego, `lateral` is **positive to
the left**. Matches `occlusion_mask.BevProjector`, `true_occupancy` and
`data_collector`.

Sensor polar:  `azimuth` is **positive to the right** (CARLA's radar
convention), so the two differ by a sign:

    forward = range * cos(azimuth)
    lateral = -range * sin(azimuth)

Nothing in this module reads simulator state. Everything here is a function of
sensor output plus fixed mounting geometry, so it stays on the runtime side of
the ground-truth / runtime split described in CLAUDE.md.
"""
import math

import numpy as np


def pixel_to_bearing(px: float, img_width: int, fov_deg: float) -> float:
    """Horizontal image coordinate -> azimuth in radians, positive to the right.

    Assumes the pinhole model `build_projection_matrix` uses and a camera
    mounted at zero yaw relative to the ego, which is how
    `sensors.spawn_sensor_rig` attaches it. This is an approximation, not a
    calibrated extrinsic: it ignores lens distortion and any mounting error.
    Adequate here because it feeds an EKF that carries its own measurement
    noise -- but the noise term has to be sized for it (see
    `tracking.DEFAULT_BEARING_STD_RAD`).
    """
    focal = img_width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return math.atan((px - img_width / 2.0) / focal)


def pixel_range_to_azimuth_range(x1: float, x2: float, img_width: int, fov_deg: float,
                                   margin_deg: float = 3.0) -> tuple[float, float]:
    """A bounding box's horizontal pixel span as an azimuth window (radians).

    The margin widens the window to absorb the approximation above plus the
    offset between the camera and radar mounts, so radar returns belonging to
    the boxed object are not missed at the edges. Too wide and neighbouring
    objects bleed in; 3 degrees is the working default.
    """
    lo, hi = sorted((pixel_to_bearing(x1, img_width, fov_deg),
                      pixel_to_bearing(x2, img_width, fov_deg)))
    margin = math.radians(margin_deg)
    return lo - margin, hi + margin


def polar_to_ego_xy(rng: np.ndarray | float, azimuth: np.ndarray | float):
    """(range, azimuth) -> (forward, lateral). See the sign note above."""
    return rng * np.cos(azimuth), -rng * np.sin(azimuth)


def ego_xy_to_polar(forward: np.ndarray | float, lateral: np.ndarray | float):
    """(forward, lateral) -> (range, azimuth). Inverse of `polar_to_ego_xy`."""
    return np.hypot(forward, lateral), np.arctan2(-lateral, forward)


def radar_to_ego_xy(radar_pts: np.ndarray) -> np.ndarray:
    """(N, 4) radar array -> (N, 2) ego-local [forward, lateral] positions.

    Input columns are [velocity, azimuth, altitude, depth] per
    `sensors.radar_to_array`. Altitude is dropped: every consumer here works on
    the ground plane, and the radar's 20-degree vertical FOV makes the
    horizontal projection error small next to the range noise.
    """
    if radar_pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    forward, lateral = polar_to_ego_xy(radar_pts[:, 3], radar_pts[:, 1])
    return np.stack([forward, lateral], axis=1).astype(np.float32)


def radar_in_azimuth_window(radar_pts: np.ndarray, az_lo: float, az_hi: float) -> np.ndarray:
    """Radar returns whose azimuth falls inside a window -- the subset belonging
    to one camera detection. Returns the filtered (M, 4) array."""
    if radar_pts.size == 0:
        return radar_pts
    az = radar_pts[:, 1]
    return radar_pts[(az >= az_lo) & (az <= az_hi)]


# Range bin width for clustering returns onto a target, metres. Narrow enough
# to separate a pedestrian from the road behind them, wide enough to hold the
# spread of returns off one body.
RANGE_CLUSTER_M = 1.5
MIN_CLUSTER_RETURNS = 2


def radar_range_for_window(radar_pts: np.ndarray, az_lo: float, az_hi: float,
                            max_range_m: float = 100.0):
    """Range and range-rate of the dominant object in an azimuth window.

    Returns (range, range_rate), or (None, None) if nothing usable is there.

    Takes the densest RANGE CLUSTER rather than the median of everything in the
    window, because the median is badly wrong on real data. CARLA's radar
    returns ~143 points per frame off road surface, buildings and background,
    and a pedestrian's angular window is a few degrees wide -- so most returns
    inside it belong to whatever is behind the pedestrian, not the pedestrian.
    Measured against ground truth on real frames, the median gave a mean range
    error of -13.3 m with 66% of observations off by more than 3 m, which
    corrupts the filter's position and therefore its velocity estimate.

    A real object produces several returns at nearly one range while background
    is spread across many; picking the tightest cluster recovers the object.
    Ties are broken toward the NEARER cluster, because for a collision-relevant
    decision the closer of two candidate objects is the one that matters.
    """
    hits = radar_in_azimuth_window(radar_pts, az_lo, az_hi)
    if hits.shape[0] == 0:
        return None, None
    if hits.shape[0] < MIN_CLUSTER_RETURNS:
        return float(hits[0, 3]), float(hits[0, 0])

    depth = hits[:, 3]
    valid = np.isfinite(depth) & (depth > 0.5) & (depth < max_range_m)
    if not valid.any():
        return None, None
    hits, depth = hits[valid], depth[valid]

    bins = np.floor(depth / RANGE_CLUSTER_M).astype(np.int64)
    labels, counts = np.unique(bins, return_counts=True)
    best = counts.max()
    # Among equally-populated clusters take the nearest.
    chosen = labels[counts == best].min()

    m = bins == chosen
    return float(np.median(hits[m, 3])), float(np.median(hits[m, 0]))


def ego_xy_to_bev_cell(forward, lateral, grid_cfg: dict):
    """Ego-local metres -> (forward_idx, lateral_idx) BEV cell indices.

    Same binning as `occlusion_grid._radar_hit_grid`. Indices may fall outside
    [0, size_cells); callers must mask, as the grid builders do.
    """
    cell = grid_cfg["cell_size_m"]
    extent = grid_cfg["extent_m"]
    fwd_idx = np.floor(np.asarray(forward) / cell).astype(np.int64)
    lat_idx = np.floor((np.asarray(lateral) + extent / 2.0) / cell).astype(np.int64)
    return fwd_idx, lat_idx


# Physical heights used to turn a detection's pixel height into a metric range.
# Adult pedestrian stature spans roughly 1.5-1.9 m, so a single figure carries
# about 12% scale error on any ONE observation -- but that error is zero-mean
# across a population, which is what makes the estimate useful for detecting a
# systematic offset even though it is poor for any single measurement.
ASSUMED_HEIGHT_M = {0: 1.5, 1: 1.7}      # 0 = vehicle, 1 = pedestrian

# Below this pixel height the estimate is too coarse to be worth anything: at
# 800x600 and 90 deg FOV the focal length is 400 px, so a 1.7 m pedestrian
# spanning 12 px sits at ~57 m and one pixel of box error moves that by 5 m.
MIN_BOX_HEIGHT_PX = 12.0


def range_from_pixel_height(box_px, cls: int, cam_cfg: dict) -> float | None:
    """Metric range from a detection's pixel height and an assumed real height.

    range = focal x real_height / pixel_height, the standard pinhole relation.

    This exists to give the health monitor a range estimate that owes the radar
    NOTHING. A constant radar range bias is invisible to every self-referential
    check: the return count is normal, the spread is normal, the distances are
    plausible, and the tracking filter absorbs the offset entirely, since an
    object 2.5 m further away at the same bearing is a perfectly consistent
    world state. Comparing against an independent estimate is the only way to
    see it, and object size is the one independent metric cue a camera has.

    Returns None when the box is too small for the estimate to mean anything.
    Accuracy is poor per observation and that is expected -- the check averages
    a systematic offset out of noise over hundreds of observations rather than
    trusting any single one.
    """
    height_px = float(box_px[3]) - float(box_px[1])
    if height_px < MIN_BOX_HEIGHT_PX:
        return None
    real_h = ASSUMED_HEIGHT_M.get(int(cls))
    if real_h is None:
        return None
    focal = cam_cfg["width"] / (2.0 * math.tan(math.radians(cam_cfg["fov"]) / 2.0))
    return float(focal * real_h / height_px)
