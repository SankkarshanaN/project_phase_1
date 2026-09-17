"""Phase 4: Three-State (+ Empty) Occlusion Detector.

Classifies each cell of the 20x20 BEV grid as VISIBLE / OCCLUDED / EMPTY /
UNKNOWN, following the spec's decision order:

    1. obstacle_map.casts_shadow_on(cell) -> OCCLUDED
    2. radar_pointcloud.has_return_at(cell) -> VISIBLE
    3. camera_depth.is_clear_at(cell) -> EMPTY
    4. else -> UNKNOWN

This module builds "obstacle_map" and "camera_depth" from the SAME MiDaS
disparity map (there is no separate 3D obstacle reconstruction step here).
`_ground_profile` fits what unoccluded ground reads as a function of range,
once per frame, from a low quantile of disparity within each range ring; a
cell whose actual disparity reads well above its own range's fitted level has
something nearer than the ground in front of it (OCCLUDED). A cell at or
below the fitted level is camera-confirmed clear (EMPTY, unless radar already
claimed it VISIBLE). This replaced an earlier cell-by-cell marching
propagation -- see `_ground_profile`'s docstring for why marching was wrong
regardless of which direction it walked. Radar detections are converted from
the sensor's native (azimuth, depth) polar form into the same ego-local
(forward, lateral) grid cells as `carla_tools.occlusion_mask.BevProjector`.
"""
import math
import numpy as np

from carla_tools.occlusion_mask import BevProjector
from perception.depth_midas import estimate_disparity, normalize_disparity
from perception.geometry import ego_xy_to_bev_cell, radar_to_ego_xy

VISIBLE, OCCLUDED, EMPTY, UNKNOWN = 0, 1, 2, 3
LABEL_NAMES = {VISIBLE: "VISIBLE", OCCLUDED: "OCCLUDED", EMPTY: "EMPTY", UNKNOWN: "UNKNOWN"}


# Quantile of per-ring disparity taken as the unoccluded ground level. Obstacles
# read NEARER than ground (disparity is inverse depth), so they sit in the upper
# tail of a ring's distribution and a low quantile estimates bare road robustly.
#
# Lowered from 0.30 to 0.01 on 2026-09-17 -- the 0.30 quantile was itself the
# recall bottleneck, not just shadow_tolerance. Against the corrected ground
# truth, a typical urban ring is 68-94% genuinely occluded (see the
# "RESOLVED" note in CLAUDE.md), so a ring's bottom-30th-percentile disparity
# is frequently estimated from mostly-occluded pixels rather than genuine
# ground: nearer objects read at HIGHER disparity, so a majority-occluded
# ring drags even its low quantile upward, and the fitted "ground" level ends
# up too close, suppressing recall on exactly the dense scenes this project
# is about. A much lower quantile is more robust to that contamination -- it
# grabs whatever thin sliver of genuine distant ground survives in a ring
# even when most of it is blocked.
#
# Swept jointly with shadow_tolerance against ground truth: first a coarse
# grid on 120 frames to find the right region, then confirmed on a disjoint
# 150-frame held-out sample with NO overlapping frames, then finalized with
# scripts/sweep_shadow_tolerance.py's official 150-frame sweep at this
# quantile (`results/shadow_tolerance_sweep.json`):
#
#     (quantile, tolerance)     precision  recall  F1     agreement
#     (0.30, 0.02)  -- old        0.846    0.482  0.614   0.686
#     (0.01, 0.001) -- new        0.727    0.917  0.811   0.780
#
# Recall nearly doubles (+90% relative) for a precision cost of ~0.12, and
# the held-out check landed within 0.01 of the tuning-set numbers, so this is
# a real improvement, not overfitting to one sample. Re-measure both
# constants together with scripts/sweep_shadow_tolerance.py before changing
# either -- they were fit as a pair, not independently.
GROUND_QUANTILE = 0.01

# Range rings the ground profile is fitted over, in cells of forward distance.
RING_CELLS = 1

# Default shadow-detection threshold. Moved to 0.001 together with
# GROUND_QUANTILE above on 2026-09-17 -- see that constant's comment for the
# joint sweep and the held-out validation. Do not tune one without the other;
# they were fit as a pair.
DEFAULT_SHADOW_TOLERANCE = 0.001


def _ground_profile(sampled: np.ndarray, ray_range: np.ndarray,
                     valid: np.ndarray) -> np.ndarray:
    """Per-frame estimate of what bare ground reads at each range.

    MiDaS returns relative inverse disparity with no metric scale, so a cell
    cannot be depth-tested against its own range directly the way the ground
    truth is (`BevProjector.compute_labels` compares the depth buffer against
    each cell's distance). What *is* stable is that unoccluded ground disparity
    falls off smoothly with range, so fitting that fall-off per frame recovers
    the same comparison: a cell reading well above its range's ground level has
    something in front of it.

    This replaced a running trend marched outward cell by cell. Marching has a
    structural cost that is easy to miss: the first cell of every track is
    consumed to seed the trend and can never be flagged. Measured on 150 frames,
    going from 20 seeded tracks to 40 dropped recall from 0.408 to 0.297 -- the
    seeding dominated, not the geometry it was marching along.
    """
    n_forward = sampled.shape[0]
    ground = np.full(sampled.shape, np.nan)
    order = np.argsort(ray_range, axis=None)
    flat_valid = valid.ravel()[order]
    flat_d = sampled.ravel()[order]
    usable = order[flat_valid]
    if usable.size == 0:
        return ground

    d_sorted = sampled.ravel()[usable]
    r_sorted = ray_range.ravel()[usable]
    n_rings = max(1, n_forward // RING_CELLS)
    edges = np.linspace(r_sorted[0], r_sorted[-1] + 1e-6, n_rings + 1)
    ring = np.clip(np.digitize(r_sorted, edges) - 1, 0, n_rings - 1)

    levels = np.full(n_rings, np.nan)
    for k in range(n_rings):
        vals = d_sorted[ring == k]
        if vals.size:
            levels[k] = float(np.quantile(vals, GROUND_QUANTILE))
    # Rings the camera never sees inherit the nearest fitted neighbour, so a
    # gap in coverage does not silently disable detection beyond it.
    idx = np.arange(n_rings)
    known = idx[~np.isnan(levels)]
    if known.size == 0:
        return ground
    levels = np.interp(idx, known, levels[known])

    ground.ravel()[usable] = levels[ring]
    return ground


def _shadow_grid_from_disparity(norm_disparity: np.ndarray, projector: BevProjector,
                                  tolerance: float = DEFAULT_SHADOW_TOLERANCE
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (shadow, valid) boolean (n, n) grids. `shadow[i, j]` is True
    where a nearer-than-ground obstacle blocks the ray to that cell."""
    px, py, x_fwd, valid = projector._project()
    h, w = norm_disparity.shape
    ix = np.clip(px, 0, w - 1).astype(np.int32)
    iy = np.clip(py, 0, h - 1).astype(np.int32)
    sampled = norm_disparity[iy, ix]

    y_right = -projector.cell_lateral
    ray_range = np.hypot(x_fwd, y_right)

    ground = _ground_profile(sampled, ray_range, valid)
    with np.errstate(invalid="ignore"):
        shadow = valid & ~np.isnan(ground) & (sampled > ground + tolerance)
    return shadow, valid


def _radar_hit_grid(radar_pts: np.ndarray, grid_cfg: dict) -> np.ndarray:
    """radar_pts: (N, 4) array of [velocity, azimuth_rad, altitude_rad, depth_m]
    (see carla_tools.sensors.radar_to_array). Returns an (n, n) bool grid,
    True where at least one radar detection falls in that cell. The radar is
    mounted forward-facing at ego yaw (no rotation offset applied at spawn in
    carla_tools.sensors.spawn_sensor_rig), so its azimuth/depth map directly
    onto the same ego-local forward/lateral frame as BevProjector.

    The polar->Cartesian conversion and the cell binning both live in
    `perception.geometry` so the tracker shares one definition of the sign
    convention with this grid.
    """
    n = grid_cfg["size_cells"]
    hits = np.zeros((n, n), dtype=bool)
    if radar_pts.size == 0:
        return hits

    xy = radar_to_ego_xy(radar_pts)
    fwd_idx, lat_idx = ego_xy_to_bev_cell(xy[:, 0], xy[:, 1], grid_cfg)

    valid = (fwd_idx >= 0) & (fwd_idx < n) & (lat_idx >= 0) & (lat_idx < n)
    hits[fwd_idx[valid], lat_idx[valid]] = True
    return hits


def classify_grid(rgb: np.ndarray, radar_pts: np.ndarray, bev_cfg: dict,
                    shadow_tolerance: float = DEFAULT_SHADOW_TOLERANCE,
                    clear_threshold: float = 0.5,
                    norm_disparity: np.ndarray | None = None) -> np.ndarray:
    """Returns an (n, n) uint8 grid of VISIBLE/OCCLUDED/EMPTY/UNKNOWN labels
    for one frame, using only camera (MiDaS) + radar -- no CARLA ground truth.

    `norm_disparity` lets a caller supply an already-computed, already-
    normalized MiDaS map instead of paying for inference here. MiDaS dominates
    this function's cost -- the shadow and radar steps are array arithmetic over
    400 cells -- so a real-time caller can refresh depth every few frames while
    still folding in fresh radar returns every frame. Geometry justifies it: the
    grid's cells are 2 m, and at city speed the scene shifts well under one cell
    between refreshes. `scripts/live_demo.py` uses this.
    """
    projector = BevProjector(bev_cfg)
    if norm_disparity is None:
        norm_disparity = normalize_disparity(estimate_disparity(rgb))

    shadow, valid = _shadow_grid_from_disparity(norm_disparity, projector, shadow_tolerance)
    radar_hits = _radar_hit_grid(radar_pts, bev_cfg["bev_grid"])

    grid = np.full(shadow.shape, UNKNOWN, dtype=np.uint8)
    grid[shadow] = OCCLUDED
    grid[~shadow & radar_hits] = VISIBLE
    px, py, _dist, _v = projector._project()
    clear = valid & ~shadow & ~radar_hits
    h, w = norm_disparity.shape
    ix = np.clip(px, 0, w - 1).astype(np.int32)
    iy = np.clip(py, 0, h - 1).astype(np.int32)
    is_clear = norm_disparity[iy, ix] < clear_threshold
    grid[clear & is_clear] = EMPTY
    # Remaining cells (valid but neither shadowed, radar-seen, nor clear-confirmed;
    # or invalid/out-of-FOV) stay UNKNOWN.
    return grid


def labels_to_rgb(grid: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
    rgb[grid == VISIBLE] = (12, 163, 12)      # green
    rgb[grid == OCCLUDED] = (208, 59, 59)     # red
    rgb[grid == EMPTY] = (66, 135, 245)       # blue
    rgb[grid == UNKNOWN] = (137, 135, 129)    # gray
    return rgb
