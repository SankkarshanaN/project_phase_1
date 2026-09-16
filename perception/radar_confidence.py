"""Phase 3: rule-based radar confidence score, per spec ("Radar confidence
score (0-1, rule-based from signal-to-noise ratio)").

CARLA's radar sensor (`sensor.other.radar`) doesn't expose a physical SNR
value through its Python API -- each detection only carries
velocity/azimuth/altitude/depth. This derives a documented, rule-based proxy
from what IS available:

  - return density: more detections in the region = a stronger aggregate
    signal, which is what SNR would reflect physically if it were exposed.
  - depth-cluster tightness: a real target reflects consistently at
    approximately one depth across nearby detections; sensor noise/clutter
    scatters depth randomly. Tight clustering -> higher confidence.

Both are standard stand-in heuristics for radar-detection confidence when
raw SNR isn't available from the sensor API.
"""
import numpy as np


def radar_confidence(radar_pts: np.ndarray, min_points_for_full_density: int = 8) -> float:
    """Overall per-frame radar confidence in [0, 1].
    radar_pts: (N, 4) array of [velocity, azimuth, altitude, depth]
    (see carla_tools.sensors.radar_to_array)."""
    if radar_pts.shape[0] == 0:
        return 0.0

    n = radar_pts.shape[0]
    depths = radar_pts[:, 3]
    density_score = min(1.0, n / min_points_for_full_density)

    if n > 1:
        depth_std = float(np.std(depths))
        depth_mean = float(np.mean(depths))
        consistency_score = 1.0 / (1.0 + depth_std / max(depth_mean, 1e-6))
    else:
        consistency_score = 0.5  # a lone return is neither clearly consistent nor scattered

    return float(np.clip(0.5 * density_score + 0.5 * consistency_score, 0.0, 1.0))


def radar_confidence_in_region(radar_pts: np.ndarray, azimuth_range: tuple | None = None,
                                 depth_range: tuple | None = None,
                                 min_points_for_full_density: int = 4) -> float:
    """Same as `radar_confidence`, scoped to an azimuth/depth window (e.g.
    the region a camera detection's bounding box projects to) -- for a
    per-detection radar confidence rather than a whole-frame summary."""
    if radar_pts.shape[0] == 0:
        return 0.0
    mask = np.ones(radar_pts.shape[0], dtype=bool)
    if azimuth_range is not None:
        mask &= (radar_pts[:, 1] >= azimuth_range[0]) & (radar_pts[:, 1] <= azimuth_range[1])
    if depth_range is not None:
        mask &= (radar_pts[:, 3] >= depth_range[0]) & (radar_pts[:, 3] <= depth_range[1])
    return radar_confidence(radar_pts[mask], min_points_for_full_density)
