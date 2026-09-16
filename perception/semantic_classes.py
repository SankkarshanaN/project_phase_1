"""Collapse CARLA's ~29 semantic tags into a small macro-class set.

Tag ids per CARLA 0.9.16 docs (CityScapesPalette + extensions):
https://carla.readthedocs.io/en/0.9.16/ref_sensors/#semantic-segmentation-camera
"""
import numpy as np

UNLABELED, DRIVABLE, VEHICLE, PEDESTRIAN, STATIC_OBSTACLE, VEGETATION = range(6)
NUM_MACRO_CLASSES = 6

_TAG_TO_MACRO = {
    0: UNLABELED,
    1: DRIVABLE,          # Roads
    24: DRIVABLE,         # RoadLine
    25: DRIVABLE,         # Ground
    2: STATIC_OBSTACLE,   # SideWalks
    3: STATIC_OBSTACLE,   # Building
    4: STATIC_OBSTACLE,   # Wall
    5: STATIC_OBSTACLE,   # Fence
    6: STATIC_OBSTACLE,   # Pole
    20: STATIC_OBSTACLE,  # Static
    26: STATIC_OBSTACLE,  # Bridge
    28: STATIC_OBSTACLE,  # GuardRail
    9: VEGETATION,
    10: VEGETATION,       # Terrain
    12: PEDESTRIAN,
    13: PEDESTRIAN,       # Rider
    14: VEHICLE,
    15: VEHICLE,          # Truck
    16: VEHICLE,          # Bus
    18: VEHICLE,          # Motorcycle
    19: VEHICLE,          # Bicycle
}

_LUT = np.full(256, UNLABELED, dtype=np.int64)
for tag, macro in _TAG_TO_MACRO.items():
    _LUT[tag] = macro


def to_macro_classes(tag_grid: np.ndarray) -> np.ndarray:
    """tag_grid: int array of raw CARLA semantic tags (or -1 for unknown/void)."""
    out = np.full(tag_grid.shape, UNLABELED, dtype=np.int64)
    valid = tag_grid >= 0
    out[valid] = _LUT[tag_grid[valid].astype(np.int64) % 256]
    return out
