"""Derive per-BEV-cell visible / occluded / unknown ground-truth labels.

Method: for every BEV grid cell (a ground-level point in the ego's local
frame), project it into the front camera's image plane and compare the
camera's rendered depth buffer at that pixel against the true distance to the
cell. If the buffer reports a *closer* surface, some object -- static (wall,
building) or dynamic (parked bus, vehicle) -- lies between the camera and
that ground point, so the cell is OCCLUDED. If the buffer's distance matches
the cell's distance (within a tolerance), the ray reaches the ground there
unobstructed, so the cell is VISIBLE. Cells that fall outside the camera's
field of view or sensor range are UNKNOWN. This reuses CARLA's off-the-shelf
depth buffer as ground truth, so no separate bounding-box raycasting is
needed -- the z-buffer already accounts for every occluder in the scene.

Semantic segmentation is sampled at the same pixel for VISIBLE cells, giving a
BEV semantic-segmentation label as a bonus signal.
"""
import math

import numpy as np

from common.config import load_yaml
from common.constants import DEPTH_TOLERANCE_M as _DEPTH_TOLERANCE_M

VISIBLE, OCCLUDED, UNKNOWN = 0, 1, 2


class BevProjector:
    def __init__(self, bev_cfg: dict | None = None):
        cfg = bev_cfg or load_yaml("bev.yaml")
        self.grid_cfg = cfg["bev_grid"]
        self.cam_cfg = cfg["camera"]

        n = self.grid_cfg["size_cells"]
        cell = self.grid_cfg["cell_size_m"]
        # Forward axis: 0..extent_m (cells in front of the ego).
        # Lateral axis: -extent_m/2..+extent_m/2.
        forward = (np.arange(n) + 0.5) * cell
        lateral = (np.arange(n) + 0.5) * cell - self.grid_cfg["extent_m"] / 2.0
        ff, ll = np.meshgrid(forward, lateral, indexing="ij")
        self.cell_forward = ff   # (n, n)
        self.cell_lateral = ll   # (n, n)

        width, height, fov = self.cam_cfg["width"], self.cam_cfg["height"], self.cam_cfg["fov"]
        self.width, self.height = width, height
        self.fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.cam_offset_x = self.cam_cfg["x"]
        self.cam_offset_z = self.cam_cfg["z"]

    def _project(self):
        """Ego-local ground cells -> camera pixel coords + forward distance."""
        x_fwd = self.cell_forward - self.cam_offset_x   # camera-local forward
        y_right = -self.cell_lateral                    # ego +lateral is left; camera +y is right
        z_up = -self.cam_offset_z                        # ground (z=0) relative to camera height

        valid_front = x_fwd > 0.1

        # CARLA sensor axes (x-fwd, y-right, z-up) -> pinhole convention
        # (x-right, y-down, z-fwd) used by the K matrix below.
        px = self.cx + self.fx * (y_right / x_fwd)
        py = self.cy + self.fy * (z_up / x_fwd)

        in_bounds = (px >= 0) & (px < self.width) & (py >= 0) & (py < self.height)
        valid = valid_front & in_bounds
        return px, py, x_fwd, valid

    def compute_labels(self, depth_m: np.ndarray, semseg_labels: np.ndarray | None = None):
        px, py, dist, valid = self._project()
        occ_grid = np.full(self.cell_forward.shape, UNKNOWN, dtype=np.uint8)
        sem_grid = np.full(self.cell_forward.shape, -1, dtype=np.int32)

        ix = np.clip(px, 0, self.width - 1).astype(np.int32)
        iy = np.clip(py, 0, self.height - 1).astype(np.int32)

        sampled_depth = depth_m[iy, ix]

        visible = valid & (sampled_depth >= (dist - _DEPTH_TOLERANCE_M))
        occluded = valid & ~visible

        occ_grid[visible] = VISIBLE
        occ_grid[occluded] = OCCLUDED
        # UNKNOWN stays wherever `valid` is False.

        if semseg_labels is not None:
            sem_grid[visible] = semseg_labels[iy, ix][visible]

        return occ_grid, sem_grid


def labels_to_rgb(occ_grid: np.ndarray) -> np.ndarray:
    """Visualization colors: visible=green, occluded=red, unknown=gray."""
    rgb = np.zeros((*occ_grid.shape, 3), dtype=np.uint8)
    rgb[occ_grid == VISIBLE] = (12, 163, 12)     # #0ca30c
    rgb[occ_grid == OCCLUDED] = (208, 59, 59)    # #d03b3b
    rgb[occ_grid == UNKNOWN] = (137, 135, 129)   # #898781
    return rgb
