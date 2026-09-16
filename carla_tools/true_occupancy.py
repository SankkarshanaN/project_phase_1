"""Privileged (occlusion-ignoring) ground truth for dynamic actors.

`occlusion_mask.py`'s occ_grid/sem_grid are camera-derived: a cell behind an
occluder is correctly marked OCCLUDED, but the camera physically cannot see
what's behind it, so there is no way to recover the true hidden class from
any sensor image. Validating the occlusion detector (Phase 4) and supervising
the evidential head's confidence (Phase 3) needs ground truth that *does* know
the truth regardless of visibility -- which only the simulator's own world
state (actor list + bounding boxes) can provide, not any sensor.

Scoped to dynamic actors (vehicles, pedestrians) only -- static geometry
(road/buildings) doesn't move, so there's no comparable "hidden object"
concept for it.
"""
import math

import numpy as np

from perception.semantic_classes import PEDESTRIAN, UNLABELED, VEHICLE


def compute_true_occupancy_grid(world, ego, bev_cfg: dict) -> np.ndarray:
    """Returns an (n, n) int array: UNLABELED (0) / VEHICLE / PEDESTRIAN,
    the ground-truth macro class occupying each BEV cell, ignoring occlusion
    entirely -- computed from actor bounding boxes, not from any sensor.
    Same ego-local (forward, lateral) grid convention as BevProjector.
    """
    grid_cfg = bev_cfg["bev_grid"]
    n = grid_cfg["size_cells"]
    cell = grid_cfg["cell_size_m"]
    extent = grid_cfg["extent_m"]

    forward = (np.arange(n) + 0.5) * cell
    lateral = (np.arange(n) + 0.5) * cell - extent / 2.0
    cell_forward, cell_lateral = np.meshgrid(forward, lateral, indexing="ij")

    ego_t = ego.get_transform()
    ego_x, ego_y = ego_t.location.x, ego_t.location.y
    ego_yaw = math.radians(ego_t.rotation.yaw)

    # +lateral is LEFT, matching BevProjector and perception.geometry.
    #
    # This previously used `- lateral*sin, + lateral*cos`, which is the offset
    # along CARLA's get_right_vector() -- so the grid was mirrored against
    # occlusion_mask's grid of the same scene, despite the docstring above
    # claiming the same convention. Verified with a probe placed to the ego's
    # right: it appeared in the right half of the camera image, and this
    # function marked cells to the right of centre while BevProjector projected
    # that same (forward, lateral) to the left half.
    world_x = ego_x + cell_forward * math.cos(ego_yaw) + cell_lateral * math.sin(ego_yaw)
    world_y = ego_y + cell_forward * math.sin(ego_yaw) - cell_lateral * math.cos(ego_yaw)

    true_grid = np.full((n, n), UNLABELED, dtype=np.int64)

    actors = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*"))
    for actor in actors:
        if actor.id == ego.id:
            continue
        bb = actor.bounding_box
        a_t = actor.get_transform()
        a_x, a_y = a_t.location.x, a_t.location.y
        a_yaw = math.radians(a_t.rotation.yaw)

        dx = world_x - a_x
        dy = world_y - a_y
        local_x = dx * math.cos(a_yaw) + dy * math.sin(a_yaw) - bb.location.x
        local_y = -dx * math.sin(a_yaw) + dy * math.cos(a_yaw) - bb.location.y

        # Dilate the footprint by half a cell width: the test below only
        # checks whether a cell's CENTER falls inside the actor's rectangle,
        # so a small actor (a pedestrian's ~0.3-0.4m half-extent vs. a 2m
        # cell here) can sit entirely between cell centers and never get
        # marked otherwise. Under-marking a pedestrian is the worse failure
        # mode for a safety-relevant occupancy signal.
        half_cell = cell / 2.0
        inside = (np.abs(local_x) <= bb.extent.x + half_cell) & (np.abs(local_y) <= bb.extent.y + half_cell)
        macro_class = VEHICLE if actor.type_id.startswith("vehicle") else PEDESTRIAN
        true_grid[inside] = macro_class

    return true_grid
