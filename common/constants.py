"""Geometry constants shared by the ground-truth labelling path.

`DEPTH_TOLERANCE_M` was previously defined twice, independently, with the same
value: once in `carla_tools.bbox_projection` (for per-actor 2D label
visibility) and once in `carla_tools.occlusion_mask` (for per-cell BEV
occupancy). Both answer the same physical question -- "is the surface the
depth buffer recorded at this pixel the thing I expected, or something nearer
blocking it?" -- so they must move together. Two module-level copies meant
changing one silently desynchronised the 2D-label occlusion criterion from the
occupancy-grid one, with nothing to catch it.
"""

# Slack on every depth-buffer occlusion test, in meters.
#
# A sampled point lies on an actor's bounding hull, but the depth buffer at
# that pixel records whatever the ray actually hit -- typically a point
# slightly nearer (the actor's own visible skin, which sits inside the hull),
# plus discretization error from CARLA's 24-bit depth encoding and integer
# pixel rounding. Without slack, self-occlusion would mark nearly every sample
# as blocked.
#
# Known consequence, documented rather than hidden: this is a fixed absolute
# distance, so it is generous at long range (relative error shrinks with
# distance), and a thin occluder sitting less than 0.75 m in front of an actor
# -- a pole, a guard rail, the near edge of another vehicle -- will not
# register as occlusion at all.
DEPTH_TOLERANCE_M = 0.75
