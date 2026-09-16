"""Graded per-actor occlusion tiers: OCCLUDED / PARTIAL / VISIBLE.

Why this exists: the pipeline previously had no graded notion of occlusion at
all. `data_collector` computed a per-actor visibility fraction, compared it
against a hardcoded 0.15, and discarded the box if it fell below -- so a
fully-hidden pedestrian, the entire subject of this project, never reached
disk. The evidential head therefore only ever trained on cleanly-visible or
cleanly-absent actors, which is why its uncertainty calibration curve came out
flat (see `docs/RESULTS.md`): the ambiguous middle band was filtered out of
the dataset before training ever saw it.

The three tiers are deliberately coarse. They are a *reporting* axis for
evaluation and dataset balancing -- "how does the detector behave when it can
see a sliver versus the whole body" -- not a training target. The underlying
continuous visibility fraction is recorded alongside the tier, so a later
analysis can re-bin without re-collecting.

A tier is assigned per actor per frame, so a single crossing episode sweeps
OCCLUDED -> PARTIAL -> VISIBLE as the walker emerges from behind an occluder.
That sweep is the point: it is the only place in the dataset where the
"partial glimpse" regime is sampled densely.
"""
from common.config import load_yaml

OCCLUDED, PARTIAL, VISIBLE = 0, 1, 2

TIER_NAMES = {OCCLUDED: "OCCLUDED", PARTIAL: "PARTIAL", VISIBLE: "VISIBLE"}

# Fallbacks used when configs/bev.yaml carries no `occlusion_tiers` block.
DEFAULT_OCCLUDED_MAX = 0.05
DEFAULT_PARTIAL_MAX = 0.65


def tier_thresholds(bev_cfg: dict | None = None) -> tuple[float, float]:
    """Returns (occluded_max, partial_max) from `configs/bev.yaml`.

    Kept in config rather than hardcoded because the PARTIAL band's upper edge
    is a real experimental knob: widening it pulls more nearly-clear actors
    into the "hard" bucket and changes every per-tier metric downstream.
    """
    cfg = bev_cfg or load_yaml("bev.yaml")
    tiers = cfg.get("occlusion_tiers", {})
    return (
        float(tiers.get("occluded_max", DEFAULT_OCCLUDED_MAX)),
        float(tiers.get("partial_max", DEFAULT_PARTIAL_MAX)),
    )


def tier_for(visibility: float, bev_cfg: dict | None = None) -> int:
    """Maps a continuous visibility fraction in [0, 1] onto a tier.

    `visibility` must be the occlusion-only fraction from
    `bbox_projection.project_actor_bbox` -- i.e. computed over sample points
    that land inside the image. Points falling outside the frame are
    truncation, not occlusion, and are reported separately; folding them in
    here would tier a perfectly-visible actor at the image edge as OCCLUDED.
    """
    occluded_max, partial_max = tier_thresholds(bev_cfg)
    if visibility <= occluded_max:
        return OCCLUDED
    if visibility <= partial_max:
        return PARTIAL
    return VISIBLE
