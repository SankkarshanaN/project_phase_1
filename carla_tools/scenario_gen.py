"""Programmatic occlusion-scenario spawner for CARLA 0.9.16.

Scenarios are generated from `configs/scenarios.yaml` parameters rather than
hand-placed: for a given RNG seed we pick a random ego spawn point, walk along
its lane using CARLA's waypoint API to place an "occluder" actor ahead (a bus
or large vehicle standing in for a parked obstruction), and place one or two
pedestrians further along whose visibility from the ego's viewpoint depends on
the occluder. Blueprint names are CARLA 0.9.16's actual catalog (verified live
via world.get_blueprint_library()).
"""
import random
from dataclasses import dataclass, field

import carla
import numpy as np

from carla_tools.sensors import set_walker_velocity
from carla_tools.walker_behavior import WalkerDriver, sample_behaviour
from common.config import load_yaml


def _pick_blueprint(bp_lib, pool: list[str], rng: random.Random):
    for name in rng.sample(pool, len(pool)):
        bps = bp_lib.filter(name)
        if len(bps) > 0:
            return bps[0]
    raise RuntimeError(f"No blueprint found in pool {pool}")


def _waypoint_ahead(start_wp: carla.Waypoint, distance: float, rng: random.Random) -> carla.Waypoint:
    nxt = start_wp.next(distance)
    if not nxt:
        return start_wp
    return rng.choice(nxt)


def _spawn_ego(world: carla.World, rng: random.Random):
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Map has no spawn points")

    ego_bp = bp_lib.filter("vehicle.lincoln.mkz")[0]  # CARLA 0.10.0 catalog has no Tesla
    for _ in range(10):
        ego_transform = rng.choice(spawn_points)
        ego = world.try_spawn_actor(ego_bp, ego_transform)
        if ego is not None:
            ego_wp = carla_map.get_waypoint(ego_transform.location)
            return ego, ego_wp
    raise RuntimeError("Failed to spawn ego after 10 attempts")


def _spawn_occluder_ahead(world, bp_lib, ego_wp, occluder_pool, occ_range, lateral_range, rng: random.Random):
    occluder_bp = _pick_blueprint(bp_lib, occluder_pool, rng)
    occ_lo, occ_hi = occ_range
    lateral_lo, lateral_hi = lateral_range

    occluder = None
    occluder_wp = None
    occluder_distance = None
    for attempt in range(6):
        occluder_distance = rng.uniform(occ_lo, occ_hi) + attempt * 2.0
        occluder_wp = _waypoint_ahead(ego_wp, occluder_distance, rng)
        lateral_jitter = rng.uniform(lateral_lo, lateral_hi)
        base_transform = occluder_wp.transform
        right = base_transform.get_right_vector()
        occ_location = carla.Location(
            x=base_transform.location.x + right.x * lateral_jitter,
            y=base_transform.location.y + right.y * lateral_jitter,
            z=base_transform.location.z + 0.3,
        )
        occ_transform = carla.Transform(occ_location, base_transform.rotation)
        occluder = world.try_spawn_actor(occluder_bp, occ_transform)
        if occluder is None:
            occluder = world.try_spawn_actor(occluder_bp, carla.Transform(
                carla.Location(base_transform.location.x, base_transform.location.y, base_transform.location.z + 0.3),
                base_transform.rotation,
            ))
        if occluder is not None:
            return occluder, occluder_wp, occluder_distance
    raise RuntimeError("Failed to spawn occluder after retries")


def _ego_camera_location(ego: carla.Actor, bev_cfg: dict | None = None) -> carla.Location:
    """Where the RGB camera will sit once `sensors.spawn_sensor_rig` attaches it.

    Occlusion has to be solved for from the *camera's* eye point, not the
    vehicle origin. The rig mounts the camera 1.5 m forward and 1.7 m up
    (configs/bev.yaml), and at the ranges these scenarios use that offset is a
    large fraction of the geometry -- using the vehicle origin instead would
    mis-site the shadow by enough to matter.
    """
    cam = (bev_cfg or load_yaml("bev.yaml"))["camera"]
    t = ego.get_transform()
    fwd = t.get_forward_vector()
    return carla.Location(
        x=t.location.x + fwd.x * cam["x"],
        y=t.location.y + fwd.y * cam["x"],
        z=t.location.z + cam["z"],
    )


def _box_half_width_along(actor: carla.Actor, direction: np.ndarray) -> float:
    """Half-width of the actor's oriented bounding box along a unit vector.

    The support function of an oriented box: sum over its three axes of
    (half-extent x |direction . axis|). Needed because an occluder's usable
    depth and width depend on how it is parked relative to the line of sight --
    a bus seen end-on shadows a very different volume than the same bus seen
    broadside, and assuming one or the other is what left walkers standing in
    plain view.
    """
    bb = actor.bounding_box
    m = (np.array(actor.get_transform().get_matrix())
         @ np.array(carla.Transform(bb.location, bb.rotation).get_matrix()))
    extents = np.array([bb.extent.x, bb.extent.y, bb.extent.z])
    return float(np.sum(extents * np.abs(m[:3, :3].T @ direction)))


def _shadow_position(camera_loc: carla.Location, occluder: carla.Actor,
                      clearance_m: float) -> tuple[carla.Location, float]:
    """A ground point hidden behind `occluder` from `camera_loc`.

    Placed on the line of sight from the camera through the occluder's centre,
    pushed past the far face by `clearance_m`. Returns (location, distance from
    camera).

    This replaces the previous approach, which offset the walker 3 m to the
    *side* of the occluder at the same longitudinal station. That gave zero
    depth separation, so nothing was ever actually behind anything: occlusion
    could only happen incidentally, when the occluder's body width happened to
    stick far enough toward the ego, and for half the random lateral jitter
    range the walker was in plain view from the first frame. The dataset's
    missing OCCLUDED tier traces directly to it.

    Where the ego approaches from roughly behind the occluder (the common
    case: same lane, kerb-parked ahead), this ray is nearly parallel to the
    occluder's own length, so it lands the target near the occluder's own
    front face already -- but only incidentally, because it depends on the
    ego's current viewing angle. `_front_corner_shadow_position` below makes
    that placement deliberate instead of a side effect of geometry that
    happens to line up.
    """
    o = occluder.get_transform().location
    d = np.array([o.x - camera_loc.x, o.y - camera_loc.y, 0.0])
    dist = float(np.linalg.norm(d))
    if dist < 1e-3:
        raise RuntimeError("Occluder is coincident with the camera")
    d /= dist

    depth = _box_half_width_along(occluder, d) + clearance_m
    return carla.Location(x=o.x + d[0] * depth, y=o.y + d[1] * depth,
                           z=o.z), dist + depth


def _front_corner_shadow_position(occluder: carla.Actor,
                                    clearance_m: float) -> carla.Location:
    """A ground point near the occluder's own front-right corner, defined
    from the occluder's geometry alone -- not the camera's viewing angle the
    way `_shadow_position` is.

    This is the classic "pedestrian steps out from in front of a parked bus"
    hazard: hidden right up until the moment they clear the bus's own front
    bumper, which is exactly where this places them (pushed `clearance_m`
    further forward so they start genuinely behind it, not straddling the
    face). The caller still has to verify with `_shadow_covers` -- this
    function only says where the classic hazard sits, not that this specific
    occluder is tall or wide enough to actually hide someone there.
    """
    t = occluder.get_transform()
    fwd, right = t.get_forward_vector(), t.get_right_vector()
    bb = occluder.bounding_box
    front = bb.extent.x + clearance_m
    # Toward the right edge of the occluder's own body, not past it -- the
    # target should still be screened by the occluder's width, only close to
    # the corner it will first clear as it walks forward and left.
    side = bb.extent.y * 0.8
    o = t.location
    return carla.Location(
        x=o.x + fwd.x * front + right.x * side,
        y=o.y + fwd.y * front + right.y * side,
        z=o.z,
    )


def _shadow_covers(camera_loc: carla.Location, occluder: carla.Actor,
                    target_loc: carla.Location, target_half_width: float = 0.35,
                    target_height: float = 1.8) -> bool:
    """Whether the occluder's silhouette is large enough to actually hide a
    target of the given size sitting at `target_loc`.

    Being *on* the line of sight is necessary but not sufficient -- a bollard
    on the same ray as a pedestrian hides almost none of them. Compares angular
    half-widths and heights as seen from the camera, so the check scales with
    range instead of relying on a fixed metric margin.
    """
    o = occluder.get_transform().location
    d = np.array([o.x - camera_loc.x, o.y - camera_loc.y, 0.0])
    r_occ = float(np.linalg.norm(d))
    if r_occ < 1e-3:
        return False
    d /= r_occ

    perpendicular = np.array([-d[1], d[0], 0.0])
    r_tgt = float(np.linalg.norm([target_loc.x - camera_loc.x, target_loc.y - camera_loc.y]))
    if r_tgt <= r_occ:
        return False  # target is nearer than the occluder; nothing is in the way

    wide_enough = (_box_half_width_along(occluder, perpendicular) / r_occ
                   > target_half_width / r_tgt)

    # Heights are measured from the camera's eye line, so an occluder that is
    # tall but mounted low can still fail to cover a standing pedestrian.
    #
    # The top of the box is origin + bounding_box.LOCATION.z + extent.z. That
    # middle term is not optional and omitting it is not a rounding error: for
    # CARLA's Fuso bus, origin 0.55 + location 2.13 + extent 2.12 gives a true
    # top of 4.80 m, while origin + extent alone gives 2.67 m. Understating
    # every occluder by more than two metres made this test reject every
    # candidate site on real geometry, and no scenario could place a walker at
    # all.
    bb = occluder.bounding_box
    occ_top = o.z + bb.location.z + bb.extent.z - camera_loc.z
    tgt_top = target_loc.z + target_height - camera_loc.z
    tall_enough = (occ_top / r_occ) > (tgt_top / r_tgt)
    return bool(wide_enough and tall_enough)


def _spawn_walker_in_shadow(world, bp_lib, camera_loc: carla.Location, occluder: carla.Actor,
                             blueprint_name: str, rng: random.Random,
                             clearance_range_m=(1.0, 3.0), attempts: int = 6):
    """Spawns a walker genuinely hidden behind `occluder`, or returns None.

    Retries at increasing depth into the shadow: the first choice can collide
    with the occluder's own collision volume or with street furniture, and a
    deeper placement is both more likely to be free and more certainly hidden.
    """
    lo, hi = clearance_range_m
    for attempt in range(attempts):
        clearance = rng.uniform(lo, hi) + attempt * 0.75
        loc, _dist = _shadow_position(camera_loc, occluder, clearance)

        # Coverage is tested at GROUND level, before the drop-spawn offset. The
        # walker ends up standing on the road whatever height it is spawned
        # from, so testing at the spawn height would credit it with an extra
        # metre of stature and demand a taller occluder than it actually needs.
        if not _shadow_covers(camera_loc, occluder, loc):
            continue

        spawn_at = carla.Transform(carla.Location(loc.x, loc.y, loc.z + 1.0))
        walker = world.try_spawn_actor(bp_lib.filter(blueprint_name)[0], spawn_at)
        if walker is not None:
            # Tick before anyone reads this actor's position.
            #
            # `try_spawn_actor` returns as soon as the server accepts the
            # request, but the actor's transform is not committed until the
            # next tick -- until then `get_location()` returns a default. That
            # default is (-6.4, 0.0, 1.09) on this build, roughly 140 m from
            # where the walker actually is, and reading it silently poisoned
            # everything downstream: the behaviour state machine's origin, and
            # `_distance_to_path_edge`, which produced a 150 m road edge and a
            # 100 m kerb distance so no walker ever reached its decision point.
            world.tick()
            return walker
    return None


# Bounds on the walker-to-corridor distance. A crossing decision made less
# than half a metre from the lane, or more than fifteen metres from it, is not
# a crossing decision.
MIN_ROAD_EDGE_M = 0.8
MAX_ROAD_EDGE_M = 15.0
DEFAULT_ROAD_EDGE_M = 3.0


def _distance_to_path_edge(walker: carla.Actor, lane_wp: carla.Waypoint,
                            direction: carla.Vector3D,
                            corridor_half_width_m: float = 1.75,
                            loc: carla.Location | None = None) -> float:
    """How far the walker must travel along `direction` to reach the edge of
    the ego's path corridor.

    Fed to `walker_behavior.sample_behaviour` so the hesitate/abort decision
    point lands before the walker enters the ego's lane rather than after it.
    Getting this wrong silently destroys the dataset's negative class -- an
    aborter who turns back only *after* stepping into the lane is labelled a
    crosser by the retrospective ground truth, so `approach_abort` stops being
    a negative at all.
    """
    # `loc` overrides the actor's own transform. A walker spawned earlier in
    # the same tick still reports its default transform until the world has
    # ticked, so callers that stage and measure without an intervening tick
    # must pass the location they actually spawned at.
    loc = loc if loc is not None else walker.get_location()
    lane = lane_wp.transform.location
    right = lane_wp.transform.get_right_vector()
    lateral_offset = (loc.x - lane.x) * right.x + (loc.y - lane.y) * right.y

    # Travelled distance is measured along `direction`; only its component
    # along the lane's lateral axis closes the gap.
    closing = -(direction.x * right.x + direction.y * right.y)

    # The occluder's lane and the ego's lane can be differently oriented -- a
    # bend, or a neighbouring segment -- which drives `closing` toward zero and
    # the quotient toward infinity. Observed producing a 150 m "road edge" and
    # hence a 100 m kerb distance, so the walker's hesitate/abort decision
    # point sat far beyond the end of the episode and no behaviour but
    # "walk straight" ever ran. Clamped to distances a crossing can plausibly
    # involve.
    if closing <= 0.25:
        return DEFAULT_ROAD_EDGE_M     # not usefully heading toward the lane
    raw = (abs(lateral_offset) - corridor_half_width_m) / closing
    return float(min(max(raw, MIN_ROAD_EDGE_M), MAX_ROAD_EDGE_M))


@dataclass
class OcclusionScenarioActors:
    ego: carla.Actor
    occluder: carla.Actor
    walker: carla.Actor
    scenario_type: str
    seed: int
    drivers: list = field(default_factory=list)


def spawn_occlusion_scenario(client, world, scenario_type: str, seed: int,
                              scenarios_cfg: dict | None = None, tm_port: int = 8000,
                              bev_cfg: dict | None = None) -> OcclusionScenarioActors:
    """Scenario A -- Occluded Pedestrian Crossing: ego + a parked-bus-style
    occluder + one pedestrian who starts genuinely hidden behind the occluder
    and walks laterally out into the road.

    `bev_cfg` is used only to locate the camera eye point for the shadow
    construction; it falls back to configs/bev.yaml.
    """
    cfg = scenarios_cfg or load_yaml("scenarios.yaml")
    sc = cfg["scenario_types"][scenario_type]
    rng = random.Random(cfg["episode"]["seed_base"] + seed)

    bp_lib = world.get_blueprint_library()
    ego, ego_wp = _spawn_ego(world, rng)

    try:
        occluder, occluder_wp, occ_dist = _spawn_occluder_ahead(
            world, bp_lib, ego_wp, sc["occluder_blueprint_pool"],
            sc["occluder_offset_range_m"], sc["jitter_lateral_m"], rng)
    except RuntimeError:
        ego.destroy()
        raise

    # Commit the ego and occluder transforms before any geometry is computed
    # from them. `try_spawn_actor` returns before the server has applied the
    # transform, so `get_transform()` on a just-spawned actor yields a default.
    # The shadow construction reads the ego's transform to locate the camera
    # eye point, and building that ray from a default put the walker at an
    # invalid site every time -- CARLA then relocated it ~140 m away, outside
    # the camera frustum, so it never appeared in a single recorded frame.
    world.tick()

    walker_bp_name = rng.choice(sc["crossing_actor_blueprint_pool"])
    right = occluder_wp.transform.get_right_vector()
    walker = _spawn_walker_in_shadow(
        world, bp_lib, _ego_camera_location(ego, bev_cfg), occluder, walker_bp_name, rng)
    if walker is None:
        ego.destroy()
        occluder.destroy()
        raise RuntimeError("Failed to spawn crossing walker in the occluder's shadow")

    # Walk across the road (toward -right) so the walker emerges laterally out
    # of the shadow. Crossing perpendicular to the line of sight is what makes
    # the visibility sweep OCCLUDED -> PARTIAL -> VISIBLE over a few seconds
    # rather than jumping between states in a single frame.
    direction = carla.Vector3D(-right.x, -right.y, 0.0)
    params = sample_behaviour(rng, cfg.get("vru_behaviour"),
                               road_edge_m=_distance_to_path_edge(walker, ego_wp, direction))
    driver = WalkerDriver(walker, direction, params)

    return OcclusionScenarioActors(ego=ego, occluder=occluder, walker=walker,
                                    scenario_type=scenario_type, seed=seed,
                                    drivers=[driver])


@dataclass
class MultiOcclusionActors:
    ego: carla.Actor
    occluder: carla.Actor
    visible_walker: carla.Actor
    hidden_walker: carla.Actor
    seed: int
    drivers: list = field(default_factory=list)


def spawn_multi_occlusion_scenario(client, world, seed: int,
                                    scenarios_cfg: dict | None = None, tm_port: int = 8000,
                                    bev_cfg: dict | None = None) -> MultiOcclusionActors:
    """Scenario C -- Multi-Occlusion: two pedestrians converging on the same
    crossing, one in clear view and one genuinely hidden behind a large parked
    occluder."""
    cfg = scenarios_cfg or load_yaml("scenarios.yaml")
    sc = cfg["scenario_types"]["multi_occlusion"]
    rng = random.Random(cfg["episode"]["seed_base"] + 1000 + seed)

    bp_lib = world.get_blueprint_library()
    ego, ego_wp = _spawn_ego(world, rng)

    try:
        occluder, occluder_wp, occ_dist = _spawn_occluder_ahead(
            world, bp_lib, ego_wp, sc["occluder_blueprint_pool"],
            sc["occluder_offset_range_m"], sc["jitter_lateral_m"], rng)
    except RuntimeError:
        ego.destroy()
        raise

    world.tick()   # commit ego/occluder transforms before reading them; see Scenario A

    right = occluder_wp.transform.get_right_vector()
    # NOTE: `sc["crossing_actor_speed_ms"]` (config) is no longer read here --
    # walker speed comes entirely from `sample_behaviour`'s own per-episode
    # sampling below, a leftover from before the walker_behavior.py rework
    # when a single fixed speed drove every walker for the whole episode.

    hidden_bp_name = rng.choice(sc["hidden_actor_blueprint_pool"])
    hidden_walker = _spawn_walker_in_shadow(
        world, bp_lib, _ego_camera_location(ego, bev_cfg), occluder, hidden_bp_name, rng)

    # The second walker is deliberately clear of the shadow: the scenario's
    # point is one pedestrian the camera can see and one it cannot, converging
    # on the same crossing, so the pipeline must carry both.
    visible_bp_name = rng.choice(sc["visible_actor_blueprint_pool"])
    stagger = sc["visible_stagger_m"]
    visible_wp = _waypoint_ahead(ego_wp, max(3.0, occ_dist - stagger), rng)
    visible_right = visible_wp.transform.get_right_vector()
    visible_loc = carla.Location(
        visible_wp.transform.location.x - visible_right.x * 3.0,
        visible_wp.transform.location.y - visible_right.y * 3.0,
        visible_wp.transform.location.z + 1.0,
    )
    visible_walker = world.try_spawn_actor(bp_lib.filter(visible_bp_name)[0], carla.Transform(visible_loc))
    if visible_walker is not None:
        world.tick()   # commit the transform before anything reads it; see above

    if hidden_walker is None or visible_walker is None:
        ego.destroy()
        occluder.destroy()
        if hidden_walker is not None:
            hidden_walker.destroy()
        if visible_walker is not None:
            visible_walker.destroy()
        raise RuntimeError("Failed to spawn one or both crossing walkers")

    # Opposite heading vectors, but the two walkers start on opposite sides of
    # the lane -- the hidden one right of centre (in the occluder's shadow), the
    # visible one left of it -- so both are walking *inward* and they converge
    # on the same crossing, as the scenario intends.
    hidden_dir = carla.Vector3D(-right.x, -right.y, 0.0)
    visible_dir = carla.Vector3D(visible_right.x, visible_right.y, 0.0)

    # Independently sampled behaviours: the point of the scenario is partly
    # that the pipeline must carry two actors whose intents differ, so one may
    # cross while the other turns back.
    vru_cfg = cfg.get("vru_behaviour")
    drivers = [
        WalkerDriver(hidden_walker, hidden_dir,
                     sample_behaviour(rng, vru_cfg,
                                       road_edge_m=_distance_to_path_edge(
                                           hidden_walker, ego_wp, hidden_dir))),
        WalkerDriver(visible_walker, visible_dir,
                     sample_behaviour(rng, vru_cfg,
                                       road_edge_m=_distance_to_path_edge(
                                           visible_walker, ego_wp, visible_dir))),
    ]

    return MultiOcclusionActors(ego=ego, occluder=occluder, visible_walker=visible_walker,
                                 hidden_walker=hidden_walker, seed=seed, drivers=drivers)


@dataclass
class BlindspotActors:
    ego: carla.Actor
    cutin_vehicle: carla.Actor
    trigger_distance_m: float
    ego_speed_ms: float
    cutin_speed_ms: float
    merge_ticks: int
    cutin_ref_wp: carla.Waypoint  # last resolved right-lane waypoint; see step_blindspot
    # Longitudinal position relative to the ego along the ego's lane, metres.
    # Negative is behind. This, not an independent waypoint chain, is what
    # places the cut-in vehicle each tick -- see step_blindspot for why.
    cutin_offset_m: float = field(default=0.0)
    max_lateral_separation_m: float = field(default=8.0)
    triggered: bool = field(default=False)
    merge_progress: int = field(default=0)


# Ticks a blind-spot episode runs for when its config does not say. Matches
# collect_dataset's --max-steps for staged scenarios.
BLINDSPOT_DEFAULT_STEPS = 65


def _right_lane_persists(start_wp: carla.Waypoint, distance_m: float,
                          step_m: float = 5.0) -> bool:
    """Does a same-direction driving lane stay to the right for `distance_m`?

    Checking only the spawn waypoint is not enough, and the difference is not
    marginal. Scenario B drives the ego 50-60 m from its spawn; on
    Town10HD_Opt a dual-carriageway stretch routinely narrows to a single lane
    well inside that. `step_blindspot` then has nowhere to put the cut-in
    vehicle and correctly refuses to teleport it onto a pavement or into
    oncoming traffic -- but that surfaces as a mid-episode failure, and five in
    a row abort the scenario's whole quota. Measured on a live collection
    chunk: blind-spot delivered 42 frames of a 300-frame target, four of its
    five failures being exactly this.

    Walking the lane at spawn time converts a mid-episode abort into a spawn
    candidate that is simply not chosen.
    """
    wp = start_wp
    walked = 0.0
    while walked < distance_m:
        nxt = wp.next(step_m)
        if not nxt:
            return False
        wp = nxt[0]
        walked += step_m
        right = wp.get_right_lane()
        if right is None or right.lane_type != carla.LaneType.Driving:
            return False
        if right.lane_id * wp.lane_id <= 0:      # oncoming, not a blind spot
            return False
    return True


def spawn_blindspot_scenario(client, world, seed: int,
                              scenarios_cfg: dict | None = None, tm_port: int = 8000) -> BlindspotActors:
    """Scenario B -- Blind-Spot Vehicle Cut-In: ego cruises at a fixed speed
    on a straight road; a second vehicle spawns in the rear-lateral blind spot
    (adjacent lane, behind) and merges into the ego's lane once it closes to
    `trigger_distance_m`.

    Both vehicles are driven kinematically (physics disabled, transform set
    directly each tick via `step_blindspot`) rather than through
    autopilot/Traffic Manager: TM's `vehicle_percentage_speed_difference`
    proved unreliable on this Town04 highway section (both vehicles were
    observed accelerating past 100 km/h instead of holding ~60-70 km/h,
    likely because this stretch's lanes aren't tagged with a real speed-limit
    landmark so `get_speed_limit()` returns a bogus fallback). Direct
    transform control gives exact, repeatable speeds and sidesteps that
    entirely -- the same scripted-motion approach already used for the
    pedestrian walkers in Scenarios A/C.
    """
    cfg = scenarios_cfg or load_yaml("scenarios.yaml")
    sc = cfg["scenario_types"]["blindspot_cutin"]
    rng = random.Random(cfg["episode"]["seed_base"] + 2000 + seed)

    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    spawn_points = carla_map.get_spawn_points()

    # How much road the episode will actually consume: the ego covers
    # speed x steps x dt, and the cut-in starts behind it. `episode.max_steps`
    # is NOT the number to use -- it is 300, while collect_dataset runs these
    # staged scenarios at 65 -- so the tick count comes from this scenario's
    # own config with a conservative default.
    dt = world.get_settings().fixed_delta_seconds or 0.1
    steps = sc.get("max_steps", BLINDSPOT_DEFAULT_STEPS)
    route_needed_m = ((sc["ego_speed_kmh"] / 3.6) * steps * dt
                       + sc["cutin_start_behind_m"])

    ego_bp = bp_lib.filter("vehicle.lincoln.mkz")[0]  # CARLA 0.10.0 catalog has no Tesla
    ego = None
    ego_wp = None
    def _try_spawn(required_m: float):
        """One pass of 40 candidates at a given route-length requirement."""
        for _ in range(40):
            t = rng.choice(spawn_points)
            wp = carla_map.get_waypoint(t.location)
            # Need a same-direction lane to the right (blind-spot lane --
            # lane_id sign must match, otherwise get_right_lane() is the
            # ONCOMING lane on a simple two-lane road) and road ahead.
            right_lane = wp.get_right_lane()
            if right_lane is None or right_lane.lane_type != carla.LaneType.Driving:
                continue
            if right_lane.lane_id * wp.lane_id <= 0:
                continue          # opposite-direction lane, not a blind spot
            if not wp.next(40.0):
                continue
            # The blind-spot lane must survive the whole drive, not merely
            # exist at the spawn point.
            if not _right_lane_persists(wp, required_m):
                continue
            actor = world.try_spawn_actor(ego_bp, t)
            if actor is not None:
                return actor, wp
        return None, None

    # Preferred requirement first, then progressively shorter. A long
    # dual-carriageway run is scarce on Town10HD_Opt, and demanding one
    # unconditionally would trade mid-episode aborts for spawn failures --
    # no improvement. Taking the longest stretch actually available keeps the
    # scenario spawnable while still avoiding most lane-runs-out aborts.
    for required_m in (route_needed_m, route_needed_m * 0.6, 25.0):
        ego, ego_wp = _try_spawn(required_m)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError(
            "Failed to spawn ego with a same-direction right lane after 40 "
            "attempts at each of three route-length requirements")
    ego.set_simulate_physics(False)

    right_wp = ego_wp.get_right_lane()
    behind_wp = right_wp.previous(sc["cutin_start_behind_m"])
    behind_wp = behind_wp[0] if behind_wp else right_wp

    def _spawn_transform(wp):
        t = wp.transform
        return carla.Transform(carla.Location(t.location.x, t.location.y, t.location.z + 0.3), t.rotation)

    cutin_bp = _pick_blueprint(bp_lib, sc["cutin_blueprint_pool"], rng)
    cutin_vehicle = world.try_spawn_actor(cutin_bp, _spawn_transform(behind_wp))
    attempts = 0
    while cutin_vehicle is None and attempts < 6:
        behind_wp = behind_wp.previous(3.0)
        behind_wp = behind_wp[0] if behind_wp else right_wp
        cutin_vehicle = world.try_spawn_actor(cutin_bp, _spawn_transform(behind_wp))
        attempts += 1
    if cutin_vehicle is None:
        ego.destroy()
        raise RuntimeError("Failed to spawn cut-in vehicle after retries")
    cutin_vehicle.set_simulate_physics(False)

    return BlindspotActors(
        ego=ego, cutin_vehicle=cutin_vehicle,
        trigger_distance_m=sc["cutin_trigger_distance_m"],
        ego_speed_ms=sc["ego_speed_kmh"] / 3.6,
        cutin_speed_ms=sc["cutin_speed_kmh"] / 3.6,
        merge_ticks=sc.get("merge_ticks", 15),
        cutin_ref_wp=behind_wp,
        cutin_offset_m=-float(sc["cutin_start_behind_m"]),
        max_lateral_separation_m=float(sc.get("max_lateral_separation_m", 8.0)),
    )


def step_blindspot(world: carla.World, actors: BlindspotActors, dt: float) -> None:
    """Advances both vehicles one tick of kinematic motion. Call once per
    `world.tick()` (e.g. as the `on_tick` callback passed to
    `data_collector.record_episode`).

    The cut-in vehicle's lane-following reference (`cutin_ref_wp`) is always
    advanced from its own previous value via `.next()`, never re-derived from
    the vehicle's actual (already laterally-blended) transform via
    `get_waypoint(get_location())` -- doing that after the merge starts snaps
    onto whichever lane centerline is nearest, and re-applying
    `get_left_lane()` to an already-shifted position compounds every tick,
    drifting the vehicle further left forever instead of settling into the
    ego's lane.
    """
    carla_map = world.get_map()

    ego_wp = carla_map.get_waypoint(actors.ego.get_location())
    ego_next = ego_wp.next(actors.ego_speed_ms * dt)
    if ego_next:
        actors.ego.set_transform(ego_next[0].transform)
    ego_wp = carla_map.get_waypoint(actors.ego.get_location())

    # The cut-in vehicle is positioned by its longitudinal OFFSET from the ego
    # along the ego's own lane, rather than by advancing an independent
    # waypoint chain.
    #
    # The independent chain was a real bug, not a stylistic matter: ego and
    # cut-in each called `.next()[0]` on their own waypoint, and `.next()`
    # returns branches in arbitrary order at a junction. On Town10HD_Opt --
    # dense with junctions, and the ego covers ~70 m per episode -- the two
    # routinely took different branches and drove apart, which accounted for
    # roughly a third of this scenario's empty frames on top of the timing
    # problem. Deriving one from the other makes divergence structurally
    # impossible.
    actors.cutin_offset_m += (actors.cutin_speed_ms - actors.ego_speed_ms) * dt
    offset = actors.cutin_offset_m
    if offset >= 0.0:
        chain = ego_wp.next(max(offset, 0.01))
    else:
        chain = ego_wp.previous(max(-offset, 0.01))
    base_wp = chain[0] if chain else ego_wp

    right_wp = base_wp.get_right_lane()
    if right_wp is None or right_wp.lane_id * base_wp.lane_id <= 0:
        # The blind-spot lane ran out (or the only neighbour is oncoming).
        # Abort rather than quietly teleporting the vehicle into oncoming
        # traffic or onto a pavement; _run_quota logs and reseeds the episode.
        raise RuntimeError("Blind-spot lane no longer available alongside the ego")
    actors.cutin_ref_wp = right_wp

    if not actors.triggered:
        # Trigger on longitudinal alignment (projected onto ego's forward
        # axis), not raw 3D distance: a euclidean check fires while the
        # cut-in vehicle is still diagonally behind in the adjacent lane, so
        # by the time the merge finishes it's settled in *behind* the ego
        # (out of the forward camera's view) instead of cutting in ahead of
        # it. Triggering once it's caught up to within `trigger_distance_m`
        # of being alongside/ahead means the merge -- plus its remaining
        # speed advantage during `merge_ticks` -- lands it visibly in front.
        ego_t = actors.ego.get_transform()
        forward = ego_t.get_forward_vector()
        delta = actors.cutin_vehicle.get_location() - ego_t.location
        longitudinal_offset = delta.x * forward.x + delta.y * forward.y
        if longitudinal_offset >= -actors.trigger_distance_m:
            actors.triggered = True

    if actors.triggered and actors.merge_progress < actors.merge_ticks:
        actors.merge_progress += 1

    lateral_frac = min(1.0, actors.merge_progress / actors.merge_ticks)
    if lateral_frac > 0.0:
        left_wp = right_wp.get_left_lane() or right_wp
        loc = carla.Location(
            right_wp.transform.location.x + (left_wp.transform.location.x - right_wp.transform.location.x) * lateral_frac,
            right_wp.transform.location.y + (left_wp.transform.location.y - right_wp.transform.location.y) * lateral_frac,
            right_wp.transform.location.z + 0.3,
        )
        actors.cutin_vehicle.set_transform(carla.Transform(loc, right_wp.transform.rotation))
    else:
        loc = carla.Location(right_wp.transform.location.x, right_wp.transform.location.y,
                              right_wp.transform.location.z + 0.3)
        actors.cutin_vehicle.set_transform(carla.Transform(loc, right_wp.transform.rotation))

    # Belt and braces on top of the offset-based positioning above: if the two
    # have somehow ended up far apart laterally (an unusual road layout, a
    # multi-lane junction), the episode is no longer the scenario it claims to
    # be. Fail it so it is logged and reseeded rather than silently recorded as
    # a blind-spot cut-in that never happened.
    ego_loc = actors.ego.get_location()
    cut_loc = actors.cutin_vehicle.get_location()
    fwd = actors.ego.get_transform().get_forward_vector()
    dx, dy = cut_loc.x - ego_loc.x, cut_loc.y - ego_loc.y
    lateral = abs(-dx * fwd.y + dy * fwd.x)
    if lateral > actors.max_lateral_separation_m:
        raise RuntimeError(
            f"Ego and cut-in vehicle diverged: {lateral:.1f} m apart laterally "
            f"(limit {actors.max_lateral_separation_m:.1f} m)")


def destroy_actors(*actors: carla.Actor) -> None:
    for actor in actors:
        if actor is not None and actor.is_alive:
            actor.destroy()
