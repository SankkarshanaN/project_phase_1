"""Open-world driving setup: autopilot ego, background traffic, and staged
occlusion encounters along whatever route the ego happens to take.

Promoted out of `scripts/live_demo.py`, where working implementations of the
first two lived but could not be reached by dataset collection. The demo and
the dataset now come from one implementation, so what the dashboard shows and
what the metrics are computed from cannot drift apart.

The encounter manager is new. Staged scenarios park the ego and build a
vignette around it; here the ego drives the city under Traffic Manager control
and occlusion encounters are constructed ahead of it as it goes. That matters
for two reasons beyond realism:

  - A moving ego makes the crossing question a real collision-risk question
    with a finite time-to-arrival, rather than "will they enter this box".
  - Scene diversity comes free. The previous dataset had ~119 distinct scenes
    sampled ~70 times each, which is what made the per-crop train/validation
    split leak so badly. A driving ego never frames the same scene twice.

Because the autopilot's route is not known in advance, occluders cannot be
placed up front. The manager watches the ego's current lane each tick, stages
an encounter when a suitable stretch appears far enough ahead, arms the
walker's behaviour, triggers it when the ego closes, and tears it down once it
is behind.
"""
import math
import random
from dataclasses import dataclass, field

import carla

from carla_tools.scenario_gen import (
    _distance_to_path_edge, _ego_camera_location, _front_corner_shadow_position,
    _shadow_covers, _shadow_position,
)
from carla_tools.walker_behavior import WalkerDriver, sample_behaviour
from common.config import load_yaml

DEFAULT_TM_PORT = 8000

# How far the ego must travel before a FAILED staging attempt is retried.
# Attempts fail routinely (junctions, bends, no kerb space); retrying every
# tick turns a normal failure into a per-tick flurry of map queries.
RETRY_COOLDOWN_M = 8.0

# Blueprints large enough to actually hide a pedestrian. Verified against
# CARLA 0.10.0's catalog, which replaced 0.9.x's vehicle set entirely.
OCCLUDER_BLUEPRINTS = [
    "vehicle.fuso.mitsubishi",     # bus
    "vehicle.sprinter.mercedes",   # large van
    "vehicle.carlacola.actors",    # box truck
    "vehicle.firetruck.actors",
]

WALKER_BLUEPRINTS = ["walker.pedestrian.00*"]

# Widest half-width among OCCLUDER_BLUEPRINTS. Used to park the occluder
# provisionally before its real bounding box can be read back (that needs a
# tick, and the manager must not tick -- see `_finish_staging`).
#
# Measured directly (world.try_spawn_actor + bounding_box.extent.y) rather
# than assumed: vehicle.fuso.mitsubishi (the bus) is 1.964m, not the 1.5m
# this constant previously assumed -- sprinter.mercedes 0.994m,
# carlacola.actors 1.456m, firetruck.actors 1.451m are all narrower and were
# never the binding case. At the old 1.5m the bus's provisional kerb spot
# undershot its real half-width by 0.46m, so `world.try_spawn_actor` failed
# on collision with sidewalk geometry whenever the bus was drawn -- 100% of
# the time in a direct test, confirmed by `stage_at_waypoint` returning
# False for every 'vehicle.fuso.mitsubishi' draw while other blueprints at
# the same waypoints succeeded. Silent in the dynamic staging path
# (`_stage`), which just retries at the next ahead-position on failure, but
# fatal for a fixed course (`live_demo.build_fixed_course`), where a failed
# waypoint is simply lost rather than retried.
PROVISIONAL_HALF_WIDTH_M = 2.0
KERB_CLEARANCE_M = 0.3

# The ego is judged stalled below this speed. Town10HD's autopilot legitimately
# waits at lights, so the stall must persist before an encounter is blamed.
STALL_SPEED_MS = 0.5
STALL_TIMEOUT_S = 12.0

# Ego autopilot tuning -- see `_tune_ego_autopilot`. Negative speed difference
# means faster than the posted limit.
EGO_IGNORE_LIGHTS_PCT = 60.0
EGO_SPEED_DIFF_PCT = -15.0


def spawn_ego_with_autopilot(world, rng: random.Random, tm_port: int = DEFAULT_TM_PORT,
                              tm=None):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    ego_bp = bp_lib.filter("vehicle.lincoln.mkz")[0]   # 0.10.0 has no Tesla
    for _ in range(20):
        ego = world.try_spawn_actor(ego_bp, rng.choice(spawn_points))
        if ego is not None:
            ego.set_autopilot(True, tm_port)
            if tm is not None:
                _tune_ego_autopilot(tm, ego)
            return ego
    raise RuntimeError("Failed to spawn ego after 20 attempts")


def _tune_ego_autopilot(tm, ego) -> None:
    """Keeps the ego rolling, because a parked ego collects nothing.

    Measured on a probe drive with the stock Traffic Manager settings: the ego
    was stationary for 54% of ticks even with no encounter anywhere near it,
    and covered 64-145 m in a 40 s episode. Town10HD_Opt is small and heavily
    signalled, so most of that was red lights -- ticks that cost the same to
    simulate and record as any other but contain no occlusion, no crossing and
    no closing speed.

    `ignore_lights_percentage` is the knob that matters. It is set part-way
    rather than to 100 so the dataset still contains junction approaches and
    stops; TM keeps its collision avoidance either way, so the ego still yields
    to cross traffic rather than driving through it.
    """
    tm.ignore_lights_percentage(ego, EGO_IGNORE_LIGHTS_PCT)
    tm.vehicle_percentage_speed_difference(ego, EGO_SPEED_DIFF_PCT)
    tm.distance_to_leading_vehicle(ego, 3.0)


def spawn_background_traffic(client, world, rng: random.Random, n_vehicles: int = 20,
                              n_walkers: int = 15, tm_port: int = DEFAULT_TM_PORT):
    """Populates the map with autopilot vehicles and AI-controlled walkers.

    Walkers are spawned BEFORE vehicles, deliberately. Spawning both together
    produced roughly 2 of 15 walkers: by the time walkers were attempted every
    vehicle spawn point was occupied and blocking nearby nav-mesh locations.
    Nav locations are also over-sampled, since many random draws land somewhere
    `try_spawn_actor` still rejects (steep terrain, inside geometry).
    """
    bp_lib = world.get_blueprint_library()

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walkers, attempts = [], 0
    while len(walkers) < n_walkers and attempts < n_walkers * 4:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(rng.choice(walker_bps), carla.Transform(loc))
        if w is not None:
            walkers.append(w)
    world.tick()

    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicles = []
    for sp in spawn_points[:n_vehicles]:
        v = world.try_spawn_actor(rng.choice(vehicle_bps), sp)
        if v is not None:
            v.set_autopilot(True, tm_port)
            vehicles.append(v)

    controller_bp = bp_lib.find("controller.ai.walker")
    controllers = []
    for w in walkers:
        c = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=w)
        if c is not None:
            controllers.append(c)
    world.tick()

    for c in controllers:
        c.start()
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            c.go_to_location(dest)
        c.set_max_speed(0.9 + rng.random() * 0.8)

    return vehicles, walkers, controllers


@dataclass
class Encounter:
    """One staged occlusion event travelling with the ego."""
    id: int
    occluder: carla.Actor
    walker: carla.Actor
    driver: WalkerDriver
    staged_at_m: float          # ego odometer reading when it was staged
    triggered: bool = False
    phase: str = "staged"       # staged -> triggered -> passed / abandoned

    def actors(self):
        return [a for a in (self.occluder, self.walker) if a is not None]


class EncounterManager:
    """Stages occlusion encounters ahead of a roaming ego.

    Encounters are constructed, not hoped for: the walker is placed on the ray
    from the camera through the occluder's body, beyond its far face, and the
    occluder's angular size is checked against the walker's before committing.
    The alternative -- scattering parked vehicles and waiting for the geometry
    to line up -- is what the staged scenarios did, and it produced a dataset
    with no fully-occluded frames at all.
    """

    def __init__(self, world, ego, rng: random.Random, scenarios_cfg: dict | None = None,
                  bev_cfg: dict | None = None,
                  stage_ahead_m: float = 35.0,
                  trigger_within_m: float = 22.0,
                  despawn_behind_m: float = 15.0,
                  min_gap_m: float = 40.0,
                  max_active: int = 2):
        self.world = world
        self.ego = ego
        self.rng = rng
        self.cfg = scenarios_cfg or load_yaml("scenarios.yaml")
        self.bev_cfg = bev_cfg or load_yaml("bev.yaml")
        self.stage_ahead_m = stage_ahead_m
        self.trigger_within_m = trigger_within_m
        self.despawn_behind_m = despawn_behind_m
        self.min_gap_m = min_gap_m
        self.max_active = max_active

        self.active: list[Encounter] = []
        self.completed = 0
        self._next_id = 0
        self._odometer = 0.0
        self._last_loc = ego.get_location()
        self._last_stage_at = -1e9
        self._next_stage_attempt_m = -1e9
        self._pending: tuple | None = None    # occluder awaiting a tick, see _stage
        self._stalled_s = 0.0

        # Fetched ONCE. `world.get_map()` serialises the entire OpenDRIVE map
        # from server to client on every call; doing it per tick dropped
        # collection from ~10 FPS to ~0.2 FPS, a 50x slowdown that made a
        # 4,000-frame run a six-hour job. The map is static for the session, so
        # there is nothing to re-fetch.
        self.carla_map = world.get_map()

    # ------------------------------------------------------------------ tick

    def step(self, dt: float) -> None:
        self._advance_odometer()
        self._track_stall(dt)
        # Completes staging begun on the PREVIOUS tick, so the occluder's
        # bounding box and transform are readable.
        if self._pending is not None:
            self._finish_staging()
        for enc in list(self.active):
            self._update(enc, dt)
        if self._should_stage():
            self._stage()

    def _track_stall(self, dt: float) -> None:
        v = self.ego.get_velocity()
        if math.hypot(v.x, v.y) < STALL_SPEED_MS:
            self._stalled_s += dt
        else:
            self._stalled_s = 0.0

    def _advance_odometer(self) -> None:
        loc = self.ego.get_location()
        self._odometer += math.hypot(loc.x - self._last_loc.x, loc.y - self._last_loc.y)
        self._last_loc = loc

    def _longitudinal_offset(self, actor) -> float:
        """Signed distance ahead of the ego along its heading."""
        t = self.ego.get_transform()
        fwd = t.get_forward_vector()
        d = actor.get_location() - t.location
        return d.x * fwd.x + d.y * fwd.y

    def _update(self, enc: Encounter, dt: float) -> None:
        offset = self._longitudinal_offset(enc.occluder)

        if not enc.triggered and offset <= self.trigger_within_m:
            enc.triggered = True
            enc.phase = "triggered"
        if enc.triggered:
            enc.driver.step(dt)

        if offset < -self.despawn_behind_m:
            enc.phase = "passed"
            self._despawn(enc)
            return

        # A parked occluder the ego cannot get past deadlocks the whole
        # episode: the autopilot queues behind it, the odometer stops, no
        # further encounter is ever staged, and every remaining frame shows a
        # stationary ego looking at a stationary bus. Measured on v2 data
        # before the kerb-offset fix: the ego was below 0.1 m/s for 79% of
        # urban frames and every episode ended still in `triggered`. The
        # offset fix should prevent it; this clears up the remainder (a lorry
        # that settles badly, a narrow lane) rather than losing the episode.
        if self._stalled_s > STALL_TIMEOUT_S and offset > 0.0:
            enc.phase = "abandoned"
            self._despawn(enc)
            self._stalled_s = 0.0
            self._next_stage_attempt_m = self._odometer + RETRY_COOLDOWN_M

    def _should_stage(self) -> bool:
        # `_next_stage_attempt_m` throttles RETRIES, separately from the
        # spacing between successful encounters. Staging legitimately fails
        # often -- the road ahead bends, or is a junction, or has no room --
        # and without a cooldown a run of failures re-attempts every single
        # tick, each attempt doing map and waypoint queries. That, not the
        # successful staging, was the bulk of the cost.
        return (self._pending is None
                and len(self.active) < self.max_active
                and self._odometer - self._last_stage_at >= self.min_gap_m
                and self._odometer >= self._next_stage_attempt_m)

    # ----------------------------------------------------------------- stage

    def _stage(self) -> None:
        """Parks an occluder at the kerb ahead. Failure is normal -- the road
        ahead may bend, or be a junction, or have no room -- so it returns
        quietly and the next attempt happens further along.

        The walker is NOT spawned here. Placing it requires reading the
        occluder's transform and bounding box back from the server, which is
        only valid after a tick, and the manager runs inside the collection
        loop's tick and must not add one of its own (an extra tick pushes an
        extra frame into every sensor queue and desynchronises the rig). So the
        occluder is held in `_pending` and the walker goes down next tick.
        """
        bp_lib = self.world.get_blueprint_library()

        # Retry no sooner than this much further along, whatever happens below.
        self._next_stage_attempt_m = self._odometer + RETRY_COOLDOWN_M

        ego_wp = self.carla_map.get_waypoint(self.ego.get_location())
        ahead = ego_wp.next(self.stage_ahead_m)
        if not ahead:
            return
        wp = ahead[0]
        # Junctions have no stable kerb to park against and the ego's route
        # through them is unpredictable, so the encounter would often end up
        # off to one side and never be seen.
        if wp.is_junction:
            return

        occ_loc = self._kerb_location(wp, PROVISIONAL_HALF_WIDTH_M)
        occ_bp = bp_lib.filter(self.rng.choice(OCCLUDER_BLUEPRINTS))
        if not occ_bp:
            return
        occluder = self.world.try_spawn_actor(
            occ_bp[0], carla.Transform(occ_loc, wp.transform.rotation))
        if occluder is None:
            return
        occluder.set_simulate_physics(False)
        self._pending = (occluder, wp)

    def stage_at_waypoint(self, wp) -> bool:
        """Stages an occluder at a caller-supplied waypoint, bypassing the
        "35m ahead of the ego" lookup `_stage` does.

        For a fixed, deterministic course built once before the drive starts
        -- the caller already has its own list of waypoints spread along the
        route and wants an occluder parked at each, not one staged reactively
        as the ego happens to approach. Reuses the same kerb-placement and
        two-tick seat/pair sequence as the dynamic path (`_stage` /
        `_finish_staging`) so a fixed course gets the identical, already-
        proven placement geometry -- only the choice of *where* differs.

        Returns False (and spawns nothing) for a junction waypoint, same
        reasoning as `_stage`: no stable kerb, unpredictable route through it.
        Caller must `world.tick()` then call `_finish_staging()` before
        staging the next one -- the pending occluder's real bounding box
        cannot be read back until a tick has passed.
        """
        if wp.is_junction:
            return False
        bp_lib = self.world.get_blueprint_library()
        occ_loc = self._kerb_location(wp, PROVISIONAL_HALF_WIDTH_M)
        occ_bp = bp_lib.filter(self.rng.choice(OCCLUDER_BLUEPRINTS))
        if not occ_bp:
            return False
        occluder = self.world.try_spawn_actor(
            occ_bp[0], carla.Transform(occ_loc, wp.transform.rotation))
        if occluder is None:
            return False
        occluder.set_simulate_physics(False)
        self._pending = (occluder, wp)
        return True

    def _kerb_location(self, wp, half_width_m: float) -> carla.Location:
        """A parking spot to the right of `wp`, fully clear of its lane.

        The offset used to be a flat 1.6 m from lane centre. Town10HD's lanes
        are ~3.5 m wide and the occluders are 2.5-2.8 m across, so a bus parked
        at 1.6 m straddled the centre of the ego's own lane and blocked it --
        the single defect responsible for the ego standing still through most
        of the flagship scenario. Clearing the lane edge by the occluder's own
        half-width is the fix, and it is also what a kerbside park looks like.
        """
        right = wp.transform.get_right_vector()
        d = wp.lane_width * 0.5 + half_width_m + KERB_CLEARANCE_M
        return carla.Location(
            wp.transform.location.x + right.x * d,
            wp.transform.location.y + right.y * d,
            wp.transform.location.z + 0.3,
        )

    def _finish_staging(self) -> None:
        """Second half of `_stage`, one tick later: re-seat the occluder using
        its real width, then place the walker in its shadow."""
        occluder, wp = self._pending
        self._pending = None
        if not occluder.is_alive:
            return

        # Now that a tick has passed the bounding box is real, so the
        # provisional half-width can be replaced with the actual one. A
        # firetruck and a van differ by ~0.4 m across, which is the difference
        # between clearing the lane and clipping it.
        half_w = float(occluder.bounding_box.extent.y)
        loc = self._kerb_location(wp, half_w)
        occluder.set_transform(carla.Transform(loc, wp.transform.rotation))

        bp_lib = self.world.get_blueprint_library()
        placed = self._spawn_hidden_walker(bp_lib, occluder, occluder_loc=loc)
        if placed is None:
            occluder.destroy()
            return
        walker, walker_loc = placed

        right = wp.transform.get_right_vector()
        direction = carla.Vector3D(-right.x, -right.y, 0.0)
        params = sample_behaviour(
            self.rng, self.cfg.get("vru_behaviour"),
            road_edge_m=_distance_to_path_edge(walker, wp, direction, loc=walker_loc))
        driver = WalkerDriver(walker, direction, params)

        self.active.append(Encounter(id=self._next_id, occluder=occluder, walker=walker,
                                      driver=driver, staged_at_m=self._odometer))
        self._next_id += 1
        self._last_stage_at = self._odometer

    def _spawn_hidden_walker(self, bp_lib, occluder, occluder_loc=None):
        """Places a walker genuinely inside the occluder's shadow.

        Tries the classic "about to step out from in front of a parked bus"
        placement first -- near the occluder's own front-right corner,
        deliberately, not as a side effect of the ego's current viewing
        angle -- and falls back to the camera-ray placement (which can hide
        a walker anywhere along the occluder's length, not just the front)
        only if that fails `_shadow_covers` a few times in a row, so an
        unusual occluder shape or approach angle still gets an encounter
        rather than none at all.

        Returns `(walker, location)` -- the caller needs the location because
        the actor's own transform is not readable until the next tick.
        """
        cam = _ego_camera_location(self.ego, self.bev_cfg)
        bp_name = self.rng.choice(WALKER_BLUEPRINTS)
        candidates = bp_lib.filter(bp_name)
        if not candidates:
            return None
        walker_bp = self.rng.choice(candidates)

        for attempt in range(6):
            clearance = self.rng.uniform(1.0, 2.5) + attempt * 0.75
            if attempt < 4:
                loc = _front_corner_shadow_position(occluder, clearance)
            else:
                loc, _dist = _shadow_position(cam, occluder, clearance)
            loc.z += 1.0
            if not _shadow_covers(cam, occluder, loc):
                continue
            w = self.world.try_spawn_actor(walker_bp, carla.Transform(loc))
            if w is not None:
                return w, loc
        return None

    # --------------------------------------------------------------- cleanup

    def _despawn(self, enc: Encounter) -> None:
        for a in enc.actors():
            if a is not None and a.is_alive:
                a.destroy()
        if enc in self.active:
            self.active.remove(enc)
        self.completed += 1

    def destroy_all(self) -> None:
        for enc in list(self.active):
            self._despawn(enc)
        if self._pending is not None:
            occluder, _ = self._pending
            if occluder is not None and occluder.is_alive:
                occluder.destroy()
            self._pending = None

    # ---------------------------------------------------------------- status

    def current_phase(self) -> tuple[int, str]:
        """(encounter_id, phase) for the nearest active encounter ahead, or
        (-1, "none"). Recorded per frame so evaluation can slice to the frames
        where something was actually happening rather than averaging over
        minutes of empty street."""
        best, best_off = None, 1e9
        for enc in self.active:
            off = self._longitudinal_offset(enc.occluder)
            if -self.despawn_behind_m < off < best_off:
                best, best_off = enc, off
        return (best.id, best.phase) if best else (-1, "none")
