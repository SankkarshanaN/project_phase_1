"""Per-tick behaviour state machines for crossing pedestrians.

Why this exists
---------------
`sensors.set_walker_velocity` was previously called exactly once, at spawn
(`scenario_gen.py`), and never again: every walker in the dataset travelled in
a straight line at a constant 1.2 m/s for the whole episode. Against actors
like that, a constant-velocity predictor is not merely good, it is *exact*, and
any reported crossing-prediction accuracy measures the scripting rather than
the method. The same flaw that made the detection task trivial -- data too
clean to distinguish approaches -- would have reappeared one level up.

So walkers here vary in three ways that matter to a predictor:

  - **Speed**, sampled per episode over a realistic pedestrian range rather
    than fixed. A child and a hurrying adult differ by a factor of three.
  - **Non-constant velocity**: hesitating at the kerb, accelerating mid-road.
    These break the constant-velocity assumption in exactly the way real
    pedestrians do, and the evaluation reports accuracy per behaviour so the
    degradation is visible instead of averaged away.
  - **Negatives.** `approach_abort` walks to the kerb and turns back. Without
    actors who approach and do *not* cross, "will they cross?" has only one
    answer and a predictor that always says yes scores perfectly.

The state machine is deliberately separate from CARLA. `CrossingBehaviour` is
pure -- it maps (elapsed time, distance travelled) to a commanded velocity --
so it can be tested without a running simulator. `WalkerDriver` is the thin
part that reads an actor's position and applies the control.

Ground truth for "did they actually cross" is NOT taken from the behaviour
label. It is computed retrospectively from the recorded trajectory by
`scripts/label_crossing_intent.py`, because what a walker was scripted to do
and what actually happened can differ -- they collide with geometry, get stuck
on a kerb, or run out of episode. The behaviour name is recorded only so
results can be sliced by it.
"""
import math
import random
from dataclasses import dataclass

import carla

from carla_tools.sensors import set_walker_velocity

# Behaviour kinds. `approach_abort` and `stop_midway` are the ones that make
# the task non-trivial; without them every actor eventually crosses.
CROSS_STEADY = "cross_steady"
CROSS_HESITATE = "cross_hesitate"
CROSS_RUN = "cross_run"
APPROACH_ABORT = "approach_abort"
STOP_MIDWAY = "stop_midway"

ALL_KINDS = (CROSS_STEADY, CROSS_HESITATE, CROSS_RUN, APPROACH_ABORT, STOP_MIDWAY)

# Roughly 60/40 crossing to non-crossing. Weighted toward crossing because that
# is the safety-relevant case, but not so far that a predictor can profit from
# always answering yes.
DEFAULT_WEIGHTS = {
    CROSS_STEADY: 0.30,
    CROSS_HESITATE: 0.20,
    CROSS_RUN: 0.10,
    APPROACH_ABORT: 0.25,
    STOP_MIDWAY: 0.15,
}

# Pedestrian walking speeds, m/s. The low end is a child or an elderly walker,
# the high end a hurrying adult. The old fixed 1.2 sat in the middle and
# removed the variable entirely.
DEFAULT_SPEED_RANGE = (0.8, 2.5)

# States
WAIT, APPROACH, HESITATE, CROSS, ABORT, DONE = "WAIT", "APPROACH", "HESITATE", "CROSS", "ABORT", "DONE"


@dataclass
class BehaviourParams:
    kind: str
    speed_ms: float
    start_delay_s: float
    kerb_distance_m: float      # distance walked before the hesitate/abort decision
    hesitate_s: float
    stop_at_m: float            # where `stop_midway` gives up
    run_multiplier: float       # speed gain once `cross_run` accelerates


def sample_behaviour(rng: random.Random, cfg: dict | None = None,
                      road_edge_m: float | None = None) -> BehaviourParams:
    """Draws one episode's walker behaviour.

    `cfg` is the `vru_behaviour` block of configs/scenarios.yaml; defaults here
    apply when it is absent so existing scenarios keep working unchanged.

    `road_edge_m` is how far the walker must travel to reach the edge of the
    ego's path. Pass it whenever the scenario knows its own geometry, because
    the hesitate/abort decision point has to sit *before* that edge to mean
    anything: a walker who steps into the ego's lane and only then turns back
    has already entered the corridor, so the retrospective label counts them as
    a crosser and `approach_abort` stops being a negative at all. Measured with
    an absolute 1.5-3.0 m decision distance against a 2.25 m corridor edge,
    91% of *all* behaviours ended up entering the corridor and the negatives
    effectively vanished from the dataset.

    Without it the absolute ranges are used, which is fine for unit tests but
    leaves that coupling to chance.
    """
    cfg = cfg or {}
    weights = cfg.get("weights", DEFAULT_WEIGHTS)
    lo, hi = cfg.get("speed_range_ms", DEFAULT_SPEED_RANGE)

    kinds = list(weights.keys())
    kind = rng.choices(kinds, weights=[weights[k] for k in kinds], k=1)[0]

    if road_edge_m is not None and road_edge_m > 0.0:
        kerb = road_edge_m * rng.uniform(*cfg.get("kerb_fraction_range", (0.55, 0.9)))
        # stop_midway deliberately straddles the edge: stopping *in* the lane
        # (a pedestrian frozen in the road) is a real and dangerous case, and
        # should count as in-path, while stopping short of it should not.
        stop_at = road_edge_m * rng.uniform(*cfg.get("stop_fraction_range", (0.7, 1.6)))
    else:
        kerb = rng.uniform(*cfg.get("kerb_distance_range_m", (1.5, 3.0)))
        stop_at = rng.uniform(*cfg.get("stop_at_range_m", (2.0, 4.0)))

    return BehaviourParams(
        kind=kind,
        speed_ms=rng.uniform(lo, hi),
        # A delay means the walker is still hidden when recording starts, so
        # the episode captures the fully-occluded tier before anything moves.
        start_delay_s=rng.uniform(*cfg.get("start_delay_range_s", (0.0, 1.5))),
        kerb_distance_m=kerb,
        hesitate_s=rng.uniform(*cfg.get("hesitate_range_s", (0.5, 2.0))),
        stop_at_m=stop_at,
        run_multiplier=rng.uniform(*cfg.get("run_multiplier_range", (1.5, 2.2))),
    )


class CrossingBehaviour:
    """Pure state machine: (dt, distance travelled) -> commanded velocity.

    `distance_m` is signed, measured along the crossing direction from the
    start point, so a walker that turns back reports a decreasing value. Speed
    is returned as a magnitude with a separate direction sign rather than a
    signed speed, because `carla.WalkerControl` takes a direction vector and a
    non-negative speed.
    """

    def __init__(self, params: BehaviourParams):
        self.p = params
        self.state = WAIT
        self.elapsed = 0.0
        self._state_entered = 0.0

    def _enter(self, state: str) -> None:
        self.state = state
        self._state_entered = self.elapsed

    @property
    def time_in_state(self) -> float:
        return self.elapsed - self._state_entered

    def step(self, dt: float, distance_m: float) -> tuple[float, float]:
        """Advances one tick. Returns (speed_ms, direction_sign)."""
        self.elapsed += dt
        p = self.p

        if self.state == WAIT:
            if self.elapsed >= p.start_delay_s:
                self._enter(APPROACH)
            return 0.0, 1.0

        if self.state == APPROACH:
            if distance_m >= p.kerb_distance_m:
                if p.kind == CROSS_HESITATE:
                    self._enter(HESITATE)
                elif p.kind == APPROACH_ABORT:
                    # Pause at the kerb first, then turn back. The pause is what
                    # makes this genuinely hard: for a second or two it is
                    # indistinguishable from cross_hesitate.
                    self._enter(HESITATE)
                else:
                    self._enter(CROSS)
            return p.speed_ms, 1.0

        if self.state == HESITATE:
            if self.time_in_state >= p.hesitate_s:
                self._enter(ABORT if p.kind == APPROACH_ABORT else CROSS)
            return 0.0, 1.0

        if self.state == CROSS:
            if p.kind == STOP_MIDWAY and distance_m >= p.stop_at_m:
                self._enter(DONE)
                return 0.0, 1.0
            speed = p.speed_ms
            if p.kind == CROSS_RUN:
                speed *= p.run_multiplier
            return speed, 1.0

        if self.state == ABORT:
            # Back to where they started, then stay put.
            if distance_m <= 0.0:
                self._enter(DONE)
                return 0.0, -1.0
            return p.speed_ms, -1.0

        return 0.0, 1.0   # DONE

    @property
    def intends_to_cross(self) -> bool:
        """What the script intended. Diagnostic only -- the dataset's
        `will_cross` label is derived from the recorded trajectory, never from
        this, because intent and outcome can diverge."""
        return self.p.kind in (CROSS_STEADY, CROSS_HESITATE, CROSS_RUN)


class WalkerDriver:
    """Binds a `CrossingBehaviour` to a CARLA walker and a crossing direction.

    Hook `step` into `data_collector.record_episode`'s `on_tick`. It fires
    during warm-up ticks too, which is harmless and in fact useful: the
    behaviour's start delay runs down while the sensors settle.
    """

    def __init__(self, walker: carla.Actor, direction: carla.Vector3D,
                  params: BehaviourParams):
        self.walker = walker
        norm = math.hypot(direction.x, direction.y) or 1.0
        self.dir = (direction.x / norm, direction.y / norm)
        self.behaviour = CrossingBehaviour(params)
        start = walker.get_location()
        self._start = (start.x, start.y)

    @property
    def distance_travelled(self) -> float:
        """Signed distance along the crossing direction. Measured from the
        actor's real position rather than integrated from commanded speed, so
        it stays correct when CARLA blocks the walker on geometry."""
        loc = self.walker.get_location()
        return ((loc.x - self._start[0]) * self.dir[0]
                + (loc.y - self._start[1]) * self.dir[1])

    def step(self, dt: float) -> None:
        speed, sign = self.behaviour.step(dt, self.distance_travelled)
        set_walker_velocity(
            self.walker,
            carla.Vector3D(self.dir[0] * sign, self.dir[1] * sign, 0.0),
            speed,
        )
