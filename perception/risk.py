"""Component 7 -- Dynamic Risk Scoring Engine.

Turns perception into a decision: a risk score and an ADAS action, with the
reasoning attached.

What makes this different from conventional ADAS risk scoring
--------------------------------------------------------------
A standard collision-risk engine scores the hazards it can see, typically on
time-to-collision. Everything invisible contributes zero, because from the
system's point of view it does not exist. That is precisely the failure this
project targets: the child behind the bus scores zero risk right up until the
moment they step out, at which point the remaining reaction time is whatever is
left after the detector finally fires.

Risk here is computed from three sources, and the second is the contribution
that conventional systems structurally cannot make:

  1. **Visible hazards** -- tracked, with a crossing probability from
     `perception.intent`. Conventional.
  2. **Hidden hazards** -- particle-filter beliefs about targets currently
     behind an occluder, scored on emergence probability, predicted emergence
     location, and how that timing lines up with the ego's arrival. A hazard
     nobody can see still produces risk.
  3. **Occlusion burden** -- how much of the road ahead is blocked at all. This
     is not about any specific hazard; it is the cost of not knowing. Driving
     fast past a line of parked vehicles is risky even with nothing detected,
     and the engine says so by recommending a speed rather than an emergency
     action.

The third source is the "explains what it can't see" claim made operational: a
system that reports elevated risk with no detections attached, and can say why,
is doing something a detector-driven pipeline cannot.

Every score carries its contributing factors so a warning can be justified
rather than asserted.
"""
import math
from dataclasses import dataclass, field

# ADAS response levels, ordered. Chosen to mirror how production systems
# escalate (inform -> warn -> act) rather than a bare score, so the output maps
# onto something a vehicle could actually do.
NONE, INFORM, WARN, BRAKE = 0, 1, 2, 3
ACTION_NAMES = {NONE: "NONE", INFORM: "INFORM", WARN: "WARN", BRAKE: "BRAKE"}

# Risk thresholds for each escalation step.
ACTION_THRESHOLDS = ((BRAKE, 0.75), (WARN, 0.50), (INFORM, 0.25))

# Time-to-collision bands, seconds. 1.5 s is roughly human reaction time plus
# brake build-up, so inside it a warning is too late to be useful and the
# system should act.
TTC_CRITICAL = 1.5
TTC_WARNING = 3.0
TTC_HORIZON = 5.0

# Weights over the three risk sources. Visible hazards dominate because they
# are certain; hidden ones are discounted for the probability that the belief
# is wrong, not ignored.
W_VISIBLE, W_HIDDEN, W_OCCLUSION = 0.5, 0.35, 0.15

# Comfortable deceleration for a speed advisory, m/s^2. Well below the ~8 m/s^2
# a car can actually manage, because this is a "you should be going slower"
# suggestion, not an emergency stop.
COMFORT_DECEL = 2.5


@dataclass
class RiskFactor:
    """One contribution to the total, kept so the score can be explained."""
    source: str          # "visible" | "hidden" | "occlusion"
    score: float
    detail: str
    track_id: int | None = None


@dataclass
class RiskAssessment:
    score: float
    action: int
    factors: list = field(default_factory=list)
    advisory_speed_ms: float | None = None
    dominant: str = "none"

    @property
    def action_name(self) -> str:
        return ACTION_NAMES[self.action]

    def explain(self) -> str:
        """Plain-language justification, strongest factor first."""
        if not self.factors:
            return "No hazards and clear visibility."
        lines = [f"{self.action_name} (risk {self.score:.2f})"]
        for f in sorted(self.factors, key=lambda x: -x.score)[:4]:
            lines.append(f"  - [{f.source}] {f.detail} (+{f.score:.2f})")
        if self.advisory_speed_ms is not None:
            lines.append(f"  - advisory speed {self.advisory_speed_ms * 3.6:.0f} km/h "
                          f"given how much of the road is hidden")
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"risk={self.score:.2f} action={self.action_name} ({self.dominant})"


def _ttc_risk(distance_m: float, closing_speed_ms: float) -> float:
    """Risk from time-to-collision, 0 when there is plenty of time."""
    if closing_speed_ms <= 0.1:
        return 0.0
    ttc = distance_m / closing_speed_ms
    if ttc <= TTC_CRITICAL:
        return 1.0
    if ttc >= TTC_HORIZON:
        return 0.0
    # Linear between critical and horizon; deliberately not exponential, so the
    # relationship between the number and the situation stays legible.
    return float((TTC_HORIZON - ttc) / (TTC_HORIZON - TTC_CRITICAL))


def _visible_risk(track, prediction, ego_speed: float) -> RiskFactor | None:
    """Risk from one tracked, currently-observable hazard."""
    forward, lateral = float(track.position[0]), float(track.position[1])
    if forward <= 0.0:
        return None                      # already behind the ego

    closing = max(ego_speed - float(track.velocity[0]), 0.0)
    ttc_part = _ttc_risk(forward, closing)
    cross_part = float(getattr(prediction, "p_cross_cautious", prediction.p_cross))

    # Both matter and neither alone is enough: someone certain to cross but
    # 200 m away is not urgent, and someone very close but walking parallel to
    # the road is not a hazard. The product keeps both necessary.
    score = ttc_part * cross_part
    if score <= 0.01:
        return None

    ttc = forward / closing if closing > 0.1 else float("inf")
    detail = (f"tracked {'VRU' if track.cls == 1 else 'vehicle'} at {forward:.0f} m, "
              f"p(cross)={cross_part:.2f}"
              + (f", TTC {ttc:.1f}s" if math.isfinite(ttc) else ""))
    return RiskFactor("visible", score, detail, track_id=getattr(track, "id", None))


def _hidden_risk(emergence, ego_speed: float, corridor_half_width_m: float = 1.75):
    """Risk from a hazard that is currently invisible.

    Scored on where the particle filter says it will reappear and whether that
    coincides with the ego arriving. A hazard predicted to emerge behind the
    ego, or well after it has passed, carries little risk however confident the
    belief is.
    """
    if emergence is None or not emergence.will_emerge or emergence.location_xy is None:
        return None

    forward, lateral = emergence.location_xy
    if forward <= 0.0:
        return None

    in_path = abs(lateral) <= corridor_half_width_m * 1.5
    time_to_arrival = forward / ego_speed if ego_speed > 0.1 else float("inf")
    overlap = 1.0
    if math.isfinite(time_to_arrival) and emergence.time_to_emerge_s is not None:
        # How closely emergence and arrival coincide. A 2 s tolerance: emerging
        # well before the ego arrives means it will be seen and handled
        # normally; well after means the ego has gone.
        gap = abs(time_to_arrival - emergence.time_to_emerge_s)
        overlap = float(math.exp(-0.5 * (gap / 2.0) ** 2))

    proximity = _ttc_risk(forward, max(ego_speed, 0.1))
    score = emergence.probability * overlap * max(proximity, 0.25)
    if not in_path:
        score *= 0.4                     # emerging off to the side is less urgent
    if score <= 0.01:
        return None

    modes = ", ".join(f"{k} {v:.0%}" for k, v in
                      sorted(emergence.modes.items(), key=lambda kv: -kv[1])[:2])
    detail = (f"hidden hazard predicted to emerge at {forward:.0f} m in "
              f"{emergence.time_to_emerge_s:.1f}s "
              f"(p={emergence.probability:.2f}; {modes})")
    return RiskFactor("hidden", float(min(score, 1.0)), detail)


def _occlusion_burden(occlusion_grid, ego_speed: float):
    """Risk from not being able to see, independent of any specific hazard.

    Uses only the near field. Occlusion 40 m away is normal urban driving and
    scoring it would leave the system permanently alarmed; occlusion within
    braking distance is the part that constrains a safe speed.
    """
    import numpy as np
    from perception.occlusion_grid import OCCLUDED

    grid = np.asarray(occlusion_grid)
    n = grid.shape[0]
    near = grid[: max(n // 2, 1)]        # the closer half of the grid
    if near.size == 0:
        return None, None

    blocked = float((near == OCCLUDED).mean())
    if blocked < 0.10:
        return None, None

    # Scaled by speed: the same blocked view is more dangerous the faster you
    # are closing on it.
    speed_factor = min(ego_speed / 14.0, 1.5)      # ~50 km/h reference
    score = min(blocked * speed_factor, 1.0)

    # Advisory speed: be able to stop within the distance actually visible.
    visible_m = (1.0 - blocked) * 20.0              # near half spans ~20 m
    advisory = math.sqrt(max(2.0 * COMFORT_DECEL * visible_m, 0.0))

    detail = (f"{blocked:.0%} of the near road is blocked from view -- hazards "
              f"could be present with no evidence either way")
    return RiskFactor("occlusion", score, detail), advisory


def assess(tracks=None, predictions=None, hidden_hazards=None, occlusion_grid=None,
            ego_speed: float = 0.0, corridor_half_width_m: float = 1.75) -> RiskAssessment:
    """Overall risk and the ADAS action it warrants.

    `tracks`/`predictions` are visible hazards (`tracking.Track` and
    `intent.CrossingPrediction`, keyed by track id); `hidden_hazards` are
    `particle_tracker.EmergencePrediction` objects for targets currently behind
    occluders; `occlusion_grid` is the runtime grid from
    `perception.occlusion_grid`.

    All are optional, so this degrades sensibly: with no grid it scores only
    hazards, and with no hazards it still reports occlusion burden.
    """
    tracks = tracks or []
    predictions = predictions or {}
    factors = []

    for t in tracks:
        pred = predictions.get(getattr(t, "id", None))
        if pred is None:
            continue
        f = _visible_risk(t, pred, ego_speed)
        if f is not None:
            factors.append(f)

    for emergence in (hidden_hazards or []):
        f = _hidden_risk(emergence, ego_speed, corridor_half_width_m)
        if f is not None:
            factors.append(f)

    advisory = None
    if occlusion_grid is not None:
        f, advisory = _occlusion_burden(occlusion_grid, ego_speed)
        if f is not None:
            factors.append(f)

    # Combine as a weighted maximum per source rather than a sum. Summing lets
    # several mild factors manufacture an emergency, and three pedestrians at a
    # safe distance is not a braking event.
    by_source = {"visible": 0.0, "hidden": 0.0, "occlusion": 0.0}
    for f in factors:
        by_source[f.source] = max(by_source[f.source], f.score)

    score = (W_VISIBLE * by_source["visible"]
             + W_HIDDEN * by_source["hidden"]
             + W_OCCLUSION * by_source["occlusion"])

    # A single certain, imminent visible hazard must be able to reach BRAKE on
    # its own, which the weighted sum alone cannot do.
    score = max(score, by_source["visible"] * 0.95)
    score = float(min(score, 1.0))

    action = NONE
    for level, threshold in ACTION_THRESHOLDS:
        if score >= threshold:
            action = level
            break

    dominant = max(by_source, key=by_source.get) if any(by_source.values()) else "none"
    # A speed advisory only makes sense while the situation is not already an
    # emergency; below WARN it is the useful output.
    if action >= WARN:
        advisory = None

    return RiskAssessment(score=score, action=action, factors=factors,
                           advisory_speed_ms=advisory, dominant=dominant)
