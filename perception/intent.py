"""Crossing-intent prediction: will this tracked actor enter the ego's path?

Takes a `tracking.Track` and answers with a probability rather than a yes/no,
because the honest answer to "will that pedestrian step out?" from two frames
of a partial glimpse is a distribution, not a decision.

Method
------
Propagate the track's state *and covariance* forward under constant velocity,
then ask what fraction of that distribution passes through the ego's path
corridor within the horizon. Evaluated by sampling rather than analytically:
the corridor test is a conjunction of conditions on correlated variables over
a time window, and Monte Carlo over the state Gaussian handles a stationary
ego, a moving ego, and a coasting track uniformly, where a closed form needs a
different special case for each.

Why constant velocity, and what it costs
----------------------------------------
A pedestrian is not a constant-velocity object -- they hesitate, accelerate,
and change their mind. Constant velocity is used anyway because the alternative
worth having is a learned trajectory model, and that needs far more scene
diversity than this dataset has to be anything but overfitted. The cost is
stated rather than hidden: predictions degrade exactly when the walker's
behaviour state machine has them hesitate or abort, and
`scripts/evaluate_prediction.py` reports accuracy split by behaviour so the
degradation is visible instead of averaged away.

The coupling to the evidential head
-----------------------------------
The EKF's covariance accounts for *sensor* noise. It knows nothing about the
classifier having only seen an ambiguous sliver of a body -- and a partial
glimpse is exactly when the box is loose, the class may be wrong, and the
association may be to the wrong object. So the evidential head's uncertainty
has to enter somewhere.

The obvious way to do it is wrong, and the failure is worth recording because
it is easy to talk oneself into. Simply inflating the predicted covariance in
proportion to classifier uncertainty does **not** make the system more
cautious: widening a distribution moves its tail mass both ways, so the
crossing probability is pushed toward 0.5 rather than upward. Measured on a
marginal track, raising uncertainty from 0 to 1 moved p_cross from 0.615 down
to 0.570 -- less alarming, not more, exactly backwards for a braking decision.

What is done instead keeps the two concerns separate:

  - `p_cross` is left honest. It comes from the filter's own covariance, which
    already grows while a track coasts behind an occluder, and it is not
    distorted by anything. It is the number to calibrate and report.
  - `p_cross_cautious` is the worst case over the ambiguity set implied by the
    classifier uncertainty -- the maximum of p_cross over inflation levels in
    [0, u]. Monotonically non-decreasing in uncertainty by construction, with a
    plain reading: "the most pessimistic interpretation consistent with how
    little we know about what we are looking at."
  - `is_risk` tests the cautious number, so a poorly-seen actor needs less
    evidence to trigger a warning than a clearly-seen one.

A property of this that is worth knowing before reading the numbers: the
cautious figure only ever exceeds the honest one when the honest one is below
0.5. Widening a distribution moves probability mass toward a coin flip from
whichever side it started, so above 0.5 every inflated reading is *lower* and
the maximum is the uninflated one. That is the desired behaviour rather than a
quirk -- once an actor is already probably going to cross, the system is
already warning, and there is nothing for caution to add. Uncertainty buys
early warning precisely in the regime where the nominal reading says "probably
fine", which is where a partial glimpse of someone stepping out actually sits.


This is the piece that makes Phase 3 do work. Without it the uncertainty number
is displayed on a dashboard and never affects a decision.
"""
import math
from dataclasses import dataclass

import numpy as np

# Half-width of the ego's path corridor, metres. The vehicle is ~1.8 m wide, so
# 0.9 m is the bodywork; the rest is the pedestrian's own width plus margin.
# Roughly a lane, which is the right scale for "would I have to brake".
DEFAULT_CORRIDOR_HALF_WIDTH = 1.75

DEFAULT_HORIZON_S = 3.0
DEFAULT_SAMPLES = 400

# Widest covariance inflation considered when forming the cautious worst case.
# u=1 (maximum evidential uncertainty) admits interpretations up to (1 + gain)
# times the filter's own covariance. 2.0 means a maximally-uncertain detection
# is considered at up to ~1.7x the positional spread -- enough to change a
# marginal call without swamping a confident one.
DEFAULT_UNCERTAINTY_GAIN = 2.0

# Inflation levels sampled across the ambiguity set. p_cross varies smoothly
# with the inflation scale, so four levels are enough for the maximum to be
# stable.
_AMBIGUITY_LEVELS = 4


@dataclass
class CrossingPrediction:
    p_cross: float                    # honest P(enters the ego corridor within the horizon)
    p_cross_cautious: float           # worst case over the classifier-uncertainty ambiguity set
    time_to_entry_s: float | None     # expected first-entry time among samples that enter
    time_to_ego_arrival_s: float | None   # when the ego reaches the actor's longitudinal position
    min_lateral_gap_m: float          # expected closest lateral approach to the corridor centre
    lead_time_s: float | None         # how long before the ego arrives the entry happens
    is_risk: bool

    def __str__(self) -> str:
        t = "--" if self.time_to_entry_s is None else f"{self.time_to_entry_s:.1f}s"
        return (f"p_cross={self.p_cross:.2f} (cautious {self.p_cross_cautious:.2f}) "
                f"entry={t} gap={self.min_lateral_gap_m:.1f}m")


def predict_crossing(track, ego_speed: float = 0.0,
                      horizon_s: float = DEFAULT_HORIZON_S,
                      dt: float = 0.1,
                      corridor_half_width_m: float = DEFAULT_CORRIDOR_HALF_WIDTH,
                      uncertainty_gain: float = DEFAULT_UNCERTAINTY_GAIN,
                      n_samples: int = DEFAULT_SAMPLES,
                      risk_threshold: float = 0.3,
                      rng: np.random.Generator | None = None) -> CrossingPrediction:
    """Probability that `track` enters the ego's path corridor within the horizon.

    `ego_speed` is the ego's own forward speed (m/s), used only to work out when
    the ego would arrive at the actor's longitudinal position. A stationary ego
    gives `time_to_ego_arrival_s = None`, and the prediction reduces to "will
    they step into my lane at all", which is still the useful question when
    creeping toward a crossing.

    Sampling is seeded by default so a given track yields the same number twice
    -- an unreproducible risk score is not something to put in a paper.
    """
    rng = np.random.default_rng(0) if rng is None else rng

    P = np.asarray(track.P, dtype=float)
    mean = np.asarray(track.x, dtype=float)
    uncertainty = float(np.clip(getattr(track, "uncertainty", 0.0), 0.0, 1.0))

    # Draw once, zero-mean, then rescale for each inflation level. A covariance
    # scaled by s has samples scaled by sqrt(s), so the whole ambiguity sweep
    # costs one Cholesky instead of one per level. Sampling whole states keeps
    # position and velocity correlated the way the filter says they are --
    # drawing them independently would understate where a fast-but-poorly-
    # observed walker can end up.
    try:
        noise = rng.multivariate_normal(np.zeros(4), P, size=n_samples, method="cholesky")
    except np.linalg.LinAlgError:
        # A degenerate covariance means the filter has effectively collapsed;
        # fall back to the point estimate rather than failing the frame.
        noise = np.zeros((n_samples, 4))

    steps = max(int(round(horizon_s / dt)), 1)
    t_grid = np.arange(1, steps + 1) * dt

    def evaluate(scale: float):
        s = mean + math.sqrt(scale) * noise
        fwd = s[:, 0:1] + s[:, 2:3] * t_grid[None, :]
        lat = s[:, 1:2] + s[:, 3:4] * t_grid[None, :]
        # "In the corridor" means laterally within it AND still ahead of the
        # ego. Dropping the forward test would count an actor who has already
        # walked past behind us.
        inside = (np.abs(lat) <= corridor_half_width_m) & (fwd > 0.0)
        return inside, lat

    inside, lat = evaluate(1.0)
    enters = inside.any(axis=1)
    p_cross = float(enters.mean())

    # Worst case over the ambiguity set the classifier uncertainty implies.
    # Monotone in `uncertainty` by construction: the set only grows. See the
    # module docstring for why simply widening the covariance is not enough.
    p_cautious = p_cross
    if uncertainty > 0.0:
        for level in range(1, _AMBIGUITY_LEVELS + 1):
            scale = 1.0 + uncertainty_gain * uncertainty * level / _AMBIGUITY_LEVELS
            p_cautious = max(p_cautious, float(evaluate(scale)[0].any(axis=1).mean()))

    if enters.any():
        first = np.argmax(inside[enters], axis=1)
        time_to_entry = float(np.mean(t_grid[first]))
    else:
        time_to_entry = None

    min_gap = float(np.mean(np.min(np.abs(lat), axis=1)))

    # When would the ego reach this actor's longitudinal position? Only
    # meaningful while actually moving forward.
    forward_now = float(mean[0])
    if ego_speed > 0.1 and forward_now > 0.0:
        time_to_arrival = forward_now / ego_speed
    else:
        time_to_arrival = None

    lead_time = (None if (time_to_entry is None or time_to_arrival is None)
                 else time_to_arrival - time_to_entry)

    # Risky when they are likely to enter, and to do so before the ego has
    # passed. A high p_cross that happens well after the ego is gone is not a
    # braking case. Tested against the cautious figure, so a poorly-seen actor
    # needs less evidence to trigger a warning than a clearly-seen one.
    is_risk = p_cautious >= risk_threshold and (lead_time is None or lead_time > -0.5)

    return CrossingPrediction(p_cross=p_cross, p_cross_cautious=p_cautious,
                               time_to_entry_s=time_to_entry,
                               time_to_ego_arrival_s=time_to_arrival,
                               min_lateral_gap_m=min_gap, lead_time_s=lead_time,
                               is_risk=bool(is_risk))


def predict_all(tracks, ego_speed: float = 0.0, **kwargs) -> dict:
    """Convenience: {track_id: CrossingPrediction} over confirmed VRU tracks.

    Restricted to confirmed tracks because a one-frame tentative detection has
    no velocity estimate worth extrapolating, and emitting a crossing
    probability from it would be noise dressed as a number.
    """
    return {t.id: predict_crossing(t, ego_speed=ego_speed, **kwargs)
            for t in tracks if t.is_confirmed}
