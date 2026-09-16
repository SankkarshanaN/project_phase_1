"""Component 6 -- Particle Filter Hidden Hazard Tracker (multi-hypothesis).

Keeps a belief over where a hazard is while it is completely invisible, and
predicts where it will reappear.

Why a particle filter rather than the Kalman filter in `tracking.py`
--------------------------------------------------------------------
While a target is *visible*, its position distribution is roughly unimodal and
an EKF is the right tool -- cheaper, exactly optimal under its assumptions, and
`tracking.py` remains in use for that. The moment the target disappears behind
a bus, the distribution stops being unimodal:

    - they may continue across and emerge at the FRONT of the bus
    - they may turn back and emerge at the REAR
    - they may stop dead behind it and not emerge at all

Those are three separated modes, not one blob with a large variance. A Gaussian
cannot represent them: its mean sits *inside the bus*, where the pedestrian
certainly is not, and its covariance implies mass spread evenly over ground the
pedestrian cannot occupy. Acting on that mean produces a warning pointed at the
wrong place.

A particle filter represents the modes directly. Each particle carries its own
motion hypothesis, so the population naturally splits into "crossing",
"returning" and "stopped" clusters; the emergence prediction is then simply
where the surviving particles say the target will appear, and the multi-modality
is visible in the output rather than averaged away.

This is also what makes re-identification on reappearance possible. When a
detection re-appears at the edge of the occluder, it can be scored against the
particle cloud -- "is this consistent with the hazard we were tracking, or is it
somebody new?" -- which a single Gaussian centred inside the occluder cannot
answer sensibly.

The occlusion grid drives the weighting: particles that drift into cells the
sensors CAN see, while nothing is detected there, are penalised. That is the
key inference. Not being seen in a visible cell is evidence of absence; not
being seen in an occluded cell is no evidence at all. Over several frames this
alone carves the belief into the genuinely-hidden region without any detection
ever arriving.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from perception.geometry import ego_xy_to_bev_cell
from perception.occlusion_grid import EMPTY, OCCLUDED, UNKNOWN, VISIBLE

DEFAULT_N_PARTICLES = 400

# Motion hypotheses assigned at spawn and kept for the particle's life. Holding
# the hypothesis fixed is what preserves distinct modes: resampling a shared
# random walk every frame would blur them back into one cloud within a second.
CONTINUE, SLOW, STOP, REVERSE = 0, 1, 2, 3
HYPOTHESIS_NAMES = {CONTINUE: "continues", SLOW: "slows", STOP: "stops",
                     REVERSE: "turns back"}

# Sampling weights. Continuing is most likely -- most people who start crossing
# finish -- but the others carry enough mass to survive if the evidence favours
# them.
HYPOTHESIS_WEIGHTS = (0.45, 0.20, 0.15, 0.20)

# Speed multipliers applied to the initial velocity per hypothesis.
HYPOTHESIS_SPEED = {CONTINUE: 1.0, SLOW: 0.45, STOP: 0.0, REVERSE: -0.8}

# Likelihood of observing NO detection at a particle's location, by occlusion
# state. The OCCLUDED value being near 1.0 is the crux: a particle hiding
# behind the bus is fully consistent with seeing nothing, so it is not
# penalised, while one standing in open view is.
NO_DETECTION_LIKELIHOOD = {
    OCCLUDED: 0.95,
    UNKNOWN: 0.60,
    VISIBLE: 0.12,
    EMPTY: 0.06,
}

PROCESS_NOISE_POS = 0.12     # m per tick
PROCESS_NOISE_VEL = 0.35     # m/s per tick -- pedestrians are erratic
RESAMPLE_RATIO = 0.5         # resample when ESS falls below this fraction


@dataclass
class EmergencePrediction:
    """Where and when the hidden hazard is expected to reappear."""
    will_emerge: bool
    time_to_emerge_s: float | None
    location_xy: tuple | None        # ego-frame (forward, lateral)
    probability: float
    modes: dict = field(default_factory=dict)   # hypothesis name -> mass

    def __str__(self) -> str:
        if not self.will_emerge:
            return f"no emergence predicted within horizon (p={self.probability:.2f})"
        f, l = self.location_xy
        return (f"emerges in {self.time_to_emerge_s:.1f}s at "
                f"({f:.1f}, {l:.1f}) p={self.probability:.2f}")


class HiddenHazardTracker:
    """Particle filter over one hazard's state while it is occluded.

    State per particle: [forward, lateral, v_forward, v_lateral] in the ego
    frame, plus a fixed motion hypothesis.

    Seeded from the last confident estimate before the target was lost --
    typically a `tracking.Track` at the moment its tier became OCCLUDED.
    """

    def __init__(self, init_xy, init_vel, init_cov=None,
                  n_particles: int = DEFAULT_N_PARTICLES,
                  rng: np.random.Generator | None = None,
                  is_vru: bool = True):
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.n = n_particles
        self.is_vru = is_vru
        self.frames_hidden = 0
        self.emerged = False

        cov = np.diag([0.5, 0.5, 0.4, 0.4]) if init_cov is None else np.asarray(init_cov)
        mean = np.array([init_xy[0], init_xy[1], init_vel[0], init_vel[1]], dtype=float)
        try:
            self.particles = self.rng.multivariate_normal(mean, cov, size=self.n)
        except np.linalg.LinAlgError:
            self.particles = np.repeat(mean[None, :], self.n, axis=0)

        self.hypotheses = self.rng.choice(4, size=self.n, p=HYPOTHESIS_WEIGHTS)
        # Apply each particle's hypothesis to its velocity once, at birth.
        for h, mult in HYPOTHESIS_SPEED.items():
            m = self.hypotheses == h
            self.particles[m, 2:] *= mult

        self.weights = np.full(self.n, 1.0 / self.n)

    # ------------------------------------------------------------------ core

    def predict(self, dt: float, ego_speed: float = 0.0, ego_yaw_rate: float = 0.0) -> None:
        """Advance particles, compensating for the ego's own motion.

        Ego speed and yaw rate are proprioceptive -- wheel encoders and IMU on
        any real vehicle -- so using them keeps this on the runtime side of the
        ground-truth/runtime split.
        """
        self.frames_hidden += 1
        p = self.particles

        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt

        if abs(ego_yaw_rate) > 1e-6 or abs(ego_speed) > 1e-6:
            c, s = math.cos(ego_yaw_rate * dt), math.sin(ego_yaw_rate * dt)
            R = np.array([[c, s], [-s, c]])
            p[:, :2] = (p[:, :2] - np.array([ego_speed * dt, 0.0])) @ R.T
            p[:, 2:] = p[:, 2:] @ R.T

        p[:, :2] += self.rng.normal(0.0, PROCESS_NOISE_POS, size=(self.n, 2))
        # Stopped particles stay stopped; adding velocity noise to them would
        # dissolve the "stopped" mode within a couple of seconds, which is
        # exactly the hypothesis a braking decision most needs kept alive.
        moving = self.hypotheses != STOP
        p[moving, 2:] += self.rng.normal(0.0, PROCESS_NOISE_VEL,
                                          size=(int(moving.sum()), 2))

    def update_from_occlusion(self, occlusion_grid, grid_cfg,
                               detections_xy=None) -> None:
        """Reweight particles against what the sensors did and did not see.

        With no detection, this is a pure negative-information update: particles
        sitting where the sensors could have seen them, but did not, lose
        weight; particles in genuinely occluded cells keep theirs. Repeated
        over several frames it concentrates the belief into the hidden region
        without a single positive measurement -- the filter reasons from the
        absence.
        """
        grid = np.asarray(occlusion_grid)
        n_cells = grid.shape[0]

        fwd_idx, lat_idx = ego_xy_to_bev_cell(self.particles[:, 0],
                                               self.particles[:, 1], grid_cfg)
        inside = ((fwd_idx >= 0) & (fwd_idx < n_cells)
                  & (lat_idx >= 0) & (lat_idx < n_cells))

        like = np.full(self.n, NO_DETECTION_LIKELIHOOD[UNKNOWN])
        if inside.any():
            states = grid[fwd_idx[inside], lat_idx[inside]]
            like[inside] = np.array([NO_DETECTION_LIKELIHOOD.get(int(s), 0.5)
                                      for s in states])
        # Off-grid particles have left the region of interest entirely.
        like[~inside] = 0.3

        if detections_xy is not None and len(detections_xy):
            det = np.asarray(detections_xy, dtype=float).reshape(-1, 2)
            d = np.linalg.norm(self.particles[:, None, :2] - det[None, :, :], axis=2)
            nearest = d.min(axis=1)
            # A detection nearby is strong positive evidence; Gaussian kernel
            # at 1.5 m, roughly a pedestrian's positional ambiguity.
            like = like * 0.2 + np.exp(-0.5 * (nearest / 1.5) ** 2) * 0.8

        self.weights *= like
        total = self.weights.sum()
        if total <= 1e-12:
            # Every hypothesis has been ruled out -- the belief has collapsed.
            # Reset to uniform rather than dividing by zero; the caller should
            # be watching `is_degenerate` and drop the track.
            self.weights = np.full(self.n, 1.0 / self.n)
        else:
            self.weights /= total

        if self.effective_sample_size < RESAMPLE_RATIO * self.n:
            self._resample()

    def _resample(self) -> None:
        """Systematic resampling -- lower variance than multinomial and O(n).

        Hypotheses are carried across with their particles, so resampling
        prunes whole motion hypotheses when the evidence stops supporting them
        rather than reshuffling them at random.
        """
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative, positions)
        idx = np.clip(idx, 0, self.n - 1)
        self.particles = self.particles[idx]
        self.hypotheses = self.hypotheses[idx]
        self.weights = np.full(self.n, 1.0 / self.n)

    # ---------------------------------------------------------------- output

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    @property
    def is_degenerate(self) -> bool:
        """Belief has collapsed to too few distinct hypotheses to be trusted."""
        return self.effective_sample_size < 0.05 * self.n

    @property
    def mean_position(self) -> np.ndarray:
        return np.average(self.particles[:, :2], axis=0, weights=self.weights)

    @property
    def position_spread(self) -> float:
        mu = self.mean_position
        d = np.linalg.norm(self.particles[:, :2] - mu, axis=1)
        return float(np.sqrt(np.average(d ** 2, weights=self.weights)))

    def mode_masses(self) -> dict:
        """Probability mass per motion hypothesis.

        The headline diagnostic of this component: it is the thing a Gaussian
        filter structurally cannot report. "60% still crossing, 25% turned
        back, 15% stopped" is actionable; a mean position inside the bus is not.
        """
        return {HYPOTHESIS_NAMES[h]: float(self.weights[self.hypotheses == h].sum())
                for h in sorted(HYPOTHESIS_NAMES)}

    def predict_emergence(self, occlusion_grid, grid_cfg, horizon_s: float = 3.0,
                           dt: float = 0.1) -> EmergencePrediction:
        """Where and when the hazard is expected to become visible again.

        Propagates a copy of the particle set forward and records where each
        first enters a cell the sensors can observe. That is the "predicts
        reappearance" capability the problem statement calls for, and it is
        available while the target is still completely invisible.
        """
        grid = np.asarray(occlusion_grid)
        n_cells = grid.shape[0]
        p = self.particles.copy()
        w = self.weights.copy()

        emerged_mass, weighted_time, locations = 0.0, 0.0, []
        still_hidden = np.ones(self.n, dtype=bool)

        for step in range(1, int(horizon_s / dt) + 1):
            p[:, 0] += p[:, 2] * dt
            p[:, 1] += p[:, 3] * dt

            fwd_idx, lat_idx = ego_xy_to_bev_cell(p[:, 0], p[:, 1], grid_cfg)
            inside = ((fwd_idx >= 0) & (fwd_idx < n_cells)
                      & (lat_idx >= 0) & (lat_idx < n_cells))
            visible = np.zeros(self.n, dtype=bool)
            if inside.any():
                states = grid[fwd_idx[inside], lat_idx[inside]]
                visible[inside] = np.isin(states, (VISIBLE, EMPTY))

            newly = visible & still_hidden
            if newly.any():
                step_mass = float(w[newly].sum())
                emerged_mass += step_mass
                # Weighted by particle mass, not particle COUNT -- a mode with
                # ten low-weight particles must not outvote one high-weight
                # particle. This was previously `times.extend([step*dt] *
                # count)` then `np.mean(times)`, an unweighted average that
                # silently ignored belief mass entirely: two particles
                # emerging at steps 1 and 30 with weights 0.9/0.1 averaged to
                # 1.55s (roughly the midpoint) instead of the mass-weighted
                # 0.39s (dominated by the heavy, early-emerging particle) --
                # a 4x error in exactly the number this component exists to
                # report accurately.
                weighted_time += step * dt * step_mass
                locations.append(np.average(p[newly, :2], axis=0,
                                             weights=w[newly]) * step_mass)
                still_hidden &= ~newly

        if emerged_mass < 1e-6:
            return EmergencePrediction(False, None, None, 0.0, self.mode_masses())

        loc = np.sum(locations, axis=0) / emerged_mass
        return EmergencePrediction(
            will_emerge=emerged_mass > 0.25,
            time_to_emerge_s=float(weighted_time / emerged_mass),
            location_xy=(float(loc[0]), float(loc[1])),
            probability=float(emerged_mass),
            modes=self.mode_masses(),
        )

    def reidentification_score(self, detection_xy) -> float:
        """How consistent a fresh detection is with this tracked hazard.

        The literature notes that occlusion trackers accumulate uncertainty and
        typically have no re-identification check on reappearance (Li et al.,
        2018). This provides one: the weighted fraction of particle mass within
        a plausible radius of the new detection. A high score says "this is who
        we lost"; a low one says "this is somebody else, and the original hazard
        is still hidden."
        """
        d = np.linalg.norm(self.particles[:, :2] - np.asarray(detection_xy, float), axis=1)
        return float(np.sum(self.weights[d < 2.5]))
