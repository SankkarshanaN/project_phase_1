"""EKF multi-object tracking in the ego frame, fusing camera bearing with radar
range and Doppler.

Why an EKF and not just differencing detections
-----------------------------------------------
The question this pipeline has to answer -- "will that pedestrian cross in
front of me?" -- is a question about *lateral* velocity. The two sensors split
that measurement between them, and neither can supply it alone:

  - Radar gives range and range-rate directly, but range-rate is the
    **radial** component. A pedestrian crossing the ego's path moves almost
    perpendicular to the line of sight, so their radial velocity is close to
    zero no matter how fast they are walking. Radar is blindest to exactly the
    component that decides the outcome.
  - The camera gives a good bearing but no range, so a bearing rate alone is
    an angular velocity, not a speed.

Together: `lateral speed ~ range (radar) x d(bearing)/dt (camera)`. The EKF is
the principled way to combine them, and the measurement Jacobian encodes the
split exactly -- look at `_jacobian`, where the range-rate row's velocity
partials are `(fx/r, ly/r)`. For a target dead ahead that is `(1, 0)`: lateral
velocity is formally unobservable from Doppler, and the filter knows it,
keeping a wide lateral-velocity covariance until camera bearing changes
constrain it.

Coasting through occlusion
--------------------------
When a tracked actor disappears behind an occluder there is no measurement at
all, from either sensor. The filter predicts anyway and its covariance grows.
That is the whole point of the project made executable: the track survives the
bus, with honestly-increasing uncertainty, instead of the object blinking out
of existence. `Track.is_coasting` exposes it.

Frames and privilege
--------------------
State is ego-local `[forward, lateral, v_forward, v_lateral]`, velocity over
ground, in the convention documented in `perception.geometry`.

The predict step needs the ego's own speed and yaw rate to compensate for ego
motion. That is proprioception -- any real vehicle reads it off its own wheel
encoders and IMU -- not privileged knowledge about other actors, so it stays
within the runtime/ground-truth split CLAUDE.md describes. Nothing here reads
the simulator's actor list, depth buffer, or semantic segmentation.
"""
import math
from dataclasses import dataclass, field

import numpy as np

# Pedestrians change speed and direction abruptly; this is the standard
# discrete white-noise-acceleration term, sized for a person rather than a
# vehicle. Too small and the filter lags a walker stepping off a kerb; too
# large and it never accumulates enough confidence to be useful.
DEFAULT_ACCEL_STD = 1.5          # m/s^2

# Measurement noise, MEASURED against ground truth on real CARLA frames rather
# than guessed. Both were previously optimistic, and an over-confident noise
# term is not a cosmetic error: the filter over-weights that measurement and
# lets it drag the state.
#
#   bearing  -- box centre vs true bearing: mean -0.58 deg, std 2.38 deg over
#               815 untruncated observations. Was set to 1.0 deg, 2.4x too
#               confident. The residual is the box centre being a crude proxy
#               for the object centre, plus the uncalibrated pinhole model in
#               `geometry.pixel_to_bearing`.
#   range    -- best-case gated radar pick vs true range: std 1.95 m, only 39%
#               within 1 m. Was set to 0.5 m, 3.9x too confident. The error is
#               irreducible clutter: just 6.8% of returns in a pedestrian's
#               azimuth window are the pedestrian.
#
# The range figure is why an earlier run had FUSED tracking worse than
# camera-only on lateral velocity (0.952 vs 0.514 m/s): a confidently-wrong
# range corrupts lateral position, since lateral = range * sin(bearing).
DEFAULT_BEARING_STD_RAD = math.radians(2.4)
DEFAULT_RANGE_STD_M = 2.0
DEFAULT_RANGE_RATE_STD = 0.8     # m/s; not separately measured, scaled with range

# Association gate, as a Mahalanobis distance in measurement space. 9.21 is the
# chi-square 99% point for 2 degrees of freedom, so a correct detection is
# rejected about 1% of the time.
DEFAULT_GATE = 9.21

# A radar return further than this from the track's predicted range belongs to
# something else -- background, or another object in the same direction -- and
# accepting it drags the track off the target.
RANGE_GATE_M = 6.0

# Birth prior for a camera-only track, from the measured distribution of ranges
# at which VRUs are actually visible in this dataset (median 37.8 m, p10 7.1,
# p90 89.4 over 1,230 observations). The spread is genuinely wide, so the
# variance is too.
CAMERA_BIRTH_RANGE_M = 35.0
CAMERA_BIRTH_RANGE_STD_M = 25.0

MAX_COAST_FRAMES = 25            # ~2.5 s at dt=0.1; long enough to cross a bus
MIN_HITS_TO_CONFIRM = 3


@dataclass
class Detection:
    """One frame's observation of one object, before association.

    `range_m` / `range_rate` are None when no radar return fell inside the
    detection's azimuth window -- a camera-only observation. `bearing_rad` is
    None for a radar-only observation. At least one must be present.

    `range_candidates` carries EVERY radar return in the detection's azimuth
    window as (range, range_rate) pairs, and when present it is preferred over
    the single pre-chosen `range_m`.

    That choice matters more than it looks. Measured on real CARLA frames, only
    6.8% of the returns inside a pedestrian's azimuth window actually come from
    the pedestrian -- the rest are road surface, buildings and whatever stands
    behind them. Picking one up front, by median or by cluster, is therefore
    wrong most of the time and feeds the filter a confidently incorrect range.
    Altitude does not separate them either: the pedestrian's returns come off
    the legs and sit LOWER (median 0.16 m) than the background (0.88 m), so a
    height gate removes the target and keeps the clutter.

    Handing the candidates to the filter instead lets it choose the one
    consistent with where it already believes the object is, which is what
    resolves the ambiguity -- standard measurement-to-track association, rather
    than committing before any state is consulted.
    """
    bearing_rad: float | None = None
    range_m: float | None = None
    range_rate: float | None = None
    range_candidates: list | None = None
    cls: int = -1
    box_px: tuple | None = None
    uncertainty: float = 0.0     # evidential head output, carried to the intent step


@dataclass
class Track:
    id: int
    x: np.ndarray                 # [forward, lateral, v_forward, v_lateral]
    P: np.ndarray                 # 4x4 covariance
    cls: int = -1
    hits: int = 1
    age: int = 0
    misses: int = 0
    frames_coasted: int = 0
    uncertainty: float = 0.0
    box_px: tuple | None = None

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.x[2:]))

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= MIN_HITS_TO_CONFIRM

    @property
    def is_coasting(self) -> bool:
        """No measurement this frame -- typically the actor is occluded."""
        return self.frames_coasted > 0

    @property
    def lateral_speed_std(self) -> float:
        """Standard deviation on lateral velocity. Rises while coasting and
        while only Doppler is available, which is the honest signal that a
        crossing prediction from this track should be treated as weak."""
        return float(math.sqrt(max(self.P[3, 3], 0.0)))


def _rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s], [-s, c]])


def _transition(dt: float, ego_speed: float, ego_yaw_rate: float):
    """Constant-velocity transition with ego-motion compensation.

    Returns (F, offset) such that `x_next = F @ x + offset`.

    A target standing still has a *changing* ego-frame position whenever the
    ego moves, so a plain constant-velocity model in the ego frame is wrong.
    Composing the target's own motion with the inverse of the ego's motion
    keeps the state's velocity meaning "over ground", which is what the
    crossing prediction needs -- a pedestrian's intent does not depend on how
    fast the observer happens to be driving.

    Documented simplification: the ego's motion is treated as exactly known.
    Real odometry drifts, and a fuller treatment would add its covariance here.
    In simulation it is exact, so this is honest for the reported numbers but
    would need revisiting on a real vehicle.
    """
    R = _rotation(ego_yaw_rate * dt)
    F = np.zeros((4, 4))
    F[:2, :2] = R
    F[:2, 2:] = R * dt
    F[2:, 2:] = R
    offset = np.zeros(4)
    offset[:2] = -R @ np.array([ego_speed * dt, 0.0])
    return F, offset


def _process_noise(dt: float, accel_std: float) -> np.ndarray:
    """Discrete white-noise acceleration, independent per axis."""
    q = accel_std ** 2
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    Q = np.zeros((4, 4))
    for i, j in ((0, 2), (1, 3)):
        Q[i, i] = dt4 / 4.0 * q
        Q[i, j] = Q[j, i] = dt3 / 2.0 * q
        Q[j, j] = dt2 * q
    return Q


def _measure(x: np.ndarray, ego_vel: np.ndarray) -> np.ndarray:
    """h(x) -> [bearing, range, range_rate]."""
    fx, ly, vx, vy = x
    r = max(math.hypot(fx, ly), 1e-3)
    dvx, dvy = vx - ego_vel[0], vy - ego_vel[1]
    return np.array([
        math.atan2(-ly, fx),
        r,
        (dvx * fx + dvy * ly) / r,     # negative closing, matching CARLA's radar
    ])


def _jacobian(x: np.ndarray, ego_vel: np.ndarray) -> np.ndarray:
    """dh/dx. The last row is where the sensor physics lives: the velocity
    partials are (fx/r, ly/r), the unit line-of-sight vector -- so Doppler
    constrains only the radial velocity component, and a target dead ahead
    (ly = 0) contributes nothing at all toward observing lateral speed."""
    fx, ly, vx, vy = x
    r = max(math.hypot(fx, ly), 1e-3)
    r2, r3 = r * r, r ** 3
    dvx, dvy = vx - ego_vel[0], vy - ego_vel[1]
    s = dvx * fx + dvy * ly
    return np.array([
        [ly / r2, -fx / r2, 0.0, 0.0],
        [fx / r, ly / r, 0.0, 0.0],
        [(dvx * r2 - s * fx) / r3, (dvy * r2 - s * ly) / r3, fx / r, ly / r],
    ])


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class MultiObjectTracker:
    """Greedy-association EKF tracker.

    Association is greedy nearest-neighbour under a Mahalanobis gate rather
    than a globally optimal assignment. With the handful of actors these
    scenarios contain, the two agree almost always, and greedy keeps the module
    free of a solver dependency. Revisit if dense traffic starts producing
    identity swaps -- `evaluate_prediction.py` reports ID switches so the
    failure would be visible rather than silent.
    """

    def __init__(self, accel_std: float = DEFAULT_ACCEL_STD,
                  bearing_std: float = DEFAULT_BEARING_STD_RAD,
                  range_std: float = DEFAULT_RANGE_STD_M,
                  range_rate_std: float = DEFAULT_RANGE_RATE_STD,
                  gate: float = DEFAULT_GATE,
                  max_coast: int = MAX_COAST_FRAMES):
        self.accel_std = accel_std
        self.meas_std = np.array([bearing_std, range_std, range_rate_std])
        self.gate = gate
        self.max_coast = max_coast
        self.tracks: list[Track] = []
        self._next_id = 0
        # Most recent range residual, consumed by `sensor_health` to detect
        # radar calibration drift. None until a radar-bearing update lands.
        self.last_range_innovation: float | None = None

    # ---------------------------------------------------------------- predict

    def predict(self, dt: float, ego_speed: float = 0.0, ego_yaw_rate: float = 0.0) -> None:
        F, offset = _transition(dt, ego_speed, ego_yaw_rate)
        Q = _process_noise(dt, self.accel_std)
        for t in self.tracks:
            t.x = F @ t.x + offset
            t.P = F @ t.P @ F.T + Q
            t.age += 1

    # ---------------------------------------------------------------- update

    def update(self, detections: list[Detection], ego_vel: np.ndarray | None = None) -> None:
        ego_vel = np.zeros(2) if ego_vel is None else np.asarray(ego_vel, dtype=float)

        unmatched = set(range(len(detections)))
        for track in self.tracks:
            best, best_dist = None, self.gate
            for i in unmatched:
                d = self._mahalanobis(track, detections[i], ego_vel)
                if d is not None and d < best_dist:
                    best, best_dist = i, d
            if best is None:
                track.misses += 1
                track.frames_coasted += 1
                continue
            unmatched.discard(best)
            self._apply(track, detections[best], ego_vel)

        for i in unmatched:
            self._spawn(detections[i], ego_vel)

        # Drop tracks that have coasted past the point of usefulness, and
        # tentative ones that never got corroborated.
        self.tracks = [t for t in self.tracks
                        if t.frames_coasted <= self.max_coast
                        and not (t.misses > 2 and not t.is_confirmed)]

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _rows_for(det: Detection) -> list[int]:
        """Which measurement rows this detection actually supplies.

        Partial observation is the normal case, not an edge case: the camera
        alone gives bearing, radar alone gives range and range-rate, and behind
        an occluder there is neither. Each is handled by slicing the same
        measurement model rather than by a separate filter.
        """
        rows = []
        if det.bearing_rad is not None:
            rows.append(0)
        if det.range_m is not None:
            rows.append(1)
        if det.range_rate is not None:
            rows.append(2)
        return rows

    @staticmethod
    def _pick_range(det: Detection, predicted_range: float):
        """Choose the radar return most consistent with where the track already
        believes the object is. See `Detection.range_candidates`.

        Falls back to the pre-chosen `range_m` when no candidate list is given,
        and refuses a candidate further than `RANGE_GATE_M` from the prediction
        -- beyond that it is another object, and accepting it would drag the
        track onto the background.
        """
        if not det.range_candidates:
            return det.range_m, det.range_rate
        best, best_d = None, RANGE_GATE_M
        for rng, rate in det.range_candidates:
            d = abs(rng - predicted_range)
            if d < best_d:
                best, best_d = (rng, rate), d
        if best is not None:
            return best
        return (None, None) if det.range_m is None else (det.range_m, det.range_rate)

    def _innovation(self, track: Track, det: Detection, ego_vel: np.ndarray):
        predicted = _measure(track.x, ego_vel)
        rng_m, rate = self._pick_range(det, predicted[1])

        # Rows are decided AFTER candidate selection: if no radar return is
        # consistent with the prediction, this frame is camera-only for this
        # track rather than a forced bad range update.
        rows = []
        if det.bearing_rad is not None:
            rows.append(0)
        if rng_m is not None:
            rows.append(1)
        if rate is not None:
            rows.append(2)
        if not rows:
            return None

        actual = np.array([
            det.bearing_rad if det.bearing_rad is not None else predicted[0],
            rng_m if rng_m is not None else predicted[1],
            rate if rate is not None else predicted[2],
        ])
        y = actual - predicted
        y[0] = _wrap_angle(y[0])
        H = _jacobian(track.x, ego_vel)[rows]
        R = np.diag(self.meas_std[rows] ** 2)
        S = H @ track.P @ H.T + R
        # Rows are returned because candidate selection can drop the range and
        # range-rate rows for this track; recomputing them in the caller would
        # build a noise matrix of the wrong shape.
        return y[rows], H, S, rows

    def _mahalanobis(self, track: Track, det: Detection, ego_vel: np.ndarray):
        if det.cls != -1 and track.cls != -1 and det.cls != track.cls:
            return None
        out = self._innovation(track, det, ego_vel)
        if out is None:
            return None
        y, _H, S, _rows = out
        try:
            return float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return None

    def _apply(self, track: Track, det: Detection, ego_vel: np.ndarray) -> None:
        out = self._innovation(track, det, ego_vel)
        if out is None:
            return
        y, H, S, rows = out

        # Expose the range residual for `sensor_health`'s calibration-drift
        # check. A constant radar range bias is invisible in the radar stream
        # itself -- normal count, normal spread, plausible distances -- and
        # shows up only here, as innovations that persistently miss zero in the
        # same direction.
        if 1 in rows:
            self.last_range_innovation = float(y[rows.index(1)])
        K = track.P @ H.T @ np.linalg.inv(S)
        track.x = track.x + K @ y
        # Joseph form: stays positive-definite under partial updates, where the
        # simpler (I - KH)P is prone to drifting asymmetric over long tracks.
        IKH = np.eye(4) - K @ H
        R = np.diag(self.meas_std[rows] ** 2)
        track.P = IKH @ track.P @ IKH.T + K @ R @ K.T

        track.hits += 1
        track.misses = 0
        track.frames_coasted = 0
        track.uncertainty = det.uncertainty
        track.box_px = det.box_px
        if track.cls == -1:
            track.cls = det.cls

    def _spawn(self, det: Detection, ego_vel: np.ndarray) -> None:
        """Initialise a track from a single detection.

        A lone detection cannot determine velocity, so velocity starts at zero
        with a deliberately large variance -- wide enough to cover a running
        pedestrian. Seeding it optimistically would make the first crossing
        prediction confidently wrong, which is the exact failure this project
        exists to avoid.
        """
        rng = det.range_m
        bearing = det.bearing_rad
        if rng is None and bearing is None:
            return
        if rng is None:
            # Camera-only birth: no range at all. Seed at the median range at
            # which VRUs are actually observed in this dataset (37.8 m measured
            # over 1,230 visible-VRU observations; p10 7.1, p90 89.4) with a
            # variance wide enough to cover that spread, and let radar or the
            # bearing history pull it in.
            #
            # The previous 20.0 m was a guess and sat well below the median, so
            # a bearing-only track systematically underestimated range -- and
            # since lateral = range * sin(bearing), it underestimated lateral
            # position and velocity with it.
            rng, range_var = CAMERA_BIRTH_RANGE_M, CAMERA_BIRTH_RANGE_STD_M ** 2
        else:
            range_var = DEFAULT_RANGE_STD_M ** 2
        if bearing is None:
            bearing, bearing_var = 0.0, math.radians(30.0) ** 2
        else:
            bearing_var = DEFAULT_BEARING_STD_RAD ** 2

        fx, ly = rng * math.cos(bearing), -rng * math.sin(bearing)
        # Propagate polar variances into Cartesian (first-order).
        pos_var = range_var + (rng ** 2) * bearing_var
        P = np.diag([pos_var, pos_var, 4.0 ** 2, 4.0 ** 2])

        self.tracks.append(Track(id=self._next_id, x=np.array([fx, ly, 0.0, 0.0]),
                                  P=P, cls=det.cls, uncertainty=det.uncertainty,
                                  box_px=det.box_px))
        self._next_id += 1

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.is_confirmed]
