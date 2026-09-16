"""Component 8 -- Sensor Health Monitor.

Estimates, per frame, how much each sensor can currently be trusted, and feeds
that into `perception.contradiction` so fusion weighting responds to conditions
instead of being fixed at design time.

Why this is needed rather than assuming healthy sensors
-------------------------------------------------------
Every likelihood in the contradiction resolver assumes a sensor performing
normally. In heavy rain at night the camera is not performing normally: it is
dark, low-contrast, and streaked, and its detections and its *silences* both
mean less than they would at noon. A resolver that does not know this will read
"camera sees nothing" as evidence of a clear road when it is really evidence of
a blinded camera. The dangerous direction of that error is obvious.

Health is deliberately estimated from the sensor streams themselves, not from
simulator state. Nothing here reads CARLA's weather parameters or actor list,
because on a real vehicle no such oracle exists -- the system must infer its own
degradation from what it is receiving. That keeps this component on the runtime
side of the ground-truth/runtime split, and it is also what makes the adversarial
testing in component 9 meaningful: injected faults have to be *detected*, not
looked up.

Signals used
------------
Camera: mean brightness (darkness, over-exposure), RMS contrast (fog, spray,
glare), Laplacian variance (defocus, rain smear), a robust noise-sigma estimate,
and a tiled featureless-region check for a physically blocked lens. The last two
exist because the first three have blind spots that injected faults exposed --
noise *raises* Laplacian variance so it reads as sharpness, and a half-covered
lens leaves global brightness and contrast looking respectable.

Radar: return rate against this sensor's own established baseline (both dropout
and flooding), rate stability, and a residual-based calibration-drift test.

Cross-sensor: sustained disagreement, which says one sensor is wrong without
saying which -- so it lowers confidence in both, correctly.

Known limitation
----------------
A constant radar **range bias** remains the hardest fault to catch. It produces
the normal number of returns, perfectly stable, at plausible distances, simply
all wrong by a fixed offset. The innovation test below catches it while tracks
are initialising, but a converged filter absorbs a constant bias into its state
and the residuals return to zero -- the bias becomes unobservable without an
independent range reference. `scripts/adversarial_test.py` reports this rather
than hiding it.
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Camera thresholds, CALIBRATED against 120 randomly sampled real CARLA frames
# (Town10HD_Opt, clear day, 800x600). Measured distributions:
#
#     brightness   min  47.8  p10  145.5  median  172.6  p90    189.6  max    198.8
#     contrast     min  49.3  p10   56.2  median   64.6  p90     77.7  max     97.8
#     sharpness    min1386.0  p10 3818.0  median 6552.3  p90  13777.8  max  18537.1
#     noise_sigma  min   0.0  p10    1.4  median    6.7  p90     12.7  max     15.0
#
# An earlier hand-guessed set was wrong by an order of magnitude on two of
# these -- "good" sharpness was set at 300 when real frames run 1400-18500, and
# noise above 12 was treated as a fault when CARLA's own clean output reaches
# 15. The effect was a healthy camera scoring 0.38 on a perfect frame, which
# would have quietly suppressed camera evidence throughout the fusion. Guessing
# these from intuition does not work; they have to come from the data.
BRIGHTNESS_DARK, BRIGHTNESS_BRIGHT = 20.0, 235.0
BRIGHTNESS_GOOD_LOW, BRIGHTNESS_GOOD_HIGH = 50.0, 200.0

# RMS contrast below this suggests fog, spray or a washed-out scene.
CONTRAST_POOR, CONTRAST_GOOD = 15.0, 50.0

# High-frequency energy: sharp scenes have plenty, defocused or rain-smeared
# ones do not.
SHARPNESS_POOR, SHARPNESS_GOOD = 200.0, 1500.0

# Robust noise-sigma bands, in grey levels. CARLA's rendering plus JPEG
# compression already produces up to ~15, so the healthy band has to extend
# past that or every frame reads as a failing imager.
NOISE_GOOD, NOISE_BAD = 16.0, 40.0

# Fraction of featureless image tiles tolerated before calling the lens
# blocked. Some flat sky or plain tarmac is normal, so this is not zero.
BLOCKED_TOLERANCE = 0.10
TILE_GRID = 8

# Residual-based calibration-drift test. A t-statistic above this, with an
# offset of at least BIAS_MIN_M, marks a systematic range error rather than
# noise.
BIAS_T_THRESHOLD = 4.0
BIAS_MIN_M = 0.8

# Radar thresholds, CALIBRATED against 50 frames of real CARLA driving
# (Town10HD_Opt, autopilot, configs/bev.yaml radar block). Measured:
#
#     returns/frame  min 111  p10 121  median 143  p90 150  max 150
#     Clark-Evans    min 0.21 p10 0.24 median 0.52 p90 0.63 max 0.72
#
# Two corrections to earlier guesses, both large:
#
#   - The return rate was assumed to be ~15/frame on the reasoning that most
#     of the beam hits nothing. It does not: CARLA's radar returns ground and
#     structure everywhere, and at 1500 points/second with dt=0.1 it SATURATES
#     at its 150-point cap for most of a drive.
#   - Real returns are far less clustered than synthetic ones. An earlier
#     "clutter" threshold of 0.60 sat below the real median of 0.52 closely
#     enough to flag a large share of perfectly healthy frames.
#
# Consequence worth stating rather than hiding: because the sensor saturates at
# its configured cap, a flood of interference CANNOT raise the return count,
# so rate-based flood detection is inert at this configuration. Only the
# coherence index can see clutter here, and its headroom is narrow.
RADAR_EXPECTED_RETURNS = 140
RADAR_SPARSE = 30

# Multiple of the sensor's OWN established return rate that counts as a flood.
# Relative rather than absolute: what is "too many" depends on the scene and
# the sensor configuration, so any fixed count is wrong somewhere.
RADAR_FLOOD_RATIO = 2.5
BASELINE_HISTORY = 200

# Spatial-coherence check, via the Clark-Evans nearest-neighbour index.
# Values well below 1 mean returns are clustered on physical objects; values
# near 1 mean they are indistinguishable from uniform scatter, i.e. clutter.
# Matches configs/bev.yaml's radar block -- the sensor footprint the index
# normalises against.
RADAR_FOV_DEG = 35.0
RADAR_RANGE_M = 100.0
COHERENCE_MIN_POINTS = 6
INCOHERENCE_GOOD, INCOHERENCE_BAD = 0.72, 0.95

# Cross-sensor range check. Compares radar range against an independent estimate
# from camera object size (`geometry.range_from_pixel_height`). Per-observation
# accuracy is poor -- assumed-height error alone is ~12% -- so the test needs a
# long window and averages a systematic offset out of the noise. Sized so the
# standard error of the mean falls well under the smallest bias worth flagging.
# Minimum paired observations before the cross-range check activates at all.
#
# NOT tunable down without re-measuring. A first attempt at 200 samples looked
# right in isolation (the aggregate 5,805-pair mean matched the calibration
# almost exactly, deviation 0.001 m) but was wrong in the way that matters: at
# a 200-sample window, episode-to-episode variance in WHICH background return
# radar association happens to pick swings the mean by up to +/-16 m -- far
# wider than the 2.5 m bias this exists to catch. Measured spread of rolling
# window means (60 episodes, 5,805 pairs):
#
#     window   spread (max - min)
#        200   23.5 m
#        500   14.5 m
#      1,000    9.2 m
#      2,000    3.1 m
#      3,000    2.8 m
#
# Running with 200 made EVERY condition in scripts/adversarial_test.py --
# including clean, fog, blur, noise, which never touch the radar at all --
# read radar health 0.54, because the check was tripping on ordinary variance
# rather than on a fault. 2500 keeps spread under the fault size with margin;
# below ~1500 the check is not trustworthy regardless of how CROSS_BIAS_T is
# tuned, because the noise floor exceeds the signal.
CROSS_RANGE_MIN_SAMPLES = 2500
CROSS_RANGE_HISTORY = 4000

# Expected radar-minus-camera-size offset on a HEALTHY sensor pair, calibrated
# over 30 clean episodes of v2: mean +5.90 m, std 13.80.
#
# The offset is not zero and should not be forced to zero. Most of it is radar
# association rather than the height assumption: the range chosen for a
# detection's azimuth window often lands on background behind the pedestrian
# rather than the pedestrian, which reads systematically FAR. That is a stable
# property of the association method, so it calibrates out cleanly -- but it
# means an ABSOLUTE test on this difference would flag a perfectly healthy
# sensor, which is exactly what the first version of this check did.
#
# Calibrating against known-good data is the same standing this module's other
# thresholds have, and it is what makes the test work from startup: a fixed
# prior is not self-referential, so it does not become the fault.
# Re-measure if the association method, camera resolution, or radar config
# changes. `scripts/adversarial_test.py` is the harness that checks it still holds.
CROSS_RANGE_EXPECTED_M = 5.9
CROSS_BIAS_MIN_M = 1.5
CROSS_BIAS_T = 3.5

# Relative coherence test. Clean driving measured 0.383 and injected clutter
# 0.499 -- a 1.30x rise that the absolute band above cannot see, since both sit
# far below it. Set between the two with margin for scene variation, and judged
# against the sensor's own running median rather than a constant.
INCOHERENCE_RISE_RATIO = 1.22
COHERENCE_BASELINE_MIN = 40

HISTORY = 30                 # frames of rolling context (~3 s at dt=0.1)
DISAGREEMENT_WINDOW = 20


@dataclass
class HealthReport:
    camera: float                    # 0-1 usable confidence multiplier
    radar: float
    camera_reasons: list = field(default_factory=list)
    radar_reasons: list = field(default_factory=list)
    degraded: bool = False

    def __str__(self) -> str:
        s = f"camera {self.camera:.2f} | radar {self.radar:.2f}"
        if self.degraded:
            s += "  DEGRADED"
        return s

    def explain(self) -> str:
        lines = [str(self)]
        for name, reasons in (("camera", self.camera_reasons),
                               ("radar", self.radar_reasons)):
            for r in reasons:
                lines.append(f"  - {name}: {r}")
        return "\n".join(lines)


def _band(value: float, poor: float, good: float) -> float:
    """Maps a raw metric onto 0-1, clamped, poor -> 0 and good -> 1."""
    if good == poor:
        return 1.0
    return float(max(0.0, min(1.0, (value - poor) / (good - poor))))


def camera_metrics(rgb: np.ndarray) -> dict:
    """Brightness, contrast, sharpness, noise level and blocked-area fraction.

    Sharpness uses Laplacian variance, the standard no-reference blur measure.
    On its own it is **not sufficient**, and the failure is counter-intuitive:
    additive sensor noise *raises* Laplacian variance, so a badly noisy frame
    scores as exceptionally sharp. Measured on injected noise, camera health
    stayed at 1.00 while the image was visibly ruined. Two extra metrics close
    that and a related hole:

      - `noise_sigma` -- a robust noise estimate from the median absolute
        deviation of the Laplacian. Real image structure is sparse and
        heavy-tailed, so the MAD tracks the noise floor rather than the edges,
        letting noise be separated from detail instead of confused with it.
      - `blocked_fraction` -- the share of image tiles that are near-flat.
        Global brightness and contrast cannot see a lens half covered in mud:
        the uncovered half keeps the average respectable. A tiled check can.
    """
    arr = np.asarray(rgb)
    grey = arr.mean(axis=2) if arr.ndim == 3 else arr.astype(np.float32)

    brightness = float(grey.mean())
    contrast = float(grey.std())

    # Discrete Laplacian without a cv2 dependency, so this runs anywhere.
    lap = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
           - 4.0 * grey[1:-1, 1:-1])
    sharpness = float(lap.var())

    # Tile the frame once and reuse the tiling for two different questions.
    # 8x8 rather than 4x4: with only four tile-columns, a lens 40% covered
    # fully occupies just one of them and reads as 25% blocked. The finer grid
    # resolves partial coverage properly.
    h, w = grey.shape
    g = TILE_GRID
    ty, tx = max(h // g, 1), max(w // g, 1)
    tiles = grey[: ty * g, : tx * g].reshape(g, ty, g, tx).transpose(0, 2, 1, 3)
    flat = tiles.reshape(g * g, -1)

    # Blocked lens: tiles that are both featureless AND markedly darker than
    # the rest of the frame.
    #
    # Flatness alone is not enough and produces constant false alarms: open sky
    # and smooth tarmac are genuinely featureless, and on a clean driving frame
    # they made 38% of tiles look "blocked". What distinguishes an obstruction
    # is that it is dead *and* out of keeping with the scene's exposure -- mud,
    # tape or dirt occlude light rather than reflecting the scene.
    #
    # This deliberately targets dark obstructions. A bright one -- ice,
    # condensation, direct glare -- washes the frame out instead, and is caught
    # by the brightness and contrast checks above.
    tile_std = flat.std(axis=1)
    tile_mean = flat.mean(axis=1)
    frame_median = float(np.median(grey))
    blocked_fraction = float(((tile_std < 3.0)
                              & (tile_mean < frame_median * 0.5)).mean())

    # Noise, estimated from the FLATTEST tiles only.
    #
    # MAD over the whole Laplacian assumes image structure is sparse, so that
    # the median tracks the noise floor rather than the edges. That assumption
    # fails on densely textured content -- foliage, brick, gravel, or a test
    # pattern -- where a large share of pixels sit on an edge and the estimate
    # inflates badly: measured 22.9 on a scene whose true noise was 1.2, which
    # would wrongly condemn a perfectly good camera in a leafy street.
    #
    # Restricting the estimate to the quietest quarter of the frame removes the
    # dependence on scene content. Sky, road and building faces supply the noise
    # floor; textured tiles are simply excluded rather than mistaken for noise.
    lap_tiles = np.abs(lap[: (lap.shape[0] // g) * g, : (lap.shape[1] // g) * g])
    lh, lw = lap_tiles.shape[0] // g, lap_tiles.shape[1] // g
    lap_flat = lap_tiles.reshape(g, lh, g, lw).transpose(0, 2, 1, 3).reshape(g * g, -1)
    tile_mad = np.median(lap_flat, axis=1)
    quiet = float(np.percentile(tile_mad, 25))
    noise_sigma = float(quiet / 0.6745 / math.sqrt(6.0))

    return {"brightness": brightness, "contrast": contrast, "sharpness": sharpness,
            "noise_sigma": noise_sigma, "blocked_fraction": blocked_fraction}


def radar_metrics(radar_pts) -> dict:
    """Return count and spatial incoherence.

    Incoherence is the fraction of returns with no neighbour within
    `COHERENCE_RADIUS_M`. It is what actually separates clutter from signal,
    and it works where rate-based checks fail:

      - An absolute count threshold is wrong somewhere. A quiet street returns
        a handful; CARLA's 1500 points/s permits 150 per frame. Any fixed
        number is either deaf on a busy road or permanently alarmed on an
        empty one.
      - A baseline-relative check cannot see a fault that is present from the
        first frame, because the baseline simply learns the faulty rate.
        Measured: with clutter injected from frame one, relative radar health
        read 1.00 -- perfectly healthy -- while position RMSE rose 1.5 m.

    Geometry does not have that problem. Real reflections cluster on physical
    objects and persist; interference and multipath scatter uniformly through
    range and azimuth. The distinction holds whatever the absolute rate and
    needs no clean reference period.
    """
    pts = np.asarray(radar_pts, dtype=np.float64).reshape(-1, 4)
    # Drop non-finite rows before any trigonometry. Raw CARLA radar was
    # measured clean over 4,142 points, but a NaN reaching np.sin poisons the
    # whole nearest-neighbour matrix and silently returns a meaningless index
    # rather than failing -- the worst way for a health metric to break.
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = int(pts.shape[0])
    if n < COHERENCE_MIN_POINTS:
        return {"count": n, "incoherence": 0.0, "clark_evans": 0.0}

    forward = pts[:, 3] * np.cos(pts[:, 1])
    lateral = -pts[:, 3] * np.sin(pts[:, 1])
    xy = np.stack([forward, lateral], axis=1)

    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    mean_nn = float(d.min(axis=1).mean())

    # Clark-Evans nearest-neighbour index: observed mean spacing divided by the
    # spacing expected if the same number of points were scattered uniformly
    # over the sensor's footprint. For a Poisson process in 2D that expectation
    # is 1 / (2*sqrt(density)).
    #
    # A fixed-radius "has a neighbour within R" test was tried first and is
    # wrong in a way that inverts the result: pile in enough scatter and every
    # point acquires a neighbour, so heavy clutter scored as MORE coherent than
    # a clean frame. The index is density-normalised and has no such saturation
    # -- R well below 1 means clustered on real objects, R near 1 means the
    # returns are indistinguishable from uniform noise.
    area = 0.5 * math.radians(RADAR_FOV_DEG) * RADAR_RANGE_M ** 2
    expected_nn = 1.0 / (2.0 * math.sqrt(max(n / area, 1e-9)))
    index = mean_nn / max(expected_nn, 1e-6)

    return {"count": n, "incoherence": float(min(index, 1.5)),
            "clark_evans": float(index)}


class SensorHealthMonitor:
    """Rolling per-sensor health estimate.

    Judgements are made over a short history rather than per frame. A single
    dark frame is a tunnel or a shadow; thirty consecutive dark frames is
    night or a failed camera, and only the second should change how the system
    fuses.
    """

    def __init__(self, history: int = HISTORY):
        self.brightness = deque(maxlen=history)
        self.contrast = deque(maxlen=history)
        self.sharpness = deque(maxlen=history)
        self.noise_sigma = deque(maxlen=history)
        self.blocked_fraction = deque(maxlen=history)
        self.radar_counts = deque(maxlen=history)
        # Long-run rate, so a flood or dropout is judged against what this
        # sensor normally produces rather than an absolute constant.
        self.radar_baseline = deque(maxlen=BASELINE_HISTORY)
        self.incoherence = deque(maxlen=history)
        # Long-run scatter norm, for the relative clutter test. Longer than the
        # rolling window so a sustained fault cannot quietly become the baseline.
        self.incoherence_baseline = deque(maxlen=BASELINE_HISTORY)
        self.disagreements = deque(maxlen=DISAGREEMENT_WINDOW)
        self.range_innovations = deque(maxlen=history)
        # (radar_range, camera_range) pairs for the cross-sensor bias check.
        # Long window: the signal is a small systematic offset inside large
        # per-observation noise, so it only emerges in the mean.
        self.range_cross = deque(maxlen=CROSS_RANGE_HISTORY)

    def update(self, rgb=None, radar_pts=None, sensors_disagreed: bool | None = None,
                range_innovation: float | None = None,
                range_pairs: list | None = None):
        """`range_innovation` is one filter residual (measured range minus
        predicted range), from `tracking`. Feeding it in enables the
        calibration-drift check in `_radar_health`, which nothing else can see.
        """
        if rgb is not None:
            m = camera_metrics(rgb)
            self.brightness.append(m["brightness"])
            self.contrast.append(m["contrast"])
            self.sharpness.append(m["sharpness"])
            self.noise_sigma.append(m["noise_sigma"])
            self.blocked_fraction.append(m["blocked_fraction"])
        if radar_pts is not None:
            rm = radar_metrics(radar_pts)
            self.radar_counts.append(rm["count"])
            self.radar_baseline.append(rm["count"])
            self.incoherence.append(rm["incoherence"])
            self.incoherence_baseline.append(rm["incoherence"])
        if sensors_disagreed is not None:
            self.disagreements.append(bool(sensors_disagreed))
        if range_innovation is not None and math.isfinite(range_innovation):
            self.range_innovations.append(float(range_innovation))
        for radar_r, cam_r in (range_pairs or ()):
            if (radar_r is not None and cam_r is not None
                    and math.isfinite(radar_r) and math.isfinite(cam_r)):
                self.range_cross.append((float(radar_r), float(cam_r)))

    # ---------------------------------------------------------------- camera

    def _camera_health(self):
        if not self.brightness:
            return 1.0, []

        reasons = []
        brightness = float(np.mean(self.brightness))
        contrast = float(np.mean(self.contrast))
        sharpness = float(np.mean(self.sharpness))

        if brightness < BRIGHTNESS_GOOD_LOW:
            b = _band(brightness, BRIGHTNESS_DARK, BRIGHTNESS_GOOD_LOW)
            reasons.append(f"low light (mean brightness {brightness:.0f})")
        elif brightness > BRIGHTNESS_GOOD_HIGH:
            b = _band(BRIGHTNESS_BRIGHT - brightness, 0.0,
                      BRIGHTNESS_BRIGHT - BRIGHTNESS_GOOD_HIGH)
            reasons.append(f"over-exposed (mean brightness {brightness:.0f})")
        else:
            b = 1.0

        c = _band(contrast, CONTRAST_POOR, CONTRAST_GOOD)
        if c < 0.6:
            reasons.append(f"low contrast ({contrast:.0f}) -- fog, spray or glare")

        s = _band(sharpness, SHARPNESS_POOR, SHARPNESS_GOOD)
        if s < 0.6:
            reasons.append(f"soft image (sharpness {sharpness:.0f}) -- defocus, "
                            f"rain on the lens or an obscured lens")

        # Noise. Checked separately from sharpness because it pushes that
        # metric the WRONG way -- noise raises Laplacian variance, so without
        # this a heavily noisy frame reads as unusually sharp and healthy.
        n = 1.0
        if self.noise_sigma:
            sigma = float(np.mean(self.noise_sigma))
            n = _band(NOISE_BAD - sigma, 0.0, NOISE_BAD - NOISE_GOOD)
            if n < 0.6:
                reasons.append(f"heavy sensor noise (sigma {sigma:.1f}) -- high gain "
                                f"or a failing imager")

        # Localised blockage. Global statistics cannot see this: half a frame
        # under mud still averages to a plausible brightness and contrast.
        k = 1.0
        if self.blocked_fraction:
            blocked = float(np.mean(self.blocked_fraction))
            if blocked > BLOCKED_TOLERANCE:
                # Squared, not linear. A camera with a blind sector is worse
                # than "proportionally less useful": nothing is known about what
                # occupies that sector, so the chance of an unseen hazard scales
                # with the blocked area while the consequence stays constant. A
                # linear penalty left a 40%-blocked lens scoring 0.77, i.e.
                # healthy.
                k = float(max(0.05, (1.0 - blocked) ** 2))
                reasons.append(f"{blocked:.0%} of the frame is featureless -- lens "
                                f"blocked by mud, ice or an obstruction")

        # Multiplied, not averaged: these degradations compound, and a camera
        # that is dark AND blurred is far worse than one with either alone.
        # Averaging would let one good score mask a blinded sensor.
        health = b * c * s * n * k
        return float(max(0.05, min(1.0, health))), reasons

    # ----------------------------------------------------------------- radar

    def _radar_health(self):
        if not self.radar_counts:
            return 1.0, []

        reasons = []
        mean_count = float(np.mean(self.radar_counts))

        if mean_count <= RADAR_SPARSE:
            reasons.append(f"very few returns ({mean_count:.1f}/frame) -- possible "
                            f"blockage, misalignment or sensor failure")
        density = _band(mean_count, 0.0, RADAR_EXPECTED_RETURNS)

        # Too MANY returns is a fault too, and an easy one to miss: interference
        # and multipath inflate the count, so a density-only score reads the
        # flood as a perfectly healthy sensor. Measured on injected clutter,
        # health sat at 1.00 while position RMSE rose 1.5 m.
        #
        # Judged against this sensor's OWN established rate rather than an
        # absolute threshold. What counts as "too many" depends entirely on the
        # scene and configuration -- CARLA's 1500 points/s allows 150 per frame,
        # while a quiet street returns a handful -- so any fixed number is
        # either deaf on a busy road or permanently alarmed on an empty one. A
        # sudden multiple of the sensor's normal rate is a fault regardless of
        # the absolute count.
        if len(self.radar_baseline) >= 20:
            baseline = float(np.median(self.radar_baseline))
            if baseline > 1.0 and mean_count > RADAR_FLOOD_RATIO * baseline:
                ratio = mean_count / baseline
                density *= float(max(0.25, 1.0 - min((ratio - RADAR_FLOOD_RATIO) / 3.0, 0.75)))
                reasons.append(f"return rate {ratio:.1f}x its established baseline "
                                f"({mean_count:.0f} vs {baseline:.0f}/frame) -- "
                                f"interference, multipath or clutter")

        # Wildly varying return counts suggest an intermittent fault rather
        # than a genuinely changing scene, which varies smoothly.
        stability = 1.0
        if len(self.radar_counts) >= 5:
            counts = np.array(self.radar_counts, dtype=float)
            cv = counts.std() / max(counts.mean(), 1e-6)
            stability = float(max(0.2, min(1.0, 1.5 - cv)))
            if stability < 0.6:
                reasons.append(f"unstable return rate (CV {cv:.2f}) -- intermittent fault")

        # Calibration drift, detected from the tracking filter's residuals.
        #
        # This is the one radar fault invisible in the radar stream itself: a
        # range bias produces the normal number of returns, perfectly stable,
        # at plausible distances -- simply all wrong by a constant. Measured on
        # an injected 2.5 m bias, position RMSE rose 2.4 m while count- and
        # stability-based health sat at 0.87 and reported nothing.
        #
        # The signature is in the innovations. An unbiased sensor produces
        # residuals scattered about zero; a biased one produces a persistent
        # offset. Testing the mean against the spread is the standard
        # residual-based fault detector, and it needs no reference sensor.
        bias_factor = 1.0
        if len(self.range_innovations) >= 10:
            inn = np.array(self.range_innovations, dtype=float)
            mean, spread = float(inn.mean()), float(inn.std())
            if spread > 1e-6:
                t_stat = abs(mean) / (spread / math.sqrt(len(inn)))
                if t_stat > BIAS_T_THRESHOLD and abs(mean) > BIAS_MIN_M:
                    bias_factor = float(max(0.2, 1.0 - min(abs(mean) / 5.0, 0.8)))
                    reasons.append(f"systematic range offset of {mean:+.1f} m in the "
                                    f"filter residuals -- calibration drift")

        # Cross-sensor range check -- the only test here that can see a bias
        # present from startup.
        #
        # Everything else in this module is self-referential: it compares the
        # radar against its own history or its own statistics. That is blind by
        # construction to a fault that was there before the monitor started,
        # because the fault IS the baseline. The residual test above has the
        # same blind spot for a different reason -- the tracking filter absorbs
        # a constant offset completely, since an object 2.5 m further away at
        # the same bearing is a consistent world state, so its residuals return
        # to zero once tracks converge.
        #
        # Camera object size gives a range estimate the radar has no say in.
        # Any single one is poor (assumed-height error alone is ~12%), but the
        # bias is systematic and the noise is not, so the mean separates them
        # given enough observations.
        cross_factor = 1.0
        if len(self.range_cross) >= CROSS_RANGE_MIN_SAMPLES:
            pairs = np.array(self.range_cross, dtype=float)
            # Deviation from the CALIBRATED offset, not from zero.
            diff = pairs[:, 0] - pairs[:, 1] - CROSS_RANGE_EXPECTED_M
            mean, spread = float(diff.mean()), float(diff.std())
            if spread > 1e-6:
                t_stat = abs(mean) / (spread / math.sqrt(len(diff)))
                if t_stat > CROSS_BIAS_T and abs(mean) > CROSS_BIAS_MIN_M:
                    cross_factor = float(max(0.2, 1.0 - min(abs(mean) / 5.0, 0.8)))
                    direction = "far" if mean > 0 else "near"
                    reasons.append(f"radar reads {abs(mean):.1f} m too {direction} "
                                    f"against camera object size over "
                                    f"{len(diff)} observations -- range calibration")

        # Spatial coherence, judged BOTH absolutely and against this sensor's
        # own established baseline.
        #
        # The absolute band alone could not see real clutter, and the margin was
        # not close. Measured over 12 episodes: clean driving scores 0.383 and a
        # 40-point interference injection moves it to 0.499 -- still below the
        # 0.52 median of ordinary driving, let alone the 0.72 where the absolute
        # band starts to react. A fixed threshold cannot separate them, because
        # what counts as coherent depends on the scene: an empty street and a
        # junction full of parked cars differ by more than the fault does.
        #
        # The relative test is the same principle already used for return rate
        # a few lines above. A sensor whose scatter jumps well above its own
        # running norm has changed, whatever the absolute number happens to be.
        coherence = 1.0
        if self.incoherence:
            inc = float(np.mean(self.incoherence))
            coherence = _band(INCOHERENCE_BAD - inc, 0.0,
                               INCOHERENCE_BAD - INCOHERENCE_GOOD)
            if coherence < 0.6:
                reasons.append(f"{inc:.0%} of returns are spatially isolated -- "
                                f"scatter rather than reflections off real objects")

            if len(self.incoherence_baseline) >= COHERENCE_BASELINE_MIN:
                base = float(np.median(self.incoherence_baseline))
                if base > 1e-6 and inc > base * INCOHERENCE_RISE_RATIO:
                    rel = float(max(0.25, 1.0 - min((inc / base - 1.0), 0.75)))
                    coherence = min(coherence, rel)
                    reasons.append(f"return scatter is {inc / base:.1f}x this sensor's "
                                    f"own norm ({inc:.2f} against {base:.2f}) -- clutter")

        health = ((0.6 * density + 0.4 * stability)
                   * bias_factor * cross_factor * coherence)
        return float(max(0.05, min(1.0, health))), reasons

    # ---------------------------------------------------------------- report

    def report(self) -> HealthReport:
        cam, cam_reasons = self._camera_health()
        rad, rad_reasons = self._radar_health()

        # Sustained disagreement means one sensor is wrong but not which. The
        # honest response is to lower confidence in both: picking a side
        # without evidence is how a fusion system talks itself into trusting
        # the failed sensor.
        if len(self.disagreements) >= 5:
            rate = float(np.mean(self.disagreements))
            if rate > 0.5:
                penalty = 1.0 - 0.4 * (rate - 0.5) / 0.5
                cam *= penalty
                rad *= penalty
                msg = (f"sensors disagreed on {rate:.0%} of recent frames -- one is "
                        f"wrong, so neither is fully trusted")
                cam_reasons.append(msg)
                rad_reasons.append(msg)

        return HealthReport(camera=float(cam), radar=float(rad),
                             camera_reasons=cam_reasons, radar_reasons=rad_reasons,
                             degraded=bool(cam < 0.6 or rad < 0.6))
