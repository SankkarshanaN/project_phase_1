"""Component 5 -- Bayesian Contradiction Resolver.

Camera and radar routinely disagree. This module decides what to believe, and
says why.

The four cases, and why a fixed fusion weight gets them wrong
-------------------------------------------------------------
    camera sees | radar sees | the honest reading
    ------------|------------|------------------------------------------
        yes     |    yes     | agreement -- high confidence
        yes     |    no      | pedestrian (small radar cross-section), or a
                |            | camera false positive
        no      |    yes     | something hidden from the camera, or radar
                |            | clutter off the road surface
        no      |    no      | empty road -- OR a fully-occluded hazard that
                |            | neither sensor can see

Static confidence weighting, which is what most fusion pipelines use, cannot
separate the pairs in rows 2-4. It has no way to ask *why* a sensor is silent.

The occlusion grid supplies exactly that missing information. If a cell is
OCCLUDED, camera silence carries almost no evidence -- of course it sees
nothing, its line of sight is blocked -- so radar should dominate and the
absence of a detection should not be read as "clear". If the cell is EMPTY
(camera-confirmed clear road), camera silence is strong evidence of absence and
an isolated radar return is more likely clutter.

So the likelihoods here are conditioned on occlusion state. That conditioning
is the entire point of the component, and it is what makes "occlusion-aware"
mean something operationally rather than being a label on the architecture
diagram.

Output is a posterior probability of existence plus a plain-language
explanation of which evidence drove it, so a downstream warning can be
justified to a human rather than asserted.
"""
import math
from dataclasses import dataclass, field

from perception.occlusion_grid import EMPTY, OCCLUDED, UNKNOWN, VISIBLE

# Prior probability that any given watched region contains a real hazard.
# Deliberately low: most of the road, most of the time, is empty. Setting this
# near 0.5 would make the resolver credulous about isolated clutter.
DEFAULT_PRIOR = 0.15

# P(camera reports a detection | an object is really there), by occlusion state.
# The OCCLUDED entry is the load-bearing one: a camera physically cannot see
# through a bus, so its silence there is close to uninformative.
CAMERA_DETECT_GIVEN_OBJECT = {
    VISIBLE: 0.92,
    EMPTY: 0.90,
    UNKNOWN: 0.60,
    OCCLUDED: 0.05,
}

# P(camera reports a detection | nothing is there) -- the false-positive rate.
# Higher in OCCLUDED/UNKNOWN cells because clutter, shadows and partial
# glimpses of the occluder itself are what generate spurious boxes.
CAMERA_DETECT_GIVEN_EMPTY = {
    VISIBLE: 0.04,
    EMPTY: 0.03,
    UNKNOWN: 0.10,
    OCCLUDED: 0.12,
}

# Radar is far less affected by optical occlusion -- it diffracts and
# multipaths around obstacles -- but it is not immune: a large metal vehicle
# genuinely does block returns from directly behind it.
RADAR_DETECT_GIVEN_OBJECT = {
    VISIBLE: 0.85,
    EMPTY: 0.80,
    UNKNOWN: 0.70,
    OCCLUDED: 0.45,
}

RADAR_DETECT_GIVEN_EMPTY = {
    VISIBLE: 0.12,
    EMPTY: 0.10,
    UNKNOWN: 0.18,
    OCCLUDED: 0.20,
}

# A pedestrian's radar cross-section is small -- roughly 0.5 m^2 against a
# vehicle's 10-100 m^2 -- so radar silence about a VRU is much weaker evidence
# of absence than radar silence about a car. Ignoring this is a classic way to
# talk yourself out of a pedestrian.
VRU_RADAR_PENALTY = 0.55

CONFLICT_THRESHOLD = 0.35   # how far apart the two sensors must be to "contradict"


@dataclass
class SensorEvidence:
    """One region's worth of evidence from both sensors."""
    camera_detected: bool = False
    camera_confidence: float = 0.0      # evidential Dirichlet mean for the class
    # Evidential uncertainty, 0 = certain. Used ONLY when camera_detected is
    # True; a non-detection has no classifier output to be uncertain about.
    camera_uncertainty: float = 0.0
    radar_detected: bool = False
    radar_confidence: float = 0.0       # perception.radar_confidence score
    occlusion_state: int = UNKNOWN      # from perception.occlusion_grid
    is_vru: bool = True
    camera_health: float = 1.0          # 0-1, from perception.sensor_health
    radar_health: float = 1.0


@dataclass
class Resolution:
    posterior: float                    # P(hazard exists | all evidence)
    prior: float
    camera_likelihood_ratio: float
    radar_likelihood_ratio: float
    contradicted: bool
    trusted: str                        # "camera", "radar", "both", "neither"
    explanation: str
    reasons: list = field(default_factory=list)

    def __str__(self) -> str:
        return f"P(hazard)={self.posterior:.2f} trust={self.trusted} -- {self.explanation}"


def _clamp(p: float, lo: float = 1e-4, hi: float = 1.0 - 1e-4) -> float:
    return max(lo, min(hi, p))


def _likelihood_ratio(detected: bool, p_given_object: float, p_given_empty: float,
                       reliability: float) -> float:
    """LR = P(observation | object) / P(observation | no object).

    `reliability` in [0, 1] shrinks the ratio toward 1 (uninformative) as the
    sensor becomes less trustworthy. A blinded camera should not be able to
    argue either for or against a hazard, and multiplying its likelihoods by a
    weight would not achieve that -- only pulling the *ratio* toward unity
    does, which is what the exponent below accomplishes.
    """
    p_obj = _clamp(p_given_object if detected else 1.0 - p_given_object)
    p_emp = _clamp(p_given_empty if detected else 1.0 - p_given_empty)
    lr = p_obj / p_emp
    r = max(0.0, min(1.0, reliability))
    return float(lr ** r)      # r=0 -> LR=1, the sensor says nothing at all


def resolve(evidence: SensorEvidence, prior: float = DEFAULT_PRIOR) -> Resolution:
    """Fuses camera and radar evidence into a posterior, with an explanation."""
    state = evidence.occlusion_state
    reasons = []

    # A camera DETECTION in a cell the grid called OCCLUDED is an internal
    # contradiction in the camera's own output, and the resolution is that the
    # grid is wrong -- not that the detection is spurious.
    #
    # Taking the OCCLUDED likelihoods at face value here inverts the component:
    # with P(detect | object, occluded) = 0.05 against P(detect | empty,
    # occluded) = 0.12, a detection yields a likelihood ratio of 0.42 and
    # therefore argues AGAINST a hazard. Measured, that put "both sensors see
    # it, cell occluded" at P(hazard) = 0.09, below camera-only in a clear
    # cell -- so a confident pedestrian detection would be suppressed exactly
    # when the occlusion grid mis-segmented, which is the dangerous direction.
    #
    # The grid is the less reliable of the two signals here: it is derived from
    # relative MiDaS disparity, its measured OCCLUDED recall is 0.55, and the
    # detection's cell assignment relies on an approximate range estimate. A
    # detection is direct evidence that the line of sight was not in fact
    # blocked, so the camera is scored under UNKNOWN instead.
    cam_state = state
    if evidence.camera_detected and state == OCCLUDED:
        cam_state = UNKNOWN
        reasons.append("camera detected something in a cell the grid called "
                        "OCCLUDED -- treating the grid as mistaken rather than "
                        "the detection as spurious")

    p_cam_obj = CAMERA_DETECT_GIVEN_OBJECT.get(cam_state, 0.6)
    p_cam_emp = CAMERA_DETECT_GIVEN_EMPTY.get(cam_state, 0.1)
    p_rad_obj = RADAR_DETECT_GIVEN_OBJECT.get(state, 0.7)
    p_rad_emp = RADAR_DETECT_GIVEN_EMPTY.get(state, 0.15)

    if evidence.is_vru:
        # Weakens radar's claim to have seen something AND its claim not to
        # have -- both directions, because a small cross-section makes the
        # sensor less informative either way.
        p_rad_obj *= VRU_RADAR_PENALTY
        reasons.append("target is a VRU: small radar cross-section, so radar "
                        "silence is weak evidence of absence")

    # Sensor reliability. Detections and non-detections are weighted
    # differently, and conflating them is a subtle but consequential error:
    #
    #   - A DETECTION carries the evidential head's own uncertainty, so an
    #     ambiguous crop should not push the posterior as hard as a clean one.
    #   - A NON-detection has no evidential output at all -- the head never ran,
    #     because there was no box to score. Folding `camera_uncertainty` in
    #     here anyway drives reliability to zero whenever it is left at its
    #     default, which silently removes the camera from the fusion in exactly
    #     the "nothing was seen" cases this component exists to reason about.
    #     A silent camera's weight is a question of whether it *could* have
    #     seen, which is health and occlusion state, not classifier confidence.
    if evidence.camera_detected:
        cam_reliability = (evidence.camera_health
                            * (1.0 - float(max(0.0, min(1.0, evidence.camera_uncertainty))))
                            * max(0.2, evidence.camera_confidence))
    else:
        cam_reliability = evidence.camera_health

    if evidence.radar_detected:
        rad_reliability = evidence.radar_health * max(0.2, evidence.radar_confidence)
    else:
        rad_reliability = evidence.radar_health

    cam_lr = _likelihood_ratio(evidence.camera_detected, p_cam_obj, p_cam_emp,
                                cam_reliability)
    rad_lr = _likelihood_ratio(evidence.radar_detected, p_rad_obj, p_rad_emp,
                                rad_reliability)

    # Naive-Bayes combination in odds form. Conditional independence given the
    # object state is an approximation -- the two sensors share the scene, so
    # heavy rain degrades both -- and it is handled where it actually bites, by
    # the sensor health monitor lowering both reliabilities together rather
    # than by a correlation term here.
    prior_odds = prior / (1.0 - prior)
    posterior_odds = prior_odds * cam_lr * rad_lr
    posterior = posterior_odds / (1.0 + posterior_odds)

    contradicted = (evidence.camera_detected != evidence.radar_detected
                     and abs(math.log(max(cam_lr, 1e-6)) - math.log(max(rad_lr, 1e-6)))
                     > CONFLICT_THRESHOLD)

    if evidence.camera_detected and evidence.radar_detected:
        trusted = "both"
        explanation = "camera and radar agree"
    elif evidence.camera_detected and not evidence.radar_detected:
        trusted = "camera"
        explanation = ("camera sees it, radar does not -- expected for a "
                        "pedestrian" if evidence.is_vru else
                        "camera sees it, radar does not")
    elif evidence.radar_detected and not evidence.camera_detected:
        if state == OCCLUDED:
            trusted = "radar"
            explanation = ("radar sees something the camera cannot -- the view "
                            "is blocked, so camera silence proves nothing")
            reasons.append("cell is OCCLUDED: camera non-detection is "
                            "uninformative and is discounted")
        else:
            trusted = "radar"
            explanation = "radar-only return in a viewable cell -- possible clutter"
    else:
        trusted = "neither"
        if state == OCCLUDED:
            explanation = ("neither sensor reports anything, but the view is "
                            "blocked -- absence of evidence is not evidence of absence")
            reasons.append("cell is OCCLUDED: 'clear' cannot be concluded here")
        else:
            explanation = "neither sensor reports anything and the view is clear"

    if state == OCCLUDED and not evidence.camera_detected:
        reasons.append(f"camera likelihood ratio held near 1.0 ({cam_lr:.2f}) "
                        f"because it could not have seen the target anyway")

    return Resolution(posterior=float(posterior), prior=prior,
                       camera_likelihood_ratio=float(cam_lr),
                       radar_likelihood_ratio=float(rad_lr),
                       contradicted=bool(contradicted), trusted=trusted,
                       explanation=explanation, reasons=reasons)


def resolve_grid(occlusion_grid, radar_hits, camera_hits=None,
                  prior: float = DEFAULT_PRIOR):
    """Posterior hazard probability for every BEV cell.

    Returns a float array the same shape as the grid. Useful as a
    hazard-probability map alongside the categorical occlusion grid: the
    categorical grid says what is *visible*, this says what is probably
    *there*, and the two differ precisely in the cells that matter.
    """
    import numpy as np

    grid = np.asarray(occlusion_grid)
    radar_hits = np.asarray(radar_hits, dtype=bool)
    camera_hits = (np.zeros_like(radar_hits) if camera_hits is None
                   else np.asarray(camera_hits, dtype=bool))

    out = np.zeros(grid.shape, dtype=np.float32)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ev = SensorEvidence(
                camera_detected=bool(camera_hits[i, j]),
                camera_confidence=0.8 if camera_hits[i, j] else 0.0,
                camera_uncertainty=0.2,
                radar_detected=bool(radar_hits[i, j]),
                radar_confidence=0.7 if radar_hits[i, j] else 0.0,
                occlusion_state=int(grid[i, j]),
                is_vru=True,
            )
            out[i, j] = resolve(ev, prior).posterior
    return out
