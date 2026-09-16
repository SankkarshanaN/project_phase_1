"""The integrated framework: all nine methodology components in one pass.

Confidence-Driven Occlusion-Aware Camera-Radar Fusion & Hidden Hazard
Prediction. Each component exists in its own module and is unit-tested there;
this is where they are wired into a single per-frame pipeline, and where the
couplings between them live.

Per-frame flow
--------------
    RGB + radar
        |
        v
    [8] sensor health          -- how much is each sensor worth right now?
        |
        v
    [3] YOLO + evidential head -- where, what, and how sure
        |
        v
    [4] occlusion grid         -- which cells can be seen at all
        |
        v
    [5] contradiction resolver -- what to believe, given what COULD be seen
        |
        v
    [EKF] tracking             -- while the target is visible
        |
        +--- target goes fully occluded --->  [6] particle filter
        |                                          (multi-hypothesis,
        |    target reappears <-- re-identify       emergence prediction)
        v
    [intent] crossing probability
        |
        v
    [7] risk scoring           -- one number, one action, and the reasons

Where the couplings actually matter
-----------------------------------
Three of these connections do real work, and they are the reason the framework
is more than nine modules in a directory:

  - **health -> contradiction.** A blinded camera's silence stops counting as
    evidence of a clear road. Without this the fusion reads a failed sensor as
    good news, which is the dangerous direction.
  - **occlusion grid -> contradiction.** "Nothing detected" means something
    entirely different in a cell the sensors can see than in one they cannot.
  - **EKF -> particle filter.** The handoff at the moment a target becomes
    fully occluded is what keeps a hazard alive after it disappears, with the
    belief splitting into separate hypotheses rather than one Gaussian blob
    centred inside the occluder.

Every heavy component is optional. Without YOLO and the evidential head the
pipeline still runs on supplied detections, and without MiDaS it accepts a
precomputed occlusion grid -- which is what makes the same code usable for
offline replay of recorded frames as for the live demo.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from common.config import load_yaml
from perception.contradiction import SensorEvidence, resolve
from perception.geometry import (
    ego_xy_to_bev_cell, pixel_range_to_azimuth_range, pixel_to_bearing,
    radar_in_azimuth_window, radar_range_for_window, range_from_pixel_height,
)
from perception.intent import predict_crossing
from perception.occlusion_grid import OCCLUDED, UNKNOWN, classify_grid
from perception.particle_tracker import HiddenHazardTracker
from perception.radar_confidence import radar_confidence, radar_confidence_in_region
from perception.risk import assess
from perception.sensor_health import SensorHealthMonitor
from perception.tracking import Detection, MultiObjectTracker

# COCO ids YOLOv8n was pretrained on that matter here, mapped onto our classes.
COCO_TO_OURS = {0: 1, 2: 0, 5: 0, 7: 0}     # person -> VRU, car/bus/truck -> vehicle

# Frames a confirmed track may go unmeasured before it is handed to the
# particle filter. Two is enough to distinguish a genuine occlusion from a
# single dropped detection, without losing so much time that the seed state is
# stale.
HANDOFF_AFTER_MISSES = 2

# Particle mass that must agree with a reappearing detection for it to be
# treated as the same hazard rather than a new one.
REID_THRESHOLD = 0.25

# Ceiling on concurrent hidden-hazard beliefs. Each is a 400-particle filter,
# and more than a handful genuinely occluded actors at once means something has
# gone wrong upstream rather than that the street is unusually busy.
MAX_HIDDEN = 6


@dataclass
class ObjectResult:
    """One detected object and everything the framework concluded about it."""
    box_px: tuple
    cls: int
    confidence: float
    uncertainty: float
    radar_confidence: float
    hazard_posterior: float
    trusted: str
    explanation: str
    track_id: int | None = None
    p_cross: float | None = None
    p_cross_cautious: float | None = None


@dataclass
class FrameResult:
    objects: list = field(default_factory=list)
    occlusion_grid: np.ndarray | None = None
    health: object = None
    risk: object = None
    tracks: list = field(default_factory=list)
    hidden: list = field(default_factory=list)      # (id, EmergencePrediction)
    frame_radar_confidence: float = 0.0

    def summary(self) -> str:
        lines = [f"objects {len(self.objects)} | tracks {len(self.tracks)} | "
                 f"hidden {len(self.hidden)} | {self.health}"]
        if self.risk is not None:
            lines.append(self.risk.explain())
        return "\n".join(lines)


class PerceptionPipeline:
    """Stateful across frames -- holds the trackers, hidden-hazard beliefs and
    sensor-health history. One instance per drive."""

    def __init__(self, bev_cfg: dict | None = None, device: str = "cpu",
                  yolo=None, evidential=None, rng=None):
        self.bev_cfg = bev_cfg or load_yaml("bev.yaml")
        self.cam_cfg = self.bev_cfg["camera"]
        self.grid_cfg = self.bev_cfg["bev_grid"]
        # Half-angle of the radar's horizontal coverage. Needed because the
        # camera sees far wider than the radar, so "radar returned nothing" is
        # only informative inside this cone. See `process`.
        self.radar_half_fov = math.radians(
            float(self.bev_cfg.get("radar", {}).get("horizontal_fov", 35.0))) / 2.0
        self.device = device
        self.yolo = yolo
        self.evidential = evidential
        self.rng = np.random.default_rng(0) if rng is None else rng

        self.tracker = MultiObjectTracker()
        self.health = SensorHealthMonitor()
        self.hidden: dict[int, HiddenHazardTracker] = {}
        self._last_seen: dict[int, tuple] = {}     # track id -> (xy, vel, P)

    # ------------------------------------------------------------- detection

    def _detect(self, rgb):
        """YOLO proposals re-scored by the evidential head.

        Returns [(box, cls, confidence, uncertainty)]. Empty if no detector is
        configured -- callers doing offline replay supply boxes directly.
        """
        if self.yolo is None or rgb is None:
            return []

        import torch
        from models.evidential_classifier import CROP_SIZE
        from models.evidential_head import EvidentialHead

        results = self.yolo.predict(rgb, conf=0.35, classes=list(COCO_TO_OURS),
                                     verbose=False)[0]
        out = []
        for box, coco in zip(results.boxes.xyxy.cpu().numpy(),
                              results.boxes.cls.cpu().numpy().astype(int)):
            cls = COCO_TO_OURS.get(int(coco))
            if cls is None:
                continue
            conf, unc = 0.5, 0.5
            if self.evidential is not None:
                import cv2
                x1, y1, x2, y2 = (int(v) for v in box)
                crop = rgb[max(0, y1):y2, max(0, x1):x2]
                if crop.size:
                    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE)).astype(np.float32) / 255.0
                    t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        alpha, u = self.evidential(t)
                        probs = EvidentialHead.expected_probability(alpha)[0]
                    conf, unc = float(probs[cls]), float(u.item())
            out.append((tuple(float(v) for v in box), cls, conf, unc))
        return out

    # ------------------------------------------------------------ occlusion

    def _cell_state(self, grid, xy) -> int:
        if grid is None:
            return UNKNOWN
        n = grid.shape[0]
        fi, li = ego_xy_to_bev_cell(np.array([xy[0]]), np.array([xy[1]]), self.grid_cfg)
        if 0 <= fi[0] < n and 0 <= li[0] < n:
            return int(grid[fi[0], li[0]])
        return UNKNOWN

    # --------------------------------------------------------------- hidden

    def _handoff(self, track) -> None:
        """Seed a particle filter from a track that has just gone fully occluded.

        Seeded from the LAST CONFIRMED estimate rather than the current coasted
        one: by the time the misses have accumulated the EKF has already
        dead-reckoned for several frames, and handing over its drifted state
        would bake that error into every particle.
        """
        if track.id in self.hidden:
            return
        seed = self._last_seen.get(track.id)
        if seed is None:
            return
        xy, vel, P = seed
        self.hidden[track.id] = HiddenHazardTracker(
            init_xy=xy, init_vel=vel, init_cov=P, rng=self.rng,
            is_vru=(track.cls == 1))

    def _prune(self) -> None:
        """Drop state belonging to tracks the tracker has retired.

        Without this both `_last_seen` and `hidden` grow without bound over a
        long drive, and every retired track leaves a particle filter running
        forever. Worse, a tracker that spawns duplicate tracks for one actor
        then spawns a hidden filter per duplicate -- observed as seven
        concurrent beliefs about a single pedestrian, each feeding the risk
        engine separately.
        """
        live = {t.id for t in self.tracker.tracks}
        for tid in list(self._last_seen):
            if tid not in live and tid not in self.hidden:
                del self._last_seen[tid]
        # A hidden hazard outlives its track by design -- that is the point --
        # but only while the belief is still meaningful; `_update_hidden`
        # retires it on re-identification, degeneracy or timeout.
        if len(self.hidden) > MAX_HIDDEN:
            oldest = sorted(self.hidden.items(), key=lambda kv: -kv[1].frames_hidden)
            for tid, _pf in oldest[: len(self.hidden) - MAX_HIDDEN]:
                del self.hidden[tid]

    def _update_hidden(self, dt, ego_speed, ego_yaw_rate, grid, detections_xy):
        """Advance every hidden-hazard belief and prune the resolved ones."""
        out = []
        for tid, pf in list(self.hidden.items()):
            pf.predict(dt, ego_speed=ego_speed, ego_yaw_rate=ego_yaw_rate)
            # Deliberately NOT `detections_xy` here, despite this method
            # receiving that exact list -- `det_xy` is every detection in the
            # frame, any class, anywhere, not ones near this particular hidden
            # hazard. Feeding it into per-frame reweighting would let an
            # unrelated pedestrian or car pull the belief toward itself on
            # every single frame. Reidentification (below) is the one place
            # detections_xy is used against a hidden hazard: a deliberate,
            # high-bar, one-shot decision (>= REID_THRESHOLD mass, then
            # revive-and-retire) rather than continuous soft contamination.
            pf.update_from_occlusion(grid, self.grid_cfg, detections_xy=None)

            # Re-identification: has this hazard reappeared?
            revived = False
            for xy in (detections_xy or []):
                if pf.reidentification_score(xy) >= REID_THRESHOLD:
                    revived = True
                    break
            if revived or pf.is_degenerate or pf.frames_hidden > 60:
                del self.hidden[tid]
                continue

            if grid is not None:
                out.append((tid, pf.predict_emergence(grid, self.grid_cfg)))
        return out

    # ----------------------------------------------------------------- main

    def process(self, rgb=None, radar_pts=None, ego_speed: float = 0.0,
                 ego_yaw_rate: float = 0.0, dt: float = 0.1,
                 occlusion_grid=None, detections=None) -> FrameResult:
        """Runs one frame through the whole framework.

        `detections` overrides the internal detector, as `[(box, cls, conf,
        uncertainty)]`; `occlusion_grid` overrides MiDaS. Both exist so
        recorded frames can be replayed through the identical code path that
        runs live -- if the offline evaluation used a different pipeline, its
        numbers would not describe the system that actually drives.
        """
        radar_pts = np.empty((0, 4), np.float32) if radar_pts is None else np.asarray(radar_pts)
        img_w, fov = self.cam_cfg["width"], self.cam_cfg["fov"]

        # [4] occlusion grid, from MiDaS + radar unless one is supplied.
        if occlusion_grid is None and rgb is not None:
            occlusion_grid = classify_grid(rgb, radar_pts, self.bev_cfg)

        # [3] detection + evidential confidence.
        dets = detections if detections is not None else self._detect(rgb)

        frame_radar_conf = radar_confidence(radar_pts)
        objects, tracker_dets, det_xy = [], [], []
        disagreed = False
        # (radar range, camera-size range) pairs, for the health monitor's
        # cross-sensor calibration check. See sensor_health._radar_health.
        range_pairs = []
        # Health does not change mid-loop -- self.health.update() runs once,
        # after every detection has been processed -- so this is computed
        # once rather than once per detection.
        report = self.health.report()

        for box, cls, conf, unc in dets:
            az_lo, az_hi = pixel_range_to_azimuth_range(box[0], box[2], img_w, fov)
            hits = radar_in_azimuth_window(radar_pts, az_lo, az_hi)
            r_conf = radar_confidence_in_region(radar_pts, azimuth_range=(az_lo, az_hi))
            bearing = pixel_to_bearing((box[0] + box[2]) / 2.0, img_w, fov)

            rng_m = rate = None
            candidates = None
            if hits.shape[0]:
                # Seed only; the tracker re-picks from the candidates using its
                # own predicted range. See tracking.Detection.range_candidates.
                rng_m, rate = radar_range_for_window(radar_pts, az_lo, az_hi)
                candidates = [(float(h[3]), float(h[0])) for h in hits]
                cam_rng = range_from_pixel_height(box, cls, self.cam_cfg)
                if rng_m is not None and cam_rng is not None:
                    range_pairs.append((rng_m, cam_rng))
            elif abs(az_lo) < self.radar_half_fov and abs(az_hi) < self.radar_half_fov:
                # Silence is only evidence of disagreement where the radar was
                # actually looking. The camera spans 90 deg and the radar 35, so
                # most of the image lies outside the radar cone entirely and a
                # detection there produces no return as a matter of geometry.
                #
                # Counting those as disagreements pinned the rate near 100% and
                # applied the cross-sensor penalty permanently: with ~5
                # detections per frame at least one sat outside the cone almost
                # every frame. Measured live, that multiplied BOTH sensors by
                # 0.6 -- a healthy camera reading 1.00 was displayed as 0.60 and
                # the radar as 0.52, with the dashboard showing DEGRADED for the
                # whole drive. Both sensors falling together was the tell; no
                # single-sensor fault does that.
                disagreed = True        # camera says here, radar looked and saw nothing

            # Approximate ego-frame position for the occlusion lookup. Uses the
            # radar range when available and a nominal one otherwise -- good
            # enough to pick a 2 m grid cell, which is all it is used for.
            r_est = rng_m if rng_m is not None else 20.0
            xy = (r_est * math.cos(bearing), -r_est * math.sin(bearing))
            state = self._cell_state(occlusion_grid, xy)

            # [5] contradiction resolution, conditioned on occlusion state and
            # current sensor health.
            res = resolve(SensorEvidence(
                camera_detected=True, camera_confidence=conf, camera_uncertainty=unc,
                radar_detected=hits.shape[0] > 0, radar_confidence=r_conf,
                occlusion_state=state, is_vru=(cls == 1),
                camera_health=report.camera, radar_health=report.radar))

            objects.append(ObjectResult(
                box_px=box, cls=cls, confidence=conf, uncertainty=unc,
                radar_confidence=r_conf, hazard_posterior=res.posterior,
                trusted=res.trusted, explanation=res.explanation))

            tracker_dets.append(Detection(
                bearing_rad=bearing, range_m=rng_m, range_rate=rate,
                range_candidates=candidates, cls=cls, box_px=box, uncertainty=unc))
            det_xy.append(xy)

        # [8] sensor health, updated with this frame's evidence.
        self.tracker.last_range_innovation = None

        # EKF predict/update.
        self.tracker.predict(dt, ego_speed=ego_speed, ego_yaw_rate=ego_yaw_rate)
        self.tracker.update(tracker_dets, ego_vel=np.array([ego_speed, 0.0]))

        self.health.update(rgb=rgb, radar_pts=radar_pts,
                            sensors_disagreed=disagreed,
                            range_innovation=self.tracker.last_range_innovation,
                            range_pairs=range_pairs)

        # Remember each confirmed track's last measured state, for handoff.
        for t in self.tracker.confirmed_tracks():
            if not t.is_coasting:
                self._last_seen[t.id] = (t.position.copy(), t.velocity.copy(), t.P.copy())

        # [6] hand fully-occluded tracks to the particle filter.
        for t in self.tracker.tracks:
            if t.is_confirmed and t.frames_coasted >= HANDOFF_AFTER_MISSES:
                self._handoff(t)
        hidden = self._update_hidden(dt, ego_speed, ego_yaw_rate, occlusion_grid, det_xy)
        self._prune()

        # Crossing intent for every confirmed VRU track.
        confirmed = self.tracker.confirmed_tracks()
        predictions = {}
        for t in confirmed:
            if t.cls != 1:
                continue
            pred = predict_crossing(t, ego_speed=ego_speed)
            predictions[t.id] = pred
            # Attach the prediction back to the matching detection, by box.
            for o in objects:
                if o.box_px == t.box_px:
                    o.track_id = t.id
                    o.p_cross = pred.p_cross
                    o.p_cross_cautious = pred.p_cross_cautious

        # [7] risk.
        risk = assess(tracks=confirmed, predictions=predictions,
                       hidden_hazards=[e for _tid, e in hidden],
                       occlusion_grid=occlusion_grid, ego_speed=ego_speed)

        return FrameResult(objects=objects, occlusion_grid=occlusion_grid,
                            health=self.health.report(), risk=risk,
                            tracks=confirmed, hidden=hidden,
                            frame_radar_confidence=frame_radar_conf)
