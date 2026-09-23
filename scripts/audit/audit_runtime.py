"""AUDIT PHASE 12 -- per-component runtime profiling on recorded frames.

Profiles the perception modules on stored sensor data, with CARLA out of the
loop entirely. That separation is the point: the manuscript's "2-9 FPS" figure
comes from the live demo, where the number is the SYNCHRONOUS-MODE tick rate
and therefore includes CARLA's own rendering and server round-trip. It is not
the perception pipeline's throughput, and the two must not be conflated.

Each stage is timed on the same frames, warm (after a warm-up pass), so the
numbers are comparable to each other.

Writes results/audit/runtime_metrics.json + figure.
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common.config import load_yaml
from common.npz_io import load_npz
from models.evidential_classifier import CROP_SIZE, EvidentialDetector
from perception.depth_midas import estimate_disparity, normalize_disparity
from perception.occlusion_grid import classify_grid
from perception.radar_confidence import radar_confidence
from perception.sensor_health import SensorHealthMonitor
from perception.tracking import Detection, MultiObjectTracker
from perception.particle_tracker import HiddenHazardTracker

DATA = Path("data/raw_v2")
OUT = Path("results/audit")
N_FRAMES = 40
WARMUP = 3


def timed(fn, n_warm=WARMUP):
    """Returns per-call wall times (s), discarding warm-up calls."""
    times = []
    for i in range(n_warm):
        fn(i)
    for i in range(N_FRAMES):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.asarray(times, float)


def summarize(t):
    return {"mean_ms": float(t.mean() * 1e3), "median_ms": float(np.median(t) * 1e3),
            "p95_ms": float(np.percentile(t, 95) * 1e3), "n_calls": int(t.size),
            "implied_fps_if_alone": float(1.0 / t.mean())}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bev_cfg = load_yaml("bev.yaml")

    metas = sorted((DATA / "meta").glob("*.npz"))[:N_FRAMES + WARMUP + 5]
    frames = []
    for p in metas:
        img = cv2.imread(str(DATA / "images" / f"{p.stem}.jpg"))
        if img is None:
            continue
        d = load_npz(p)
        frames.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d["radar_pts"], d))
        if len(frames) >= N_FRAMES + WARMUP:
            break
    if len(frames) < N_FRAMES + WARMUP:
        raise SystemExit("not enough frames to profile")

    stages = {}

    # --- YOLOv8n detection
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        stages["yolov8n_detect"] = summarize(timed(
            lambda i: yolo.predict(frames[i][0], conf=0.35, verbose=False, device=device)))
    except Exception as e:                                     # noqa: BLE001
        stages["yolov8n_detect"] = {"error": f"{type(e).__name__}: {e}"}

    # --- evidential head on one 64x64 crop
    try:
        model = EvidentialDetector(num_classes=3).to(device).eval()
        ck = Path("models/evidential_detector.pt")
        if ck.exists():
            model.load_state_dict(torch.load(ck, map_location=device))
        crop = torch.rand(1, 3, CROP_SIZE, CROP_SIZE, device=device)

        def _ev(i):
            with torch.no_grad():
                model(crop)
        stages["evidential_head_per_crop"] = summarize(timed(_ev))
    except Exception as e:                                     # noqa: BLE001
        stages["evidential_head_per_crop"] = {"error": f"{type(e).__name__}: {e}"}

    # --- MiDaS monocular depth
    try:
        stages["midas_depth"] = summarize(timed(
            lambda i: estimate_disparity(frames[i][0])))
    except Exception as e:                                     # noqa: BLE001
        stages["midas_depth"] = {"error": f"{type(e).__name__}: {e}"}

    # --- occlusion grid, with disparity precomputed (isolates the grid maths)
    try:
        disp = [normalize_disparity(estimate_disparity(frames[i][0]))
                for i in range(min(6, len(frames)))]
        stages["occlusion_grid_given_disparity"] = summarize(timed(
            lambda i: classify_grid(frames[i][0], frames[i][1], bev_cfg,
                                     norm_disparity=disp[i % len(disp)])))
    except Exception as e:                                     # noqa: BLE001
        stages["occlusion_grid_given_disparity"] = {"error": f"{type(e).__name__}: {e}"}

    # --- radar confidence
    stages["radar_confidence"] = summarize(timed(
        lambda i: radar_confidence(frames[i][1])))

    # --- EKF predict+update with a handful of detections
    tracker = MultiObjectTracker()
    dets = [Detection(cls=1, bearing_rad=0.05, range_m=20.0, range_rate=-1.0),
            Detection(cls=0, bearing_rad=-0.10, range_m=35.0, range_rate=-3.0)]

    def _ekf(i):
        tracker.predict(0.1, ego_speed=8.0)
        tracker.update(dets, ego_vel=np.array([8.0, 0.0]))
    stages["ekf_tracking_step"] = summarize(timed(_ekf))

    # --- particle filter step (400 particles)
    pf = HiddenHazardTracker(init_xy=(15.0, 2.0), init_vel=(0.0, 1.2),
                              rng=np.random.default_rng(0))
    grid_for_pf = classify_grid(frames[0][0], frames[0][1], bev_cfg)

    # `bev_cfg["bev_grid"]`, matching pipeline.PerceptionPipeline.grid_cfg --
    # the particle filter indexes cells and needs the grid block, not the root.
    grid_cfg = bev_cfg["bev_grid"]

    def _pf(i):
        pf.predict(0.1, ego_speed=8.0)
        pf.update_from_occlusion(grid_for_pf, grid_cfg)
    stages["particle_filter_step_400p"] = summarize(timed(_pf))

    # --- sensor health monitor
    mon = SensorHealthMonitor()
    stages["sensor_health_update"] = summarize(timed(
        lambda i: mon.update(rgb=frames[i][0], radar_pts=frames[i][1])))

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    try:
        vram = (f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
                if torch.cuda.is_available() else "n/a")
    except Exception:                                          # noqa: BLE001
        vram = "unknown"

    serial_ms = sum(v["mean_ms"] for v in stages.values()
                    if isinstance(v, dict) and "mean_ms" in v)

    out = {
        "experiment_id": "AUDIT-J-RUNTIME",
        "scope": ("perception modules only, replayed on recorded frames; CARLA is NOT "
                   "in the loop and contributes nothing to these timings"),
        "n_frames_timed": N_FRAMES,
        "warmup_calls": WARMUP,
        "hardware": {
            "gpu": gpu, "vram_total": vram, "device_used": device,
            "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "input": {"camera_resolution": [bev_cfg["camera"]["width"],
                                         bev_cfg["camera"]["height"]],
                   "crop_size": CROP_SIZE,
                   "bev_grid": bev_cfg["bev_grid"]["size_cells"]},
        "stages": stages,
        "serial_sum_mean_ms": serial_ms,
        "implied_fps_serial_sum": (1000.0 / serial_ms) if serial_ms else None,
        "important_distinction": (
            "implied_fps_serial_sum assumes every stage runs every frame in series. "
            "The live demo does NOT do that -- MiDaS is refreshed every 3rd frame "
            "(DEPTH_EVERY), and the evidential head runs per DETECTION, not once. So "
            "this figure is an upper bound on cost, not the demo's measured rate."),
        "vs_manuscript": (
            "The manuscript's 2-9 FPS is the live demo's synchronous-mode tick rate, "
            "which includes CARLA rendering and the server round-trip. It is not "
            "comparable to these module timings and should not be presented as "
            "perception throughput."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runtime_metrics.json").write_text(json.dumps(out, indent=2))

    ok = {k: v for k, v in stages.items() if "mean_ms" in v}
    if ok:
        figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=200)
        names = sorted(ok, key=lambda k: -ok[k]["mean_ms"])
        ax.barh(names, [ok[n]["mean_ms"] for n in names], color="#2e86c1")
        ax.set_xscale("log"); ax.set_xlabel("mean latency per call (ms, log scale)")
        ax.set_title("Perception module latency (CARLA not in the loop)",
                     fontsize=9, loc="left")
        ax.grid(alpha=0.25, axis="x")
        fig.tight_layout(); fig.savefig(figdir / "audit_runtime_breakdown.png")
        plt.close(fig)

    for k, v in sorted(stages.items(), key=lambda kv: -(kv[1].get("mean_ms", 0))):
        if "mean_ms" in v:
            print(f"{k:<34} {v['mean_ms']:>9.2f} ms  p95 {v['p95_ms']:>8.2f}  "
                  f"({v['implied_fps_if_alone']:.1f} FPS alone)")
        else:
            print(f"{k:<34} {v.get('error')}")
    print(f"\nserial sum = {serial_ms:.1f} ms -> {1000.0/serial_ms:.2f} FPS upper-bound cost")


if __name__ == "__main__":
    main()
