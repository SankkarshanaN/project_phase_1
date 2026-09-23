"""SECOND PASS -- dose-response for the three SILENT failures.

The first audit established a binary result: radar_bias, radar_clutter and
extrinsic_drift degrade the pipeline while the health monitor stays nominal.
A binary result cannot say how bad it gets, or whether the monitor would ever
wake up at a larger magnitude. This sweeps the severity parameter that each
fault function already exposes and measures degradation against health.

No new fault model is created. Each fault is the SHIPPED function from
`scripts.adversarial_test`, called through `functools.partial` with a
different value of its own existing keyword:

    radar_bias(offset_m=...)      default 2.5
    radar_clutter(n_extra=...)    default 40
    extrinsic_drift(yaw_deg=...)  default 4.0

All three perturb radar only, so images are not loaded and camera faults are
inert -- which also makes the sweep affordable.

Writes results/second_pass/fault_severity_metrics.json + figure.
"""
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import load_yaml
from common.npz_io import load_npz
import scripts.adversarial_test as AT

DATA = Path("data/raw_v2")
OUT = Path("results/second_pass")
MAX_EPISODES = 24           # same budget the first-pass fault run used

SWEEPS = {
    "radar_bias": ("offset_m", [0.5, 1.0, 2.5, 5.0, 10.0, 20.0], 2.5),
    "radar_clutter": ("n_extra", [10, 20, 40, 80, 160, 320], 40),
    "extrinsic_drift": ("yaw_deg", [1.0, 2.0, 4.0, 8.0, 16.0, 32.0], 4.0),
}
HEALTH_DEGRADED_BELOW = 0.6     # the threshold adversarial_test itself uses


def main():
    bev = load_yaml("bev.yaml")
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])
    lab = load_npz(DATA / "intent_labels.npz")
    # adversarial_test.run_pipeline expects labels[key] -> BOOL, not the dict
    # form evaluate_prediction uses. Passing the dict yields AUC = nan.
    labels = {(str(lab["episode"][i]), int(lab["frame_idx"][i]), int(lab["actor_id"][i])):
              bool(lab["will_cross"][i])
              for i in range(len(lab["episode"]))}

    by_ep = defaultdict(list)
    for p in sorted((DATA / "meta").glob("*.npz")):
        by_ep[AT._episode_key(p)].append(p) if hasattr(AT, "_episode_key") else None
    if not by_ep:                      # adversarial_test may not export the helper
        for p in sorted((DATA / "meta").glob("*.npz")):
            by_ep[p.stem.rsplit("_f", 1)[0]].append(p)
    keys = AT._sample_episodes(by_ep, MAX_EPISODES) if hasattr(AT, "_sample_episodes") \
        else sorted(by_ep)[:MAX_EPISODES]
    episodes = {k: [load_npz(p) for p in sorted(by_ep[k],
                                                 key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
                for k in keys}
    print(f"sweeping over {len(episodes)} episodes (radar-side faults, no images)")

    rng = np.random.default_rng(0)
    clean = AT.run_pipeline(episodes, "clean", bev["camera"], dt, labels, rng, image_dir=None)
    print(f"clean baseline: pos {clean['pos_rmse']:.3f}  v_lat {clean['lat_vel_mae']:.3f}  "
          f"AUC {clean['auc']:.3f}  radar health {clean['health_radar']:.2f}")

    results = {}
    for fault, (param, values, default) in SWEEPS.items():
        base_fn = AT.FAULTS[fault]
        rows = []
        for v in values:
            name = f"__sweep_{fault}_{param}_{v}"
            AT.FAULTS[name] = partial(base_fn, **{param: v})
            try:
                r = AT.run_pipeline(episodes, name, bev["camera"], dt, labels,
                                     np.random.default_rng(0), image_dir=None)
            finally:
                AT.FAULTS.pop(name, None)
            row = {
                param: v, "is_default": v == default,
                "pos_rmse_m": r["pos_rmse"], "v_lat_mae_ms": r["lat_vel_mae"],
                "hidden_rmse_m": r.get("occluded_rmse"),
                "intent_auc": r["auc"], "n_detections": r["n_detections"],
                "camera_health": r["health_camera"], "radar_health": r["health_radar"],
                "health_flags_degraded": bool(r["health_radar"] < HEALTH_DEGRADED_BELOW),
                "delta_pos_rmse_m": r["pos_rmse"] - clean["pos_rmse"],
                "delta_v_lat_mae": r["lat_vel_mae"] - clean["lat_vel_mae"],
                "delta_intent_auc": (r["auc"] - clean["auc"]
                                      if np.isfinite(r["auc"])
                                      and np.isfinite(clean["auc"]) else None),
            }
            rows.append(row)
            print(f"  {fault:<16} {param}={v:<6} pos {r['pos_rmse']:.3f} "
                  f"(d_{row['delta_pos_rmse_m']:+.3f})  v_lat {r['lat_vel_mae']:.3f}  "
                  f"AUC {r['auc']:.3f}  radar hp {r['health_radar']:.2f}"
                  f"{'  <-- DETECTED' if row['health_flags_degraded'] else ''}")

        detected = [r[param] for r in rows if r["health_flags_degraded"]]
        results[fault] = {
            "severity_parameter": param,
            "default_value": default,
            "values_swept": values,
            "rows": rows,
            "monitor_first_flags_at": min(detected) if detected else None,
            "monitor_never_flags": not detected,
            "max_delta_pos_rmse_m": max(r["delta_pos_rmse_m"] for r in rows),
            "max_abs_delta_intent_auc": max(
                (abs(r["delta_intent_auc"]) for r in rows
                 if r["delta_intent_auc"] is not None), default=None),
        }

    payload = {
        "experiment_id": "SP-04-FAULT-SEVERITY",
        "status": "EXECUTED",
        "question": ("How much do the three SILENT failures degrade the pipeline as "
                      "severity rises, and does the health monitor ever notice?"),
        "method": ("each fault is the shipped function from scripts.adversarial_test, "
                    "invoked via functools.partial with a different value of its own "
                    "existing severity keyword; no new fault model was written"),
        "episodes": len(episodes),
        "health_degraded_threshold": HEALTH_DEGRADED_BELOW,
        "clean_baseline": {
            "pos_rmse_m": clean["pos_rmse"], "v_lat_mae_ms": clean["lat_vel_mae"],
            "intent_auc": clean["auc"], "camera_health": clean["health_camera"],
            "radar_health": clean["health_radar"]},
        "sweeps": results,
        "denominator_note": ("metrics are aggregated over all VRU observations in the "
                              "sampled episodes; n_detections is reported per condition"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fault_severity_metrics.json").write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), dpi=200)
    for ax, (fault, d) in zip(axes, results.items()):
        xs = [r[d["severity_parameter"]] for r in d["rows"]]
        ax2 = ax.twinx()
        ax.plot(xs, [r["delta_pos_rmse_m"] for r in d["rows"]], marker="o",
                color="#c0392b", label="d_ position RMSE (m)")
        ax2.plot(xs, [r["radar_health"] for r in d["rows"]], marker="s",
                 color="#2e86c1", label="radar health")
        ax2.axhline(HEALTH_DEGRADED_BELOW, color="#2e86c1", linestyle=":", linewidth=1)
        ax2.set_ylim(0, 1.05)
        dv = d["default_value"]
        ax.axvline(dv, color="#888", linestyle="--", linewidth=1)
        ax.annotate("shipped\nseverity", (dv, ax.get_ylim()[1]), fontsize=6.5,
                    color="#888", ha="center", va="top")
        ax.set_xscale("log")
        ax.set_xlabel(f"{fault} ({d['severity_parameter']})")
        ax.set_ylabel("d_ position RMSE (m)", color="#c0392b")
        ax2.set_ylabel("radar health", color="#2e86c1")
        ax.set_title(fault, loc="left", fontweight="bold", fontsize=9.5)
        ax.grid(alpha=0.22)
    fig.suptitle("Silent failures: damage rises with severity while health stays nominal "
                 "(dotted line = 0.6 degraded threshold)", fontsize=9, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fault_severity.png", bbox_inches="tight")
    plt.close(fig)

    print("\n=== SUMMARY ===")
    for fault, d in results.items():
        first = d["monitor_first_flags_at"]
        print(f"{fault:<16} max d_pos {d['max_delta_pos_rmse_m']:+.3f} m   "
              f"monitor flags at {d['severity_parameter']}="
              f"{first if first is not None else 'NEVER (across entire sweep)'}")


if __name__ == "__main__":
    main()
