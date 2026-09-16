"""Figures for components 5-9 and the crossing-intent work.

`generate_report_figures.py` covers Phases 1-4 (dataset composition, training
curves, confusion matrix, uncertainty calibration, occlusion-grid validation).
It produces nothing for the framework components added since, which is what
this covers:

  figures/occlusion_tiers.png        -- tier distribution and the visibility histogram
  figures/crossing_timeline.png      -- one episode: visibility, tier and p_cross over time
  figures/sensor_ablation.png        -- camera / radar / fused, the headline result
  figures/contradiction_matrix.png   -- P(hazard) by sensor agreement x occlusion state
  figures/particle_modes.png         -- multi-hypothesis mass while a hazard is hidden
  figures/robustness.png             -- degradation per injected fault, and whether it was noticed

Everything is computed from the collected dataset or from the components
themselves; nothing is illustrative. Figures whose inputs are missing are
skipped with a message rather than faked.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carla_tools.occlusion_tiers import OCCLUDED, PARTIAL, TIER_NAMES, VISIBLE
from common.config import load_yaml
from perception.contradiction import SensorEvidence, resolve
from perception.occlusion_grid import EMPTY as G_EMPTY
from perception.occlusion_grid import OCCLUDED as G_OCCLUDED
from perception.occlusion_grid import UNKNOWN as G_UNKNOWN
from perception.occlusion_grid import VISIBLE as G_VISIBLE
from perception.particle_tracker import HiddenHazardTracker

# One palette, used consistently so the same concept is the same colour in
# every figure -- tiers, sensors and verdicts each keep their hue throughout.
C_OCC, C_PART, C_VIS = "#c0392b", "#e39a25", "#2e86c1"
C_CAM, C_RAD, C_FUSED = "#7d8ea3", "#8e6fb3", "#1f9e6e"
C_GOOD, C_BAD = "#1f9e6e", "#c0392b"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def _load_frames(data_dir: Path):
    by_ep = defaultdict(list)
    for p in sorted((data_dir / "meta").glob("*.npz")):
        by_ep[p.stem.rsplit("_f", 1)[0]].append(p)
    for k in by_ep:
        by_ep[k].sort(key=lambda q: int(q.stem.rsplit("_f", 1)[1]))
    return by_ep


# ------------------------------------------------------------------ figures

def fig_occlusion_tiers(by_ep, out: Path):
    """Tier distribution and the continuous visibility histogram.

    The histogram is the evidence that visibility is actually *graded* now.
    The previous 8-corner measure could only return multiples of 1/8, so its
    histogram was eight spikes; a smooth distribution between the extremes is
    what makes per-tier analysis meaningful.
    """
    tiers, vis = Counter(), []
    for frames in by_ep.values():
        for p in frames:
            d = np.load(p, allow_pickle=True)
            if "obj_tier" not in d.files:
                continue
            tiers.update(d["obj_tier"].tolist())
            vis.extend(d["obj_visibility"].tolist())
    if not vis:
        print("  occlusion_tiers: no per-object records, skipped")
        return None

    vis = np.asarray(vis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))

    names = [TIER_NAMES[t] for t in (OCCLUDED, PARTIAL, VISIBLE)]
    counts = [tiers.get(t, 0) for t in (OCCLUDED, PARTIAL, VISIBLE)]
    total = max(sum(counts), 1)
    bars = ax1.bar(names, counts, color=[C_OCC, C_PART, C_VIS], width=0.62)
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{c}\n{c / total:.1%}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("observations")
    ax1.set_title("Per-object occlusion tier", loc="left", fontweight="bold")
    ax1.set_ylim(0, max(counts) * 1.25)

    mid = vis[(vis > 0.01) & (vis < 0.99)]
    ax2.hist(vis, bins=40, color="#b9c6d4", edgecolor="white", linewidth=0.4,
             label=f"all ({len(vis)})")
    if len(mid):
        ax2.hist(mid, bins=40, color=C_PART, edgecolor="white", linewidth=0.4,
                 label=f"partial ({len(mid)})")
    ax2.set_xlabel("visibility fraction")
    ax2.set_ylabel("observations")
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("Visibility is continuous, not quantised", loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "occlusion_tiers.png", bbox_inches="tight")
    plt.close(fig)
    return {"tier_counts": {TIER_NAMES[t]: int(tiers.get(t, 0))
                            for t in (OCCLUDED, PARTIAL, VISIBLE)},
            "distinct_partial_values": int(len({round(float(v), 3) for v in mid}))}


def fig_crossing_timeline(by_ep, out: Path):
    """One episode's VRU: visibility and tier over time.

    Picks the episode with the cleanest OCCLUDED -> VISIBLE sweep. This is the
    figure that shows the dataset contains the event the project is about; the
    v1 dataset could not produce it at all.
    """
    best, best_score = None, -1
    for key, frames in by_ep.items():
        vis, tiers, t = [], [], []
        for p in frames:
            d = np.load(p, allow_pickle=True)
            if "obj_is_vru" not in d.files:
                continue
            m = np.where(d["obj_is_vru"])[0]
            if len(m) == 0:
                continue
            j = m[0]
            vis.append(float(d["obj_visibility"][j]))
            tiers.append(int(d["obj_tier"][j]))
            t.append(float(d["frame_idx"]))
        if len(vis) < 20:
            continue
        seen = set(tiers)
        score = len(seen) * 100 + sum(1 for a, b in zip(tiers, tiers[1:]) if a != b)
        if {OCCLUDED, VISIBLE} <= seen and score > best_score:
            best, best_score = (key, np.array(t), np.array(vis), np.array(tiers)), score

    if best is None:
        print("  crossing_timeline: no episode sweeps OCCLUDED -> VISIBLE, skipped")
        return None

    key, t, vis, tiers = best
    t = (t - t[0]) * 0.1          # ticks -> seconds

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for tier, colour, label in ((OCCLUDED, C_OCC, "OCCLUDED"),
                                 (PARTIAL, C_PART, "PARTIAL"),
                                 (VISIBLE, C_VIS, "VISIBLE")):
        m = tiers == tier
        if m.any():
            ax.fill_between(t, 0, 1, where=m, color=colour, alpha=0.16,
                            step="mid", label=label)
    ax.plot(t, vis, color="#20242b", linewidth=1.8, zorder=3)
    ax.set_xlabel("time within episode (s)")
    ax.set_ylabel("visibility fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(t[0], t[-1])
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax.set_title("A pedestrian emerging from behind a parked occluder",
                 loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "crossing_timeline.png", bbox_inches="tight")
    plt.close(fig)
    return {"episode": key, "frames": int(len(t))}


def fig_contradiction_matrix(out: Path):
    """P(hazard) for each sensor-agreement case, by occlusion state.

    The column that carries the argument is "neither sensor reports anything":
    the same null observation means very different things depending on whether
    the sensors could have seen. Fixed-weight fusion returns one number for
    that whole row.
    """
    cases = [("both see it", True, True), ("camera only", True, False),
             ("radar only", False, True), ("neither", False, False)]
    states = [(G_VISIBLE, "VISIBLE"), (G_EMPTY, "EMPTY"),
              (G_UNKNOWN, "UNKNOWN"), (G_OCCLUDED, "OCCLUDED")]

    M = np.zeros((len(cases), len(states)))
    for i, (_n, cam, rad) in enumerate(cases):
        for j, (state, _s) in enumerate(states):
            M[i, j] = resolve(SensorEvidence(
                camera_detected=cam, camera_confidence=0.9 if cam else 0.0,
                camera_uncertainty=0.1, radar_detected=rad,
                radar_confidence=0.8 if rad else 0.0,
                occlusion_state=state, is_vru=True)).posterior

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(states)), [s for _c, s in states], fontsize=8)
    ax.set_yticks(range(len(cases)), [n for n, _a, _b in cases], fontsize=8)
    for i in range(len(cases)):
        for j in range(len(states)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.6 or M[i, j] < 0.12 else "#20242b")
    ax.set_xlabel("occlusion state of the cell")
    ax.set_title("P(hazard) after Bayesian contradiction resolution",
                 loc="left", fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out / "contradiction_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return {"neither_visible": float(M[3, 0]), "neither_occluded": float(M[3, 3])}


def fig_particle_modes(out: Path):
    """Hypothesis mass over time while a hazard is fully hidden.

    The point a Gaussian filter cannot make: the belief stays genuinely
    multi-modal for seconds after the target disappears, so there is no single
    "where it is" to report -- only a set of possibilities with weights.
    """
    grid = np.full((20, 20), G_VISIBLE, dtype=np.uint8)
    grid[7:11, 10:14] = G_OCCLUDED
    cfg = {"size_cells": 20, "cell_size_m": 2.0, "extent_m": 40.0}

    pf = HiddenHazardTracker(init_xy=(16.0, 2.0), init_vel=(0.0, -1.3),
                              rng=np.random.default_rng(1))
    series, spread = defaultdict(list), []
    for _ in range(40):
        pf.predict(0.1)
        pf.update_from_occlusion(grid, cfg)
        for name, mass in pf.mode_masses().items():
            series[name].append(mass)
        spread.append(pf.position_spread)

    t = np.arange(len(spread)) * 0.1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))

    ax1.stackplot(t, *[series[k] for k in sorted(series)],
                  labels=sorted(series),
                  colors=["#2e86c1", "#59a9d6", "#e39a25", "#c0392b"], alpha=0.9)
    ax1.set_xlabel("time hidden (s)")
    ax1.set_ylabel("probability mass")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, t[-1])
    ax1.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax1.set_title("Motion hypotheses stay separate", loc="left", fontweight="bold")

    ax2.plot(t, spread, color=C_OCC, linewidth=1.9)
    ax2.set_xlabel("time hidden (s)")
    ax2.set_ylabel("positional spread (m)")
    ax2.set_xlim(0, t[-1])
    ax2.set_title("Uncertainty grows honestly while blind", loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "particle_modes.png", bbox_inches="tight")
    plt.close(fig)
    return {"final_modes": {k: float(v[-1]) for k, v in series.items()},
            "final_spread_m": float(spread[-1])}


def fig_sensor_ablation(metrics: dict, out: Path):
    """Camera / radar / fused. The headline result."""
    if not metrics:
        print("  sensor_ablation: no ablation metrics supplied, skipped")
        return None

    modes = ["camera", "radar", "fused"]
    colours = [C_CAM, C_RAD, C_FUSED]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

    panels = [("vel_lat_mae", "lateral velocity MAE (m/s)",
               "Lateral velocity error\n(radar is blind to this)"),
              ("pos_rmse", "position RMSE (m)", "Position error"),
              ("auc", "AUC", "Crossing-intent AUC")]

    for ax, (key, ylabel, title) in zip(axes, panels):
        vals = [metrics.get(m, {}).get(key, np.nan) for m in modes]
        bars = ax.bar(modes, vals, color=colours, width=0.6)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.28)
        if key == "auc":
            ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
            ax.text(2.45, 0.51, "chance", fontsize=7, color="#888", ha="right")

    fig.tight_layout()
    fig.savefig(out / "sensor_ablation.png", bbox_inches="tight")
    plt.close(fig)
    return metrics


def fig_robustness(rows: list, out: Path):
    """Degradation per injected fault, and whether the health monitor saw it."""
    if not rows:
        print("  robustness: no adversarial results supplied, skipped")
        return None

    names = [r["fault"] for r in rows]
    delta = [r["delta_pos"] for r in rows]
    noticed = [r["noticed"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    bars = ax.barh(names, delta,
                   color=[C_GOOD if n else C_BAD for n in noticed], height=0.62)
    for b, d, n in zip(bars, delta, noticed):
        ax.text(b.get_width() + max(delta) * 0.02, b.get_y() + b.get_height() / 2,
                "detected" if n else "SILENT", va="center", fontsize=8,
                color=C_GOOD if n else C_BAD, fontweight="bold")
    ax.set_xlabel("added position RMSE vs clean (m)")
    ax.set_xlim(0, max(delta) * 1.35 if max(delta) > 0 else 1)
    ax.invert_yaxis()
    ax.set_title("Injected sensor faults: damage done, and whether the system noticed",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "robustness.png", bbox_inches="tight")
    plt.close(fig)
    return {"silent": [n for n, ok in zip(names, noticed) if not ok]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw_v2")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    print(f"Figures -> {out}")

    # Component-only figures need no dataset.
    print("  contradiction_matrix ...")
    results["contradiction"] = fig_contradiction_matrix(out)
    print("  particle_modes ...")
    results["particle_modes"] = fig_particle_modes(out)

    if (data_dir / "meta").is_dir():
        by_ep = _load_frames(data_dir)
        print(f"  loaded {sum(len(v) for v in by_ep.values())} frames "
              f"across {len(by_ep)} episodes")
        print("  occlusion_tiers ...")
        results["tiers"] = fig_occlusion_tiers(by_ep, out)
        print("  crossing_timeline ...")
        results["timeline"] = fig_crossing_timeline(by_ep, out)
    else:
        print(f"  no dataset at {data_dir} -- dataset figures skipped")

    import json
    (Path("results") / "framework_metrics.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nWrote {len([v for v in results.values() if v])} figures + "
          f"results/framework_metrics.json")


if __name__ == "__main__":
    main()
