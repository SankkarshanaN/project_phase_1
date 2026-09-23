"""AUDIT PHASE 5 -- occlusion detector evaluated on frames DISJOINT from the
sweep that selected its operating point.

Why this exists
---------------
`scripts/sweep_shadow_tolerance.py` draws a 120-frame sample with
`random.seed(0)`, sweeps `shadow_tolerance`, and the reported operating point
(q=0.01, tau=0.001 -> precision 0.727 / recall 0.917 / F1 0.811 / agreement
0.780) is read off THAT SAME sample at the tolerance chosen as best-F1 on it.
Selecting a hyperparameter and reporting its score on one set of frames is
test-set tuning, and the reported figure is optimistically biased by an
unknown amount.

This script does not change the operating point, retune anything, or alter the
method. It evaluates the ALREADY-FIXED operating point on frames the sweep
never saw, which is the only way to say what that operating point is actually
worth. The sweep's sample is reconstructed with the identical seed and
excluded.

Also reports performance by occlusion severity and by scenario, and the full
3x3 confusion over the mapped label sets.

Writes results/audit/occlusion_metrics.json + figures.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carla_tools.occlusion_mask import OCCLUDED as GT_OCCLUDED
from carla_tools.occlusion_mask import UNKNOWN as GT_UNKNOWN
from carla_tools.occlusion_mask import VISIBLE as GT_VISIBLE
from common.config import load_yaml
from perception.occlusion_grid import (DEFAULT_SHADOW_TOLERANCE, GROUND_QUANTILE,
                                        OCCLUDED as PRED_OCCLUDED, classify_grid)

DATA = Path("data/raw_v2")
OUT = Path("results/audit")
SWEEP_SEED, SWEEP_N = 0, 120        # must match sweep_shadow_tolerance.py exactly


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p and r and np.isfinite(p) and np.isfinite(r)) else float("nan")
    return float(p), float(r), float(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=300,
                     help="held-out frames to evaluate (disjoint from the sweep sample)")
    ap.add_argument("--seed", type=int, default=1234,
                     help="deliberately NOT the sweep's seed")
    args = ap.parse_args()

    bev_cfg = load_yaml("bev.yaml")
    metas = sorted((DATA / "meta").glob("*.npz"))

    # Reconstruct EXACTLY the frames the sweep used, and exclude them.
    random.seed(SWEEP_SEED)
    sweep_sample = set(p.stem for p in random.sample(metas, min(SWEEP_N, len(metas))))
    pool = [p for p in metas if p.stem not in sweep_sample]
    rng = random.Random(args.seed)
    held = rng.sample(pool, min(args.n_frames, len(pool)))
    print(f"sweep sample: {len(sweep_sample)} frames (excluded)")
    print(f"held-out pool: {len(pool)}  evaluating: {len(held)} frames")

    tp = fp = fn = tn = 0
    agree = total = 0
    conf = np.zeros((3, 3), dtype=np.int64)   # gt {VIS,OCC,UNK} x pred {VIS,OCC,other}
    by_scenario = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "agree": 0, "total": 0})
    per_frame_f1 = []
    sev_bins = [(0.0, 0.05), (0.05, 0.20), (0.20, 0.50), (0.50, 0.65), (0.65, 1.01)]
    by_sev = {f"{lo:.2f}-{hi:.2f}": {"tp": 0, "fp": 0, "fn": 0, "frames": 0}
              for lo, hi in sev_bins}

    used = 0
    for p in held:
        img = cv2.imread(str(DATA / "images" / f"{p.stem}.jpg"))
        if img is None:
            continue
        d = np.load(p, allow_pickle=True)
        gt = d["occ_grid"]
        pred = classify_grid(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d["radar_pts"], bev_cfg)
        used += 1

        g_occ, p_occ = (gt == GT_OCCLUDED), (pred == PRED_OCCLUDED)
        f_tp = int((g_occ & p_occ).sum()); f_fp = int((~g_occ & p_occ).sum())
        f_fn = int((g_occ & ~p_occ).sum()); f_tn = int((~g_occ & ~p_occ).sum())
        tp += f_tp; fp += f_fp; fn += f_fn; tn += f_tn
        agree += int((g_occ == p_occ).sum()); total += g_occ.size
        per_frame_f1.append(prf(f_tp, f_fp, f_fn)[2])

        scen = p.stem.split("_ep")[0]
        s = by_scenario[scen]
        s["tp"] += f_tp; s["fp"] += f_fp; s["fn"] += f_fn
        s["agree"] += int((g_occ == p_occ).sum()); s["total"] += g_occ.size

        # Occlusion severity = fraction of the GT grid that is OCCLUDED in this
        # frame. Severity is a property of the SCENE, so it bins frames, not cells.
        sev = float(g_occ.mean())
        for lo, hi in sev_bins:
            if lo <= sev < hi:
                b = by_sev[f"{lo:.2f}-{hi:.2f}"]
                b["tp"] += f_tp; b["fp"] += f_fp; b["fn"] += f_fn; b["frames"] += 1
                break

        for gi, gval in enumerate((GT_VISIBLE, GT_OCCLUDED, GT_UNKNOWN)):
            gm = (gt == gval)
            conf[gi, 0] += int((gm & ~p_occ).sum())
            conf[gi, 1] += int((gm & p_occ).sum())

    precision, recall, f1 = prf(tp, fp, fn)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")

    # Episode-level bootstrap is not meaningful here (frames are sampled across
    # episodes), so bootstrap over FRAMES, which are the sampling unit.
    pf = np.array([v for v in per_frame_f1 if np.isfinite(v)], float)
    boot = np.random.default_rng(0)
    means = [float(boot.choice(pf, size=len(pf), replace=True).mean()) for _ in range(2000)]
    ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    stored = json.loads(Path("results/shadow_tolerance_sweep.json").read_text())
    at_op = next((r for r in stored if abs(r["tolerance"] - DEFAULT_SHADOW_TOLERANCE) < 1e-9), None)

    out = {
        "experiment_id": "AUDIT-B-OCCLUSION-HOLDOUT",
        "operating_point": {"GROUND_QUANTILE_q": GROUND_QUANTILE,
                             "shadow_tolerance_tau": DEFAULT_SHADOW_TOLERANCE},
        "operating_point_provenance": (
            "q and tau were selected as best-F1 on the sweep's 120-frame seed-0 "
            "sample and the manuscript reports the score from that same sample. "
            "This run holds the operating point FIXED and evaluates it on frames "
            "excluded from that sample."),
        "evaluation_set": {
            "n_frames_evaluated": used,
            "n_cells": int(total),
            "sampling_seed": args.seed,
            "disjoint_from_sweep": True,
            "sweep_frames_excluded": len(sweep_sample),
            "denominator": "BEV grid CELLS (20x20=400 per frame) for P/R/F1; frames for CI",
        },
        "heldout": {
            "precision_occluded": precision, "recall_occluded": recall,
            "f1_occluded": f1, "iou_occluded": float(iou),
            "cell_agreement": float(agree / total) if total else float("nan"),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "per_frame_f1_mean": float(pf.mean()), "per_frame_f1_ci95": list(ci),
            "n_bootstrap": 2000,
        },
        "sweep_sample_reported": (
            {"precision_occluded": at_op["precision"], "recall_occluded": at_op["recall"],
             "f1_occluded": at_op["f1"], "cell_agreement": at_op["cell_agreement"],
             "source": "results/shadow_tolerance_sweep.json (tuning sample)"}
            if at_op else None),
        "optimism_gap": (
            {"precision": at_op["precision"] - precision,
             "recall": at_op["recall"] - recall,
             "f1": at_op["f1"] - f1,
             "note": "positive = tuning sample flattered the operating point"}
            if at_op else None),
        "confusion_gt_vs_pred_occluded": {
            "rows_gt": ["VISIBLE", "OCCLUDED", "UNKNOWN"],
            "cols_pred": ["not-OCCLUDED", "OCCLUDED"],
            "matrix": conf[:, :2].tolist(),
            "note": ("The two modules' label enums differ (occlusion_mask.UNKNOWN==2 vs "
                     "occlusion_grid.EMPTY==2), so only the OCCLUDED class is comparable "
                     "and the prediction axis is collapsed to OCCLUDED / not-OCCLUDED."),
        },
        "by_scenario": {
            k: {**dict(zip(("precision", "recall", "f1"), prf(v["tp"], v["fp"], v["fn"]))),
                "cell_agreement": v["agree"] / v["total"] if v["total"] else float("nan"),
                "n_cells": v["total"]}
            for k, v in sorted(by_scenario.items())},
        "by_occlusion_severity": {
            k: {**dict(zip(("precision", "recall", "f1"), prf(v["tp"], v["fp"], v["fn"]))),
                "n_frames": v["frames"]}
            for k, v in by_sev.items()},
        "severity_definition": "fraction of GT grid cells that are OCCLUDED in that frame",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "occlusion_metrics.json").write_text(json.dumps(out, indent=2))

    figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    sev_keys = [k for k, v in out["by_occlusion_severity"].items() if v["n_frames"] > 0]
    if sev_keys:
        fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=200)
        xs = np.arange(len(sev_keys))
        for name, colour in (("precision", "#2e86c1"), ("recall", "#c0392b"), ("f1", "#1f9e6e")):
            ax.plot(xs, [out["by_occlusion_severity"][k][name] for k in sev_keys],
                    marker="o", label=name, color=colour)
        ax.set_xticks(xs, [f"{k}\nn={out['by_occlusion_severity'][k]['n_frames']}"
                            for k in sev_keys], fontsize=7)
        ax.set_xlabel("fraction of scene occluded (ground truth)")
        ax.set_ylabel("score"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Held-out occlusion detection vs scene occlusion severity",
                     fontsize=9, loc="left")
        fig.tight_layout(); fig.savefig(figdir / "audit_occlusion_by_severity.png")
        plt.close(fig)

    print(json.dumps({"heldout": out["heldout"], "sweep_reported": out["sweep_sample_reported"],
                      "optimism_gap": out["optimism_gap"],
                      "by_severity": out["by_occlusion_severity"]}, indent=2))


if __name__ == "__main__":
    main()
