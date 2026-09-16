"""Precision/recall of the occlusion detector across its one tuning parameter.

`docs/RESULTS.md` lists this as the outstanding quantitative gap: the detector
is reported at a single operating point, so its recall limitation reads as a
fixed property rather than as a choice. Sweeping `shadow_tolerance` turns one
number into a curve, which is standard for a detector with a single threshold.

It is now more than a nicety. On the v1 dataset the detector measured precision
0.98 / recall 0.55 at the default tolerance of 0.12. On v2 -- collected with a
roaming ego rather than a parked one -- the same default gives precision 0.96
but recall **0.17**. Either the operating point no longer suits the scene
distribution, or the detector genuinely degrades on realistic driving. A sweep
distinguishes those: if a lower tolerance recovers recall, it was the operating
point; if the whole curve has collapsed, it is the detector.

Ground truth is `occ_grid` from CARLA's depth buffer
(`carla_tools.occlusion_mask`); the prediction is `perception.occlusion_grid`,
which sees only MiDaS disparity and radar. Note the two modules' label enums
differ -- `occlusion_mask.UNKNOWN == 2` but `occlusion_grid.EMPTY == 2` -- so
they are imported under distinct names and only the OCCLUDED class is compared.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2

from carla_tools.occlusion_mask import OCCLUDED as GT_OCCLUDED
from common.config import load_yaml
from perception.occlusion_grid import OCCLUDED as PRED_OCCLUDED
from perception.occlusion_grid import classify_grid

DEFAULT_TOLERANCES = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40]


def evaluate(frames, tolerances, bev_cfg):
    """Returns per-tolerance precision/recall/F1 for the OCCLUDED class.

    MiDaS runs once per frame and the disparity is reused across every
    tolerance -- the tolerance only affects the shadow decision downstream, and
    re-running the network per setting would make the sweep an order of
    magnitude slower for identical results.
    """
    rows = []
    for tol in tolerances:
        tp = fp = fn = 0
        agree = total = 0
        for rgb, radar, gt in frames:
            pred = classify_grid(rgb, radar, bev_cfg, shadow_tolerance=tol)
            p_occ = pred == PRED_OCCLUDED
            g_occ = gt == GT_OCCLUDED
            tp += int((p_occ & g_occ).sum())
            fp += int((p_occ & ~g_occ).sum())
            fn += int((~p_occ & g_occ).sum())
            agree += int((p_occ == g_occ).sum())
            total += int(g_occ.size)

        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if tp else float("nan")
        rows.append({"tolerance": tol, "precision": prec, "recall": rec, "f1": f1,
                      "cell_agreement": agree / max(total, 1)})
        print(f"  tolerance {tol:.2f}: precision {prec:.3f}  recall {rec:.3f}  "
              f"F1 {f1:.3f}  agreement {rows[-1]['cell_agreement']:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw_v2")
    ap.add_argument("--n-frames", type=int, default=120,
                     help="Frames sampled. Each runs MiDaS once, so this dominates runtime.")
    ap.add_argument("--out", default="results/figures/shadow_tolerance_sweep.png")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    bev_cfg = load_yaml("bev.yaml")

    metas = sorted((data_dir / "meta").glob("*.npz"))
    if not metas:
        raise SystemExit(f"No frames under {data_dir}")
    random.seed(0)
    sample = random.sample(metas, min(args.n_frames, len(metas)))

    print(f"Loading {len(sample)} frames from {data_dir} ...")
    frames = []
    for p in sample:
        img = cv2.imread(str(data_dir / "images" / f"{p.stem}.jpg"))
        if img is None:
            continue
        d = np.load(p, allow_pickle=True)
        frames.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d["radar_pts"], d["occ_grid"]))
    print(f"Sweeping shadow_tolerance over {len(frames)} frames "
          f"({len(frames) * 400} cells per setting)\n")

    rows = evaluate(frames, DEFAULT_TOLERANCES, bev_cfg)

    tol = [r["tolerance"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=130)
    ax1.plot(tol, prec, marker="o", label="precision", color="#2e86c1")
    ax1.plot(tol, rec, marker="s", label="recall", color="#c0392b")
    ax1.plot(tol, f1, marker="^", label="F1", color="#1f9e6e")
    ax1.axvline(0.12, color="#888", linestyle="--", linewidth=1)
    ax1.text(0.125, 0.02, "default", fontsize=7, color="#888")
    ax1.set_xlabel("shadow_tolerance")
    ax1.set_ylabel("score")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("OCCLUDED class vs. detector threshold", loc="left", fontweight="bold")
    ax1.grid(alpha=0.25)

    ax2.plot(rec, prec, marker="o", color="#20242b")
    for r in rows:
        ax2.annotate(f"{r['tolerance']:.2f}", (r["recall"], r["precision"]),
                     fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Precision-recall operating points", loc="left", fontweight="bold")
    ax2.grid(alpha=0.25)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    import json
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "shadow_tolerance_sweep.json").write_text(json.dumps(rows, indent=2))

    best = max((r for r in rows if np.isfinite(r["f1"])), key=lambda r: r["f1"], default=None)
    print(f"\nWrote {out} and results/shadow_tolerance_sweep.json")
    if best:
        print(f"Best F1 at tolerance {best['tolerance']:.2f}: "
              f"precision {best['precision']:.3f} recall {best['recall']:.3f} "
              f"F1 {best['f1']:.3f}")
        print("If that tolerance differs from the 0.12 default, the reported recall was an")
        print("operating-point choice rather than a limit of the detector -- say which.")


if __name__ == "__main__":
    main()
