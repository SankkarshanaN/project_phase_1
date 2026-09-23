"""SECOND PASS -- error distributions, effect sizes, and difficulty breakdowns.

Covers the remaining supported gaps in one replay, because they all need the
same raw per-observation errors:

  * tracking error DISTRIBUTIONS (median / IQR / p90 / p95 / worst) -- the
    tail is the safety-relevant part and the first pass reported only means;
  * EFFECT SIZES for the fusion comparisons (paired Cohen's d and the
    non-parametric probability of superiority), because a p-value says
    whether a difference exists, not whether it matters;
  * tracking error by RANGE band;
  * intent performance by SCENARIO and by RANGE;
  * classifier accuracy/uncertainty by OBJECT SIZE and by RANGE, using the
    raw validation predictions the first pass already saved.

Writes results/second_pass/{distributions,effect_sizes,breakdowns}.json + figures.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import load_yaml
from common.npz_io import load_npz
from scripts.evaluate_prediction import MODES, _auc, _episode_key, _prf, evaluate_mode

DATA = Path("data/raw_v2")
OUT = Path("results/second_pass")
RANGE_BANDS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 1e9)]
MIN_N = 30


def dist(a):
    a = np.asarray(a, float)
    q1, q3 = np.percentile(a, [25, 75])
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "iqr": [float(q1), float(q3)], "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "std": float(a.std(ddof=1)) if a.size > 1 else None}


def paired_effect(a, b):
    """a, b are paired error vectors (lower = better). Returns effect size of
    a relative to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1)
    cohen_dz = float(d.mean() / sd) if sd > 0 else None
    # Probability that a randomly chosen pair has a < b (a better), ties at 0.5.
    wins = float((a < b).mean() + 0.5 * (a == b).mean())
    return {"mean_difference": float(d.mean()), "cohens_dz_paired": cohen_dz,
            "prob_a_better_than_b": wins,
            "interpretation": ("|dz| < 0.2 negligible, < 0.5 small, < 0.8 medium, "
                                "else large (Cohen's conventions)")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bev = load_yaml("bev.yaml")
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])
    lab = load_npz(DATA / "intent_labels.npz")

    labels_by_key, meta_by_key = {}, {}
    for i in range(len(lab["episode"])):
        k = (str(lab["episode"][i]), int(lab["frame_idx"][i]), int(lab["actor_id"][i]))
        labels_by_key[k] = {"will_cross": bool(lab["will_cross"][i]),
                            "tier": int(lab["tier"][i])}
        meta_by_key[k] = {"forward_m": float(lab["forward_m"][i]),
                          "scenario": str(lab["scenario"][i]),
                          "tier": int(lab["tier"][i])}

    by_ep = defaultdict(list)
    for p in sorted((DATA / "meta").glob("*.npz")):
        by_ep[_episode_key(p)].append(p)
    eps = {k: [load_npz(p) for p in sorted(v, key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
           for k, v in by_ep.items()}
    print(f"replaying {len(eps)} episodes ...")
    res = {m: evaluate_mode(eps, m, bev["camera"], dt, 3.0, labels_by_key) for m in MODES}

    per_obs = {m: res[m]["per_obs"] for m in MODES}
    shared = sorted(set.intersection(*[set(per_obs[m]) for m in MODES]))
    arr = {m: np.array([per_obs[m][k] for k in shared], float) for m in MODES}

    # ------------------------------------------------- 1. error distributions
    distributions = {
        "experiment_id": "SP-05-ERROR-DISTRIBUTIONS",
        "n_paired_observations": len(shared),
        "denominator": "paired observation = (episode, frame, actor) tracked by all "
                       "three modes AND moving laterally >= 0.3 m/s",
        "by_mode": {m: {"v_lat": dist(arr[m][:, 0]), "v_fwd": dist(arr[m][:, 1]),
                         "position": dist(arr[m][:, 2])} for m in MODES},
    }
    tails = {}
    for i, field in enumerate(("v_lat", "v_fwd", "position")):
        tails[field] = {
            "p95_camera": float(np.percentile(arr["camera"][:, i], 95)),
            "p95_radar": float(np.percentile(arr["radar"][:, i], 95)),
            "p95_fused": float(np.percentile(arr["fused"][:, i], 95)),
        }
        tails[field]["fusion_reduces_p95_vs_camera"] = bool(
            tails[field]["p95_fused"] < tails[field]["p95_camera"])
        tails[field]["fusion_reduces_p95_vs_radar"] = bool(
            tails[field]["p95_fused"] < tails[field]["p95_radar"])
    distributions["tail_comparison"] = tails
    distributions["why_tails_matter"] = (
        "A mean hides whether a configuration is consistently adequate or "
        "occasionally catastrophic. For a braking decision the 95th percentile "
        "error is the operative number, not the average.")
    (OUT / "distributions.json").write_text(json.dumps(distributions, indent=2))

    # --------------------------------------------------------- 2. effect sizes
    effects = {"experiment_id": "SP-06-EFFECT-SIZES",
               "n_paired_observations": len(shared),
               "note": ("complements the first pass's bootstrap p-values: a p-value "
                         "reports whether a difference is distinguishable from zero, "
                         "an effect size reports whether it is large enough to care about"),
               "tests": {}}
    for i, field in enumerate(("v_lat", "v_fwd", "position")):
        effects["tests"][field] = {
            "fused_vs_camera": paired_effect(arr["fused"][:, i], arr["camera"][:, i]),
            "fused_vs_radar": paired_effect(arr["fused"][:, i], arr["radar"][:, i]),
        }
    (OUT / "effect_sizes.json").write_text(json.dumps(effects, indent=2))

    # ------------------------------------------------------- 3. breakdowns
    breakdowns = {"experiment_id": "SP-07-BREAKDOWNS", "min_n_for_report": MIN_N}

    # 3a. tracking error by range band
    rng_rows = {}
    for lo, hi in RANGE_BANDS:
        sel = [j for j, k in enumerate(shared)
               if lo <= meta_by_key.get(k, {}).get("forward_m", -1) < hi]
        if len(sel) < MIN_N:
            continue
        name = f"{lo}-{hi if hi < 1e8 else 'inf'}m"
        rng_rows[name] = {"n": len(sel),
                          **{m: {"v_lat_mae": float(arr[m][sel, 0].mean()),
                                 "position_mae": float(arr[m][sel, 2].mean())}
                             for m in MODES}}
    breakdowns["tracking_by_range"] = rng_rows

    # 3b. intent by scenario and by range
    def intent_subset(mode, keyfilter):
        sbo = res[mode]["scored_by_obs"]
        ks = [k for k in sbo if keyfilter(k)]
        if len(ks) < MIN_N:
            return None
        p = np.array([sbo[k][0] for k in ks], float)
        y = np.array([sbo[k][2] for k in ks], bool)
        if y.all() or (~y).all():
            return {"n": len(ks), "note": "single-class subset; AUC undefined",
                    "positive_rate": float(y.mean())}
        pr, rc, f1 = _prf(p, y, 0.5)
        return {"n": len(ks), "positive_rate": float(y.mean()), "roc_auc": _auc(p, y),
                "precision": pr, "recall": rc, "f1": f1}

    scenarios = sorted({v["scenario"] for v in meta_by_key.values()})
    breakdowns["intent_by_scenario"] = {
        sc: {m: intent_subset(m, lambda k, sc=sc: meta_by_key.get(k, {}).get("scenario") == sc)
             for m in MODES} for sc in scenarios}
    breakdowns["intent_by_range"] = {}
    for lo, hi in RANGE_BANDS:
        name = f"{lo}-{hi if hi < 1e8 else 'inf'}m"
        row = {m: intent_subset(
            m, lambda k, lo=lo, hi=hi: lo <= meta_by_key.get(k, {}).get("forward_m", -1) < hi)
            for m in MODES}
        if any(v for v in row.values()):
            breakdowns["intent_by_range"][name] = row

    # 3c. classifier accuracy + uncertainty by object size and range, from the
    # raw validation predictions the first pass saved.
    raw_p = Path("results/audit/raw_predictions_val.npz")
    if raw_p.exists():
        r = np.load(raw_p)
        y, pred, unc, vis = r["y_true"], r["y_pred"], r["uncertainty"], r["visibility"]
        correct = (y == pred)
        vis_ok = np.isfinite(vis)
        bands = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
        by_vis = {}
        for lo, hi in bands:
            m = vis_ok & (vis >= lo) & (vis < hi)
            if m.sum() >= MIN_N:
                by_vis[f"{lo:.2f}-{min(hi,1.0):.2f}"] = {
                    "n": int(m.sum()), "accuracy": float(correct[m].mean()),
                    "mean_uncertainty": float(unc[m].mean())}
        breakdowns["classifier_by_visibility_fraction"] = by_vis
        # Spearman between uncertainty and being wrong, on matched crops.
        sub = vis_ok
        u, w = unc[sub], (~correct[sub]).astype(float)
        ru = np.argsort(np.argsort(u)); rw = np.argsort(np.argsort(w))
        rho = float(np.corrcoef(ru, rw)[0, 1])
        breakdowns["spearman_uncertainty_vs_error"] = {
            "rho": rho, "n": int(sub.sum()),
            "interpretation": ("positive rho means higher uncertainty accompanies more "
                                "errors, i.e. the uncertainty is ordered with difficulty")}
        breakdowns["classifier_size_analysis"] = {
            "status": "SKIPPED",
            "reason": ("raw_predictions_val.npz stores tier and visibility but not the "
                        "crop's pixel dimensions, and re-deriving them would require "
                        "re-running inference. Visibility fraction is reported instead, "
                        "which is the difficulty axis the project itself defines.")}
    else:
        breakdowns["classifier_by_visibility_fraction"] = {
            "status": "SKIPPED", "reason": "results/audit/raw_predictions_val.npz absent"}

    (OUT / "breakdowns.json").write_text(json.dumps(breakdowns, indent=2))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5), dpi=200)
    for ax, (i, field, unit) in zip(axes, [(0, "v_lat", "m/s"), (1, "v_fwd", "m/s"),
                                            (2, "position", "m")]):
        data = [arr[m][:, i] for m in MODES]
        bp = ax.boxplot(data, labels=list(MODES), showfliers=False, patch_artist=True,
                        medianprops=dict(color="#20242b", linewidth=1.4))
        for patch, c in zip(bp["boxes"], ["#2e86c1", "#c0392b", "#1f9e6e"]):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        for j, m in enumerate(MODES, start=1):
            ax.scatter([j], [np.percentile(arr[m][:, i], 95)], marker="_", s=260,
                       color="#8a1538", zorder=5)
        ax.set_ylabel(f"absolute error ({unit})")
        ax.set_title(f"{field}  (bar = p95)", loc="left", fontweight="bold", fontsize=9.5)
        ax.grid(alpha=0.22, axis="y")
    fig.suptitle(f"Paired tracking error distributions, n={len(shared)} observations "
                 f"(outliers hidden; dark bar = 95th percentile)", fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "error_distributions.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\npaired observations: {len(shared)}")
    print(f"{'mode':<8}{'v_lat med':>11}{'v_lat p95':>11}{'pos med':>10}{'pos p95':>10}")
    for m in MODES:
        d = distributions["by_mode"][m]
        print(f"{m:<8}{d['v_lat']['median']:>11.3f}{d['v_lat']['p95']:>11.3f}"
              f"{d['position']['median']:>10.3f}{d['position']['p95']:>10.3f}")
    print("\neffect sizes (fused vs X, negative dz = fusion better):")
    for field, t in effects["tests"].items():
        for name, e in t.items():
            dz = e["cohens_dz_paired"]
            print(f"  {field:<9}{name:<18} dz={dz:+.3f}  P(fused better)="
                  f"{e['prob_a_better_than_b']:.3f}")
    if "spearman_uncertainty_vs_error" in breakdowns:
        print(f"\nSpearman(uncertainty, error) = "
              f"{breakdowns['spearman_uncertainty_vs_error']['rho']:.4f}")


if __name__ == "__main__":
    main()
