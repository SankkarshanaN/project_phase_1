"""AUDIT PHASE 6+8 -- camera / radar / fusion tracking and crossing intent,
recomputed from raw replay, with episode-level bootstrap confidence intervals.

Two things this adds over `scripts/evaluate_prediction.py`:

1. PAIRED metrics with episode-level bootstrap CIs. Consecutive frames of one
   episode are strongly correlated, so treating thousands of frames as
   independent samples produces CIs that are far too narrow. Resampling
   EPISODES with replacement respects that correlation.

2. Extra error statistics the manuscript does not currently report -- RMSE,
   median absolute error and the 95th percentile -- because a mean alone
   hides whether an estimator is consistently decent or occasionally awful.

Reuses evaluate_prediction's own functions so the replay path is identical to
the one the manuscript's numbers came from; nothing is re-implemented.

Writes results/audit/tracking_metrics.json and results/audit/intent_metrics.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_yaml
from common.npz_io import load_npz
from scripts.evaluate_prediction import (MODES, _auc, _episode_key, _prf, evaluate_mode)

DATA = Path("data/raw_v2")
OUT = Path("results/audit")
N_BOOT = 2000


def episode_bootstrap(per_obs_by_mode, modes, keys_shared, n_boot=N_BOOT, seed=0):
    """Bootstrap the PAIRED means by resampling episodes with replacement.

    `keys_shared` are (episode, frame, actor) tuples every mode tracked. We
    group them by episode, resample episodes, and recompute each mode's mean
    over the observations belonging to the resampled episodes.
    """
    by_ep = defaultdict(list)
    for k in keys_shared:
        by_ep[k[0]].append(k)
    episodes = sorted(by_ep)
    rng = np.random.default_rng(seed)

    draws = {m: {"lat": [], "fwd": [], "pos": []} for m in modes}
    for _ in range(n_boot):
        picked = rng.choice(len(episodes), size=len(episodes), replace=True)
        keys = [k for i in picked for k in by_ep[episodes[i]]]
        if not keys:
            continue
        for m in modes:
            a = np.array([per_obs_by_mode[m][k] for k in keys], float)
            draws[m]["lat"].append(a[:, 0].mean())
            draws[m]["fwd"].append(a[:, 1].mean())
            draws[m]["pos"].append(a[:, 2].mean())
    out = {}
    for m in modes:
        out[m] = {}
        for field in ("lat", "fwd", "pos"):
            d = np.array(draws[m][field], float)
            out[m][field] = {"mean": float(d.mean()), "std": float(d.std(ddof=1)),
                              "ci95": [float(np.percentile(d, 2.5)),
                                       float(np.percentile(d, 97.5))]}
    return out, len(episodes)


def err_stats(a):
    a = np.asarray(a, float)
    return {"mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(a ** 2))),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)}


def main():
    bev_cfg = load_yaml("bev.yaml")
    town_cfg = load_yaml("town.yaml")
    dt = float(town_cfg["fixed_delta_seconds"])

    lab = load_npz(DATA / "intent_labels.npz")
    labels_by_key = {
        (str(lab["episode"][i]), int(lab["frame_idx"][i]), int(lab["actor_id"][i])):
            {"will_cross": bool(lab["will_cross"][i]), "tier": int(lab["tier"][i])}
        for i in range(len(lab["episode"]))}

    by_episode = defaultdict(list)
    for p in sorted((DATA / "meta").glob("*.npz")):
        by_episode[_episode_key(p)].append(p)
    episodes = {k: [load_npz(p) for p in sorted(v, key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
                for k, v in by_episode.items()}
    print(f"replaying {len(episodes)} episodes")

    results = {m: evaluate_mode(episodes, m, bev_cfg["camera"], dt, 3.0, labels_by_key)
               for m in MODES}

    per_obs = {m: results[m]["per_obs"] for m in MODES}
    shared = sorted(set.intersection(*[set(per_obs[m]) for m in MODES]))
    boot, n_ep_shared = episode_bootstrap(per_obs, list(MODES), shared)

    paired = {}
    for m in MODES:
        a = np.array([per_obs[m][k] for k in shared], float)
        paired[m] = {
            "v_lat": err_stats(a[:, 0]), "v_fwd": err_stats(a[:, 1]),
            "position": err_stats(a[:, 2]),
            "bootstrap_episode_level": boot[m],
        }

    tracking = {
        "experiment_id": "AUDIT-C-TRACKING",
        "detections": "ground-truth boxes (occluded actors withheld, filter coasts)",
        "paired_observations": len(shared),
        "episodes_contributing_to_paired": n_ep_shared,
        "episodes_replayed": len(episodes),
        "denominator": ("paired observation = one (episode, frame, actor) tracked by ALL "
                        "THREE modes AND moving laterally >= 0.3 m/s"),
        "bootstrap": {"n_resamples": N_BOOT, "unit": "episode", "seed": 0,
                       "why": "consecutive frames within an episode are correlated; "
                              "frame-level CIs would be far too narrow"},
        "paired_metrics": paired,
        "unpaired_counts": {m: {"n_obs": results[m]["n_obs"],
                                 "n_moving": results[m]["n_moving"],
                                 "id_switches": results[m]["id_switches"],
                                 "n_occluded_obs": results[m]["n_occluded"]}
                             for m in MODES},
        "unpaired_warning": (
            "Unpaired means are NOT comparable across modes -- each mode tracks a "
            "different subset of actors. Counts are reported because coverage is a "
            "real property; the paired block is the comparison."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tracking_metrics.json").write_text(json.dumps(tracking, indent=2))

    # ------------------------------------------------------------------ intent
    intent = {"experiment_id": "AUDIT-F-INTENT", "threshold": 0.5, "by_mode": {}}
    for m in MODES:
        sc = results[m]["scored"]
        if not sc:
            intent["by_mode"][m] = {"n": 0, "note": "no scored observations"}
            continue
        p, pc, y, tiers, leads = map(np.array, zip(*sc))
        prec, rec, f1 = _prf(p, y, 0.5)
        tn = int(((p < 0.5) & ~y.astype(bool)).sum())
        fp = int(((p >= 0.5) & ~y.astype(bool)).sum())
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        good = [l for l, yy in zip(leads, y) if yy and l is not None]
        # PR-AUC via step integration over sorted thresholds.
        order = np.argsort(-p)
        ys = y.astype(bool)[order]
        tp_c = np.cumsum(ys); fp_c = np.cumsum(~ys)
        rec_c = tp_c / max(int(ys.sum()), 1)
        prec_c = tp_c / np.maximum(tp_c + fp_c, 1)
        pr_auc = float(np.sum(np.diff(np.concatenate([[0.0], rec_c])) * prec_c))
        by_tier = {}
        for t in sorted(set(tiers.tolist())):
            mt = tiers == t
            if mt.sum() < 5:
                continue
            pt, rt, ft = _prf(p[mt], y[mt], 0.5)
            by_tier[str(int(t))] = {"n": int(mt.sum()), "positive_rate": float(y[mt].mean()),
                                     "auc": _auc(p[mt], y[mt]), "precision": pt,
                                     "recall": rt, "f1": ft}
        intent["by_mode"][m] = {
            "n": int(len(y)), "positive_rate": float(y.mean()),
            "roc_auc": _auc(p, y), "pr_auc": pr_auc, "precision": prec, "recall": rec,
            "f1": f1, "specificity": float(spec),
            "balanced_accuracy": float((rec + spec) / 2) if np.isfinite(rec) else float("nan"),
            "mean_lead_time_s": float(np.mean(good)) if good else float("nan"),
            "n_with_lead_time": len(good),
            "by_tier": by_tier,
            "tier_key": "0=OCCLUDED 1=PARTIAL 2=VISIBLE",
        }
    # Paired intent comparison on observations all three modes scored.
    shared_sc = set.intersection(*[set(results[m]["scored_by_obs"]) for m in MODES])
    intent["paired"] = {"n": len(shared_sc)}
    if shared_sc:
        ordk = sorted(shared_sc)
        for m in MODES:
            rows = [results[m]["scored_by_obs"][k] for k in ordk]
            p = np.array([r[0] for r in rows]); y = np.array([r[2] for r in rows])
            pr, rc, f1 = _prf(p, y, 0.5)
            intent["paired"][m] = {"roc_auc": _auc(p, y), "precision": pr,
                                    "recall": rc, "f1": f1}
    (OUT / "intent_metrics.json").write_text(json.dumps(intent, indent=2))

    print("\n=== PAIRED TRACKING (episode-bootstrap 95% CI) ===")
    print(f"paired obs={len(shared)} from {n_ep_shared} episodes")
    for m in MODES:
        b = paired[m]["bootstrap_episode_level"]
        print(f"{m:<8} v_lat {paired[m]['v_lat']['mae']:.3f} "
              f"CI[{b['lat']['ci95'][0]:.3f},{b['lat']['ci95'][1]:.3f}]  "
              f"v_fwd {paired[m]['v_fwd']['mae']:.3f} "
              f"CI[{b['fwd']['ci95'][0]:.3f},{b['fwd']['ci95'][1]:.3f}]  "
              f"pos {paired[m]['position']['mae']:.3f} "
              f"CI[{b['pos']['ci95'][0]:.3f},{b['pos']['ci95'][1]:.3f}]")
    print("\n=== INTENT ===")
    for m in MODES:
        d = intent["by_mode"][m]
        if d.get("n"):
            print(f"{m:<8} n={d['n']:<6} ROC-AUC {d['roc_auc']:.3f}  PR-AUC {d['pr_auc']:.3f}  "
                  f"P {d['precision']:.3f}  R {d['recall']:.3f}  F1 {d['f1']:.3f}")


if __name__ == "__main__":
    main()
