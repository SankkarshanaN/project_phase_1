"""AUDIT PHASE 13 -- paired significance for the fusion claims.

Comparing two marginal confidence intervals and asking whether they overlap is
the wrong test for paired data: it ignores the per-observation pairing and is
badly under-powered. The correct analysis resamples EPISODES and recomputes the
DIFFERENCE of the paired means, then asks whether that difference's CI excludes
zero.

This does not change any method or metric. It only asks, of numbers already
computed, whether the dataset actually supports the comparative claims.

Writes results/audit/statistics.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_yaml
from common.npz_io import load_npz
from scripts.evaluate_prediction import MODES, _episode_key, evaluate_mode

DATA = Path("data/raw_v2")
OUT = Path("results/audit")
N_BOOT = 10000
FIELDS = {"v_lat": 0, "v_fwd": 1, "position": 2}


def paired_diff_test(per_obs, keys, mode_a, mode_b, field_idx, n_boot=N_BOOT, seed=0):
    """Bootstrap CI for mean(a) - mean(b) over the SAME observations, resampling
    episodes. Negative means `mode_a` has lower error, i.e. is better."""
    by_ep = defaultdict(list)
    for k in keys:
        by_ep[k[0]].append(k)
    episodes = sorted(by_ep)
    rng = np.random.default_rng(seed)

    observed = (np.mean([per_obs[mode_a][k][field_idx] for k in keys])
                - np.mean([per_obs[mode_b][k][field_idx] for k in keys]))
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(episodes), size=len(episodes), replace=True)
        ks = [k for i in pick for k in by_ep[episodes[i]]]
        a = np.mean([per_obs[mode_a][k][field_idx] for k in ks])
        b = np.mean([per_obs[mode_b][k][field_idx] for k in ks])
        diffs.append(a - b)
    d = np.asarray(diffs, float)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    # Two-sided bootstrap p-value: how often the resampled difference crosses 0.
    p = 2.0 * min((d >= 0).mean(), (d <= 0).mean())
    return {
        "comparison": f"{mode_a} minus {mode_b}",
        "observed_difference": float(observed),
        "ci95": [lo, hi],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "bootstrap_p_two_sided": float(min(p, 1.0)),
        "better": (mode_a if observed < 0 else mode_b),
        "interpretation": ("difference is statistically distinguishable from zero"
                            if (lo > 0 or hi < 0) else
                            "difference is NOT distinguishable from zero at this sample size"),
    }


def main():
    bev_cfg = load_yaml("bev.yaml")
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])
    lab = load_npz(DATA / "intent_labels.npz")
    labels_by_key = {
        (str(lab["episode"][i]), int(lab["frame_idx"][i]), int(lab["actor_id"][i])):
            {"will_cross": bool(lab["will_cross"][i]), "tier": int(lab["tier"][i])}
        for i in range(len(lab["episode"]))}

    by_ep = defaultdict(list)
    for p in sorted((DATA / "meta").glob("*.npz")):
        by_ep[_episode_key(p)].append(p)
    episodes = {k: [load_npz(p) for p in sorted(v, key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
                for k, v in by_ep.items()}

    res = {m: evaluate_mode(episodes, m, bev_cfg["camera"], dt, 3.0, labels_by_key)
           for m in MODES}
    per_obs = {m: res[m]["per_obs"] for m in MODES}
    keys = sorted(set.intersection(*[set(per_obs[m]) for m in MODES]))
    n_ep = len({k[0] for k in keys})

    tests = {}
    for field, idx in FIELDS.items():
        tests[field] = {
            "fused_vs_camera": paired_diff_test(per_obs, keys, "fused", "camera", idx),
            "fused_vs_radar": paired_diff_test(per_obs, keys, "fused", "radar", idx),
        }

    out = {
        "experiment_id": "AUDIT-STAT-PAIRED",
        "n_paired_observations": len(keys),
        "n_episodes_contributing": n_ep,
        "n_bootstrap": N_BOOT,
        "resampling_unit": "episode",
        "method": ("paired difference of means over identical observations, with "
                    "episodes resampled with replacement; CI from the 2.5/97.5 "
                    "percentiles of the bootstrap difference distribution"),
        "why_not_marginal_cis": (
            "Overlapping marginal CIs do not imply a non-significant difference for "
            "PAIRED data. The paired difference removes the between-episode variance "
            "that dominates each marginal CI, so it is the appropriate test here."),
        "tests": tests,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "statistics.json").write_text(json.dumps(out, indent=2))

    print(f"paired obs={len(keys)}  episodes={n_ep}  boot={N_BOOT}\n")
    for field in FIELDS:
        for name, t in tests[field].items():
            flag = "SIGNIFICANT" if t["excludes_zero"] else "not significant"
            print(f"{field:<9} {name:<18} diff={t['observed_difference']:+.4f} "
                  f"CI[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] p={t['bootstrap_p_two_sided']:.4f}"
                  f"  -> {flag}")


if __name__ == "__main__":
    main()
