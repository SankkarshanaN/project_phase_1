"""SECOND PASS -- crossing-intent performance as a function of LEAD TIME.

The safety question the first audit never answered: *how early* can the system
flag a crossing before the pedestrian actually enters the ego's corridor?

This needs no new label definition. `scripts/label_crossing_intent.py` already
records `time_to_entry_s` per observation -- the ground-truth number of seconds
until that actor first occupies the corridor -- computed with the project's own
existing corridor test. This script only stratifies by it.

Design of the evaluation, and why
---------------------------------
`time_to_entry_s` is defined only for positives (an actor that never crosses
has no entry time). So for each lead-time band we form:

    positives = observations whose TRUE time-to-entry falls in that band
    negatives = ALL observations labelled will_cross == False

and score the predictor on that pooled set. Recall in a band is then literally
"of the crossings that were T seconds away, what fraction did we flag", which
is the operative safety number. Precision/AUC share one common negative pool
across bands, so they are comparable band to band; this is stated rather than
hidden, because a per-band negative pool would make the columns incomparable.

Writes results/second_pass/intent_vs_lead_time.{json,csv} and a figure.
"""
import csv
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
# Bands are upper-exclusive; 3.0 s is the labelling horizon, so nothing exists
# beyond it by construction.
BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.01)]
MIN_POS = 20            # below this a band is reported but flagged unreliable


def pr_auc(p, y):
    order = np.argsort(-p)
    ys = np.asarray(y, bool)[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    rec = tp / max(int(ys.sum()), 1)
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def main():
    bev = load_yaml("bev.yaml")
    dt = float(load_yaml("town.yaml")["fixed_delta_seconds"])
    lab = load_npz(DATA / "intent_labels.npz")

    labels_by_key, tte = {}, {}
    for i in range(len(lab["episode"])):
        k = (str(lab["episode"][i]), int(lab["frame_idx"][i]), int(lab["actor_id"][i]))
        labels_by_key[k] = {"will_cross": bool(lab["will_cross"][i]),
                            "tier": int(lab["tier"][i])}
        tte[k] = float(lab["time_to_entry_s"][i])

    finite = np.asarray([v for v in tte.values() if np.isfinite(v)], float)
    print(f"observations with a ground-truth entry time: {finite.size:,} "
          f"(min {finite.min():.2f}s, max {finite.max():.2f}s)")

    by_ep = defaultdict(list)
    for p in sorted((DATA / "meta").glob("*.npz")):
        by_ep[_episode_key(p)].append(p)
    eps = {k: [load_npz(p) for p in sorted(v, key=lambda q: int(q.stem.rsplit("_f", 1)[1]))]
           for k, v in by_ep.items()}
    print(f"replaying {len(eps)} episodes for {len(MODES)} configurations ...")

    res = {m: evaluate_mode(eps, m, bev["camera"], dt, 3.0, labels_by_key) for m in MODES}

    rows, payload = [], {"experiment_id": "SP-02-INTENT-LEAD-TIME", "status": "EXECUTED",
                          "bands_s": [[a, b] for a, b in BANDS],
                          "ground_truth": "time_to_entry_s from intent_labels.npz "
                                           "(project's own corridor definition)",
                          "evaluation_design": (
                              "per band: positives = observations whose true "
                              "time-to-entry falls in the band; negatives = the common "
                              "pool of all will_cross==False observations scored by that "
                              "configuration. Recall is band-specific; precision and AUC "
                              "share the common negative pool and are comparable across "
                              "bands for a given configuration."),
                          "min_positives_for_reliability": MIN_POS,
                          "by_mode": {}}

    for m in MODES:
        sbo = res[m]["scored_by_obs"]
        if not sbo:
            payload["by_mode"][m] = {"n_scored": 0}
            continue
        keys = list(sbo)
        p_all = np.array([sbo[k][0] for k in keys], float)
        y_all = np.array([sbo[k][2] for k in keys], bool)
        t_all = np.array([tte.get(k, np.nan) for k in keys], float)

        neg = ~y_all
        n_neg = int(neg.sum())
        band_rows = {}
        for lo, hi in BANDS:
            sel_pos = y_all & np.isfinite(t_all) & (t_all >= lo) & (t_all < hi)
            n_pos = int(sel_pos.sum())
            if n_pos == 0:
                continue
            mask = sel_pos | neg
            p, y = p_all[mask], y_all[mask]
            prec, rec, f1 = _prf(p, y, 0.5)
            tn = int(((p < 0.5) & ~y).sum()); fp = int(((p >= 0.5) & ~y).sum())
            spec = tn / (tn + fp) if (tn + fp) else float("nan")
            band = {
                "band_s": [lo, min(hi, 3.0)], "n_positives": n_pos, "n_negatives": n_neg,
                "roc_auc": _auc(p, y), "pr_auc": pr_auc(p, y),
                "precision": prec, "recall": rec, "f1": f1, "specificity": float(spec),
                "balanced_accuracy": float((rec + spec) / 2) if np.isfinite(rec) else None,
                "reliable": n_pos >= MIN_POS,
            }
            band_rows[f"{lo:.1f}-{min(hi,3.0):.1f}s"] = band
            rows.append({"mode": m, "band_s": f"{lo:.1f}-{min(hi,3.0):.1f}",
                          "n_positives": n_pos, "n_negatives": n_neg,
                          "roc_auc": band["roc_auc"], "pr_auc": band["pr_auc"],
                          "precision": prec, "recall": rec, "f1": f1,
                          "balanced_accuracy": band["balanced_accuracy"],
                          "reliable": band["reliable"]})
        payload["by_mode"][m] = {"n_scored": len(keys), "n_negatives_pool": n_neg,
                                  "bands": band_rows}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intent_vs_lead_time.json").write_text(json.dumps(payload, indent=2))
    if rows:
        with (OUT / "intent_vs_lead_time.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), dpi=200)
    colours = {"camera": "#2e86c1", "radar": "#c0392b", "fused": "#1f9e6e"}
    for m in MODES:
        b = payload["by_mode"][m].get("bands", {})
        if not b:
            continue
        xs = [np.mean(v["band_s"]) for v in b.values()]
        axes[0].plot(xs, [v["recall"] for v in b.values()], marker="o",
                     color=colours[m], label=m)
        axes[1].plot(xs, [v["roc_auc"] for v in b.values()], marker="o",
                     color=colours[m], label=m)
    for ax, ttl, yl in ((axes[0], "Recall vs lead time", "recall @ p>=0.5"),
                        (axes[1], "ROC-AUC vs lead time", "ROC-AUC")):
        ax.set_xlabel("ground-truth time until corridor entry (s)")
        ax.set_ylabel(yl); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        ax.invert_xaxis()      # later = further left: reads as "time running out"
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("How early is a crossing detected? (x-axis inverted: right = earliest)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "intent_lead_time.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'mode':<8}{'band':>10}{'n_pos':>7}{'AUC':>8}{'prec':>8}{'rec':>8}{'F1':>8}")
    for r in rows:
        flag = "" if r["reliable"] else "  (n<20)"
        print(f"{r['mode']:<8}{r['band_s']:>10}{r['n_positives']:>7}{r['roc_auc']:>8.3f}"
              f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}{flag}")


if __name__ == "__main__":
    main()
