"""AUDIT PHASE 4 -- evidential classifier: classification + uncertainty quality.

Re-runs the SHIPPED checkpoint (models/evidential_detector.pt) over the exact
episode-disjoint validation split the trainer uses, saves RAW per-crop outputs,
and computes every metric from those raw outputs rather than from any stored
summary.

What this can and cannot cover, stated up front because it bounds the claim:
`CropDataset` is built from `labels/*.txt`, which by design carries only
camera-observable actors (tier != OCCLUDED). Fully-occluded objects therefore
never enter the classifier's train or validation set at all, so the
"uncertainty rises under occlusion" question can only be asked of VISIBLE vs
PARTIAL crops here, never of fully-hidden ones.

Writes results/audit/classification_metrics.json,
results/audit/uncertainty_metrics.json, raw predictions as .npz, and figures.
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
import torch

from carla_tools.occlusion_tiers import TIER_NAMES
from common.npz_io import load_npz
from models.evidential_classifier import EvidentialDetector
from models.evidential_head import EvidentialHead
from models.train_evidential import CropDataset, PAD_FRAC, episode_split

DATA = Path("data/raw_v2")
OUT = Path("results/audit")
CKPT = Path("models/evidential_detector.pt")
CLASS_NAMES = ["vehicle", "pedestrian", "background"]
EPS = 1e-12


def _padded_box(x1, y1, x2, y2, w, h):
    """Reproduce CropDataset's PAD_FRAC expansion, to match a crop back to the
    meta record it came from (which carries visibility/tier)."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pw, ph = (x2 - x1) * (1 + PAD_FRAC), (y2 - y1) * (1 + PAD_FRAC)
    return (max(0.0, cx - pw / 2), max(0.0, cy - ph / 2),
            min(float(w), cx + pw / 2), min(float(h), cy + ph / 2))


def build_tier_lookup(stems):
    """stem -> list of (padded_box, tier, visibility) from the meta records."""
    out = {}
    for stem in stems:
        p = DATA / "meta" / f"{stem}.npz"
        if not p.exists():
            continue
        d = load_npz(p)
        rows = []
        for j in range(len(d["obj_actor_id"])):
            b = d["obj_box_px"][j]
            rows.append((tuple(float(v) for v in b), int(d["obj_tier"][j]),
                          float(d["obj_visibility"][j])))
        out[stem] = rows
    return out


def _ece(conf, correct, n_bins=15):
    """Expected Calibration Error, equal-width bins on confidence."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                          "acc": None, "conf": None})
            continue
        acc, avg_conf = float(correct[m].mean()), float(conf[m].mean())
        ece += (m.sum() / len(conf)) * abs(acc - avg_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                      "acc": acc, "conf": avg_conf})
    return float(ece), bins


def _auroc(scores, labels):
    """Rank-based AUROC with tie handling (same estimator as evaluate_prediction)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    if not CKPT.exists():
        raise SystemExit(f"SKIPPED -- checkpoint {CKPT} not present")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CropDataset(str(DATA))
    _train_ds, val_ds, n_train_ep, n_val_ep = episode_split(ds, val_frac=0.15, seed=0)
    val_idx = list(val_ds.indices)
    print(f"val crops: {len(val_idx)}  val episodes: {n_val_ep}  device: {device}")

    model = EvidentialDetector(num_classes=3).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    # ---- raw inference over the validation split
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, val_idx), batch_size=256, shuffle=False, num_workers=0)
    probs_all, unc_all, y_all, alpha_all = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            alpha, unc = model(xb.to(device))
            p = EvidentialHead.expected_probability(alpha)
            probs_all.append(p.cpu().numpy())
            alpha_all.append(alpha.cpu().numpy())
            unc_all.append(unc.squeeze(-1).cpu().numpy())
            y_all.append(yb.numpy())
    probs = np.concatenate(probs_all)
    alphas = np.concatenate(alpha_all)
    unc = np.concatenate(unc_all)
    y = np.concatenate(y_all).astype(int)
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y)

    # ---- tier/visibility for each val crop, matched back to the meta record
    stems = sorted({Path(ds.samples[i][0]).stem for i in val_idx})
    lookup = build_tier_lookup(stems)
    tiers = np.full(len(val_idx), -1, dtype=int)
    vis = np.full(len(val_idx), np.nan)
    for k, i in enumerate(val_idx):
        path, x1, y1, x2, y2, cls = ds.samples[i]
        if cls == 2:            # background crop -- no meta object behind it
            continue
        stem = Path(path).stem
        rows = lookup.get(stem, [])
        best, best_d = None, 12.0
        for raw_box, tier, v in rows:
            pb = _padded_box(*raw_box, 800, 600)
            d = abs(pb[0] - x1) + abs(pb[1] - y1) + abs(pb[2] - x2) + abs(pb[3] - y2)
            if d < best_d:
                best, best_d = (tier, v), d
        if best is not None:
            tiers[k], vis[k] = best[0], best[1]

    np.savez_compressed(OUT / "raw_predictions_val.npz", probs=probs, alphas=alphas,
                         uncertainty=unc, y_true=y, y_pred=pred, tier=tiers,
                         visibility=vis)

    # ---- classification metrics, computed from raw predictions
    cm = np.zeros((3, 3), dtype=int)
    for t, p_ in zip(y, pred):
        cm[t, p_] += 1
    per_class = {}
    for c in range(3):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and np.isfinite(prec)
                                                and np.isfinite(rec)) else float("nan")
        per_class[CLASS_NAMES[c]] = {"precision": prec, "recall": rec, "f1": f1,
                                      "support": int(cm[c, :].sum())}
    support = np.array([per_class[c]["support"] for c in CLASS_NAMES], float)
    f1s = np.array([per_class[c]["f1"] for c in CLASS_NAMES], float)
    precs = np.array([per_class[c]["precision"] for c in CLASS_NAMES], float)
    recs = np.array([per_class[c]["recall"] for c in CLASS_NAMES], float)

    accuracy = float(correct.mean())
    classification = {
        "experiment_id": "AUDIT-A-CLASSIFICATION",
        "split": "episode-disjoint validation (val_frac=0.15, seed=0)",
        "n_crops": int(len(y)),
        "n_val_episodes": int(n_val_ep),
        "n_train_episodes": int(n_train_ep),
        "checkpoint": str(CKPT),
        "accuracy": accuracy,
        "macro_precision": float(np.nanmean(precs)),
        "macro_recall": float(np.nanmean(recs)),
        "macro_f1": float(np.nanmean(f1s)),
        "weighted_f1": float(np.nansum(f1s * support) / support.sum()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_rows": "true", "confusion_matrix_cols": "predicted",
        "class_order": CLASS_NAMES,
    }

    # ---- uncertainty / calibration, all from raw outputs
    brier = float(np.mean([(probs[i] - np.eye(3)[y[i]]) ** 2 for i in range(len(y))]).sum()
                  if False else np.mean(np.sum((probs - np.eye(3)[y]) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))
    ece, rel_bins = _ece(conf, correct)
    # Error detection: can uncertainty separate the crops the model got wrong?
    err_auroc = _auroc(unc, ~correct)

    # Risk-coverage: sort by confidence descending, accumulate error rate.
    order = np.argsort(-conf)
    cum_err = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)
    aurc = float(np.trapezoid(cum_err, coverage)) if hasattr(np, "trapezoid") \
        else float(np.trapz(cum_err, coverage))

    by_tier = {}
    for t in (0, 1, 2):
        m = tiers == t
        if not m.any():
            continue
        by_tier[TIER_NAMES.get(t, str(t))] = {
            "n": int(m.sum()),
            "accuracy": float(correct[m].mean()),
            "mean_uncertainty": float(unc[m].mean()),
            "median_uncertainty": float(np.median(unc[m])),
            "mean_confidence": float(conf[m].mean()),
        }
    unmatched = int((tiers == -1).sum())

    uncertainty = {
        "experiment_id": "AUDIT-A-UNCERTAINTY",
        "n_crops": int(len(y)),
        "expected_calibration_error": ece,
        "ece_bins": 15,
        "brier_score_multiclass": brier,
        "negative_log_likelihood": nll,
        "error_detection_auroc": err_auroc,
        "area_under_risk_coverage": aurc,
        "reliability_bins": rel_bins,
        "uncertainty_summary": {
            "mean": float(unc.mean()), "median": float(np.median(unc)),
            "p05": float(np.percentile(unc, 5)), "p95": float(np.percentile(unc, 95)),
        },
        "uncertainty_by_tier": by_tier,
        "crops_without_tier_match": unmatched,
        "tier_coverage_note": (
            "Background crops have no meta object and are unmatched by construction. "
            "Fully-OCCLUDED objects are absent from labels/ by dataset design, so "
            "this table cannot compare against fully-hidden crops -- only VISIBLE "
            "vs PARTIAL."),
        "interpretation": (
            "error_detection_auroc is the operative calibration claim: it is the "
            "probability that a randomly chosen misclassified crop carries higher "
            "uncertainty than a randomly chosen correct one. 0.5 would mean the "
            "uncertainty output carries no information about correctness."),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classification_metrics.json").write_text(json.dumps(classification, indent=2))
    (OUT / "uncertainty_metrics.json").write_text(json.dumps(uncertainty, indent=2))

    # ---------------------------------------------------------------- figures
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#20242b", fontsize=9)
    ax.set_xticks(range(3), CLASS_NAMES, rotation=20)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Validation confusion matrix (n={len(y)})", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout(); fig.savefig(figdir / "audit_confusion_matrix.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    xs = [b["conf"] for b in rel_bins if b["n"] > 0]
    ys = [b["acc"] for b in rel_bins if b["n"] > 0]
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1, label="perfect calibration")
    ax.plot(xs, ys, marker="o", color="#c0392b", label="observed")
    ax.set_xlabel("mean confidence in bin"); ax.set_ylabel("accuracy in bin")
    ax.set_title(f"Reliability diagram (ECE={ece:.4f})", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figdir / "audit_reliability_diagram.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    ax.hist(unc[correct], bins=40, alpha=0.65, label="correct", color="#1f9e6e")
    ax.hist(unc[~correct], bins=40, alpha=0.65, label="incorrect", color="#c0392b")
    ax.set_xlabel("predicted uncertainty"); ax.set_ylabel("crops")
    ax.set_yscale("log")
    ax.set_title(f"Uncertainty by correctness (AUROC={err_auroc:.3f})", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "audit_uncertainty_hist.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    ax.plot(coverage, cum_err, color="#20242b")
    ax.set_xlabel("coverage (fraction of crops kept, most-confident first)")
    ax.set_ylabel("error rate on kept crops")
    ax.set_title(f"Risk-coverage (AURC={aurc:.4f})", fontsize=9, loc="left")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figdir / "audit_risk_coverage.png"); plt.close(fig)

    print(json.dumps({
        "accuracy": accuracy, "macro_f1": classification["macro_f1"],
        "weighted_f1": classification["weighted_f1"], "ECE": ece, "Brier": brier,
        "NLL": nll, "error_detection_AUROC": err_auroc, "AURC": aurc,
        "by_tier": by_tier, "n": len(y)}, indent=2))


if __name__ == "__main__":
    main()
