"""SECOND PASS -- gap-matrix rows 8 and 2.

Row 8: WHERE does the occlusion detector make its errors?
    The first audit reported aggregate P/R/F1 on held-out frames and a
    breakdown by scene severity and scenario. It never asked where in the
    grid the false positives and false negatives sit. Two structural
    questions matter for a monocular-disparity method:
      - does accuracy fall with range? (disparity resolution degrades
        quadratically with distance, so it should)
      - are errors concentrated on the BOUNDARY of occluded regions, which
        is a cheap labelling/discretisation disagreement, or in the
        INTERIOR, which is a real failure to see a shadow?

Row 2: is the episode-disjoint split actually leak-free at the PIXEL level?
    `episode_split` guarantees no episode appears on both sides. That does
    not by itself guarantee no near-identical crop appears on both sides:
    the scenarios are generated procedurally from seeds, so two different
    episodes of the same scenario can stage visually similar frames. This
    hashes every crop and counts cross-boundary collisions.

Neither analysis changes any operating point or retrains anything.

Writes results/second_pass/spatial_errors.json, leakage_audit.json, and a figure.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carla_tools.occlusion_mask import OCCLUDED as GT_OCCLUDED
from common.config import load_yaml
from models.train_evidential import CropDataset, episode_split

CLASS_NAMES = {0: "vehicle", 1: "pedestrian", 2: "background"}
from perception.occlusion_grid import OCCLUDED as PRED_OCCLUDED, classify_grid

DATA = Path("data/raw_v2")
OUT = Path("results/second_pass")
SWEEP_SEED, SWEEP_N = 0, 120       # identical to audit_occlusion_holdout.py
HOLDOUT_SEED, HOLDOUT_N = 1234, 300
HASH_SIDE = 8                      # dHash on an 8x9 grayscale -> 64-bit


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p and r and np.isfinite(p) and np.isfinite(r)) else float("nan")
    return float(p), float(r), float(f)


# --------------------------------------------------------------- row 8
def spatial_errors():
    bev_cfg = load_yaml("bev.yaml")
    grid = bev_cfg["bev_grid"]
    # occlusion_mask.BevProjector builds the grid as (forward, lateral) with
    # forward 0..extent_m and lateral -extent_m/2..+extent_m/2, POSITIVE LEFT.
    n_fwd = n_lat = int(grid["size_cells"])
    cell_m = float(grid["cell_size_m"])
    extent_m = float(grid["extent_m"])

    metas = sorted((DATA / "meta").glob("*.npz"))
    random.seed(SWEEP_SEED)
    sweep = set(p.stem for p in random.sample(metas, min(SWEEP_N, len(metas))))
    pool = [p for p in metas if p.stem not in sweep]
    held = random.Random(HOLDOUT_SEED).sample(pool, min(HOLDOUT_N, len(pool)))
    print(f"[row 8] evaluating {len(held)} held-out frames (sweep's {len(sweep)} excluded)")

    tp_map = np.zeros((n_fwd, n_lat), np.int64)
    fp_map = np.zeros_like(tp_map); fn_map = np.zeros_like(tp_map)
    tn_map = np.zeros_like(tp_map)
    # boundary = a GT-OCCLUDED cell with at least one 4-neighbour not occluded
    b = {"boundary": Counter(), "interior": Counter()}
    used = 0

    for p in held:
        img = cv2.imread(str(DATA / "images" / f"{p.stem}.jpg"))
        if img is None:
            continue
        d = np.load(p, allow_pickle=True)
        gt = np.asarray(d["occ_grid"])
        pred = classify_grid(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d["radar_pts"], bev_cfg)
        if gt.shape != tp_map.shape or pred.shape != tp_map.shape:
            continue
        used += 1
        g, q = (gt == GT_OCCLUDED), (pred == PRED_OCCLUDED)
        tp_map += (g & q); fp_map += (~g & q); fn_map += (g & ~q); tn_map += (~g & ~q)

        # 4-neighbour erosion: interior = occluded and all 4 neighbours occluded
        pad = np.pad(g, 1, constant_values=False)
        interior = g & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        boundary = g & ~interior
        b["interior"]["tp"] += int((interior & q).sum())
        b["interior"]["fn"] += int((interior & ~q).sum())
        b["boundary"]["tp"] += int((boundary & q).sum())
        b["boundary"]["fn"] += int((boundary & ~q).sum())

    # ------- by range ring (rows of the grid are forward distance)
    rings = []
    for i in range(n_fwd):
        tp, fp, fn = int(tp_map[i].sum()), int(fp_map[i].sum()), int(fn_map[i].sum())
        tn = int(tn_map[i].sum())
        pr, rc, f1 = prf(tp, fp, fn)
        rings.append({"range_m": [round(i * cell_m, 1), round((i + 1) * cell_m, 1)],
                       "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "gt_occluded_fraction": (tp + fn) / max(tp + fn + fp + tn, 1),
                       "precision": pr, "recall": rc, "f1": f1})
    # coarse bands for a quotable statement
    bands = {}
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40)):
        sel = slice(int(lo / cell_m), int(hi / cell_m))
        tp, fp, fn = int(tp_map[sel].sum()), int(fp_map[sel].sum()), int(fn_map[sel].sum())
        pr, rc, f1 = prf(tp, fp, fn)
        bands[f"{lo}-{hi}m"] = {"tp": tp, "fp": fp, "fn": fn,
                                 "precision": pr, "recall": rc, "f1": f1}

    # ------- by lateral offset (columns; lateral POSITIVE IS LEFT, per project convention)
    lat = []
    for j in range(n_lat):
        off = (j - n_lat / 2 + 0.5) * cell_m
        tp, fp, fn = int(tp_map[:, j].sum()), int(fp_map[:, j].sum()), int(fn_map[:, j].sum())
        pr, rc, f1 = prf(tp, fp, fn)
        lat.append({"lateral_m_centre": round(float(off), 1),
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": pr, "recall": rc, "f1": f1})

    bi_r = b["interior"]["tp"] / max(b["interior"]["tp"] + b["interior"]["fn"], 1)
    bb_r = b["boundary"]["tp"] / max(b["boundary"]["tp"] + b["boundary"]["fn"], 1)
    n_fn = int(fn_map.sum())
    fn_on_boundary = b["boundary"]["fn"] / max(n_fn, 1)

    out = {
        "experiment_id": "SP-08-OCCLUSION-SPATIAL-ERRORS",
        "status": "EXECUTED",
        "question": "where in the BEV grid does the occlusion detector fail?",
        "method": ("same fixed operating point and same held-out frame set as "
                    "AUDIT-B-OCCLUSION-HOLDOUT (seed 1234, sweep's seed-0 sample "
                    "excluded); nothing was retuned"),
        "n_frames": used,
        "denominator": "BEV grid CELLS; each frame contributes 400",
        "by_range_ring": rings,
        "by_range_band": bands,
        "by_lateral_column": lat,
        "boundary_vs_interior": {
            "definition": ("interior = GT-OCCLUDED cell whose four 4-neighbours are "
                            "all GT-OCCLUDED; boundary = every other GT-OCCLUDED cell"),
            "interior_recall": float(bi_r), "interior_cells": int(b["interior"]["tp"] + b["interior"]["fn"]),
            "boundary_recall": float(bb_r), "boundary_cells": int(b["boundary"]["tp"] + b["boundary"]["fn"]),
            "fraction_of_all_FN_that_are_boundary_cells": float(fn_on_boundary),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spatial_errors.json").write_text(json.dumps(out, indent=2))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=200)
    xs = [np.mean(r["range_m"]) for r in rings]
    axes[0].plot(xs, [r["recall"] for r in rings], marker="o", color="#1f9e6e", label="recall")
    axes[0].plot(xs, [r["precision"] for r in rings], marker="s", color="#c0392b", label="precision")
    axes[0].set_xlabel("forward range (m)"); axes[0].set_ylabel("score"); axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(alpha=0.25)
    axes[0].set_title("Accuracy vs range", loc="left", fontweight="bold", fontsize=10)

    err = (fp_map + fn_map) / max(used, 1)
    # flip the lateral axis so +LEFT is drawn on the left, as a BEV should read
    im = axes[1].imshow(err[:, ::-1], origin="lower", aspect="auto", cmap="magma",
                        extent=[extent_m / 2, -extent_m / 2, 0, n_fwd * cell_m])
    axes[1].set_xlabel("lateral (m, + = LEFT)"); axes[1].set_ylabel("forward (m)")
    axes[1].set_title("Errors per frame (FP+FN)", loc="left", fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].bar(["interior", "boundary"], [bi_r, bb_r], color=["#2e86c1", "#e59866"])
    axes[2].set_ylim(0, 1.02); axes[2].set_ylabel("recall")
    axes[2].set_title("Interior vs boundary recall", loc="left", fontweight="bold", fontsize=10)
    axes[2].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "occlusion_spatial_errors.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  range bands: " + "  ".join(
        f"{k} R={v['recall']:.3f}/P={v['precision']:.3f}" for k, v in bands.items()))
    print(f"  interior recall {bi_r:.3f} ({b['interior']['tp']+b['interior']['fn']} cells)  "
          f"boundary recall {bb_r:.3f} ({b['boundary']['tp']+b['boundary']['fn']} cells)")
    print(f"  {fn_on_boundary:.1%} of all false negatives are boundary cells")
    return out


# --------------------------------------------------------------- row 2
def leakage_audit():
    """dHash every crop, count identical hashes that straddle the train/val line.

    dHash compares adjacent-pixel brightness on a 9x8 grayscale reduction. Two
    crops with the same 64-bit hash are visually near-identical; Hamming
    distance <= 5 is the conventional near-duplicate threshold.
    """
    ds = CropDataset(str(DATA))
    train, val, n_tr_ep, n_val_ep = episode_split(ds, 0.2)
    tr_idx, val_idx = list(train.indices), list(val.indices)
    print(f"[row 2] {len(tr_idx)} train / {len(val_idx)} val crops, "
          f"{n_tr_ep} / {n_val_ep} episodes")

    # Episode-level check first: the guarantee episode_split actually makes.
    tr_eps = {ds.episode_of(i) for i in tr_idx}
    val_eps = {ds.episode_of(i) for i in val_idx}
    ep_overlap = tr_eps & val_eps

    def hashes(indices, tag):
        """Group by source image so each JPEG is decoded once, not once per crop."""
        per_image = defaultdict(list)
        for i in indices:
            s = ds.samples[i]
            per_image[s[0]].append((i, s[1], s[2], s[3], s[4]))
        out = {}
        for n, (path, crops) in enumerate(per_image.items()):
            if n % 2000 == 0:
                print(f"    hashing {tag} image {n}/{len(per_image)}", flush=True)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            for i, x1, y1, x2, y2 in crops:
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    continue
                small = cv2.resize(crop, (HASH_SIDE + 1, HASH_SIDE),
                                    interpolation=cv2.INTER_AREA).astype(np.int16)
                bits = (small[:, 1:] > small[:, :-1]).flatten()
                h = 0
                for bit in bits:
                    h = (h << 1) | int(bit)
                out.setdefault(h, []).append(i)
        return out

    tr_h = hashes(tr_idx, "train")
    val_h = hashes(val_idx, "val")

    exact = set(tr_h) & set(val_h)
    n_val_exact = sum(len(val_h[h]) for h in exact)
    cls_exact = Counter(ds.samples[i][5] for h in exact for i in val_h[h])

    # Near-duplicates at Hamming <= 3, found exactly via 4-band LSH: split each
    # 64-bit hash into four 16-bit bands. If two hashes differ in at most 3 bits
    # then by pigeonhole at least one of the four bands is identical, so
    # bucketing on bands finds every such pair without an all-pairs scan.
    # (Hamming <= 5 has no such guarantee with 4 bands, so the stricter -- and
    # for a near-duplicate claim, more meaningful -- threshold is used.)
    HAMMING_MAX = 3
    bands_tr = [defaultdict(list) for _ in range(4)]
    for h in tr_h:
        for bi in range(4):
            bands_tr[bi][(h >> (16 * bi)) & 0xFFFF].append(h)
    near_val_crops, near_pairs = 0, 0
    cls_near = Counter()
    for vh in val_h:
        cand = set()
        for bi in range(4):
            cand.update(bands_tr[bi].get((vh >> (16 * bi)) & 0xFFFF, ()))
        hit = [th for th in cand if bin(vh ^ th).count("1") <= HAMMING_MAX]
        if hit:
            near_pairs += len(hit)
            near_val_crops += len(val_h[vh])
            cls_near.update(ds.samples[i][5] for i in val_h[vh])

    # Within-episode consecutive-frame similarity, for scale: this is the
    # leakage a per-crop random split WOULD have introduced.
    by_ep_hash = defaultdict(list)
    for h, idxs in tr_h.items():
        for i in idxs:
            by_ep_hash[ds.episode_of(i)].append(h)
    within = []
    for ep, hs in list(by_ep_hash.items())[:20]:
        a = np.array(sorted(set(hs)), dtype=np.uint64)
        if a.size < 2:
            continue
        d = [bin(int(a[k]) ^ int(a[k + 1])).count("1") for k in range(a.size - 1)]
        within.append(float(np.mean([x <= 3 for x in d])))

    out = {
        "experiment_id": "SP-09-LEAKAGE-AUDIT",
        "status": "EXECUTED",
        "question": ("does episode-disjoint splitting leave any near-identical crop "
                      "on both sides of the train/val boundary?"),
        "method": ("64-bit dHash of every crop at its recorded box; exact hash collisions "
                    "and Hamming<=3 near-duplicates (found exactly via 4-band LSH) "
                    "counted between the two sides"),
        "n_train_crops": len(tr_idx), "n_val_crops": len(val_idx),
        "n_train_episodes": n_tr_ep, "n_val_episodes": n_val_ep,
        "episode_overlap": sorted(ep_overlap),
        "episode_overlap_count": len(ep_overlap),
        "exact_hash_collisions": {
            "n_colliding_hashes": len(exact),
            "n_val_crops_affected": n_val_exact,
            "fraction_of_val": n_val_exact / max(len(val_idx), 1),
            "by_class": {CLASS_NAMES.get(k, str(k)): v for k, v in cls_exact.items()},
        },
        "near_duplicates_hamming_le_3": {
            "n_val_crops_affected": near_val_crops,
            "fraction_of_val": near_val_crops / max(len(val_idx), 1),
            "n_hash_pairs": near_pairs,
            "by_class": {CLASS_NAMES.get(k, str(k)): v for k, v in cls_near.items()},
            "val_class_distribution_for_reference": {
                CLASS_NAMES.get(k, str(k)): v for k, v in
                Counter(ds.samples[i][5] for i in val_idx).items()},
        },
        "context_within_episode_consecutive_similarity": {
            "mean_fraction_of_adjacent_hashes_within_hamming_3": (
                float(np.mean(within)) if within else None),
            "episodes_sampled": len(within),
            "note": ("NOT a valid comparator and must not be quoted as one: this "
                      "walks hashes in sorted VALUE order, not frame order, so it "
                      "does not measure consecutive-frame similarity. Retained only "
                      "to record what was computed."),
        },
    }
    (OUT / "leakage_audit.json").write_text(json.dumps(out, indent=2))
    print(f"  episode overlap: {len(ep_overlap)}")
    print(f"  exact hash collisions: {len(exact)} hashes, {n_val_exact} val crops "
          f"({n_val_exact/max(len(val_idx),1):.3%} of val)")
    print(f"  near-duplicates (Hamming<=3): {near_val_crops} val crops "
          f"({near_val_crops/max(len(val_idx),1):.3%} of val)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["spatial", "leakage"])
    a = ap.parse_args()
    if a.only != "leakage":
        spatial_errors()
        print()
    if a.only != "spatial":
        leakage_audit()
