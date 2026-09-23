"""AUDIT PHASE 2 -- dataset statistics and split validation.

Computes every denominator the results package needs, directly from
`data/raw_v2`, and validates the train/val split the evidential classifier
actually uses for episode-level leakage.

Writes results/audit/dataset_statistics.json and results/audit/splits.json.
Reads only; never modifies the dataset.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carla_tools.occlusion_tiers import TIER_NAMES
from common.config import load_yaml
from common.npz_io import load_npz

DATA = Path("data/raw_v2")
OUT = Path("results/audit")


def episode_of(stem: str) -> str:
    return stem.rsplit("_f", 1)[0]


def scenario_of(stem: str) -> str:
    return stem.split("_ep")[0]


def main():
    metas = sorted((DATA / "meta").glob("*.npz"))
    images = sorted((DATA / "images").glob("*.jpg"))
    labels = sorted((DATA / "labels").glob("*.txt"))
    if not metas:
        raise SystemExit(f"No frames under {DATA}")

    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")
    dt = float(town_cfg["fixed_delta_seconds"])

    by_episode = defaultdict(list)
    by_scenario = Counter()
    for p in metas:
        by_episode[episode_of(p.stem)].append(p)
        by_scenario[scenario_of(p.stem)] += 1

    # ---- per-object statistics, streamed (the whole dataset does not fit
    # comfortably in memory as loaded arrays).
    tier_counts = Counter()
    cls_counts = Counter()
    vru_obs = 0
    n_objects = 0
    visibility_vals = []
    radar_pts_per_frame = []
    actors_per_episode = defaultdict(set)
    frames_per_episode = {k: len(v) for k, v in by_episode.items()}

    for p in metas:
        d = load_npz(p)
        ep = episode_of(p.stem)
        n = len(d["obj_actor_id"])
        n_objects += n
        for j in range(n):
            tier_counts[int(d["obj_tier"][j])] += 1
            cls_counts[int(d["obj_class"][j])] += 1
            visibility_vals.append(float(d["obj_visibility"][j]))
            actors_per_episode[ep].add(int(d["obj_actor_id"][j]))
            if bool(d["obj_is_vru"][j]):
                vru_obs += 1
        rp = d["radar_pts"]
        radar_pts_per_frame.append(0 if rp is None else int(np.asarray(rp).shape[0]))

    visibility_vals = np.asarray(visibility_vals, dtype=float)
    radar_pts_per_frame = np.asarray(radar_pts_per_frame, dtype=float)

    # ---- crossing-intent labels, if present
    intent = {}
    lab_path = DATA / "intent_labels.npz"
    if lab_path.exists():
        lab = load_npz(lab_path)
        wc = np.asarray(lab["will_cross"], dtype=bool)
        tiers = np.asarray(lab["tier"], dtype=int)
        intent = {
            "labelled_observations": int(len(wc)),
            "crossing_events_positive_obs": int(wc.sum()),
            "non_crossing_obs": int((~wc).sum()),
            "positive_rate": float(wc.mean()),
            "episodes_with_labels": int(len({str(e) for e in lab["episode"]})),
            "distinct_actors": int(len({(str(lab["episode"][i]), int(lab["actor_id"][i]))
                                        for i in range(len(wc))})),
            "by_tier": {TIER_NAMES.get(int(t), str(int(t))): int((tiers == t).sum())
                        for t in sorted(set(tiers.tolist()))},
            "note": ("denominator is OBSERVATIONS (actor x frame), not episodes or "
                     "crossing events; one crossing spans many observations"),
        }

    episode_frame_counts = np.array(sorted(frames_per_episode.values()), dtype=float)

    stats = {
        "experiment_id": "AUDIT-DATASET",
        "data_dir": str(DATA),
        "counts": {
            "frames_meta": len(metas),
            "frames_images": len(images),
            "frames_labels": len(labels),
            "episodes": len(by_episode),
            "object_instances_total": n_objects,
            "vru_object_instances": vru_obs,
            "note_denominators": {
                "frame": "one simulation tick with a full sensor read",
                "object_instance": "one actor observed in one frame (actor x frame)",
                "episode": "one contiguous scenario run",
            },
        },
        "integrity": {
            "meta_image_label_counts_match": (len(metas) == len(images) == len(labels)),
            "frames_with_zero_radar_returns": int((radar_pts_per_frame == 0).sum()),
        },
        "scenarios": {k: int(v) for k, v in sorted(by_scenario.items())},
        "episode_duration_frames": {
            "mean": float(episode_frame_counts.mean()),
            "median": float(np.median(episode_frame_counts)),
            "min": int(episode_frame_counts.min()),
            "max": int(episode_frame_counts.max()),
            "seconds_mean": float(episode_frame_counts.mean() * dt),
        },
        "object_class_distribution": {
            {0: "vehicle", 1: "pedestrian"}.get(k, str(k)): int(v)
            for k, v in sorted(cls_counts.items())
        },
        "occlusion_tiers_object_instances": {
            TIER_NAMES.get(k, str(k)): int(v) for k, v in sorted(tier_counts.items())
        },
        "visibility_fraction": {
            "mean": float(visibility_vals.mean()),
            "median": float(np.median(visibility_vals)),
            "frac_exactly_zero": float((visibility_vals <= 0.0).mean()),
            "frac_exactly_one": float((visibility_vals >= 1.0).mean()),
            "distinct_values": int(len(np.unique(visibility_vals))),
        },
        "radar_returns_per_frame": {
            "mean": float(radar_pts_per_frame.mean()),
            "median": float(np.median(radar_pts_per_frame)),
            "p05": float(np.percentile(radar_pts_per_frame, 5)),
            "p95": float(np.percentile(radar_pts_per_frame, 95)),
        },
        "sensor_configuration": {
            "camera_resolution": [bev_cfg["camera"]["width"], bev_cfg["camera"]["height"]],
            "camera_fov_deg": bev_cfg["camera"]["fov"],
            "radar": bev_cfg["radar"],
            "bev_grid": bev_cfg["bev_grid"],
            "fixed_delta_seconds": dt,
            "sensor_rate_hz": 1.0 / dt,
            "note": ("all sensors are read in lock-step synchronous mode, so camera and "
                     "radar share one rate; this is SIMULATION rate, not pipeline FPS"),
        },
        "crossing_intent_labels": intent,
        "towns": {
            "town": town_cfg["town"],
            "note": "single town by design; CARLA 0.10.0 ships only Town10HD_Opt",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

    # ---------------------------------------------------------------- splits
    # Reproduce the classifier's actual split WITHOUT retraining, by calling
    # the same function the trainer uses, so the audit reports the real split
    # rather than a re-derivation of it.
    import torch  # noqa: F401  (train_evidential imports torch at module level)
    from models.train_evidential import CropDataset, episode_split

    ds = CropDataset(str(DATA))
    train_ds, val_ds, n_train_ep, n_val_ep = episode_split(ds, val_frac=0.15, seed=0)

    train_eps = {ds.episode_of(i) for i in train_ds.indices} if hasattr(train_ds, "indices") \
        else {ds.episode_of(i) for i in getattr(train_ds, "idx", [])}
    # AugmentedCrops wraps a base dataset + index list; recover indices robustly.
    if not train_eps:
        base_idx = getattr(train_ds, "indices", None) or getattr(train_ds, "index", None)
        train_eps = {ds.episode_of(i) for i in (base_idx or [])}
    val_eps = {ds.episode_of(i) for i in val_ds.indices}

    overlap = sorted(train_eps & val_eps)
    splits = {
        "experiment_id": "AUDIT-SPLITS",
        "method": "episode_split(val_frac=0.15, seed=0) -- grouped by EPISODE",
        "n_crops_total": len(ds),
        "n_train_episodes": int(n_train_ep),
        "n_val_episodes": int(n_val_ep),
        "n_val_crops": len(val_ds),
        "n_train_crops": len(ds) - len(val_ds),
        "val_episodes": sorted(val_eps),
        "episode_overlap_train_val": overlap,
        "leakage_detected": bool(overlap),
        "leakage_check": ("PASS -- no episode appears on both sides" if not overlap
                          else f"FAIL -- {len(overlap)} episodes on both sides"),
        "independent_test_set_exists": False,
        "test_set_note": (
            "There is NO third, untouched test split. The classifier reports "
            "validation accuracy on the 15% episode-disjoint validation split, which "
            "is also the split the best-epoch checkpoint is selected on. Model "
            "selection and reporting therefore share a split; see the audit report."),
    }
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2))

    print(json.dumps({"dataset": stats["counts"], "splits": {
        k: splits[k] for k in ("n_train_episodes", "n_val_episodes", "n_val_crops",
                                "leakage_check", "independent_test_set_exists")}}, indent=2))


if __name__ == "__main__":
    main()
