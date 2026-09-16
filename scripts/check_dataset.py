"""Gate a collected dataset before committing to a long run.

Run this on a small pilot collection. Every check here corresponds to a defect
that actually shipped in the previous 8,280-frame dataset and was only found
afterwards, by which point the collection hours were spent:

  - 2,160 frames tagged `heavy_rain_night_fog` were bright clear-day images,
    because nothing ever verified the weather preset took.
  - 85% of blind-spot frames had empty label files, because the cut-in vehicle
    was behind the camera for the first 40 of 60 recorded ticks.
  - No frame anywhere contained a fully-occluded actor, because the recorder
    discarded them and the scenarios never produced them in the first place.
  - Visibility was quantised to eighths, too coarse to grade occlusion.

Exits non-zero if any check fails, so it can gate a collection script.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.occlusion_tiers import OCCLUDED, PARTIAL, TIER_NAMES, VISIBLE

ADVERSE_MAX_BRIGHTNESS = 80.0

# Share of urban ticks the ego may spend stationary. Red lights are legitimate;
# 79% was a bus parked in the ego's own lane. Measured after the fix: 15%.
MAX_STATIONARY_SHARE = 0.35


class Report:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok

    def warn(self, ok: bool, name: str, detail: str = "") -> None:
        if not ok:
            print(f"  WARN  {name}" + (f"  [{detail}]" if detail else ""))
            self.warnings.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw_v2")
    ap.add_argument("--skip-brightness", action="store_true",
                     help="Skip the adverse-weather pixel check (needs images/ and cv2).")
    args = ap.parse_args()

    data = Path(args.data_dir)
    meta_dir, label_dir, image_dir = data / "meta", data / "labels", data / "images"
    frames = sorted(meta_dir.glob("*.npz"))
    if not frames:
        raise SystemExit(f"No frames under {meta_dir}")

    rep = Report()
    print(f"Checking {len(frames)} frames in {data}\n")

    vis_all, tier_counts = [], Counter()
    by_group, by_episode = defaultdict(list), defaultdict(list)
    missing_record = 0

    # Copy out the few fields the checks need and let the file close.
    #
    # `np.load` on an .npz returns a lazy NpzFile holding an open handle, and
    # retaining one per frame exhausts the process file-handle limit -- this
    # died with OSError 24 partway through a 15,420-frame dataset. The
    # per-object arrays are small; the grids (occ_grid, true_occ_grid,
    # radar_pts) are not, and are deliberately not kept.
    KEEP = ("obj_tier", "obj_visibility", "obj_is_vru", "obj_actor_id",
            "ego_vx", "ego_vy", "scenario_type", "weather_name")
    for p in frames:
        with np.load(p, allow_pickle=True) as z:
            if "obj_tier" not in z.files:
                missing_record += 1
                continue
            d = {k: z[k] for k in KEEP if k in z.files}
        group = f"{d['scenario_type']}_{d['weather_name']}"
        by_group[group].append(p)
        by_episode[p.stem.rsplit("_f", 1)[0]].append(d)
        vis_all.extend(d["obj_visibility"].tolist())
        tier_counts.update(d["obj_tier"].tolist())

    print("1. Per-object record present")
    rep.check(missing_record == 0, "every frame carries the per-object arrays",
              f"{missing_record} without")

    print("\n2. All three occlusion tiers are populated")
    total_obj = sum(tier_counts.values())
    for t in (OCCLUDED, PARTIAL, VISIBLE):
        n = tier_counts.get(t, 0)
        share = n / total_obj if total_obj else 0.0
        print(f"     {TIER_NAMES[t]:<9} {n:7d}  {share:6.1%}")
    rep.check(tier_counts.get(OCCLUDED, 0) > 0,
              "OCCLUDED tier is non-empty (the whole point of the rework)")
    rep.check(tier_counts.get(PARTIAL, 0) > 0, "PARTIAL tier is non-empty")
    rep.check(tier_counts.get(VISIBLE, 0) > 0, "VISIBLE tier is non-empty")

    print("\n3. Visibility is finely graded, not quantised")
    # Only the PARTIAL band is tested. Fully-hidden (0.0) and fully-visible
    # (1.0) actors dominate any real dataset and both sit exactly on the old
    # k/8 lattice, so including them would make this check fail on correct data
    # while telling us nothing: quantisation only matters where the value is
    # actually grading something.
    vis = np.asarray(vis_all, dtype=float)
    mid = vis[(vis > 0.01) & (vis < 0.99)]
    eighths = {round(i / 8, 4) for i in range(9)}
    distinct = len({round(float(v), 3) for v in mid})
    print(f"     {len(mid)} partially-visible observations of {len(vis)} total")
    rep.check(len(mid) > 0, "the dataset contains partially-visible actors at all")
    if len(mid):
        off = sum(1 for v in mid if round(float(v), 4) not in eighths)
        rep.check(distinct > 15, "partial visibilities take many distinct values",
                  f"{distinct} distinct")
        rep.check(off > 0.5 * len(mid),
                  "partial values mostly land off the old 8-corner k/8 lattice",
                  f"{off}/{len(mid)} off-lattice")

    print("\n4. Frames are not empty of actors entirely")
    # Deliberately counts frames whose per-object record is EMPTY -- nothing in
    # the scene at all -- rather than frames with no YOLO label. A frame whose
    # actors are all occluded has an empty label file BY DESIGN and is exactly
    # the data this rework exists to collect. The defect being screened for is
    # the blind-spot one: the scenario's only actor being behind the camera, so
    # the frame shows bare road and teaches nothing.
    for group in sorted(by_group):
        paths = by_group[group]
        n_empty = 0
        for p in paths:
            with np.load(p, allow_pickle=True) as d:
                if len(d["obj_actor_id"]) == 0:
                    n_empty += 1
        share = n_empty / len(paths)
        ok = share <= 0.20
        print(f"     {'PASS' if ok else 'FAIL'}  {group:<44} {n_empty}/{len(paths)} "
              f"with no actors at all ({share:.0%})")
        if not ok:
            rep.failures.append(f"{group}: {share:.0%} of frames contain no actors")

    print("\n4b. Label files reflect the tiers")
    if label_dir.is_dir():
        occluded_only = labelled = 0
        for p in frames[:: max(1, len(frames) // 200)]:
            with np.load(p, allow_pickle=True) as d:
                if "obj_tier" not in d.files or len(d["obj_tier"]) == 0:
                    continue
                all_occluded = bool((d["obj_tier"] == OCCLUDED).all())
            txt = (label_dir / f"{p.stem}.txt")
            has_label = txt.exists() and bool(txt.read_text().strip())
            if all_occluded and not has_label:
                occluded_only += 1
            elif has_label:
                labelled += 1
        print(f"     {labelled} sampled frames carry labels, {occluded_only} are "
              f"correctly label-free because every actor was hidden")

    print("\n5. Adverse-weather frames actually look like adverse weather")
    adverse = [g for g in by_group if "clear" not in g]
    if not adverse:
        print("     (no adverse-weather frames in this sample)")
    elif args.skip_brightness or not image_dir.is_dir():
        print("     (skipped)")
    else:
        import cv2
        for group in adverse:
            sample = by_group[group][:: max(1, len(by_group[group]) // 30)]
            means = []
            for p in sample:
                img = cv2.imread(str(image_dir / f"{p.stem}.jpg"))
                if img is not None:
                    means.append(float(img.mean()))
            if not means:
                continue
            mean = float(np.mean(means))
            rep.check(mean <= ADVERSE_MAX_BRIGHTNESS,
                      f"{group} is dark enough",
                      f"mean brightness {mean:.1f}, need <= {ADVERSE_MAX_BRIGHTNESS}")

    print("\n6. Crossing episodes sweep the tiers")
    swept = 0
    for ep, ds in by_episode.items():
        tiers = set()
        for d in ds:
            vru = d["obj_is_vru"]
            if vru.any():
                tiers.update(d["obj_tier"][vru].tolist())
        if {OCCLUDED, VISIBLE} <= tiers:
            swept += 1
    share = swept / len(by_episode) if by_episode else 0.0
    print(f"     {swept}/{len(by_episode)} episodes contain both a hidden and a visible VRU frame")
    rep.warn(share >= 0.3, "at least 30% of episodes sweep OCCLUDED -> VISIBLE",
             f"{share:.0%}")

    print("\n7. The roaming ego actually roamed")
    # The first v2 collection wrote 3,713 well-formed urban frames in which the
    # ego was stationary 79% of the time, blocked by its own staged occluder.
    # Nothing here caught it: every frame had valid labels, valid grids and a
    # plausible tier mix. Only ego motion tells a drive apart from a very long
    # look at a parked bus.
    urban = [d for ep, ds in by_episode.items() if ep.startswith("urban_crossing")
             for d in ds]
    if not urban:
        print("     no urban_crossing frames sampled -- skipped")
    else:
        speeds = np.array([float(np.hypot(d["ego_vx"], d["ego_vy"])) for d in urban])
        stopped = float((speeds < 0.1).mean())
        print(f"     {len(urban)} urban frames, mean {speeds.mean():.2f} m/s, "
              f"stationary {stopped:.0%} of ticks")
        rep.check(stopped <= MAX_STATIONARY_SHARE,
                  "urban ego is not stalled",
                  f"stationary {stopped:.0%}, need <= {MAX_STATIONARY_SHARE:.0%}")
        rep.warn(speeds.mean() >= 3.0, "urban ego averages a city speed",
                 f"{speeds.mean():.2f} m/s")

    print("\n8. Scene diversity")
    print(f"     {len(by_episode)} episodes, {len(frames)} frames "
          f"({len(frames) / max(len(by_episode), 1):.0f} frames/episode)")
    rep.warn(len(by_episode) >= 10, "at least 10 distinct episodes in the sample",
             f"{len(by_episode)}")

    print()
    if rep.failures:
        print(f"FAILED ({len(rep.failures)}):")
        for f in rep.failures:
            print(f"  - {f}")
        print("\nDo not start a full collection until these pass.")
        return 1
    if rep.warnings:
        print(f"Passed with {len(rep.warnings)} warning(s).")
    else:
        print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
