"""Phase 2 orchestrator: runs all 4 scenarios end-to-end to build the pilot
dataset (~1,000-1,500 frames, split roughly evenly across A/B/C/D).

CARLA 0.10.0 only ships Town10HD_Opt, so all scenarios (including D's weather
variants) share a single connection -- no town reloads needed, which also
sidesteps the level-reload-triggered instability seen on 0.9.16.

Run with the CARLA server already started, e.g.:
    E:\\Carla-0.10.0\\Carla-0.10.0-Win64-Shipping\\CarlaUnreal.exe -quality-level=Low -ResX=800 -ResY=600
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import carla

from carla_tools.client import connect, disconnect
from common.config import load_yaml
from common.wallclock import WallclockBudget
from scenarios import scenario_a_occluded_pedestrian as scen_a
from scenarios import scenario_b_blindspot_cutin as scen_b
from scenarios import scenario_c_multi_occlusion as scen_c
from scenarios import scenario_e_urban_crossing as scen_e
from scenarios.scenario_d_adverse_weather import PRESET_NAME, WeatherNotApplied, _apply_weather


MAX_CONSECUTIVE_EPISODE_FAILURES = 5


def _run_quota(run_episode_fn, client, world, bev_cfg, scenarios_cfg, target_frames: int,
                out_dir: str, max_steps: int, budget: WallclockBudget, weather_name=None,
                seed_offset: int = 0) -> int:
    """A single dropped sensor frame (transient CARLA stall -- observed in
    practice on a long unattended run) used to crash the whole multi-thousand
    -frame collection and lose everything after it. Each episode is now
    isolated: a failure is logged and skipped (that episode's partial
    frames, if any, stay on disk -- harmless, just slightly fewer frames for
    that seed) rather than taking down the run. `MAX_CONSECUTIVE_EPISODE_FAILURES`
    still aborts if something is persistently broken (e.g. the server actually
    died) rather than spinning forever."""
    total = 0
    ep = 0
    consecutive_failures = 0
    while total < target_frames and not budget.expired():
        try:
            n = run_episode_fn(client, world, bev_cfg, scenarios_cfg, seed_offset + ep,
                                out_dir, max_steps, weather_name=weather_name)
            total += n
            ep += 1
            consecutive_failures = 0
            print(f"  episode {ep}: +{n} frames (total {total}/{target_frames})", flush=True)
        except Exception as e:
            consecutive_failures += 1
            ep += 1
            print(f"  episode {ep}: FAILED ({type(e).__name__}: {e}) -- skipping "
                  f"({consecutive_failures}/{MAX_CONSECUTIVE_EPISODE_FAILURES} consecutive)", flush=True)
            if consecutive_failures >= MAX_CONSECUTIVE_EPISODE_FAILURES:
                print("  too many consecutive failures, aborting this scenario's quota.", flush=True)
                break
    return total


# Share of the clear-weather budget per scenario. Scenario E (roaming ego) gets
# the largest share: it is the only one where the ego drives, so it supplies
# nearly all of the dataset's scene diversity. The v1 dataset's worst structural
# flaw was ~119 distinct scenes sampled ~70 times each, and a parked-ego
# scenario cannot fix that however many frames it contributes.
CLEAR_SPLIT = {"e": 0.40, "a": 0.20, "c": 0.20, "b": 0.20}

# Fraction of the total budget spent on adverse weather. Below the 1:1 the v1
# collection nominally targeted, because adverse frames are a robustness check
# rather than the primary training set.
ADVERSE_SHARE = 0.25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-frames", type=int, default=15000,
                         help="Total frames, split across scenarios per CLEAR_SPLIT.")
    parser.add_argument("--max-steps", type=int, default=65,
                         help="Ticks per staged episode (A/B/C). Scenario E uses "
                              "--urban-max-steps, since a drive is far longer.")
    parser.add_argument("--urban-max-steps", type=int, default=400)
    parser.add_argument("--out-dir", default="data/raw_v2")
    parser.add_argument("--max-wallclock-seconds", type=float, default=14400.0)
    parser.add_argument("--pilot", action="store_true",
                         help="Small gated run: ~300 frames across every scenario and "
                              "both weathers, to be checked with scripts/check_dataset.py "
                              "BEFORE committing hours to a full collection.")
    parser.add_argument("--seed-offset", type=int, default=0,
                         help="Added to every episode seed. A long collection must be run "
                              "in chunks with a CARLA restart between them -- the server "
                              "degrades badly over uptime (measured: 9.8 FPS falling to "
                              "0.17 after ~20 min and many spawn/destroy cycles, restored "
                              "by a restart). Give each chunk a distinct offset or the "
                              "later ones overwrite the earlier ones' frames.")
    parser.add_argument("--skip-adverse", action="store_true",
                         help="Clear weather only. Use if the weather preset cannot be "
                              "applied on this server rather than writing mislabelled frames.")
    args = parser.parse_args()

    if args.pilot:
        args.target_frames = min(args.target_frames, 300)
        args.max_wallclock_seconds = min(args.max_wallclock_seconds, 1800.0)
        args.urban_max_steps = min(args.urban_max_steps, 150)
        print("PILOT RUN -- collect, then gate with:\n"
              f"  python scripts/check_dataset.py --data-dir {args.out_dir}\n"
              "Do not start a full collection until it passes.\n", flush=True)

    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")
    scenarios_cfg = load_yaml("scenarios.yaml")
    weather_preset = scenarios_cfg["weather_presets"][PRESET_NAME]

    adverse_total = 0 if args.skip_adverse else int(args.target_frames * ADVERSE_SHARE)
    clear_total = args.target_frames - adverse_total
    budget = WallclockBudget(args.max_wallclock_seconds)
    grand_total = 0

    runners = {"e": (scen_e, args.urban_max_steps), "a": (scen_a, args.max_steps),
               "c": (scen_c, args.max_steps), "b": (scen_b, args.max_steps)}

    client, world = connect(town_cfg, town_override="Town10HD_Opt")
    try:
        for key, share in CLEAR_SPLIT.items():
            module, steps = runners[key]
            target = int(clear_total * share)
            if target <= 0:
                continue
            print(f"=== Clear weather: {module.SCENARIO_TYPE} "
                  f"(target {target} frames) ===", flush=True)
            grand_total += _run_quota(module.run_episode, client, world, bev_cfg,
                                       scenarios_cfg, target, args.out_dir, steps, budget,
                                       seed_offset=args.seed_offset)

        if adverse_total > 0:
            print(f"=== Adverse weather ({PRESET_NAME}) ===", flush=True)
            try:
                # Verifies the server actually applied it, and raises if not --
                # the v1 collection wrote 2,160 frames tagged as heavy rain at
                # night that are bright clear-day images, because nothing checked.
                _apply_weather(world, weather_preset)
            except WeatherNotApplied as exc:
                print(f"\n  {exc}\n  Skipping adverse-weather collection rather than "
                      f"writing mislabelled frames.\n", flush=True)
                adverse_total = 0

            if adverse_total > 0:
                per_sub = adverse_total // len(CLEAR_SPLIT)
                for key in CLEAR_SPLIT:
                    module, steps = runners[key]
                    print(f"--- {module.SCENARIO_TYPE} under {PRESET_NAME} ---", flush=True)
                    grand_total += _run_quota(module.run_episode, client, world, bev_cfg,
                                               scenarios_cfg, per_sub, args.out_dir, steps,
                                               budget, weather_name=PRESET_NAME,
                                               seed_offset=args.seed_offset)
                world.set_weather(carla.WeatherParameters.ClearNoon)
    finally:
        disconnect(client, world)

    print(f"\nDone. {grand_total} frames written to {args.out_dir} in {budget.elapsed():.0f}s.")
    print(f"\nNext: python scripts/check_dataset.py --data-dir {args.out_dir}")


if __name__ == "__main__":
    main()
