"""Scenario E -- Urban Occluded Crossing (roaming ego).

The ego drives Town10HD_Opt under Traffic Manager autopilot through live
background traffic. Occlusion encounters are staged ahead of it as it goes
(`carla_tools.urban_world.EncounterManager`): a large vehicle is parked at the
kerb and a pedestrian placed genuinely inside its shadow, then triggered to
cross as the ego closes.

How this differs from Scenarios A and C, and why it exists
----------------------------------------------------------
Those park the ego and build a vignette in front of it. This drives. Three
things follow:

  - The crossing question becomes a real collision-risk question. A stationary
    ego has an undefined time-to-arrival, so "will they cross" degenerates into
    "will they enter this box". Here there is a closing speed and a deadline.
  - Scene diversity comes free, which is the direct fix for the previous
    dataset's worst structural problem: ~119 distinct scenes sampled ~70 times
    each, near-duplicate frames that leaked across every train/validation
    split.
  - Occlusions are encountered rather than posed, which is the honest setting
    for a claim about general validity.

An episode is a continuous drive of `max_steps` ticks, not a short vignette, so
`--max-steps` should be considerably larger here than for A/C.

Standalone run starts its own connection; also used as a library by
scripts/collect_dataset.py.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.client import connect, disconnect
from carla_tools.data_collector import record_episode
from carla_tools.urban_world import (
    EncounterManager, spawn_background_traffic, spawn_ego_with_autopilot,
)
from common.config import load_yaml

SCENARIO_TYPE = "urban_crossing"


def run_episode(client, world, bev_cfg, scenarios_cfg, seed: int, out_dir: str, max_steps: int,
                 tm_port: int = 8000, weather_name: str | None = None) -> int:
    rng = random.Random(scenarios_cfg["episode"]["seed_base"] + 3000 + seed)
    dt = world.get_settings().fixed_delta_seconds or 0.1

    tm = client.get_trafficmanager(tm_port)
    ego = spawn_ego_with_autopilot(world, rng, tm_port, tm=tm)
    traffic = ([], [], [])
    manager = None

    try:
        traffic = spawn_background_traffic(client, world, rng, tm_port=tm_port)
        manager = EncounterManager(world, ego, rng, scenarios_cfg, bev_cfg)

        def on_tick(step, frame):
            manager.step(dt)

        def annotate():
            enc_id, phase = manager.current_phase()
            return {"encounter_id": enc_id, "encounter_phase": phase}

        frames = record_episode(out_dir, world, ego, bev_cfg, SCENARIO_TYPE, seed, max_steps,
                                 weather_name=weather_name, on_tick=on_tick, annotate=annotate)
        print(f"  encounters staged: {manager.completed + len(manager.active)}", flush=True)
        return frames
    finally:
        if manager is not None:
            manager.destroy_all()
        _cleanup(ego, traffic)


def _cleanup(ego, traffic):
    """Controllers must be stopped before their walkers are destroyed, or CARLA
    leaves orphaned AI controllers driving nothing."""
    vehicles, walkers, controllers = traffic
    for c in controllers:
        if c is not None and c.is_alive:
            c.stop()
            c.destroy()
    for a in list(walkers) + list(vehicles) + [ego]:
        if a is not None and a.is_alive:
            a.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=400,
                         help="Ticks per drive. Long, unlike the staged scenarios -- "
                              "encounters are staged every ~40 m of travel.")
    parser.add_argument("--out-dir", default="data/raw_v2")
    args = parser.parse_args()

    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")
    scenarios_cfg = load_yaml("scenarios.yaml")

    client, world = connect(town_cfg)
    total = 0
    try:
        for ep in range(args.episodes):
            n = run_episode(client, world, bev_cfg, scenarios_cfg, ep,
                            args.out_dir, args.max_steps)
            total += n
            print(f"episode {ep}: +{n} frames (total {total})", flush=True)
    finally:
        disconnect(client, world)
    print(f"Done. {total} frames written to {args.out_dir}.")


if __name__ == "__main__":
    main()
