"""Scenario B -- Blind-Spot Vehicle Cut-In (Town10HD_Opt -- CARLA 0.10.0 ships
no other town, so this runs on ordinary city streets, not a highway).

Ego cruises at a fixed 35 km/h (Town10's streets are tighter than a highway,
so the originally-planned 60 was reduced -- see configs/scenarios.yaml); a
second vehicle spawns in the rear-lateral blind spot (adjacent lane, behind,
6 m back) at 60 km/h and merges into the ego's lane once it closes to the
configured trigger distance -- producing a track with zero prior camera
history at the moment of the cut-in. Both vehicles are driven kinematically
(see carla_tools.scenario_gen.step_blindspot) rather than via autopilot/
Traffic Manager, which proved unreliable for holding a precise speed.

Standalone run: starts its own connection, runs a few episodes, and exits.
Also used as a library by scripts/collect_dataset.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.client import connect, disconnect
from carla_tools.data_collector import record_episode
from carla_tools.scenario_gen import destroy_actors, spawn_blindspot_scenario, step_blindspot
from common.config import load_yaml

SCENARIO_TYPE = "blindspot_cutin"


def run_episode(client, world, bev_cfg, scenarios_cfg, seed: int, out_dir: str, max_steps: int,
                 tm_port: int = 8000, weather_name: str | None = None) -> int:
    actors = spawn_blindspot_scenario(client, world, seed, scenarios_cfg, tm_port)
    dt = world.get_settings().fixed_delta_seconds or 0.1

    def on_tick(step, frame):
        step_blindspot(world, actors, dt)

    try:
        return record_episode(out_dir, world, actors.ego, bev_cfg, SCENARIO_TYPE, seed, max_steps,
                               weather_name=weather_name, on_tick=on_tick)
    finally:
        destroy_actors(actors.ego, actors.cutin_vehicle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--out-dir", default="data/raw_v2")
    args = parser.parse_args()

    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")
    scenarios_cfg = load_yaml("scenarios.yaml")

    client, world = connect(town_cfg, town_override=scenarios_cfg["scenario_types"][SCENARIO_TYPE]["town"])
    total = 0
    try:
        for ep in range(args.episodes):
            n = run_episode(client, world, bev_cfg, scenarios_cfg, ep, args.out_dir, args.max_steps)
            total += n
            print(f"[{SCENARIO_TYPE}] episode {ep}: {n} frames (total {total})")
    finally:
        disconnect(client, world)
    print(f"Done. {total} frames written to {args.out_dir}")


if __name__ == "__main__":
    main()
