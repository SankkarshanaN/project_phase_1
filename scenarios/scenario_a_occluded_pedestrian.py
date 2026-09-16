"""Scenario A -- Occluded Pedestrian Crossing (Town10HD_Opt).

A bus/van occluder parks ahead of the ego on a crosswalk approach; a
pedestrian walks laterally at 1.2 m/s from behind it into the road. Ground
truth (carla_tools.true_occupancy) keeps the pedestrian's real position every
tick even while the RGB camera can't see them -- the point of the scenario is
that the track must continue through the occlusion.

Standalone run: starts its own connection, runs a few episodes, and exits.
Also used as a library by scripts/collect_dataset.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.client import connect, disconnect
from carla_tools.data_collector import record_episode
from carla_tools.scenario_gen import destroy_actors, spawn_occlusion_scenario
from common.config import load_yaml

SCENARIO_TYPE = "occluded_pedestrian_crossing"


def run_episode(client, world, bev_cfg, scenarios_cfg, seed: int, out_dir: str, max_steps: int,
                 tm_port: int = 8000, weather_name: str | None = None) -> int:
    actors = spawn_occlusion_scenario(client, world, SCENARIO_TYPE, seed, scenarios_cfg, tm_port,
                                       bev_cfg=bev_cfg)
    dt = world.get_settings().fixed_delta_seconds or 0.1

    # The walker's behaviour state machine has to be stepped every tick. It
    # previously received one velocity command at spawn and coasted for the
    # whole episode, which made every walker a constant-velocity actor.
    def on_tick(step, frame):
        for driver in actors.drivers:
            driver.step(dt)

    try:
        return record_episode(out_dir, world, actors.ego, bev_cfg, SCENARIO_TYPE, seed, max_steps,
                               weather_name=weather_name, on_tick=on_tick)
    finally:
        destroy_actors(actors.ego, actors.occluder, actors.walker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--out-dir", default="data/raw")
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
