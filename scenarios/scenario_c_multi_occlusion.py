"""Scenario C -- Multi-Occlusion (Town10HD_Opt).

Two pedestrians converge on the same crossing: one in clear view, one hidden
behind a parked-car occluder. Tests whether the pipeline can track both --
maintaining the hidden one's ground-truth position while YOLO only ever gets
a bbox label for the visible one.

Standalone run: starts its own connection, runs a few episodes, and exits.
Also used as a library by scripts/collect_dataset.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.client import connect, disconnect
from carla_tools.data_collector import record_episode
from carla_tools.scenario_gen import destroy_actors, spawn_multi_occlusion_scenario
from common.config import load_yaml

SCENARIO_TYPE = "multi_occlusion"


def run_episode(client, world, bev_cfg, scenarios_cfg, seed: int, out_dir: str, max_steps: int,
                 tm_port: int = 8000, weather_name: str | None = None) -> int:
    actors = spawn_multi_occlusion_scenario(client, world, seed, scenarios_cfg, tm_port,
                                             bev_cfg=bev_cfg)
    dt = world.get_settings().fixed_delta_seconds or 0.1

    # Both walkers carry their own independently-sampled behaviour, so one may
    # cross while the other turns back -- stepped every tick.
    def on_tick(step, frame):
        for driver in actors.drivers:
            driver.step(dt)

    try:
        return record_episode(out_dir, world, actors.ego, bev_cfg, SCENARIO_TYPE, seed, max_steps,
                               weather_name=weather_name, on_tick=on_tick)
    finally:
        destroy_actors(actors.ego, actors.occluder, actors.visible_walker, actors.hidden_walker)


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
