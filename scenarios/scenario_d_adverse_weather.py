"""Scenario D -- Adverse Weather Stack.

STATUS: blocked by a simulator defect. CARLA 0.10.0's weather API does not
work on this build, and this was established by measurement, not inference:

  - `get_weather()` returns all-zero fields (sun_altitude_angle, fog_density,
    precipitation, cloudiness) no matter what has been set -- including
    immediately after applying CARLA's own built-in presets, and before
    anything has been set at all. The read-back is inoperative.
  - `set_weather()` does not change the render either. Measured mean frame
    brightness over an ALTERNATING A/B with a 40-tick settle, which rules out
    the scene-warm-up drift that a single sweep would confound:

        rep 1:  ClearNoon 180.5   HardRainNight 177.4   delta -3.0
        rep 2:  ClearNoon 176.6   HardRainNight 173.5   delta -3.1
        rep 3:  ClearNoon 170.8   HardRainNight 171.2   delta +0.3

    A genuine night scene is 100+ brightness points darker. These deltas are
    noise. A naive single sweep additionally showed ClearNight reading
    *brighter* than ClearNoon, which is the giveaway.

This is the root cause of the v1 dataset's 2,160 frames tagged
`heavy_rain_night_fog` that are bright clear-day images. The bug was never in
the collection code; the simulator silently ignored every request.

What to do instead
------------------
Adverse-condition robustness is evaluated by applying calibrated photometric
degradations to captured frames -- `scripts/adversarial_test.py`'s `darkness`,
`fog`, `blur` and `noise` faults -- rather than by asking the simulator to
render them. That is not merely a workaround: the degradation magnitude is then
known exactly and is reproducible, where a simulator-rendered condition is
whatever the renderer happened to produce. State it as a methodological choice
forced by a documented tool defect, and report the A/B above as the evidence.

`_apply_weather` is retained with its verification intact so that a future
CARLA build which fixes this is detected automatically: if the preset ever
starts applying, the assertion stops firing and collection proceeds.
"""
import argparse
import sys
from pathlib import Path

import carla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.client import connect, disconnect
from common.config import load_yaml
from scenarios import scenario_a_occluded_pedestrian as scen_a
from scenarios import scenario_b_blindspot_cutin as scen_b
from scenarios import scenario_c_multi_occlusion as scen_c

PRESET_NAME = "heavy_rain_night_fog"


class WeatherNotApplied(RuntimeError):
    """The server did not take the weather preset it was given."""


def _apply_weather(world: carla.World, preset: dict, verify: bool = True,
                    settle_ticks: int = 10) -> None:
    """Applies a weather preset and, by default, verifies the server took it.

    The verification is not paranoia. The previous 8,280-frame collection wrote
    2,160 frames tagged `heavy_rain_night_fog` that are, measurably, bright
    clear-day images: mean frame brightness 155-171 against 168-172 for the
    clear-day set, where a scene with `sun_altitude_angle = -30` should sit
    near 20-50. A quarter of the dataset was mislabelled and nothing noticed,
    because nothing ever checked. Failing loudly here costs one round trip and
    saves hours of unusable collection.
    """
    weather = carla.WeatherParameters(
        cloudiness=preset["cloudiness"],
        precipitation=preset["precipitation"],
        precipitation_deposits=preset["precipitation_deposits"],
        wind_intensity=preset["wind_intensity"],
        sun_altitude_angle=preset["sun_altitude_angle"],
        fog_density=preset["fog_density"],
        fog_distance=preset["fog_distance"],
        wetness=preset["wetness"],
    )
    world.set_weather(weather)

    if not verify:
        return

    # Weather changes are not instantaneous in UE5; let the sky and
    # post-processing settle before reading anything back.
    for _ in range(settle_ticks):
        world.tick()

    applied = world.get_weather()
    mismatches = []
    for field_name, wanted in (
        ("sun_altitude_angle", preset["sun_altitude_angle"]),
        ("fog_density", preset["fog_density"]),
        ("precipitation", preset["precipitation"]),
        ("cloudiness", preset["cloudiness"]),
    ):
        got = getattr(applied, field_name, None)
        if got is None or abs(float(got) - float(wanted)) > 1.0:
            mismatches.append(f"{field_name}: asked {wanted}, server reports {got}")

    if mismatches:
        raise WeatherNotApplied(
            "CARLA did not apply the weather preset:\n  " + "\n  ".join(mismatches)
            + "\nCollecting now would write frames tagged as adverse weather that are "
              "not. Fix the server/preset before continuing."
        )


def frame_is_dark_enough(rgb, max_mean_brightness: float = 80.0) -> bool:
    """Second, independent check: does the rendered image actually look like
    night?

    `get_weather()` reporting the right numbers only proves the server accepted
    them, not that the render changed -- which is exactly the failure mode that
    produced the mislabelled frames. This tests the pixels. Run it on the first
    recorded frame of an adverse-weather episode.
    """
    import numpy as np
    return float(np.asarray(rgb, dtype=np.float32).mean()) <= max_mean_brightness


def _run_town(town: str, sub_scenarios, episodes: int, max_steps: int, out_dir: str,
              town_cfg: dict, scenarios_cfg: dict, bev_cfg: dict, preset: dict) -> int:
    client, world = connect(town_cfg, town_override=town)
    _apply_weather(world, preset)
    total = 0
    try:
        for mod in sub_scenarios:
            for ep in range(episodes):
                n = mod.run_episode(client, world, bev_cfg, scenarios_cfg, ep, out_dir, max_steps,
                                     weather_name=PRESET_NAME)
                total += n
                print(f"[adverse_weather/{mod.SCENARIO_TYPE}] episode {ep}: {n} frames (total {total})")
    finally:
        world.set_weather(carla.WeatherParameters.ClearNoon)
        disconnect(client, world)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")
    scenarios_cfg = load_yaml("scenarios.yaml")
    preset = scenarios_cfg["weather_presets"][PRESET_NAME]

    # CARLA 0.10.0 only ships Town10HD_Opt, so all three sub-scenarios share
    # one connection/weather application instead of switching towns.
    total = 0
    total += _run_town("Town10HD_Opt", [scen_a, scen_c, scen_b], args.episodes, args.max_steps,
                        args.out_dir, town_cfg, scenarios_cfg, bev_cfg, preset)

    print(f"Done. {total} frames written to {args.out_dir} under '{PRESET_NAME}' weather.")


if __name__ == "__main__":
    main()
