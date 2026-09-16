"""Does set_weather() change the RENDER, even though get_weather() reads back
zeros on this build?

get_weather() returning all-zero for every preset proves the read-back is
broken; it does not prove the setter is. The pixels decide.
"""
import sys

import numpy as np

sys.path.insert(0, r"E:\Project Phase 1")

import carla

from carla_tools.client import connect
from carla_tools.sensors import rgb_to_array, spawn_sensor_rig
from common.config import load_yaml

town, bev = load_yaml("town.yaml"), load_yaml("bev.yaml")
client, world = connect(town)
bl = world.get_blueprint_library()
sp = world.get_map().get_spawn_points()

ego = None
for t in sp[:25]:
    ego = world.try_spawn_actor(bl.filter("vehicle.lincoln.mkz")[0], t)
    if ego:
        break
world.tick()
rig = spawn_sensor_rig(world, ego, bev)


def brightness(settle=15, n=5):
    for _ in range(settle):
        world.tick()
    vals = []
    for _ in range(n):
        f = world.tick()
        vals.append(float(rgb_to_array(rig.rgb_buf.get(f)).mean()))
    return float(np.mean(vals))


try:
    print(f"{'weather':30s} {'brightness':>11s}  {'read-back sun':>14s}")
    print(f"{'(as spawned)':30s} {brightness():11.1f}  {world.get_weather().sun_altitude_angle:14.1f}")

    for name in ("ClearNoon", "ClearNight", "HardRainNoon", "HardRainNight",
                  "MidRainyNight", "CloudyNight"):
        preset = getattr(carla.WeatherParameters, name, None)
        if preset is None:
            continue
        world.set_weather(preset)
        b = brightness()
        print(f"{name:30s} {b:11.1f}  {world.get_weather().sun_altitude_angle:14.1f}")

    wp = carla.WeatherParameters(
        cloudiness=100.0, precipitation=90.0, precipitation_deposits=90.0,
        wind_intensity=60.0, sun_altitude_angle=-30.0, fog_density=60.0,
        fog_distance=15.0, wetness=90.0)
    world.set_weather(wp)
    b = brightness()
    print(f"{'our heavy_rain_night_fog':30s} {b:11.1f}  {world.get_weather().sun_altitude_angle:14.1f}")

    world.set_weather(carla.WeatherParameters.ClearNoon)
    print(f"{'back to ClearNoon':30s} {brightness():11.1f}")
finally:
    rig.destroy()
    ego.destroy()
    world.tick()
