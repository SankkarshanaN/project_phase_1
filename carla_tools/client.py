"""Connect to an already-running CarlaUE4.exe server and configure synchronous mode.

CarlaUE4.exe must be launched separately, e.g.:
    E:\\Carla\\CarlaUE4.exe -quality-level=Low -ResX=800 -ResY=600

This module does not launch the server itself so that the same client code
works whether the server is windowed, off-screen, or running on another
machine.
"""
import carla

from common.config import load_yaml


def connect(town_cfg: dict | None = None, town_override: str | None = None) -> tuple[carla.Client, carla.World]:
    cfg = town_cfg or load_yaml("town.yaml")
    client = carla.Client(cfg["carla_host"], cfg["carla_port"])
    client.set_timeout(cfg["carla_timeout_s"])

    target_town = town_override or cfg["town"]
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != target_town:
        world = client.load_world(target_town)

    settings = world.get_settings()
    settings.synchronous_mode = cfg["synchronous_mode"]
    settings.fixed_delta_seconds = cfg["fixed_delta_seconds"]
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(cfg["synchronous_mode"])

    return client, world


def disconnect(client: carla.Client, world: carla.World) -> None:
    """Intentionally a no-op.

    Calling `world.apply_settings(...)` here to flip synchronous_mode back
    off reliably hard-crashes the process (STATUS_STACK_BUFFER_OVERRUN) once
    any Traffic-Manager-controlled actor (set_autopilot + a
    vehicle_percentage_speed_difference/auto_lane_change call, as in
    scenario_gen.spawn_blindspot_scenario) has been spawned and destroyed
    earlier in the session -- reproduced empirically on this machine's
    CARLA 0.9.16 server, the same failure signature a prior project
    (E:\\New OAPS\\carla_tools\\client.py) documented for 0.10.0. Scenarios
    with no Traffic-Manager actors (A, C) don't trigger it, but there's no
    reliable way to know in advance from here, so this is unconditionally a
    no-op. Harmless: every `connect()` call unconditionally re-applies
    synchronous_mode=True for its own session, so there's no correctness
    need to flip it off on the way out -- the server is left running in
    synchronous mode between script invocations, which is fine for this
    single-client pipeline.
    """
    return


def tick(world: carla.World) -> int:
    return world.tick()
