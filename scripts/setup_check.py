"""Phase 1 verification: connect to a running CarlaUE4.exe (0.9.16), load
Town10HD_Opt, spawn an ego vehicle, attach the full sensor rig (RGB + depth +
semseg + radar), tick for ~100 frames, and confirm frame-synced data with no
dropped-frame warnings while reporting achieved FPS.

Run with the CARLA server already started, e.g.:
    E:\\Carla\\CarlaUE4.exe -quality-level=Low -ResX=800 -ResY=600
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.client import connect, disconnect, tick
from carla_tools.sensors import (
    depth_to_meters,
    radar_to_array,
    rgb_to_array,
    semseg_to_labels,
    spawn_sensor_rig,
)
from common.config import load_yaml

N_FRAMES = 100


def main():
    town_cfg = load_yaml("town.yaml")
    bev_cfg = load_yaml("bev.yaml")

    print(f"Connecting to {town_cfg['carla_host']}:{town_cfg['carla_port']} ...")
    client, world = connect(town_cfg)
    print(f"Connected. Map: {world.get_map().name}")

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    ego_bp = bp_lib.filter("vehicle.lincoln.mkz")[0]  # CARLA 0.10.0 catalog has no Tesla
    ego = world.try_spawn_actor(ego_bp, spawn_points[0])
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle")
    print("Ego vehicle spawned.")

    rig = spawn_sensor_rig(world, ego, bev_cfg)
    print("Sensor rig attached: RGB, depth, semseg, radar, collision, lane-invasion.")

    dropped = 0
    radar_hits_total = 0
    t0 = time.time()
    try:
        for step in range(N_FRAMES):
            frame = tick(world)
            try:
                rgb_img = rig.rgb_buf.get(frame, timeout=5.0)
                depth_img = rig.depth_buf.get(frame, timeout=5.0)
                semseg_img = rig.semseg_buf.get(frame, timeout=5.0)
                radar_data = rig.radar_buf.get(frame, timeout=5.0)
            except Exception as e:
                dropped += 1
                print(f"  [frame {frame}] dropped: {e}")
                continue

            if rgb_img.frame != frame or depth_img.frame != frame or semseg_img.frame != frame:
                dropped += 1
                print(f"  [frame {frame}] frame-sync mismatch: rgb={rgb_img.frame} depth={depth_img.frame} semseg={semseg_img.frame}")

            if step == N_FRAMES - 1:
                rgb = rgb_to_array(rgb_img)
                depth_m = depth_to_meters(depth_img)
                sem = semseg_to_labels(semseg_img)
                radar_pts = radar_to_array(radar_data)
                print(f"  Sample shapes -> rgb={rgb.shape} depth={depth_m.shape} sem={sem.shape} radar_pts={radar_pts.shape}")

            radar_hits_total += len(radar_data)

        elapsed = time.time() - t0
        fps = N_FRAMES / elapsed
        print(f"\nTicked {N_FRAMES} frames in {elapsed:.2f}s -> {fps:.1f} FPS")
        print(f"Dropped/mismatched frames: {dropped}/{N_FRAMES}")
        print(f"Total radar detections across run: {radar_hits_total}")
        if dropped == 0:
            print("PASS: sensor rig is frame-synced with no drops.")
        else:
            print("WARNING: some frames were dropped or mismatched -- see log above.")
    finally:
        rig.destroy()
        ego.destroy()
        disconnect(client, world)
        print("Cleaned up actors and restored asynchronous mode.")


if __name__ == "__main__":
    main()
