"""Runs a long collection as a series of short chunks, restarting the CARLA
server between each.

Why this exists rather than one long run
----------------------------------------
The CARLA 0.10.0 server degrades severely over uptime, and the degradation is
not subtle. Measured on this machine, with the same scene and the same code:

    fresh server, no traffic              14.1 FPS
    fresh server, 30 vehicles + 15 walkers 9.8 FPS
    same server after ~20 min of
      spawn/destroy cycles                 0.17 FPS

That is an 80x slowdown, and it is not caused by the traffic -- at the degraded
point, zero traffic and full traffic both measured 0.17 FPS. It is the server
process itself. A restart restores full speed immediately. Leftover actors
accumulate too: a "clean" world after several killed runs held 128 actors.

Left unattended, a 15,000-frame run therefore starts at 10 FPS, decays toward
0.2, and takes many hours instead of ~25 minutes. Chunking with restarts keeps
every frame collected at full speed.

Each chunk gets a distinct `--seed-offset` so its episode filenames do not
collide with the previous chunk's.
"""
import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml

PYTHON = sys.executable


def carla_is_up(host: str, port: int, timeout: float = 2.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def stop_carla() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "CarlaUnreal.exe"],
                    capture_output=True, check=False)
    subprocess.run(["taskkill", "/F", "/IM", "CarlaUnreal-Win64-Shipping.exe"],
                    capture_output=True, check=False)
    # Generous settle. Killing a UE5 process and relaunching immediately leaves
    # the GPU still releasing resources, and the next launch then dies silently
    # during startup -- observed killing a whole collection run, since a failed
    # start used to be fatal.
    time.sleep(15)


def start_carla(town_cfg: dict, wait_s: float = 150.0, attempts: int = 3) -> bool:
    """Starts the server, retrying a failed launch rather than giving up.

    Returns True once the RPC port is answering. A launch failing occasionally
    is normal after repeated kill/restart cycles; treating the first failure as
    fatal threw away every remaining chunk of a long collection.
    """
    exe = Path(town_cfg["carla_root"]) / town_cfg["carla_exe"]
    if not exe.exists():
        raise SystemExit(f"CARLA executable not found at {exe}")
    host, port = town_cfg["carla_host"], town_cfg["carla_port"]

    for attempt in range(attempts):
        subprocess.Popen([str(exe), "-quality-level=Low", "-ResX=1280", "-ResY=720",
                           "-carla-server", f"-carla-rpc-port={port}"],
                          cwd=str(exe.parent))
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if carla_is_up(host, port):
                # The port opens before the world is actually loadable; give the
                # level time to finish streaming in or the first client call fails.
                time.sleep(20)
                return True
            time.sleep(3)
        print(f"  CARLA did not come up within {wait_s:.0f}s "
              f"(attempt {attempt + 1}/{attempts}) -- retrying", flush=True)
        stop_carla()
    return False


def frames_in(out_dir: Path) -> int:
    d = out_dir / "images"
    return len(list(d.glob("*.jpg"))) if d.is_dir() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-frames", type=int, default=15000)
    ap.add_argument("--chunk-frames", type=int, default=1200,
                     help="Frames per chunk. Sized so a chunk finishes well before the "
                          "server degrades -- a few minutes at ~10 FPS.")
    ap.add_argument("--out-dir", default="data/raw_v2")
    ap.add_argument("--chunk-timeout", type=float, default=900.0)
    ap.add_argument("--skip-adverse", action="store_true", default=True)
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--seed-base", type=int, default=0,
                     help="Starting seed block. Set this above every block a previous "
                          "run used when EXTENDING an existing dataset, or the new "
                          "chunks reuse episode ids and silently overwrite earlier "
                          "frames. Each chunk consumes one block of 1000.")
    args = ap.parse_args()

    town_cfg = load_yaml("town.yaml")
    out_dir = Path(args.out_dir)
    root = Path(__file__).resolve().parent.parent

    print(f"Target {args.target_frames} frames in chunks of {args.chunk_frames}, "
          f"restarting CARLA between each.\n")

    chunk = 0
    barren = 0            # consecutive chunks that produced nothing
    MAX_BARREN = 3
    while frames_in(out_dir) < args.target_frames and chunk < args.max_chunks:
        have = frames_in(out_dir)
        want = min(args.chunk_frames, args.target_frames - have)
        print(f"--- chunk {chunk}: have {have}, collecting {want} more ---", flush=True)

        stop_carla()
        if not start_carla(town_cfg):
            print("  server would not start for this chunk; skipping it", flush=True)
            barren += 1
            chunk += 1
            if barren >= MAX_BARREN:
                print(f"  {barren} chunks in a row produced nothing; stopping.", flush=True)
                break
            continue

        # -u so the child's progress reaches the log as it happens. Without it
        # Python block-buffers when stdout is a file rather than a terminal, and
        # a detached run looks hung for minutes at a time even while collecting
        # normally.
        cmd = [PYTHON, "-u", str(root / "scripts" / "collect_dataset.py"),
               "--target-frames", str(want),
               "--out-dir", str(out_dir),
               "--seed-offset", str(args.seed_base + chunk * 1000),
               "--max-wallclock-seconds", str(args.chunk_timeout)]
        if args.skip_adverse:
            cmd.append("--skip-adverse")

        t0 = time.time()
        try:
            subprocess.run(cmd, cwd=str(root), timeout=args.chunk_timeout + 300, check=False)
        except subprocess.TimeoutExpired:
            print("  chunk timed out -- restarting and continuing", flush=True)

        got = frames_in(out_dir) - have
        rate = got / max(time.time() - t0, 1e-6)
        print(f"--- chunk {chunk}: +{got} frames in {time.time() - t0:.0f}s "
              f"({rate:.2f} frames/s), total {frames_in(out_dir)} ---\n", flush=True)

        # A single empty chunk is usually a transient server fault, not a dead
        # run -- bail out only once several in a row fail, so one bad launch
        # cannot discard the thousands of frames still to collect.
        barren = barren + 1 if got == 0 else 0
        if barren >= MAX_BARREN:
            print(f"  {barren} chunks in a row produced nothing; stopping.", flush=True)
            break
        chunk += 1

    stop_carla()
    print(f"\nDone. {frames_in(out_dir)} frames in {out_dir}.")
    print(f"Next: {PYTHON} scripts/check_dataset.py --data-dir {out_dir}")


if __name__ == "__main__":
    main()
