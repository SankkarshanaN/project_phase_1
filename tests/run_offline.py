"""Runs every offline test in this directory (no CARLA server needed) and
reports a single pass/fail summary.

These were written during development to pin down specific defects --
`test_lateral_sign.py` and `test_weather.py` are excluded here because they
need a live CARLA server; run them individually against one:

    python tests/test_lateral_sign.py
    python tests/test_weather.py

Each test file is a standalone script (no pytest, no fixtures) that prints
its own PASS/FAIL lines and exits non-zero on failure -- this runner just
invokes each as a subprocess and aggregates the exit codes, so a single
broken test cannot take down the rest of the suite.

Usage, from the project root:

    python tests/run_offline.py
"""
import subprocess
import sys
from pathlib import Path

NEEDS_CARLA = {"test_lateral_sign.py", "test_weather.py"}


def main() -> int:
    test_dir = Path(__file__).resolve().parent
    tests = sorted(p for p in test_dir.glob("test_*.py") if p.name not in NEEDS_CARLA)
    if not tests:
        print("No offline tests found.")
        return 1

    print(f"Running {len(tests)} offline tests ({', '.join(sorted(NEEDS_CARLA))} "
          f"need a live CARLA server -- run those separately)\n")

    failures = []
    for t in tests:
        print(f"--- {t.name} " + "-" * max(1, 60 - len(t.name)))
        result = subprocess.run([sys.executable, str(t)], cwd=str(test_dir.parent))
        if result.returncode != 0:
            failures.append(t.name)
        print()

    print("=" * 60)
    if failures:
        print(f"FAILED: {', '.join(failures)} ({len(failures)}/{len(tests)})")
        return 1
    print(f"ALL PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
