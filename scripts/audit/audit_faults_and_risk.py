"""AUDIT PHASE 9+10 -- parse the fault-injection run into machine-readable form,
and determine whether the risk engine can be evaluated at all.

Fault injection: `scripts/adversarial_test.py` prints a table and persists
nothing, so its stdout is captured to results/audit/logs/fault_injection.log
and parsed here. Nothing is retyped by hand.

Risk: the risk engine (`perception.risk`) produces a score and an action, but
scoring it requires a ground-truth notion of "was this frame actually
dangerous", and the dataset defines no such label. This script checks for one
rather than assuming, and records a SKIP with the reason if absent.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.npz_io import load_npz

DATA = Path("data/raw_v2")
OUT = Path("results/audit")
LOG = OUT / "logs" / "fault_injection.log"

ROW = re.compile(
    r"^(?P<cond>[a-z_]+)\s+(?P<dets>\d+)\s+(?P<pos>[\d.]+)\s+(?P<vlat>[\d.]+)\s+"
    r"(?P<hid>[\d.nan]+)\s+(?P<auc>[\d.nan]+)\s+(?P<cam>[\d.]+)\s+"
    r"(?P<rad>[\d.>\-]+)\s+(?P<noticed>YES|NO|--)\s*$")


def parse_faults():
    if not LOG.exists():
        return {"status": "SKIPPED", "reason": f"{LOG} not present"}
    rows, degradation, silent = {}, {}, []
    for line in LOG.read_text(errors="replace").splitlines():
        m = ROW.match(line.strip())
        if m:
            g = m.groupdict()
            def f(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            rows[g["cond"]] = {
                "n_detections": int(g["dets"]),
                "pos_rmse_m": f(g["pos"]),
                "v_lat_mae_ms": f(g["vlat"]),
                "hidden_rmse_m": f(g["hid"]),
                "intent_auc": f(g["auc"]),
                "camera_health": f(g["cam"]),
                "radar_health_raw": g["rad"],
                "detected_by_health_monitor": (None if g["noticed"] == "--"
                                                else g["noticed"] == "YES"),
            }
        # The script formats these with an explicit sign, so the value itself
        # carries '+' or '-' after the literal "pos +" prefix, and the AUC delta
        # may be '+0.007', '-0.008' or '+nan'.
        dm = re.match(r"^\s*([a-z_]+)\s+pos \+([+-][\d.]+) m\s+"
                      r"AUC ([+-]?(?:[\d.]+|nan))\s+detections lost (\d+)", line)
        if dm:
            degradation[dm.group(1)] = {
                "delta_pos_rmse_m": float(dm.group(2)),
                "delta_intent_auc": (None if "nan" in dm.group(3) else float(dm.group(3))),
                "detections_lost": int(dm.group(4)),
            }
        sm = re.match(r"^\s*SILENT FAILURES:\s*(.+)$", line)
        if sm:
            silent = [s.strip() for s in sm.group(1).split(",") if s.strip()]

    faults = {k: v for k, v in rows.items() if k != "clean"}
    detected = [k for k, v in faults.items() if v["detected_by_health_monitor"]]
    undetected = [k for k, v in faults.items() if v["detected_by_health_monitor"] is False]
    return {
        "experiment_id": "AUDIT-H-FAULTS",
        "status": "EXECUTED",
        "source_log": str(LOG),
        "command": "scripts/adversarial_test.py --data-dir data/raw_v2 --max-episodes 24",
        "n_conditions_total": len(rows),
        "n_faults_excluding_clean": len(faults),
        "clean_baseline": rows.get("clean"),
        "per_condition": rows,
        "degradation_vs_clean": degradation,
        "detected_by_health_monitor": sorted(detected),
        "not_detected_by_health_monitor": sorted(undetected),
        "n_detected": len(detected),
        "n_not_detected": len(undetected),
        "silent_failures_reported_by_script": silent,
        "silent_failure_definition": (
            "the script counts a fault as a SILENT FAILURE only if it measurably "
            "degraded something AND health stayed >= 0.6; a fault that changed "
            "nothing measurable is not counted as survived"),
        "denominator_warning": (
            "The manuscript's '5 of 7 faults detected' uses a different denominator "
            "from this run, which injects 10 fault conditions (darkness, fog, blur, "
            "noise, lens_blocked, radar_dropout, radar_clutter, radar_bias, "
            "extrinsic_drift, radar_clutter_onset). Any claim of the form 'N of M' "
            "must state which M."),
    }


def check_risk():
    """Is there any ground truth against which a risk score could be scored?"""
    p = sorted((DATA / "meta").glob("*.npz"))[0]
    keys = sorted(load_npz(p).keys())
    risk_like = [k for k in keys
                 if any(t in k.lower() for t in ("risk", "ttc", "hazard", "danger",
                                                  "collision", "brake", "warn"))]
    extra = [f.name for f in DATA.iterdir() if f.is_file()]
    return {
        "experiment_id": "AUDIT-G-RISK",
        "status": "SKIPPED",
        "reason": (
            "No risk / TTC / hazard ground truth exists in the dataset. perception."
            "risk.assess() produces a score and an ADAS action, but scoring those "
            "requires a per-frame label for whether the situation was genuinely "
            "dangerous, and no such label is recorded. Deriving one (e.g. by "
            "thresholding distance or TTC) would be inventing the ground truth the "
            "metric is meant to test against, which this audit will not do."),
        "meta_keys_present": keys,
        "risk_like_keys_found": risk_like,
        "dataset_level_files": extra,
        "what_would_be_needed": [
            "a per-frame or per-event label of genuine hazard/no-hazard, or",
            "recorded time-to-collision ground truth, or",
            "logged collision/near-miss events with timestamps to score warnings against",
        ],
        "note": (
            "The collision sensor exists in the sensor rig (carla_tools.sensors), so "
            "such labels COULD be recorded in a future collection, but they are not "
            "present in data/raw_v2 and cannot be recovered from it after the fact."),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    faults = parse_faults()
    risk = check_risk()
    (OUT / "fault_metrics.json").write_text(json.dumps(faults, indent=2))
    (OUT / "risk_metrics.json").write_text(json.dumps(risk, indent=2))

    print("=== FAULT INJECTION ===")
    if faults.get("status") == "EXECUTED":
        print(f"conditions parsed: {faults['n_conditions_total']} "
              f"(faults: {faults['n_faults_excluding_clean']})")
        print(f"detected:     {faults['n_detected']}  {faults['detected_by_health_monitor']}")
        print(f"NOT detected: {faults['n_not_detected']}  "
              f"{faults['not_detected_by_health_monitor']}")
        print(f"script-flagged silent failures: {faults['silent_failures_reported_by_script']}")
    else:
        print(faults)
    print("\n=== RISK ENGINE ===")
    print(f"{risk['status']}: {risk['reason'][:150]}...")
    print(f"risk-like keys in meta: {risk['risk_like_keys_found']}")


if __name__ == "__main__":
    main()
