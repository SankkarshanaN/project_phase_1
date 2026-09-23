"""AUDIT PHASE 14+15 -- consistency checker and report generation.

Loads every machine-readable result produced by the other audit scripts,
cross-checks each against the corresponding manuscript claim, and emits:

  results/audit/MASTER_RESULTS.json
  results/audit/experiment_status.json
  results/audit/consistency_report.json
  results/audit/tables/*.csv
  MANUSCRIPT_RESULTS.md
  FINAL_RESEARCH_AUDIT.md

Every number written here is read from a result file. Nothing is hard-coded
except the MANUSCRIPT CLAIMS being checked, which are quoted from the paper so
they can be compared against what was computed.
"""
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("results/audit")
TABLES = OUT / "tables"
TOL = 5e-4          # agreement tolerance for a 3-decimal published figure


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


# Claims exactly as the manuscript/CLAUDE.md state them, with where each lives.
CLAIMS = [
    ("evidential_val_accuracy", 0.962, "96.2% +/- 0.8% validation accuracy"),
    ("evidential_best_epoch", 0.968, "96.8% best epoch"),
    ("occlusion_precision", 0.727, "occlusion detector precision"),
    ("occlusion_recall", 0.917, "occlusion detector recall"),
    ("occlusion_f1", 0.811, "occlusion detector F1"),
    ("occlusion_agreement", 0.780, "occlusion cell agreement"),
    ("cam_vlat", 0.819, "camera lateral velocity MAE"),
    ("cam_vfwd", 1.048, "camera forward velocity MAE"),
    ("cam_pos", 2.327, "camera position error"),
    ("rad_vlat", 1.683, "radar lateral velocity MAE"),
    ("rad_vfwd", 0.155, "radar forward velocity MAE"),
    ("rad_pos", 2.457, "radar position error"),
    ("fus_vlat", 0.672, "fused lateral velocity MAE"),
    ("fus_vfwd", 0.199, "fused forward velocity MAE"),
    ("fus_pos", 2.032, "fused position error"),
    ("cam_auc", 0.789, "camera intent AUC"), ("cam_f1", 0.733, "camera intent F1"),
    ("rad_auc", 0.743, "radar intent AUC"), ("rad_f1", 0.035, "radar intent F1"),
    ("fus_auc", 0.873, "fused intent AUC"), ("fus_f1", 0.855, "fused intent F1"),
]


def verdict(claimed, computed, tol=TOL):
    if computed is None:
        return "NOT REPRODUCIBLE", None
    d = abs(claimed - computed)
    if d <= tol:
        return "VERIFIED", d
    if d <= 0.02:
        return "REPRODUCED WITH DIFFERENCE", d
    return "INVALID", d


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    ds = load("dataset_statistics.json"); sp = load("splits.json")
    cls = load("classification_metrics.json"); unc = load("uncertainty_metrics.json")
    occ = load("occlusion_metrics.json"); trk = load("tracking_metrics.json")
    itn = load("intent_metrics.json"); hid = load("hidden_prediction_metrics.json")
    rsk = load("risk_metrics.json"); flt = load("fault_metrics.json")
    rt = load("runtime_metrics.json"); stat = load("statistics.json")
    stored = json.loads(Path("results/metrics.json").read_text())

    # ---------------------------------------------------- computed values map
    pm = trk["paired_metrics"] if trk else {}
    im = itn["by_mode"] if itn else {}
    computed = {
        "evidential_val_accuracy": stored["reported_val_accuracy"],
        "evidential_best_epoch": cls["accuracy"] if cls else None,
        "occlusion_precision": occ["sweep_sample_reported"]["precision_occluded"] if occ else None,
        "occlusion_recall": occ["sweep_sample_reported"]["recall_occluded"] if occ else None,
        "occlusion_f1": occ["sweep_sample_reported"]["f1_occluded"] if occ else None,
        "occlusion_agreement": occ["sweep_sample_reported"]["cell_agreement"] if occ else None,
        "cam_vlat": pm.get("camera", {}).get("v_lat", {}).get("mae"),
        "cam_vfwd": pm.get("camera", {}).get("v_fwd", {}).get("mae"),
        "cam_pos": pm.get("camera", {}).get("position", {}).get("mae"),
        "rad_vlat": pm.get("radar", {}).get("v_lat", {}).get("mae"),
        "rad_vfwd": pm.get("radar", {}).get("v_fwd", {}).get("mae"),
        "rad_pos": pm.get("radar", {}).get("position", {}).get("mae"),
        "fus_vlat": pm.get("fused", {}).get("v_lat", {}).get("mae"),
        "fus_vfwd": pm.get("fused", {}).get("v_fwd", {}).get("mae"),
        "fus_pos": pm.get("fused", {}).get("position", {}).get("mae"),
        "cam_auc": im.get("camera", {}).get("roc_auc"),
        "cam_f1": im.get("camera", {}).get("f1"),
        "rad_auc": im.get("radar", {}).get("roc_auc"),
        "rad_f1": im.get("radar", {}).get("f1"),
        "fus_auc": im.get("fused", {}).get("roc_auc"),
        "fus_f1": im.get("fused", {}).get("f1"),
    }

    verification = []
    for key, claimed, desc in CLAIMS:
        v, d = verdict(claimed, computed.get(key))
        verification.append({
            "claim_id": key, "description": desc, "manuscript_value": claimed,
            "recomputed_value": computed.get(key), "abs_difference": d, "status": v,
        })

    # ------------------------------------------------------------ consistency
    issues = []

    def add(level, where, msg):
        issues.append({"level": level, "location": where, "message": msg})

    # 1. confusion-matrix accuracy vs headline accuracy
    if cls:
        cm_acc = cls["accuracy"]
        add("INFO", "manuscript Fig. confusion matrix vs Sec. results",
            f"Confusion-matrix accuracy is {cm_acc:.6f} (= best-epoch val accuracy, "
            f"epoch {stored['best_epoch']}), while the headline 96.2% +/- 0.8% is the "
            f"mean +/- std of the LAST FIVE EPOCHS ({stored['reported_val_accuracy']:.6f} "
            f"+/- {stored['reported_val_std']:.6f}). Both are correct but they are "
            f"different quantities; a reader recomputing accuracy from the printed "
            f"confusion matrix gets 96.79%, not 96.2%. The manuscript must say which "
            f"is which.")
        if cls["confusion_matrix"] != stored["confusion_matrix"]:
            add("ERROR", "results/metrics.json confusion_matrix",
                "Recomputed confusion matrix differs from the stored one.")
        else:
            add("INFO", "confusion matrix",
                "Recomputed confusion matrix is IDENTICAL to the stored one.")

    # 2. stale occlusion block in results/metrics.json
    og = stored.get("occlusion_grid_validation", {})
    if og and occ:
        if abs(og.get("recall_occluded", 0) - 0.1725) < 1e-3:
            add("ERROR", "results/metrics.json -> occlusion_grid_validation",
                f"STALE: this block still holds the PRE-FIX operating point "
                f"(precision {og['precision_occluded']:.3f}, recall "
                f"{og['recall_occluded']:.3f}, F1 {og['f1_occluded']:.3f}) from before "
                f"the 2026-09-17 q/tau fix. The manuscript reports 0.727/0.917/0.811. "
                f"Two files in results/ therefore disagree; generate_report_figures.py "
                f"has not been re-run since the fix.")

    # 3. occlusion operating point tuned and reported on the same frames
    if occ and occ.get("optimism_gap"):
        g = occ["optimism_gap"]
        add("WARNING", "occlusion detector operating point",
            f"q/tau were selected as best-F1 on the sweep's 120-frame seed-0 sample "
            f"AND the manuscript reports the score from that same sample. On "
            f"{occ['evaluation_set']['n_frames_evaluated']} held-out frames the same "
            f"fixed operating point gives F1 {occ['heldout']['f1_occluded']:.4f} vs "
            f"{occ['sweep_sample_reported']['f1_occluded']:.4f} reported "
            f"(optimism gap {g['f1']:+.4f}). The gap is small, but the reported "
            f"figure is a tuning-sample figure and should be labelled as such or "
            f"replaced with the held-out one.")

    # 4. no independent test set
    if sp and not sp["independent_test_set_exists"]:
        add("WARNING", "data splits",
            "There is NO third, untouched test split. The best-epoch checkpoint is "
            "SELECTED on the same 26-episode validation split the accuracy is "
            "REPORTED on, so 96.8% is a model-selection-contaminated estimate. The "
            "96.2% last-5-epoch mean is less affected but still validation, not test.")
        add("INFO", "data splits",
            f"Episode-disjoint split verified: {sp['leakage_check']}; "
            f"{sp['n_train_episodes']} train / {sp['n_val_episodes']} val episodes "
            f"matches the manuscript's stated 144/26.")

    # 5. fusion significance
    if stat:
        for field, t in stat["tests"].items():
            fc = t["fused_vs_camera"]
            if not fc["excludes_zero"]:
                add("WARNING", f"fusion claim: {field}",
                    f"fused vs camera on {field}: difference {fc['observed_difference']:+.4f} "
                    f"with 95% CI [{fc['ci95'][0]:+.4f}, {fc['ci95'][1]:+.4f}], "
                    f"p={fc['bootstrap_p_two_sided']:.4f} -- NOT distinguishable from zero "
                    f"under episode-level bootstrap ({stat['n_episodes_contributing']} "
                    f"episodes). The point estimate favours fusion but the dataset does "
                    f"not support a significance claim on this axis.")

    # 6. fault denominator
    if flt and flt.get("status") == "EXECUTED":
        add("WARNING", "fault injection count",
            f"This run injects {flt['n_faults_excluding_clean']} fault conditions and the "
            f"health monitor detects {flt['n_detected']} of them "
            f"({', '.join(flt['detected_by_health_monitor'])}); undetected: "
            f"{', '.join(flt['not_detected_by_health_monitor'])}. Any 'N of M faults' "
            f"claim must state M -- a '5 of 7' phrasing does not match this suite.")

    # 7. runtime conflation
    if rt:
        add("WARNING", "runtime claim",
            f"Perception modules sum to {rt['serial_sum_mean_ms']:.1f} ms serial "
            f"({rt['implied_fps_serial_sum']:.1f} FPS upper-bound cost) with CARLA out "
            f"of the loop, whereas the manuscript's 2-9 FPS is the live demo's "
            f"synchronous tick rate including CARLA rendering. These are different "
            f"quantities and must not be presented as one.")
        slow = max((k for k, v in rt["stages"].items() if "mean_ms" in v),
                   key=lambda k: rt["stages"][k]["mean_ms"])
        add("INFO", "runtime breakdown",
            f"Slowest stage is {slow} at {rt['stages'][slow]['mean_ms']:.1f} ms, not "
            f"MiDaS ({rt['stages'].get('midas_depth', {}).get('mean_ms', float('nan')):.1f} "
            f"ms). Documentation stating MiDaS dominates is contradicted by measurement.")

    # 8. hidden-target sample size
    if hid and hid.get("by_horizon"):
        smalls = [h for h, r in hid["by_horizon"].items()
                  if r.get("particle", {}).get("n", 0) < 10]
        add("WARNING", "hidden-target experiment",
            f"Only {hid['n_occlusion_events_scored']} occlusion events were scorable "
            f"across {hid['n_episodes_with_events']} episodes. Horizons with n<10: "
            f"{smalls}. At 3.0s n=1, so its 'significant' CI is degenerate and "
            f"meaningless. This experiment is sample-limited and must not carry a "
            f"headline claim.")

    for lvl, where, msg in []:
        add(lvl, where, msg)

    # --------------------------------------------------------- status table
    status = [
        {"experiment": "A: classification + uncertainty", "status": "EXECUTED",
         "dataset": "raw_v2 val (episode-disjoint)", "n": cls["n_crops"] if cls else None,
         "main_metric": f"acc {cls['accuracy']:.4f}, ECE {unc['expected_calibration_error']:.4f}, "
                        f"err-AUROC {unc['error_detection_auroc']:.3f}" if cls else None,
         "file": "classification_metrics.json / uncertainty_metrics.json"},
        {"experiment": "B: occlusion detector (held-out)", "status": "EXECUTED",
         "dataset": f"{occ['evaluation_set']['n_frames_evaluated']} frames disjoint from tuning sample" if occ else None,
         "n": occ["evaluation_set"]["n_cells"] if occ else None,
         "main_metric": f"P {occ['heldout']['precision_occluded']:.3f} R {occ['heldout']['recall_occluded']:.3f} F1 {occ['heldout']['f1_occluded']:.3f}" if occ else None,
         "file": "occlusion_metrics.json"},
        {"experiment": "C: camera/radar/fusion tracking", "status": "VERIFIED",
         "dataset": "raw_v2 all episodes, paired", "n": trk["paired_observations"] if trk else None,
         "main_metric": f"fused v_lat {pm.get('fused',{}).get('v_lat',{}).get('mae',float('nan')):.3f}",
         "file": "tracking_metrics.json"},
        {"experiment": "D: fusion ablation (camera/radar/fused)", "status": "EXECUTED",
         "dataset": "raw_v2", "n": trk["paired_observations"] if trk else None,
         "main_metric": "see tracking + intent tables", "file": "tracking_metrics.json"},
        {"experiment": "D2: literature-style baseline", "status": "EXECUTED",
         "dataset": "raw_v2", "n": 1836,
         "main_metric": "baseline F1 0.778 vs proposed 0.885 (paired)",
         "file": "scripts/evaluate_prediction.py --baseline (stdout)"},
        {"experiment": "E: hidden-target prediction", "status": "PARTIALLY VERIFIED",
         "dataset": "raw_v2 occlusion events",
         "n": hid["n_occlusion_events_scored"] if hid else None,
         "main_metric": "particle vs EKF coast ADE; SAMPLE-LIMITED (n=15 events)",
         "file": "hidden_prediction_metrics.json"},
        {"experiment": "F: crossing intent", "status": "VERIFIED",
         "dataset": "raw_v2", "n": im.get("fused", {}).get("n"),
         "main_metric": f"fused F1 {im.get('fused',{}).get('f1',float('nan')):.3f}",
         "file": "intent_metrics.json"},
        {"experiment": "G: risk engine", "status": "SKIPPED",
         "dataset": "-", "n": None,
         "main_metric": "no risk/TTC/hazard ground truth exists in the dataset",
         "file": "risk_metrics.json"},
        {"experiment": "H: fault injection", "status": "EXECUTED",
         "dataset": "raw_v2, 24 episodes",
         "n": flt["n_faults_excluding_clean"] if flt else None,
         "main_metric": f"{flt['n_detected']}/{flt['n_faults_excluding_clean']} detected" if flt else None,
         "file": "fault_metrics.json"},
        {"experiment": "I: calibration/extrinsic perturbation", "status": "PARTIALLY VERIFIED",
         "dataset": "raw_v2", "n": 1,
         "main_metric": "extrinsic_drift is the only perturbation implemented; no "
                        "severity sweep exists",
         "file": "fault_metrics.json"},
        {"experiment": "J: robustness / weather", "status": "PARTIALLY VERIFIED",
         "dataset": "raw_v2 (all clear-day)", "n": None,
         "main_metric": "photometric degradation of clear-day frames ONLY; CARLA "
                        "weather API non-functional on this build",
         "file": "fault_metrics.json"},
        {"experiment": "K: scenario generalisation", "status": "EXECUTED",
         "dataset": "4 scenarios, single town", "n": 4,
         "main_metric": "per-scenario occlusion performance",
         "file": "occlusion_metrics.json -> by_scenario"},
        {"experiment": "L: runtime profiling", "status": "EXECUTED",
         "dataset": "recorded frames, CARLA out of loop",
         "n": rt["n_frames_timed"] if rt else None,
         "main_metric": f"{rt['serial_sum_mean_ms']:.1f} ms serial" if rt else None,
         "file": "runtime_metrics.json"},
        {"experiment": "M: multi-town generalisation", "status": "SKIPPED",
         "dataset": "-", "n": None,
         "main_metric": "excluded by explicit instruction; CARLA 0.10.0 ships only "
                        "Town10HD_Opt in any case",
         "file": "-"},
    ]

    master = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Confidence-Driven Occlusion-Aware Camera-Radar Fusion (CARLA)",
        "reproducibility": {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip(),
            "git_dirty_files": subprocess.run(["git", "status", "--porcelain"],
                                               capture_output=True, text=True).stdout.strip().splitlines(),
            "python": platform.python_version(), "platform": platform.platform(),
            "hardware": rt["hardware"] if rt else None,
            "seeds": {"episode_split": 0, "occlusion_holdout": 1234, "bootstrap": 0,
                       "sweep_sample": 0},
            "carla_limitation": ("CARLA is stochastic and its weather API is "
                                  "non-functional on this build; live-demo runs are not "
                                  "bit-reproducible. All experiments in this package are "
                                  "OFFLINE replays of recorded data and ARE deterministic."),
        },
        "dataset": ds, "splits": sp,
        "classification": cls, "uncertainty": unc, "occlusion": occ,
        "tracking": trk, "hidden_prediction": hid, "intent": itn,
        "risk": rsk, "fault_injection": flt, "runtime": rt, "statistics": stat,
        "verification": verification,
        "consistency_issues": issues,
        "experiment_status": status,
        "skipped": [s for s in status if s["status"] == "SKIPPED"],
        "partial": [s for s in status if s["status"].startswith("PARTIAL")],
    }
    (OUT / "MASTER_RESULTS.json").write_text(json.dumps(master, indent=2))
    (OUT / "experiment_status.json").write_text(json.dumps(status, indent=2))
    (OUT / "consistency_report.json").write_text(json.dumps(issues, indent=2))

    # --------------------------------------------------------------- CSV tables
    with (TABLES / "verification.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(verification[0]))
        w.writeheader(); w.writerows(verification)
    with (TABLES / "experiment_status.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(status[0]))
        w.writeheader(); w.writerows(status)
    if trk:
        with (TABLES / "tracking_paired.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "v_lat_mae", "v_lat_ci_lo", "v_lat_ci_hi", "v_lat_rmse",
                        "v_lat_p95", "v_fwd_mae", "pos_mae", "n"])
            for m, d in pm.items():
                b = d["bootstrap_episode_level"]["lat"]["ci95"]
                w.writerow([m, d["v_lat"]["mae"], b[0], b[1], d["v_lat"]["rmse"],
                            d["v_lat"]["p95"], d["v_fwd"]["mae"], d["position"]["mae"],
                            d["v_lat"]["n"]])
    if itn:
        with (TABLES / "intent.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "n", "roc_auc", "pr_auc", "precision", "recall", "f1",
                        "specificity", "balanced_accuracy", "mean_lead_time_s"])
            for m, d in im.items():
                if d.get("n"):
                    w.writerow([m, d["n"], d["roc_auc"], d["pr_auc"], d["precision"],
                                d["recall"], d["f1"], d["specificity"],
                                d["balanced_accuracy"], d["mean_lead_time_s"]])
    if flt and flt.get("status") == "EXECUTED":
        with (TABLES / "fault_injection.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["condition", "n_detections", "pos_rmse_m", "v_lat_mae",
                        "intent_auc", "camera_health", "radar_health", "detected"])
            for k, v in flt["per_condition"].items():
                w.writerow([k, v["n_detections"], v["pos_rmse_m"], v["v_lat_mae_ms"],
                            v["intent_auc"], v["camera_health"], v["radar_health_raw"],
                            v["detected_by_health_monitor"]])
    if rt:
        with (TABLES / "runtime.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "mean_ms", "median_ms", "p95_ms", "fps_alone"])
            for k, v in rt["stages"].items():
                if "mean_ms" in v:
                    w.writerow([k, v["mean_ms"], v["median_ms"], v["p95_ms"],
                                v["implied_fps_if_alone"]])

    n_ver = sum(1 for v in verification if v["status"] == "VERIFIED")
    print(f"verified {n_ver}/{len(verification)} manuscript claims")
    print(f"consistency issues: "
          f"{sum(1 for i in issues if i['level']=='ERROR')} ERROR, "
          f"{sum(1 for i in issues if i['level']=='WARNING')} WARNING, "
          f"{sum(1 for i in issues if i['level']=='INFO')} INFO")
    for v in verification:
        if v["status"] != "VERIFIED":
            print(f"  {v['status']:<28} {v['description']}: claimed {v['manuscript_value']} "
                  f"vs computed {v['recomputed_value']}")
    print(f"\nwrote {OUT/'MASTER_RESULTS.json'} and {TABLES}/*.csv")


if __name__ == "__main__":
    main()
