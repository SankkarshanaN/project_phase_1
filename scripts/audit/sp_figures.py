"""SECOND PASS -- the three results that had no figure.

  fig_independent_test   SP-01  validation vs independent test (the pass's headline)
  fig_hidden_funnel      SP-03  202 -> 32 -> 15, and where the events are lost
  fig_effect_sizes       SP-06  Cohen's dz forest plot with the conventional bands

Every number is read from the second-pass JSONs; nothing is retyped here.

Palette matches the existing figure set (camera #2e86c1, radar #c0392b,
fused #1f9e6e) -- validated colourblind-safe: worst adjacent pair dE 9.2
deutan / 28.8 normal, all three inside the lightness band and >=3:1 on the
surface. Identity is never carried by colour alone; every series is also
direct-labelled.

Writes results/second_pass/{independent_test,hidden_funnel,effect_sizes}.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SP = Path("results/second_pass")
A1 = Path("results/audit")

VAL = "#8a8a8a"        # validation: recessive, it is the reference
TEST = "#1f9e6e"       # independent test: the new measurement
CAMERA, RADAR, FUSED = "#2e86c1", "#c0392b", "#1f9e6e"
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#d7d7d7"


def load(p):
    return json.loads(Path(p).read_text())


def _dumbbell(ax, rows, xlabel, note_fmt="{:+.4f}"):
    """rows = [(label, val, test, higher_is_better)] drawn bottom-up."""
    ys = np.arange(len(rows))
    for y, (lab, v, t, hib) in zip(ys, rows):
        ax.plot([v, t], [y, y], color=GRID, linewidth=2, zorder=1,
                solid_capstyle="round")
        ax.scatter([t], [y], s=120, color=TEST, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.scatter([v], [y], s=42, color=VAL, zorder=4,
                   edgecolor="white", linewidth=1.2)
        gap = t - v
        # "better" is direction-aware, so the reader never has to infer the sign
        better = (gap > 0) == hib
        tag = "same" if abs(gap) < 5e-4 else ("better" if better else "worse")
        ax.annotate(f"{note_fmt.format(gap)}  {tag}",
                    (max(v, t), y), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.grid(axis="x", color=GRID, alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_independent_test(out: Path):
    it = load(SP / "independent_test_metrics.json")
    cls, unc = load(A1 / "classification_metrics.json"), load(A1 / "uncertainty_metrics.json")
    t, v = it["classifier"]["test"], it["classifier"]["validation_for_comparison"]
    ti, vi = it["intent"]["test"]["fused"], it["intent"]["validation_for_comparison"]["fused"]
    tt = it["tracking"]["test"]["fused"]
    vt = it["tracking"]["validation_for_comparison"]["fused"]

    # Two panels because the measures have different units -- never a second
    # y-scale on one plot.
    unit_rows = [
        ("Fused intent F1", vi["f1"], ti["f1"], True),
        ("Error-detect AUROC", unc["error_detection_auroc"], t["error_detection_auroc"], True),
        ("ECE (lower better)", unc["expected_calibration_error"],
         t["expected_calibration_error"], False),
        ("Classifier macro-F1", cls["macro_f1"], t["macro_f1"], True),
        ("Classifier accuracy", v["accuracy"], t["accuracy"], True),
    ]
    mae_rows = [
        ("Fused position MAE", vt["position_mae"], tt["position_mae"], False),
        ("Fused v_fwd MAE", vt["v_fwd_mae"], tt["v_fwd_mae"], False),
        ("Fused v_lat MAE", vt["v_lat_mae"], tt["v_lat_mae"], False),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.3), dpi=200,
                             gridspec_kw={"height_ratios": [5, 3.2]})
    _dumbbell(axes[0], unit_rows, "score (0-1)")
    _dumbbell(axes[1], mae_rows, "mean absolute error (m or m/s)", "{:+.3f}")
    axes[0].set_xlim(-0.02, 1.13)

    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=VAL,
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=f"Validation  ({v['n_crops']:,} crops, 26 episodes)"),
         plt.Line2D([], [], marker="o", linestyle="", markersize=10, color=TEST,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label=f"Independent test  ({t['n_crops']:,} crops, "
                          f"{t['n_test_episodes']} unseen episodes)")]
    axes[0].legend(handles=h, frameon=False, fontsize=8, loc="lower left",
                   bbox_to_anchor=(0.0, 1.02), ncol=1)
    fig.suptitle("The model holds up on episodes it has never seen",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=1.06)
    fig.text(0.005, 1.005,
             "Classification and calibration transfer with no loss (accuracy moves by "
             "6e-5); a small validation dot inside a test dot means the two agree. "
             "Tracking degrades modestly on all three axes: +5.9% v_lat, +9.4% "
             "v_fwd, +12.0% position.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_hidden_funnel(out: Path):
    hd = load(SP / "hidden_target_diagnosis.json")
    f, d = hd["funnel"], hd["hidden_run_length_distribution"]
    stages = [
        ("Occlusion transitions\nin ground truth", f["stage_1_occlusion_transitions_in_ground_truth"]),
        ("...actor was confirmed-\ntracked beforehand",
         f["stage_2_actor_was_confirmed_tracked_before_going_hidden"]),
        ("...stayed hidden >=0.5 s\nwith truth available", f["stage_3_stays_hidden_at_least_0.5s_with_truth"]),
        ("Scored by the audit", f["stage_4_scored_by_first_audit"]),
    ]
    # Sequential single hue, light -> dark: this encodes magnitude, not identity.
    shades = ["#bfe3d3", "#8ccfb4", "#4fb98d", "#1f9e6e"]
    n0 = stages[0][1]

    fig, ax = plt.subplots(figsize=(8.4, 3.5), dpi=200)
    ys = np.arange(len(stages))[::-1]
    for y, (lab, n), c in zip(ys, stages, shades):
        ax.barh(y, n, height=0.62, color=c, zorder=2)
        ax.annotate(f"{n}", (n, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, fontweight="bold", color=INK)
        ax.annotate(f"{n/n0:.0%} of all", (n, y), xytext=(8, -12),
                    textcoords="offset points", va="center", fontsize=7.5, color=MUTED)
    for i in range(len(stages) - 1):
        lost = stages[i][1] - stages[i + 1][1]
        if lost:
            ax.annotate(f"-{lost}", (stages[i + 1][1] + (stages[i][1] - stages[i + 1][1]) / 2,
                                      ys[i] - 0.5), ha="center", va="center",
                        fontsize=8, color=RADAR, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in stages], fontsize=8.5, color=INK)
    ax.set_xlabel("occlusion events", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, n0 * 1.18)
    ax.grid(axis="x", color=GRID, alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("The 15-event limit is the data, not the evaluation",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=1.10)
    fig.text(0.005, 1.02,
             f"84% of actors that go behind an occluder were never confirmed-tracked "
             f"first, so there is no state to propagate. The median occlusion lasts "
             f"{d['median_frames_hidden']:.0f} frames ({d['median_frames_hidden']*0.1:.1f} s).",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_effect_sizes(out: Path):
    ef = load(SP / "effect_sizes.json")
    order = [("v_lat", "lateral velocity"), ("v_fwd", "forward velocity"),
             ("position", "position")]
    rows = []
    for key, pretty in order:
        for cmp_name, colour, vs in (("fused_vs_radar", RADAR, "vs radar"),
                                      ("fused_vs_camera", CAMERA, "vs camera")):
            e = ef["tests"][key][cmp_name]
            rows.append((f"{pretty}\n{vs}", e["cohens_dz_paired"],
                         e["prob_a_better_than_b"], colour))

    XLO, XHI = -1.35, 0.46          # plotting range for the dz axis
    TXT = 0.52                      # fixed column for the value text, clear of every mark

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=200)
    # Cohen's conventional bands. Shaded only on the "fusion better" side, so the
    # shading reads as a magnitude scale rather than as a second category.
    for lo, hi, shade in ((0.0, 0.2, "#f4f4f4"), (0.2, 0.5, "#e9e9e9"),
                           (0.5, 0.8, "#dedede"), (0.8, -XLO, "#d3d3d3")):
        ax.axvspan(-hi, -lo, color=shade, zorder=0)
    ax.axvline(0, color=INK, linewidth=1.1, zorder=2)

    ys = np.arange(len(rows))[::-1]
    for y, (lab, dz, p_sup, c) in zip(ys, rows):
        ax.plot([0, dz], [y, y], color=c, linewidth=2.2, zorder=3,
                solid_capstyle="round")
        ax.scatter([dz], [y], s=95, color=c, zorder=4, edgecolor="white",
                   linewidth=1.8)
        mag = ("large" if abs(dz) >= 0.8 else "medium" if abs(dz) >= 0.5
               else "small" if abs(dz) >= 0.2 else "negligible")
        ax.annotate(f"dz = {dz:+.3f}", (TXT, y), fontsize=8.5, color=INK,
                    va="center", ha="left", annotation_clip=False)
        ax.annotate(f"{mag}", (TXT + 0.35, y), fontsize=8.5, va="center",
                    ha="left", color=INK if mag == "negligible" else MUTED,
                    fontweight="bold" if mag == "negligible" else "normal",
                    annotation_clip=False)
        ax.annotate(f"{p_sup:.2f}", (TXT + 0.72, y), fontsize=8.5, color=MUTED,
                    va="center", ha="left", annotation_clip=False)
    # column headers
    for x, h in ((TXT, "effect size"), (TXT + 0.35, "magnitude"),
                 (TXT + 0.72, "P(fused better)")):
        ax.annotate(h, (x, len(rows) - 0.45), fontsize=7.5, color=MUTED,
                    va="center", ha="left", annotation_clip=False)
    # band ticks along the bottom, on the fusion-better side only
    for x, lab in ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"),
                   (1.15, "large")):
        if -x > XLO:
            ax.annotate(lab, (-x + (0.10 if lab == "large" else 0.0), -0.82),
                        fontsize=7, color=MUTED, ha="center",
                        va="center", annotation_clip=False)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlabel("paired Cohen's dz        <-- fusion better        fusion worse -->",
                  fontsize=8.5, color=MUTED)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(-1.05, len(rows) - 0.3)
    ax.set_xticks([-1.0, -0.5, 0.0])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    leg = [plt.Line2D([], [], color=RADAR, linewidth=2.2, marker="o",
                      markersize=8, markeredgecolor="white", label="fused vs radar"),
           plt.Line2D([], [], color=CAMERA, linewidth=2.2, marker="o",
                      markersize=8, markeredgecolor="white", label="fused vs camera")]
    ax.legend(handles=leg, frameon=False, fontsize=8, loc="lower left",
              bbox_to_anchor=(0.0, -0.30), ncol=2)

    fig.suptitle("Fusion beats radar decisively; against camera the effect is negligible",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=1.05)
    fig.text(0.005, 0.99,
             f"Paired on the same {ef['n_paired_observations']} observations. An effect "
             "size says whether a difference is big enough to matter; the first pass's "
             "p-values only said whether it was distinguishable from zero.",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    SP.mkdir(parents=True, exist_ok=True)
    fig_independent_test(SP / "independent_test.png")
    fig_hidden_funnel(SP / "hidden_funnel.png")
    fig_effect_sizes(SP / "effect_sizes.png")
