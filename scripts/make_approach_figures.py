"""Figures for the approach comparison: adapter-only (previous) against classifier + adapter prose (current).

Every number is read from `outputs/analysis/phase3a/hybrid/comparison.json` (the three-way validation
comparison) or `crossfit.json` (the cross-fitted, out-of-fold numbers). The architecture panels are drawn from
the component names in the protocol, with no number typed by hand.

Usage:  python scripts/make_approach_figures.py
Writes: docs/reports/figures/fig5_architecture.png, fig6_diagnosis_by_stratum.png,
        fig7_abstention.png, fig8_surrogate.png, approach_data.json
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/reports/figures"
COMPARISON = ROOT / "outputs/analysis/phase3a/hybrid/comparison.json"
CROSSFIT = ROOT / "outputs/analysis/phase3a/hybrid/crossfit.json"

INK = "#1f2933"
ACCENT = "#0b6e99"
MUTED = "#8a94a6"
OLD = "#b5651d"
GATE = "#b03a2e"


def style(ax, title, ylabel="", ymax=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def box(ax, x, y, w, h, text, face="#ffffff", edge=INK, size=7.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.0, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight=weight, wrap=True)


def arrow(ax, xy_from, xy_to, color=MUTED):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=color, shrinkA=2, shrinkB=2))


def fig_architecture():
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), dpi=200)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("Previous approach — the adapter decides and writes", loc="left", fontsize=10.5, color=OLD, pad=6)
    box(ax, 0.02, 0.30, 0.17, 0.40, "evidence page\n(sensors, z, lag,\nagent scores)")
    box(ax, 0.24, 0.30, 0.19, 0.40, "InternVL3-2B\n+ LoRA adapter\n(seed 42, greedy)", edge=OLD)
    box(ax, 0.48, 0.55, 0.18, 0.28, "diagnosis from\nits own prose", face="#f7efe8", edge=OLD)
    box(ax, 0.48, 0.18, 0.18, 0.28, "prose body", face="#f7efe8", edge=OLD)
    box(ax, 0.71, 0.30, 0.15, 0.40, "schema parse\n+ validators")
    box(ax, 0.89, 0.36, 0.09, 0.28, "report")
    arrow(ax, (0.19, 0.50), (0.24, 0.50), OLD)
    arrow(ax, (0.43, 0.53), (0.48, 0.66), OLD)
    arrow(ax, (0.43, 0.47), (0.48, 0.32), OLD)
    arrow(ax, (0.66, 0.62), (0.71, 0.58), OLD)
    arrow(ax, (0.66, 0.32), (0.71, 0.42), OLD)
    arrow(ax, (0.86, 0.50), (0.89, 0.50), OLD)
    ax.text(0.30, 0.06, "the label and the words come from the same stochastic decode: one bad label corrupts both",
            fontsize=7, color=OLD, ha="left")

    ax = axes[1]
    ax.set_title("Current approach — the classifier decides, the adapter explains", loc="left",
                 fontsize=10.5, color=ACCENT, pad=6)
    box(ax, 0.02, 0.30, 0.17, 0.40, "evidence page\n(323 features,\none row per event)")
    box(ax, 0.23, 0.30, 0.19, 0.40, "LightGBM + isotonic\ncalibration\n(cross-fitted)", edge=ACCENT)
    box(ax, 0.46, 0.30, 0.17, 0.40, "menu rule +\nthreshold 0.979", edge=ACCENT)
    box(ax, 0.67, 0.55, 0.17, 0.28, "diagnosis\n(deterministic)")
    box(ax, 0.67, 0.18, 0.17, 0.28, "adapter writes\nforced-prefix prose", edge=ACCENT)
    box(ax, 0.87, 0.34, 0.11, 0.32, "validators\n→ report")
    arrow(ax, (0.19, 0.50), (0.23, 0.50), ACCENT)
    arrow(ax, (0.42, 0.50), (0.46, 0.50), ACCENT)
    arrow(ax, (0.63, 0.60), (0.67, 0.68), ACCENT)
    arrow(ax, (0.63, 0.40), (0.67, 0.32), ACCENT)
    arrow(ax, (0.84, 0.68), (0.87, 0.60), ACCENT)
    arrow(ax, (0.84, 0.32), (0.87, 0.42), ACCENT)
    ax.text(0.23, 0.06, "the label is a function of the evidence; the adapter cannot change it and is measured against it",
            fontsize=7, color=ACCENT, ha="left")

    fig.tight_layout()
    fig.savefig(FIG / "fig5_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig_accuracy(data):
    order = [("observable", "observable (822)"), ("menu_present", "menu present (497)"),
             ("fault_named", "fault named (325)"), ("normal", "normal (47)"),
             ("open_set_on_menu", "open-set on menu (125)"), ("unobservable", "unobservable (12)")]
    labels = [name for _, name in order]
    old = [data["strata"][k]["adapter"]["accuracy"] * 100 for k, _ in order]
    new = [data["strata"][k]["classifier_menu_rule"]["accuracy"] * 100 for k, _ in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    y = range(len(order))
    h = 0.36
    b1 = ax.barh([i + h / 2 for i in y], old, height=h, color=OLD, label="previous: adapter alone")
    b2 = ax.barh([i - h / 2 for i in y], new, height=h, color=ACCENT, label="current: classifier + menu rule")
    for bars, vals in ((b1, old), (b2, new)):
        for bar, v in zip(bars, vals):
            ax.text(min(v + 1.2, 99.2), bar.get_y() + bar.get_height() / 2, "%.1f" % v,
                    va="center", fontsize=7, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "Diagnosis accuracy by stratum, validation (834 rows)", "")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_diagnosis_by_stratum.png", bbox_inches="tight")
    plt.close(fig)


def fig_abstention(data):
    strata = [("observable", "observable (n=822)"), ("menu_present", "menu present (n=497)")]
    metrics = [("abstention_recall", "abstention recall"), ("missed_abstention_rate", "missed abstention")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=200)
    for ax, (key, title) in zip(axes, metrics):
        x = list(range(len(strata)))
        w = 0.36
        old = [data["strata"][s]["adapter"]["abstention"][key] * 100 for s, _ in strata]
        new = [data["strata"][s]["classifier_menu_rule"]["abstention"][key] * 100 for s, _ in strata]
        b1 = ax.bar([i - w / 2 for i in x], old, width=w, color=OLD, label="adapter alone")
        b2 = ax.bar([i + w / 2 for i in x], new, width=w, color=ACCENT, label="classifier + rule")
        for bars, vals in ((b1, old), (b2, new)):
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, "%.1f" % v,
                        ha="center", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([n.split(" (")[0] for _, n in strata])
        style(ax, title, "%", ymax=max(old + new) * 1.35)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=7.5, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Abstention behaviour: who declines when the evidence cannot support an answer",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_abstention.png", bbox_inches="tight")
    plt.close(fig)


def fig_surrogate(data):
    items = sorted(data["surrogate"].items())
    labels, rates, notes = [], [], []
    for name, entry in items:
        labels.append(name.replace("FAULT_", "fault "))
        rates.append(entry["abstained_rate"] * 100)
        renamed = entry.get("mislabelled") or {}
        if isinstance(renamed, dict):
            top = ", ".join("%s×%d" % (k.replace("FAULT_", "f"), v) for k, v in list(renamed.items())[:2])
        else:
            top = str(renamed)
        notes.append(top or "declined")
    fig, ax = plt.subplots(figsize=(5.8, 3.1), dpi=200)
    bars = ax.bar(labels, rates, color=ACCENT, width=0.55)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3.0, "%.0f %%" % rate,
                ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["%s\n→ %s" % (lab, note.replace("UNCLASSIFIED_OPEN_SET", "open-set")
                                      .replace("FAULT_", "f").replace("f0", "f0"))
                        for lab, note in zip(labels, notes)], fontsize=7.5)
    style(ax, "A never-seen fault: does the classifier decline?\n(leave-one-fault-out, 25 items each)",
          "items declined (%)", ymax=118)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_surrogate.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    data = json.loads(COMPARISON.read_text(encoding="utf-8"))
    crossfit = json.loads(CROSSFIT.read_text(encoding="utf-8"))
    fig_architecture()
    fig_accuracy(data)
    fig_abstention(data)
    fig_surrogate(data)
    (FIG / "approach_data.json").write_text(json.dumps({
        "observable": {"previous": data["strata"]["observable"]["adapter"],
                       "current": data["strata"]["observable"]["classifier_menu_rule"]},
        "menu_present": {"previous": data["strata"]["menu_present"]["adapter"],
                         "current": data["strata"]["menu_present"]["classifier_menu_rule"]},
        "unobservable": {"previous": data["strata"]["unobservable"]["adapter"],
                         "current": data["strata"]["unobservable"]["classifier_menu_rule"]},
        "crossfit_observable": crossfit["validation"]["observable"],
        "threshold_crossfit": crossfit["threshold"],
        "threshold_in_sample": data["threshold"],
        "saturation_share": crossfit["validation"]["saturation_share"],
        "surrogate": data["surrogate"],
    }, indent=2), encoding="utf-8")
    print(json.dumps({"figures": 4, "saturated": crossfit["validation"]["saturated"],
                      "rows": data["rows"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
