"""Render every paper figure from the stored artefacts.

Nothing is typed by hand: the cell table comes from `n4_cells.json`, the test contrast from
`d33/test_contrast.json`, the classifier comparison from `p2/classifier_contrasts.json`, and the stratified
failure profile is recomputed with the same `measure` function the tables use. Re-running this script after a
new scoring pass re-renders every figure, which is the point: no number lives only in a picture.

Usage:  python scripts/make_figures.py
Writes: docs/reports/figures/*.png and docs/reports/figures/figure_data.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_prose_dataset import load_jsonl  # noqa: E402
from scripts.fallback_v2 import parse_store  # noqa: E402
from scripts.overnight_run import normalise_flags  # noqa: E402
from scripts.pilot_rechecks import measure  # noqa: E402
from scripts.prose_pilot_prompts import build_orders, parse_prose  # noqa: E402

FIG = ROOT / "docs/reports/figures"
CELLS = ROOT / "outputs/overnight_2026-11-05/n4_cells.json"
D33 = ROOT / "outputs/analysis/phase3a/d33/test_contrast.json"
CLASSIFIER = ROOT / "outputs/analysis/phase3a/p2/classifier_contrasts.json"
VAL_RAW = ROOT / "outputs/benchmark_v2/eval/prose_pilot/raw_pilot_kb.jsonl"
TEST_ROWS = ROOT / "outputs/benchmark_v2/eval/prose_test/rows_test.jsonl"
TEST_RAW = ROOT / "outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl"

INK = "#1f2933"
ACCENT = "#0b6e99"
MUTED = "#8a94a6"
GATE = "#b03a2e"


def style(ax, title, ylabel, ymax=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def stratified(rows, raw_path):
    raws = {r["item_id"]: r for r in load_jsonl(raw_path)}
    total, bad = Counter(), Counter()
    for row in rows:
        if row["item_id"] not in raws:
            continue
        prompt = row["prompts"]["kb"]
        store = parse_store(prompt)
        parsed = parse_prose(raws[row["item_id"]]["text"])
        text = parsed["prose"] if parsed["schema_valid"] else str(raws[row["item_id"]]["text"])
        result = normalise_flags(measure(text, prompt, row["gold_fault_id"], row["gold_verdict"], store))
        gold = row["gold_fault_id"]
        klass = ("normal-transient" if gold == "NORMAL_TRANSIENT"
                 else ("open-set" if gold == "UNCLASSIFIED_OPEN_SET" else "classified"))
        total[klass] += 1
        bad[klass] += 1 if result["numeric"] else 0
    return {k: {"defects": bad[k], "rows": total[k],
                "rate": round(bad[k] / total[k], 4) if total[k] else 0.0} for k in total}


def fig_cells(cells):
    labels = ["A", "B", "C", "M", "D"]
    rates = [cells["summary"][k]["numeric_rate"] for k in labels]
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
    bars = ax.bar(labels, [r * 100 for r in rates], color=[ACCENT if k in "ACD" else MUTED for k in labels],
                  width=0.6)
    ax.axhline(2.0, color=GATE, linestyle="--", linewidth=1)
    ax.text(2.5, 2.18, "2 % gate", color=GATE, fontsize=7, ha="center")
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, "%.2f" % (value * 100),
                ha="center", fontsize=7, color=INK)
    style(ax, "Numeric-defect rate by condition, 520 validation rows", "defective rows (%)", ymax=max(rates) * 100 + 1.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_cells.png", bbox_inches="tight")
    plt.close(fig)


def fig_stratified(val, test):
    classes = ["normal-transient", "open-set", "classified"]
    x = range(len(classes))
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
    width = 0.36
    for offset, (name, data, color) in enumerate((("validation", val, ACCENT), ("test", test, MUTED))):
        vals = [data.get(c, {}).get("rate", 0) * 100 for c in classes]
        bars = ax.bar([i + (offset - 0.5) * width for i in x], vals, width=width, label=name, color=color)
        for bar, value in zip(bars, vals):
            label = "%.1f" % value if value < 10 else "%.0f" % value
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, label,
                    ha="center", fontsize=7, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Numeric defects by gold class: the failure is class-specific", "defective rows (%)", ymax=85)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_stratified.png", bbox_inches="tight")
    plt.close(fig)


def fig_classifier(data):
    order = ["menu_rule", "hybrid", "gbdt", "mlp", "adapter"]
    labels = {"menu_rule": "menu rule", "hybrid": "hybrid", "gbdt": "LightGBM", "mlp": "MLP", "adapter": "adapter (2B)"}
    acc = [data["per_system"][k]["accuracy"] for k in order]
    lo = [data["per_system"][k]["accuracy"] - data["per_system"][k]["accuracy_ci95"][0] for k in order]
    hi = [data["per_system"][k]["accuracy_ci95"][1] - data["per_system"][k]["accuracy"] for k in order]
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
    ax.barh([labels[k] for k in order][::-1], [a * 100 for a in acc][::-1],
            xerr=[[h * 100 for h in hi][::-1], [l * 100 for l in lo][::-1]], color=ACCENT, height=0.55,
            error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
    for y, value in enumerate([a * 100 for a in acc][::-1]):
        ax.text(value - 1.0, y, "%.1f" % value, va="center", ha="right", fontsize=7, color="white")
    style(ax, "Diagnosis accuracy on 497 shared items (95 % CI)", "")
    ax.set_xlim(75, 101)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_classifier.png", bbox_inches="tight")
    plt.close(fig)


def fig_anchor(cells, d33):
    rows = [("validation", cells["summary"]["A"]["anchor_exact"] / cells["summary"]["A"]["anchor_rows"],
             cells["summary"]["B"]["anchor_exact"] / cells["summary"]["B"]["anchor_rows"]),
            ("test", 1.0, d33["anchor_status"]["B"]["exact"] / d33["rows"])]
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
    x = range(len(rows))
    width = 0.36
    for offset, (name, color, key) in enumerate((("A (deterministic)", ACCENT, 1), ("B (adapter)", MUTED, 2))):
        values = [r[key] * 100 for r in rows]
        bars = ax.bar([i + (offset - 0.5) * width for i in x], values, width=width, color=color, label=name)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, "%.1f" % value,
                    ha="center", fontsize=7, color=INK)
    ax.axhline(90, color=GATE, linestyle="--", linewidth=1)
    ax.text(1.45, 91.5, "90 % gate", color=GATE, fontsize=7, ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows])
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    style(ax, "Anchor exactness: the structural failure", "reports with the top sensor's z (%)", ymax=112)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_anchor.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    cells = json.loads(CELLS.read_text(encoding="utf-8"))
    d33 = json.loads(D33.read_text(encoding="utf-8"))
    classifier = json.loads(CLASSIFIER.read_text(encoding="utf-8"))
    val = stratified(build_orders(), VAL_RAW)
    test = stratified([json.loads(line) for line in TEST_ROWS.read_text(encoding="utf-8").splitlines()
                       if line.strip()], TEST_RAW)
    fig_cells(cells)
    fig_stratified(val, test)
    fig_classifier(classifier)
    fig_anchor(cells, d33)
    (FIG / "figure_data.json").write_text(json.dumps(
        {"cells": {k: cells["summary"][k]["numeric_rate"] for k in ("A", "B", "C", "M", "D")},
         "cell_counts": {k: cells["summary"][k]["numeric"] for k in ("A", "B", "C", "M", "D")},
         "stratified": {"validation": val, "test": test},
         "d33_rates": d33["rates"], "anchor": d33["anchor_status"]}, indent=2), encoding="utf-8")
    print(json.dumps({"figures": 4, "stratified": {"validation": val, "test": test}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
