"""Render fig8 from a surrogate-wide artefact (default: the retrained ``vocabfix`` generation).

The published figure was drawn from ``comparison.json``'s surrogate block, which was produced before the feature
vocabulary was fixed. This reads the surrogate artefact directly, so the figure and the table cannot diverge.

    python scripts/make_surrogate_figure.py
    python scripts/make_surrogate_figure.py --wide .../surrogate_wide.json --out docs/reports/figures/fig8_surrogate.png
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402

from make_approach_figures import ACCENT, INK, style  # noqa: E402

DEFAULT_WIDE = ROOT / "outputs/analysis/phase3a/hybrid/vocabfix/surrogate_wide.json"
DEFAULT_OUT = ROOT / "docs/reports/figures/fig8_surrogate.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=pathlib.Path, default=DEFAULT_WIDE)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    wide = json.loads(args.wide.read_text(encoding="utf-8"))
    faults = sorted(wide["faults"])
    labels, rates, notes = [], [], []
    for fault in faults:
        entry = wide["per_fault"][fault]["by_variant"]["menu"]
        labels.append(fault.replace("FAULT_", "fault "))
        rates.append(entry["declined_rate"] * 100)
        named = wide["per_fault"][fault]["named_as"]["menu"]
        notes.append(", ".join(f"{name.replace('FAULT_', 'f')}×{count}"
                               for name, count in list(named.items())[:2]) or "declined")

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
    bars = ax.bar(labels, rates, color=ACCENT, width=0.58)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3.0, f"{rate:.0f} %",
                ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{lab}\n→ {note}" for lab, note in zip(labels, notes)], fontsize=7)
    style(ax, "A never-seen fault: does the classifier decline?\n(leave-one-fault-out, retrained generation, 25 items each)",
          "items declined (%)", ymax=118)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"out": str(args.out), "faults": len(faults),
                      "declined": {f: wide["per_fault"][f]["by_variant"]["menu"]["declined"] for f in faults}},
                     indent=1))


if __name__ == "__main__":
    main()
