"""Run the current diagnosis architecture on held-out validation events and print what it decides.

This is the shipped pipeline up to the label: the evidence page is rendered, the 323 features are parsed from
that same page, the LightGBM classifier scores 15 classes, isotonic calibration makes the confidence
comparable, and the menu rule turns the prediction into a verdict. No adapter is involved at this layer — the
label is printed by code before any prose exists.

Usage:  python scripts/demo_diagnosis.py [--n 6]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lightgbm as lgb  # noqa: E402

from llm.hybrid.calibration import apply_calibrator  # noqa: E402
from llm.hybrid.menu_rule import menu_rule  # noqa: E402
from scripts.evidence_classifier import prompt_features  # noqa: E402

ART = ROOT / "outputs/analysis/phase3a/hybrid/protocolA"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"


def load_rows() -> list[dict]:
    return [json.loads(line) for line in VAL.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_of(row: dict) -> tuple[str, str]:
    target = row.get("target") or {}
    return str(target.get("fault_id") or "NORMAL_TRANSIENT"), str(target.get("verdict") or "?")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="how many example rows to print")
    args = parser.parse_args()

    rows = load_rows()
    booster = lgb.Booster(model_file=str(ART / "classifier.txt"))
    calibration = json.loads((ART / "calibration.json").read_text(encoding="utf-8"))
    threshold = float(json.loads((ART / "threshold.json").read_text(encoding="utf-8"))["threshold"])
    labels = calibration["labels"]

    features = np.asarray([prompt_features(row["prompt"]) for row in rows], dtype=np.float64)
    raw = booster.predict(features)
    raw_rows = [{label: float(value) for label, value in zip(labels, row)} for row in raw]
    calibrated = apply_calibrator(calibration, raw_rows)

    results = []
    for row, raw_row, cal_row in zip(rows, raw_rows, calibrated):
        top = max(cal_row, key=cal_row.get)
        raw_top = max(raw_row, key=raw_row.get)
        decision = menu_rule(top, cal_row[top], row.get("menu_ids") or [], threshold)
        gold, gold_verdict = gold_of(row)
        results.append({"row": row, "raw_top": raw_top, "raw_conf": raw_row[raw_top],
                        "top": top, "conf": cal_row[top], "decision": decision,
                        "gold": gold, "gold_verdict": gold_verdict,
                        "correct": decision["diagnosis"] == gold})

    print("artefacts")
    print("  classifier   %s (%.1f MB)" % (ART / "classifier.txt", (ART / "classifier.txt").stat().st_size / 1e6))
    print("  calibration  %s (%s over %d classes)" % (ART / "calibration.json", calibration["method"], len(labels)))
    print("  threshold    %.4f (in-sample rule, budget 5 %%)" % threshold)
    print("  features     %d parsed from the rendered evidence page" % features.shape[1])
    print("  rows         %d validation events\n" % len(rows))

    wanted = [
        ("menu", lambda r: r["row"]["fault"] == 1 and r["row"]["variant"] == "menu"),
        ("open-set twin", lambda r: r["row"]["fault"] == 1 and r["row"]["variant"] == "removed"),
        ("normal", lambda r: r["row"]["fault"] is None and r["row"]["variant"] == "menu"),
        ("fault 15", lambda r: r["row"]["fault"] == 15 and r["row"]["variant"] == "menu"),
        ("unobservable 3", lambda r: r["row"]["fault"] == 3 and r["row"]["variant"] == "menu"),
        ("true open-set 17", lambda r: r["row"]["fault"] == 17 and r["row"]["variant"] == "removed"),
    ]
    shown = []
    for name, predicate in wanted[: args.n]:
        match = next((r for r in results if predicate(r)), None)
        if match is None:
            continue
        shown.append((name, match))

    header = "%-16s %-22s %-18s %-8s %-24s %-7s %s" % (
        "case", "item", "gold", "conf", "rule -> diagnosis", "reason", "ok")
    print(header)
    print("-" * len(header))
    for name, r in shown:
        print("%-16s %-22s %-18s %-8.4f %-24s %-7s %s" % (
            name, r["row"]["item_id"], r["gold"], r["conf"], r["decision"]["diagnosis"],
            r["decision"]["reason"][:7], "yes" if r["correct"] else "NO"))

    decisions = Counter(r["decision"]["reason"] for r in results)
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    strata = {
        "all 834": results,
        "observable": [r for r in results if not r["row"]["unobservable"]],
        "menu-present": [r for r in results if r["row"]["variant"] == "menu"
                         and not r["row"]["unobservable"]],
        "removed twins": [r for r in results if r["row"]["variant"] == "removed"],
        "unobservable": [r for r in results if r["row"]["unobservable"]],
    }
    print("\naggregate (recomputed live, not read from a table)")
    for name, subset in strata.items():
        acc = sum(1 for r in subset if r["correct"]) / max(len(subset), 1)
        print("  %-16s n=%4d accuracy %.4f" % (name, len(subset), acc))
    print("  all rows          %d correct / %d" % (correct, total))
    print("  rule reasons      %s" % dict(decisions))
    print("  threshold arms    below_threshold %d, not_on_menu %d, no_fault %d, classified %d" % (
        decisions.get("below_threshold", 0), decisions.get("not_on_menu", 0),
        decisions.get("no_fault_predicted", 0), decisions.get("on_menu_above_threshold", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
