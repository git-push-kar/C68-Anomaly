"""Score every stored live event through the live page layer, and say why a page is thin.

Runs the shipped hybrid up to the label — page -> 323 features -> Protocol A LightGBM -> isotonic
calibration -> ``menu_rule`` with the deployment menu (decision A: 15 faults + explicit abstain) — over the
events in ``outputs/events.db``, and reports answers, confidence, rule agreement and page population.

A page can be thin for three unrelated reasons, and conflating them is how a mapping bug hides inside a
"sparse evidence" story. Each thin page is therefore attributed to one of:

* ``LEGACY_NO_DETECTOR`` — the record has no ``detector_evidence`` at all (schema v1: ``severity``,
  ``candidate_subsystem``). Nothing to render; not a mapping failure, but the classifier was never trained on
  such a page.
* ``VOCAB_MISMATCH`` — the page lists a sensor whose tag has no feature column. ``evidence_classifier.TAGS``
  names ``XMV_1..11`` while the page prints ``XMV_42..52``, so those sensors' z/deviation/trend/present are
  silently dropped. This one also weakens the benchmark measurements.
* ``ZERO_ENCODED`` — every value is legitimately zero: trend ``stable`` encodes to 0.0 and an onset at
  ``+0.0 min`` encodes to 0.0, indistinguishable from "absent" in the vector.

Usage::  python scripts/report_live_classifier.py [--limit 489] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sqlite3
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lightgbm as lgb  # noqa: E402

from llm.benchmark.live_page import (  # noqa: E402
    detector_evidence_ready,
    render_live_page,
)
from llm.benchmark.menu import build_menu  # noqa: E402
from llm.hybrid.calibration import apply_calibrator  # noqa: E402
from llm.hybrid.menu_rule import menu_rule  # noqa: E402
from scripts.evidence_classifier import TAGS, prompt_features  # noqa: E402

ART = ROOT / "outputs/analysis/phase3a/hybrid/protocolA"
DB = ROOT / "outputs/events.db"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
#: Stored benchmark pages carry 19-39 populated features (mean 28.4).
STORED_FLOOR = 25
FLAG_LOW, FLAG_HIGH = 18, 20
#: The paper's cross-fitted abstention threshold; the shipped Protocol A artefact carries 1.0.
PAPER_THRESHOLD = 0.9792
VOCABULARY = set(TAGS)


def load_events(limit: int) -> list[dict]:
    con = sqlite3.connect(DB)
    rows = [json.loads(value) for (value,) in
            con.execute("select evidence_json from events order by rowid desc limit ?", (limit,))]
    con.close()
    return rows


def listed_tags(page: str) -> list[str]:
    return re.findall(r"^- .+? \[(X(?:MEAS|MV)_\d{1,2}), [^\]]+\]: z=", page, re.M)


def section_counts(page: str) -> dict:
    return {
        "sensors": len(listed_tags(page)),
        "in_vocabulary": sum(1 for tag in listed_tags(page) if tag in VOCABULARY),
        "onsets": len(re.findall(r"^\d+\. .+ at \+[\d.]+ min", page, re.M)),
        "residual": page.count("residual share"),
        "cva": "CVA Q-statistic" in page,
        "fused": "Fused trigger" in page,
    }


def cause_of(row: dict) -> str:
    if not row["has_detector"]:
        return "LEGACY_NO_DETECTOR"
    if row["page"]["in_vocabulary"] < row["page"]["sensors"]:
        return "VOCAB_MISMATCH"
    if row["trend_nonzero"] == 0 and row["onset_nonzero"] == 0:
        return "ZERO_ENCODED"
    return "OTHER"


def block_counts(vector: np.ndarray, names: list[str]) -> dict:
    suffix = {"trend_nonzero": ":trend", "onset_nonzero": ":onset", "residual_nonzero": ":residual",
              "present_nonzero": ":present"}
    return {key: int(np.count_nonzero([vector[i] for i, name in enumerate(names)
                                       if name.endswith(tail)])) for key, tail in suffix.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=489)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    booster = lgb.Booster(model_file=str(ART / "classifier.txt"))
    calibration = json.loads((ART / "calibration.json").read_text(encoding="utf-8"))
    labels = calibration["labels"]
    artefact_threshold = float(json.loads(
        (ART / "threshold.json").read_text(encoding="utf-8"))["threshold"])
    menu_ids = [entry["fault_id"] for entry in build_menu()]

    from scripts.evidence_classifier import feature_names
    names = feature_names()

    events = load_events(args.limit)
    pages = [render_live_page(event) for event in events]
    features = np.asarray([prompt_features(page) for page in pages], dtype=np.float64)
    # The guard: events with no detector score never reach the classifier.
    ready = [detector_evidence_ready(event) for event in events]
    ready_index = [index for index, flag in enumerate(ready) if flag]
    if ready_index:
        raw = booster.predict(features[ready_index])
        raw_rows = [{label: float(value) for label, value in zip(labels, row)} for row in raw]
        calibrated = dict(zip(ready_index, apply_calibrator(calibration, raw_rows)))
    else:
        calibrated = {}

    results = []
    for index, (event, page) in enumerate(zip(events, pages)):
        prob = calibrated.get(index)
        top = max(prob, key=prob.get) if prob else None
        blocks = block_counts(features[index], names)
        results.append({
            "event": str(event.get("event_id") or index),
            "classified": prob is not None,
            "top": top, "confidence": float(prob[top]) if prob else None,
            "has_detector": bool(event.get("detector_evidence")),
            "populated": int(np.count_nonzero(features[index])),
            "page": section_counts(page),
            **blocks,
        })

    scored = [row for row in results if row["classified"]]
    decisions = {}
    for threshold in (artefact_threshold, PAPER_THRESHOLD):
        rows = [menu_rule(row["top"], row["confidence"], menu_ids, threshold) for row in scored]
        non_abstained = [row for row, decision in zip(scored, rows) if not decision["abstained"]]
        decisions[f"{threshold:.4f}"] = {
            "threshold": threshold,
            "abstentions": sum(1 for decision in rows if decision["abstained"]),
            "reasons": dict(Counter(decision["reason"] for decision in rows)),
            "diagnoses": dict(Counter(decision["diagnosis"] for decision in rows).most_common()),
            "argmax_matches_rule": sum(1 for row, decision in zip(scored, rows)
                                       if decision["diagnosis"] == row["top"]),
            "argmax_matches_rule_non_abstained": sum(
                1 for row, decision in zip(scored, rows)
                if not decision["abstained"] and decision["diagnosis"] == row["top"]),
            "non_abstained": len(non_abstained),
        }

    for row in results:
        row["cause"] = cause_of(row)

    populated = [row["populated"] for row in results]
    legacy = [row for row in results if not row["has_detector"]]
    bearing = [row for row in results if row["has_detector"]]
    flagged = [row for row in results if FLAG_LOW <= row["populated"] <= FLAG_HIGH]

    val_pages = [json.loads(line)["prompt"] for line in
                 VAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    stored_loss = sum(len(tags) - sum(1 for tag in tags if tag in VOCABULARY)
                      for tags in (listed_tags(page) for page in val_pages))
    stored_total = sum(len(listed_tags(page)) for page in val_pages)
    live_loss = sum(row["page"]["sensors"] - row["page"]["in_vocabulary"] for row in results)
    live_total = sum(row["page"]["sensors"] for row in results)

    summary = {
        "events": len(results),
        "thresholds": {"artefact": artefact_threshold, "paper": PAPER_THRESHOLD},
        "populated": {"mean": round(float(np.mean(populated)), 2), "min": int(min(populated)),
                      "max": int(max(populated))},
        "cohorts": {
            "legacy_no_detector": {"n": len(legacy),
                                   "populated_mean": round(float(np.mean([r["populated"] for r in legacy])), 2)
                                   if legacy else None,
                                   "range": [min(r["populated"] for r in legacy),
                                             max(r["populated"] for r in legacy)] if legacy else None},
            "detector_bearing": {"n": len(bearing),
                                 "populated_mean": round(float(np.mean([r["populated"] for r in bearing])), 2)
                                 if bearing else None,
                                 "range": [min(r["populated"] for r in bearing),
                                           max(r["populated"] for r in bearing)] if bearing else None},
        },
        "vocabulary_loss": {
            "stored": f"{stored_loss}/{stored_total} listed sensors have no column "
                      f"({stored_loss / max(stored_total, 1):.1%})",
            "live": f"{live_loss}/{live_total} listed sensors have no column "
                    f"({live_loss / max(live_total, 1):.1%})",
        },
        "classified": len(scored),
        "refused_incomplete_evidence": len(results) - len(scored),
        "answers": dict(Counter(row["top"] for row in scored).most_common()),
        "confidence": {"mean": round(float(np.mean([r["confidence"] for r in scored])), 4),
                       "min": round(min(r["confidence"] for r in scored), 4),
                       "max": round(max(r["confidence"] for r in scored), 4)} if scored else {},
        "decisions": decisions,
        "flagged": {"count": len(flagged),
                    "causes": dict(Counter(row["cause"] for row in flagged)),
                    "populated_histogram": dict(sorted(Counter(row["populated"] for row in flagged).items()))},
        "flags": [{key: row[key] for key in ("event", "top", "confidence", "populated", "page", "cause")}
                  for row in flagged],
    }

    printed = {key: value for key, value in summary.items() if key != "flags"}
    print(json.dumps(printed, indent=2))
    print("\nflagged events (18-20 populated features)")
    header = "%-14s %-24s %-6s %-4s %-7s %-6s %-8s %s" % (
        "event", "answer", "conf", "pop", "sensors", "invoc", "trend/onset", "cause")
    print(header)
    print("-" * len(header))
    for row in flagged:
        print("%-14s %-24s %-6s %-4d %-7d %-6d %-8s %s" % (
            row["event"][-14:], row["top"] or "(refused)",
            f'{row["confidence"]:.3f}' if row["confidence"] is not None else "-", row["populated"],
            row["page"]["sensors"], row["page"]["in_vocabulary"],
            f'{row["trend_nonzero"]}/{row["onset_nonzero"]}', row["cause"]))

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
