"""Why does FAULT_15 dominate the live verdicts — distribution shift, windowing, or scoring?

Three competing explanations, each with a signature in the stored records:

* **shift** — the live events come from a different operating regime, so the class prior differs;
* **windowing** — a handful of physical episodes produce many overlapping monitor windows, so one incident is
  counted many times;
* **scoring** — the model keys on a feature block (CVA lag contributions) rather than on the sensors the fault
  actually moves.

The script prints the evidence for each and deliberately does not conclude on its own: there are no live labels,
so a causal verdict is not derivable here, only which explanations the records are consistent with.

    python scripts/diagnose_fault15_skew.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_event import DEFAULT_ARTIFACTS, DEFAULT_DB, reconstruct  # noqa: E402
from llm.hybrid.menu_rule import menu_rule  # noqa: E402
from main import TEPApp  # noqa: E402
from utils import load_config  # noqa: E402

#: Sensors that a condenser-cooling fault moves directly. FAULT_05/12/15 act on the cooling water; if the
#: model is right, these should appear among the leading deviations.
CONDENSER_TAGS = {"XMEAS_22", "XMV_11", "XMEAS_21"}
FEED_TAGS = {"XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4", "XMV_42", "XMV_44"}


def _tags(sensor: dict) -> set[str]:
    name = str(sensor.get("name") or "")
    display = str(sensor.get("display_name") or "")
    found = {tag for tag in CONDENSER_TAGS | FEED_TAGS if tag == name}
    if not found:
        lowered = display.lower()
        if "condenser_cooling" in lowered:
            found.add("condenser")
        elif "feed" in lowered:
            found.add("feed")
    return found or {"other"}


def load_rows(app: TEPApp, db: pathlib.Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        payloads = [p for (p,) in con.execute("SELECT event_json FROM events ORDER BY rowid").fetchall()]
    finally:
        con.close()
    rows = []
    for payload in payloads:
        record = json.loads(payload)
        evidence_payload = record.get("evidence") or {}
        if evidence_payload.get("detector_evidence") is None:
            continue
        event = reconstruct(record)
        try:
            probabilities = app.hybrid.probabilities(event.evidence.to_dict())
        except Exception:
            continue
        predicted = max(probabilities, key=probabilities.get)
        decision = menu_rule(predicted, float(probabilities[predicted]), app.hybrid.menu_ids,
                             app.hybrid.threshold)
        top = (evidence_payload.get("top_anomalous_sensors") or [])[:5]
        tags = set().union(*[_tags(sensor) for sensor in top]) if top else {"other"}
        detector = evidence_payload.get("detector_evidence") or {}
        rows.append({
            "event_id": event.event_id,
            "predicted": predicted,
            "confidence": float(probabilities[predicted]),
            "verdict": decision["verdict"],
            "start": datetime.fromisoformat(record["start_time"]),
            "start_sample": record.get("start_sample"),
            "end_sample": record.get("end_sample"),
            "triggered_by": detector.get("triggered_by"),
            "evidence": event.evidence.to_dict(),
            "change_type": detector.get("change_type"),
            "tags": tags,
            "cva_top": sorted(
                ((detector.get("dynamic") or {}).get("cva") or {}).get("sensor_contributions") or [],
                reverse=True)[:3],
        })
    return rows


def clusters(rows: list[dict], gap: timedelta = timedelta(minutes=10)) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda row: row["start"])
    groups: list[list[dict]] = []
    for row in ordered:
        if groups and row["start"] - groups[-1][-1]["start"] <= gap:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


LSTM_LINE = re.compile(
    r"LSTM autoencoder reconstruction error ([\d.]+) \(threshold ([\d.]+), ratio [\d.]+x, alarmed (True|False)\)")
CVA_LINE = re.compile(
    r"CVA Q-statistic ([\d.]+) \(threshold ([\d.]+), ratio [\d.]+x, alarmed (True|False)\)")


def prompt_trigger(prompt: str) -> tuple[bool, bool] | None:
    """``(lstm_alarmed, cva_alarmed)`` from a benchmark page, or None if the page has no detector block."""
    lstm = LSTM_LINE.search(prompt)
    cva = CVA_LINE.search(prompt)
    if not lstm or not cva:
        return None
    return lstm.group(3) == "True", cva.group(3) == "True"


def trigger_label(flags: tuple[bool, bool]) -> str:
    lstm, cva = flags
    if lstm and cva:
        return "both"
    if cva:
        return "cva-only (dynamics)"
    if lstm:
        return "lstm-only"
    return "neither"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    parser.add_argument("--artifacts", type=pathlib.Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()

    config = load_config()
    config["hybrid"] = {"enabled": True, "artifact_dir": str(args.artifacts)}
    app = TEPApp(config, enable_llm=False)
    rows = load_rows(app, args.db)

    fifteen = [row for row in rows if row["predicted"] == "FAULT_15"]
    named = [row for row in rows if row["predicted"].startswith("FAULT_")]
    others = [row for row in named if row["predicted"] != "FAULT_15"]

    print(f"events with usable scores        {len(rows)}")
    print(f"FAULT_15 argmax                  {len(fifteen)} ({len(fifteen)/len(rows):.1%})")
    print(f"other named faults               {len(others)}")

    print("\n-- windowing --")
    groups = clusters(fifteen)
    print(f"FAULT_15 predicted events        {len(fifteen)}")
    print(f"episodes (10-minute gap)         {len(groups)}  sizes {[len(g) for g in groups]}")
    print(f"distinct start samples           {len({r['start_sample'] for r in fifteen})}")
    print(f"time span                        {min(r['start'] for r in fifteen)} .. {max(r['start'] for r in fifteen)}")
    for group in groups[:8]:
        print(f"   {group[0]['start']}  n={len(group):2d}  samples "
              f"{group[0]['start_sample']}..{group[-1]['end_sample']}")

    print("\n-- trigger composition --")
    for label, subset in (("FAULT_15", fifteen), ("other named", others)):
        triggers = Counter((r["triggered_by"], r["change_type"]) for r in subset)
        print(f"{label:12s} n={len(subset):3d}  {dict(triggers.most_common())}")

    print("\n-- leading sensors --")
    for label, subset in (("FAULT_15", fifteen), ("other named", others)):
        tag_counts = Counter(tag for r in subset for tag in r["tags"])
        condenser_share = sum(1 for r in subset if {"XMEAS_22", "XMV_11", "XMEAS_21", "condenser"} & r["tags"])
        feed_share = sum(1 for r in subset if {"XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4", "XMV_42", "XMV_44",
                                               "feed"} & r["tags"])
        print(f"{label:12s} condenser-sensor in top5 {condenser_share}/{len(subset)}  "
              f"feed-sensor in top5 {feed_share}/{len(subset)}  tags {dict(tag_counts.most_common(6))}")

    print("\n-- confidence --")
    for label, subset in (("FAULT_15", fifteen), ("other named", others)):
        values = sorted(r["confidence"] for r in subset)
        print(f"{label:12s} median {values[len(values)//2]:.4f}  min {values[0]:.4f}  "
              f"at 1.0 {sum(1 for v in values if v == 1.0)}/{len(values)}")

    print("\n-- benchmark base rate (validation split) --")
    preds_path = pathlib.Path(args.artifacts) / "preds_protocolA_val.jsonl"
    preds = [json.loads(line) for line in preds_path.read_text(encoding="utf-8").splitlines()]
    menu_rows = [row for row in preds if row.get("variant") == "menu"]
    gold = Counter(row.get("gold") for row in menu_rows)
    predicted = Counter(row.get("classifier_label") for row in menu_rows)
    print(f"gold counts       {dict(gold.most_common())}")
    print(f"predicted counts  {dict(predicted.most_common())}")
    fifteen_gold = [row for row in menu_rows if row.get("gold") == "FAULT_15"]
    if fifteen_gold:
        confusion = Counter(row.get("classifier_label") for row in fifteen_gold)
        print(f"gold FAULT_15 (n={len(fifteen_gold)}) predicted as {dict(confusion.most_common())}")

    print("\n-- benchmark trigger composition (does the live page type exist in training?) --")
    for name in ("train", "val"):
        path = ROOT / f"outputs/benchmark_v2/dataset/{name}.jsonl"
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        parsed = [(row.get("fault") or row.get("label"), prompt_trigger(row.get("prompt") or ""))
                  for row in items]
        parsed = [(gold, flags) for gold, flags in parsed if flags]
        counts = Counter(trigger_label(flags) for _, flags in parsed)
        print(f"{name:5s} n={len(parsed):5d}  {dict(counts.most_common())}")
        fifteen_pages = [(gold, flags) for gold, flags in parsed if gold in (15, "FAULT_15")]
        if fifteen_pages:
            print(f"      fault-15 pages   {dict(Counter(trigger_label(f) for _, f in fifteen_pages).most_common())}")

    print("\n-- novelty (Euclidean distance to 10 nearest training events) --")
    try:
        import numpy as np

        from llm.hybrid import calibration as cal
        import hybrid_crossfit as hc

        train_matrix, _train_labels, _train_items = hc.dataset(hc.TRAIN, dedupe=False)
        live_pages = [app.hybrid.page(row["evidence"]) for row in rows]
        from scripts.evidence_classifier import prompt_features
        live_matrix = np.asarray([prompt_features(page) for page in live_pages], dtype=np.float64)
        live_novelty = cal.novelty_scores(live_matrix, train_matrix, k=10)
        val_rows = [json.loads(line) for line in preds_path.with_name("crossfit_val_rows.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
        val_novelty = [row["signals"]["novelty"] for row in val_rows if row.get("signals")]

        def quantiles(values):
            ordered = sorted(float(v) for v in values)
            at = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
            return {"p50": round(at(0.5), 4), "p90": round(at(0.9), 4), "p99": round(at(0.99), 4),
                    "max": round(ordered[-1], 4)}

        print(f"benchmark val   n={len(val_novelty)}  {quantiles(val_novelty)}")
        print(f"live            n={len(live_novelty)}  {quantiles(live_novelty)}")
        threshold = sorted(val_novelty)[int(0.99 * (len(val_novelty) - 1))]
        above = sum(1 for value in live_novelty if value > threshold)
        print(f"live above val p99 ({threshold:.4f}): {above}/{len(live_novelty)} ({above/len(live_novelty):.1%})")
        dynamics_live = [value for value, row in zip(live_novelty, rows)
                         if row["triggered_by"] == "dynamic"]
        if dynamics_live:
            print(f"live dynamics-only pages n={len(dynamics_live)}  {quantiles(dynamics_live)}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"novelty unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
