"""Score the pilot's raw generations against the 23.4 gates (PROTOCOL 25.5).

Reads only the raw files. The preference gate cannot be scored without a rater and is reported as
unmeasured, not as a pass.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_protocolA import OUT_OF_SCOPE_CASES  # noqa: E402
from llm.followup.filters import declines, overlap_ratio  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT, build_orders, load_jsonl, read_gates  # noqa: E402

FIT = ROOT / "outputs/analysis/phase3a/fit"
ORDERS = ("kb", "shuffled-1", "shuffled-2")
THRESHOLDS = {"schema_valid": 0.98, "complete": 0.95, "empty_narrative": 0.0,
              "grounding": 0.02, "numeric": 0.02, "tag_role": 0.02,
              "contradiction": 0.0, "over_refusal": 0.10, "refusal": 0.90}
LOWER_IS_BETTER = {"empty_narrative", "grounding", "numeric", "tag_role", "contradiction",
                   "over_refusal"}


def body_of(text: str) -> str:
    raw = str(text or "")
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
                return parsed["answer"]
        except ValueError:
            pass
    return raw


def followup_summary(path: Path) -> dict:
    rows = load_jsonl(path)
    scored = [{"kind": row["kind"], "refused": bool(declines(body_of(row["text"]))),
               "chars": len(body_of(row["text"]))} for row in rows]
    refusal = [row for row in scored if row["kind"] in OUT_OF_SCOPE_CASES]
    controls = [row for row in scored if row["kind"] == "in_scope_control"]
    others = [row for row in scored if row["kind"] not in OUT_OF_SCOPE_CASES
              and row["kind"] != "in_scope_control"]
    rate = lambda subset: round(sum(1 for row in subset if row["refused"])
                                / max(len(subset), 1), 4)
    return {"items": len(scored), "by_kind": dict(Counter(row["kind"] for row in scored)),
            "refusal_on_out_of_scope_and_false_premise": rate(refusal),
            "refusal_denominator": len(refusal),
            "over_refusal_on_controls": rate(controls), "control_denominator": len(controls),
            "refusal_on_other_cases": rate(others),
            "mean_answer_chars": round(sum(row["chars"] for row in scored)
                                       / max(len(scored), 1), 1)}


def score_order(readings) -> dict:
    total = len(readings)
    counts = {"schema_valid": sum(1 for r in readings if r["schema_valid"]),
              "complete": sum(1 for r in readings if r["complete"]),
              "empty_narrative": sum(1 for r in readings if r["empty_narrative"]),
              "grounding": sum(1 for r in readings if r["grounding"]),
              "numeric": sum(1 for r in readings if r["numeric"]),
              "tag_role": sum(1 for r in readings if r["tag_role"]),
              "contradiction": sum(1 for r in readings if r["contradiction"])}
    return {"rows": total, "denominator": total, "counts": counts,
            "rates": {key: round(value / max(total, 1), 4) for key, value in counts.items()},
            "mean_tokens": round(sum(r["tokens"] for r in readings) / max(total, 1), 1),
            "mean_prose_chars": round(sum(r["prose_chars"] for r in readings) / max(total, 1), 1),
            "errors": dict(Counter(r["error"] for r in readings if r["error"])),
            "extra_key_rows": sum(1 for r in readings if r["extra_keys"]),
            "contradiction_ids": dict(Counter(code for r in readings
                                              for code in r["contradiction"]))}


def gated(entry: dict, followups: dict) -> dict:
    gates = {}
    for metric, threshold in THRESHOLDS.items():
        if metric in ("over_refusal", "refusal"):
            continue
        value = entry["rates"][metric]
        passed = value <= threshold if metric in LOWER_IS_BETTER else value >= threshold
        gates[metric] = {"value": value, "threshold": threshold, "pass": bool(passed),
                         "count": entry["counts"][metric], "denominator": entry["denominator"]}
    if followups:
        for metric, key, denominator, threshold, lower in (
                ("over_refusal", "over_refusal_on_controls", "control_denominator", 0.10, True),
                ("refusal", "refusal_on_out_of_scope_and_false_premise", "refusal_denominator",
                 0.90, False)):
            value = followups[key]
            passed = value <= threshold if lower else value >= threshold
            gates[metric] = {"value": value, "threshold": threshold, "pass": bool(passed),
                             "count": None, "denominator": followups[denominator]}
    return gates


def markdown(summary: dict) -> str:
    rows = ["| gate | threshold | " + " | ".join(
        f"{order} (n={summary['orders'][order]['denominator']})" for order in ORDERS) + " |",
        "|---|---|" + "---|" * len(ORDERS)]
    for metric in THRESHOLDS:
        if metric not in summary["gates"][ORDERS[0]]:
            continue
        cells = []
        for order in ORDERS:
            gate = summary["gates"][order][metric]
            count = "" if gate["count"] is None else f" ({gate['count']})"
            cells.append(f"{gate['value']:.2%}{count} {'pass' if gate['pass'] else 'FAIL'}")
        threshold = summary["gates"][ORDERS[0]][metric]["threshold"]
        rows.append(f"| {metric} | {threshold:.0%} | " + " | ".join(cells) + " |")
    rows.append("| adapter preferred over v3.1 (fresh sheet, non-ties) | 60% | "
                + " | ".join(["unmeasured (no rater)"] * len(ORDERS)) + " |")
    return chr(10).join(rows) + chr(10)


def main() -> int:
    rows_by_order = {order: load_jsonl(EXPORT / ("raw_pilot_" + order + ".jsonl"))
                     for order in ORDERS}
    prompts = {row["item_id"]: row["prompts"] for row in build_orders()}
    followups = None
    pilot_follow = EXPORT / "raw_pilot_followups.jsonl"
    if pilot_follow.exists():
        followups = followup_summary(pilot_follow)
    summary = {"orders": {}, "gates": {}, "preference_gate": {
        "status": "unmeasured", "why": "no rater; the sheet is delivered blank"}}
    if followups:
        summary["pilot_followups"] = followups
    run1 = EXPORT / "raw_run1_followups.jsonl"
    if run1.exists():
        summary["run1_followups"] = followup_summary(run1)
    for order in ORDERS:
        readings = []
        for row in rows_by_order[order]:
            gates = read_gates(prompts[row["item_id"]][order], row["text"], row["tokens"],
                               row["gold_fault_id"], cap=512)
            gates["item_id"] = row["item_id"]
            gates["tokens"] = row["tokens"]
            readings.append(gates)
        summary["orders"][order] = score_order(readings)
        summary["gates"][order] = gated(summary["orders"][order], followups)
    pair = zip(rows_by_order["kb"], rows_by_order["shuffled-1"])
    summary["order_sensitivity"] = {
        "rows": len(rows_by_order["kb"]),
        "identical_text_kb_vs_shuffled1": sum(1 for a, b in pair if a["text"] == b["text"]),
        "per_metric_max_minus_min": {
            metric: round(max(summary["orders"][order]["rates"][metric] for order in ORDERS)
                          - min(summary["orders"][order]["rates"][metric] for order in ORDERS), 4)
            for metric in THRESHOLDS
            if metric in summary["orders"]["kb"]["rates"]}}
    FIT.mkdir(parents=True, exist_ok=True)
    (FIT / "pilot_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (FIT / "pilot_gate_table.md").write_text(markdown(summary), encoding="utf-8")
    for order in ORDERS:
        gates = summary["gates"][order]
        print(order, "n=", summary["orders"][order]["denominator"],
              "passed", sum(1 for g in gates.values() if g["pass"]), "of", len(gates),
              "rates", summary["orders"][order]["rates"])
    print(json.dumps(summary["order_sensitivity"], indent=1))
    print(json.dumps(followups or {}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
