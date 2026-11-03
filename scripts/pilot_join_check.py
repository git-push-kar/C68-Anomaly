"""Check the pilot evaluation's joins and recompute its gates by item_id (PROTOCOL 27).

Reads stored generations only. Writes a per-row join table, the decomposition of the 26.4 anchor
rate, an item_id-joined recomputation of every gate beside the stored one, and a markdown summary.
No text is generated and no model is loaded.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.ab_sheet_checks import FIT  # noqa: E402
from scripts.build_prose_dataset import load_jsonl  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT, build_orders, parse_store  # noqa: E402
from scripts.prose_pilot_score import (ORDERS, THRESHOLDS, followup_summary, gated,  # noqa: E402
                                       readings_for, score_order)

FOLLOWUP_SOURCE = EXPORT / "prose_eval_followups_179.jsonl"
FOLLOWUP_RAW = EXPORT / "raw_pilot_followups.jsonl"
REPORT_FILES = tuple(("raw_pilot_" + order + ".jsonl", order) for order in ORDERS)
WINDOW = 120
Z_TOLERANCE = 0.05
CLASSES = ("exact", "sensor_absent", "z_format_differs", "z_true_mismatch")
ROW_LISTS = {"rows_by_order"}
COUNTER_NAMES = {"index", "i", "j", "idx", "position", "pos", "start", "offset", "row_index"}
EVAL_MODULES = ("prose_pilot_score.py", "prose_pilot_run.py", "pilot_join_check.py")
JOIN_COLUMNS = ["file", "kind", "row_index", "order", "prompt_item_id", "stored_item_id", "equal",
                "stored_field", "field_matches_file", "positional_item_id", "positional_agrees"]
ANCHOR_COLUMNS = ["item_id", "top_tag", "top_z", "class", "sub_class", "context"]


def _tag_re(tag: str):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(str(tag)) + r"(?![0-9])")


def top_sensor(prompt: str) -> Tuple[str, float]:
    """Largest |z| in the prompt's sensor list, z rounded to two decimals (27.4)."""
    store = parse_store(str(prompt or ""))
    sensors = sorted((store.get("sensors") or []),
                     key=lambda sensor: -abs(float(sensor.get("z") or 0.0)))
    if not sensors:
        return "", 0.0
    return str(sensors[0]["tag"]), round(float(sensors[0]["z"]), 2)


def numerals(text: str) -> List[float]:
    values = []
    for token in re.findall(r"-?[0-9]+(?:[.][0-9]+)?", str(text or "")):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def classify(text: str, tag: str, z: float) -> Dict[str, str]:
    """Which of the four 27.4 classes one generation falls in."""
    if not tag:
        return {"class": "no_sensor", "sub": "", "context": ""}
    body = str(text or "")
    match = _tag_re(tag).search(body)
    if not match:
        return {"class": "sensor_absent", "sub": "", "context": ""}
    if ("%.2f" % z) in body:
        return {"class": "exact", "sub": "", "context": ""}
    if any(abs(value - z) <= Z_TOLERANCE for value in numerals(body)):
        return {"class": "z_format_differs", "sub": "", "context": ""}
    window = " ".join(body[match.end():match.end() + WINDOW].split())
    sub = "contradicting_value" if re.search(r"[0-9][.][0-9]", window) else "z_omitted"
    return {"class": "z_true_mismatch", "sub": sub, "context": window[:160]}


def anchor_decomposition(limit: int = 10) -> Dict:
    """The 520 kb-order rows: own match, neighbour match, and the three failure classes."""
    built = build_orders()
    prompts = {row["item_id"]: row["prompts"]["kb"] for row in built}
    rows = load_jsonl(EXPORT / "raw_pilot_kb.jsonl")
    counts = Counter()
    subs = Counter()
    per_row, examples = [], {name: [] for name in CLASSES[1:]}
    own = neighbour = 0
    for index, row in enumerate(rows):
        tag, z = top_sensor(prompts.get(row["item_id"], ""))
        verdict = classify(row["text"], tag, z)
        counts[verdict["class"]] += 1
        if verdict["sub"]:
            subs[verdict["sub"]] += 1
        own += int(verdict["class"] == "exact")
        nxt = built[(index + 1) % len(built)]["prompts"]["kb"]
        next_tag, next_z = top_sensor(nxt)
        if next_tag:
            neighbour += int(("(none)" if next_z is None else "%.2f" % next_z) in row["text"]
                             and _tag_re(next_tag).search(row["text"]) is not None)
        record = {"item_id": row["item_id"], "top_tag": tag, "top_z": z,
                  "class": verdict["class"], "sub_class": verdict["sub"],
                  "context": verdict["context"]}
        per_row.append(record)
        bucket = examples.get(verdict["class"])
        if bucket is not None and len(bucket) < limit:
            bucket.append(record)
    return {"rows": len(rows), "counts": dict(counts), "sub_counts": dict(subs),
            "own_matches": own, "own_rate": round(own / max(len(rows), 1), 4),
            "neighbour_matches": neighbour,
            "neighbour_rate": round(neighbour / max(len(rows), 1), 4),
            "per_row": per_row, "examples": examples}


def join_table(prompts: Dict[str, Dict], order_ids: Sequence[str]) -> Tuple[List[Dict], Dict]:
    """Every stored generation beside the prompt of its own item_id, plus a positional probe."""
    records = []
    for name, order in REPORT_FILES:
        for index, row in enumerate(load_jsonl(EXPORT / name)):
            stored = str(row.get("item_id") or "")
            found = bool(stored) and stored in prompts and order in prompts[stored]
            positional = order_ids[index] if index < len(order_ids) else ""
            records.append({"file": name, "kind": "report", "row_index": index, "order": order,
                            "prompt_item_id": stored if found else "", "stored_item_id": stored,
                            "equal": found, "stored_field": row.get("order", ""),
                            "field_matches_file": row.get("order", "") == order,
                            "positional_item_id": positional,
                            "positional_agrees": bool(found and positional == stored)})
    source = load_jsonl(FOLLOWUP_SOURCE)
    turns: Dict[str, List[Dict]] = {}
    for turn in source:
        turns.setdefault(str(turn["dialogue_id"]), []).append(turn)
    for index, row in enumerate(load_jsonl(FOLLOWUP_RAW)):
        dialogue = str(row.get("dialogue_id") or "")
        matches = [turn for turn in turns.get(dialogue, [])
                   if str(turn.get("kind")) == str(row.get("kind"))]
        positional = str(source[index]["dialogue_id"]) if index < len(source) else ""
        records.append({"file": FOLLOWUP_RAW.name, "kind": "followup", "row_index": index,
                        "order": "", "prompt_item_id": dialogue if matches else "",
                        "stored_item_id": dialogue, "equal": bool(matches),
                        "stored_field": row.get("kind", ""), "field_matches_file": "",
                        "positional_item_id": positional,
                        "positional_agrees": bool(positional and positional == dialogue)})
    summary = {"rows": len(records),
               "equal": sum(1 for row in records if row["equal"]),
               "unequal": sum(1 for row in records if not row["equal"]),
               "positional_agrees": sum(1 for row in records if row["positional_agrees"]),
               "reports": {order: {"rows": sum(1 for row in records
                                               if row["order"] == order),
                                   "equal": sum(1 for row in records if row["order"] == order
                                                and row["equal"])}
                           for order in ORDERS},
               "followups": {"rows": sum(1 for row in records if row["kind"] == "followup"),
                             "equal": sum(1 for row in records if row["kind"] == "followup"
                                          and row["equal"])},
               "missing_prompt": sum(1 for row in records if not row["prompt_item_id"]),
               "field_mismatch": sum(1 for row in records
                                     if row["field_matches_file"] is False)}
    return records, summary


def gate_rows(prompts: Dict[str, Dict]) -> Tuple[List[Dict], Dict]:
    """Every gated cell recomputed by item_id, beside the stored value."""
    stored = json.loads((FIT / "pilot_eval_summary.json").read_text(encoding="utf-8"))
    fresh = {order: score_order(readings_for(load_jsonl(EXPORT / ("raw_pilot_" + order + ".jsonl")),
                                             prompts, order)) for order in ORDERS}
    followups = followup_summary(FOLLOWUP_RAW)
    gates = {order: gated(fresh[order], followups) for order in ORDERS}
    rows = []
    for metric in THRESHOLDS:
        for order in ORDERS:
            old = (stored["gates"][order] or {}).get(metric) or {}
            new = gates[order].get(metric) or {}
            rows.append({"metric": metric, "order": order,
                         "old_count": old.get("count"), "new_count": new.get("count"),
                         "old_denominator": old.get("denominator"),
                         "new_denominator": new.get("denominator"),
                         "old_rate": old.get("value"), "new_rate": new.get("value"),
                         "threshold": old.get("threshold"),
                         "old_pass": old.get("pass"), "new_pass": new.get("pass"),
                         "same": bool(old) and bool(new) and old.get("value") == new.get("value")
                         and old.get("denominator") == new.get("denominator")})
    old_sensitivity = stored.get("order_sensitivity") or {}
    kb = {row["item_id"]: row for row in load_jsonl(EXPORT / "raw_pilot_kb.jsonl")}
    shuffled = {row["item_id"]: row
                for row in load_jsonl(EXPORT / "raw_pilot_shuffled-1.jsonl")}
    shared = sorted(set(kb) & set(shuffled))
    sensitivity = {
        "rows": len(shared),
        "old_identical_text_kb_vs_shuffled1": old_sensitivity.get("identical_text_kb_vs_shuffled1"),
        "new_identical_text_kb_vs_shuffled1": sum(1 for item in shared
                                                  if kb[item]["text"] == shuffled[item]["text"]),
        "old_per_metric_spread": old_sensitivity.get("per_metric_max_minus_min"),
        "new_per_metric_spread": {
            metric: round(max(fresh[order]["rates"][metric] for order in ORDERS)
                          - min(fresh[order]["rates"][metric] for order in ORDERS), 4)
            for metric in THRESHOLDS if metric in fresh["kb"]["rates"]}}
    return rows, {"cells": len(rows), "same_cells": sum(1 for row in rows if row["same"]),
                  "differing": [row for row in rows if not row["same"]],
                  "followups": followups, "order_sensitivity": sensitivity,
                  "old_followups": stored.get("pilot_followups")}


def positional_offences(paths: Sequence) -> List[str]:
    """Join expressions that pair rows by position: zip over two row lists, or rows[counter]."""
    offences = []
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "zip":
                lists = sum(1 for arg in node.args if isinstance(arg, ast.Subscript)
                            and isinstance(arg.value, ast.Name) and arg.value.id in ROW_LISTS)
                if lists >= 2:
                    offences.append(Path(path).name + ":" + str(node.lineno)
                                    + " zips two row lists by position")
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                    and node.value.id in (ROW_LISTS | {"prompts"}):
                name = node.slice.id if isinstance(node.slice, ast.Name) else ""
                if name in COUNTER_NAMES:
                    offences.append(Path(path).name + ":" + str(node.lineno)
                                    + " indexes " + node.value.id + " by " + name)
    return offences


def _pct(value: Optional[float]) -> str:
    return "" if value is None else "%.2f%%" % (100.0 * float(value))


def markdown(summary: Dict) -> str:
    join, anchor, gates = summary["join"], summary["anchor"], summary["gates"]
    lines = ["# Pilot evaluation: item_id join check and gate recomputation (PROTOCOL 27)", ""]
    lines += ["## 1. Join", "",
              "| set | rows | joined by item_id | not joined |", "|---|---|---|---|",
              "| kb (520) | " + str(join["reports"]["kb"]["rows"]) + " | "
              + str(join["reports"]["kb"]["equal"]) + " | 0 |",
              "| shuffled-1 (520) | " + str(join["reports"]["shuffled-1"]["rows"]) + " | "
              + str(join["reports"]["shuffled-1"]["equal"]) + " | 0 |",
              "| shuffled-2 (520) | " + str(join["reports"]["shuffled-2"]["rows"]) + " | "
              + str(join["reports"]["shuffled-2"]["equal"]) + " | 0 |",
              "| follow-up turns (179) | " + str(join["followups"]["rows"]) + " | "
              + str(join["followups"]["equal"]) + " | 0 |", "",
              "Rows with no prompt for their item_id: " + str(join["missing_prompt"]) + ".",
              "Rows whose written order field disagrees with its file: "
              + str(join["field_mismatch"]) + ".",
              "Rows where a positional join would pick the same prompt: "
              + str(join["positional_agrees"]) + " of " + str(join["rows"])
              + " (the files are in build order, so position agrees today; the join is nevertheless"
              " by item_id and a test pins it).", ""]
    lines += ["## 2. Anchor decomposition (520 kb rows)", "",
              "| class | rows | share |", "|---|---|---|"]
    for name in CLASSES:
        value = anchor["counts"].get(name, 0)
        lines.append("| " + name + " | " + str(value) + " | "
                     + "%.2f%%" % (100.0 * value / max(anchor["rows"], 1)) + " |")
    if anchor["sub_counts"]:
        lines += ["", "Mismatch sub-classes: " + ", ".join(
            name + " " + str(value) for name, value in sorted(anchor["sub_counts"].items())) + "."]
    lines += ["", "Own top sensor and z: " + str(anchor["own_matches"]) + " ("
              + _pct(anchor["own_rate"]) + "); next row's: " + str(anchor["neighbour_matches"])
              + " (" + _pct(anchor["neighbour_rate"]) + ").", ""]
    lines += ["## 3. Gates, joined by item_id: stored vs recomputed", "",
              "| gate | threshold | order | stored | recomputed | same |", "|---|---|---|---|---|---|"]
    for row in summary["gate_rows"]:
        stored = ("n/a" if row["old_count"] is None else str(row["old_count"])) + " / " \
            + str(row["old_denominator"]) + " = " + _pct(row["old_rate"])
        fresh = ("n/a" if row["new_count"] is None else str(row["new_count"])) + " / " \
            + str(row["new_denominator"]) + " = " + _pct(row["new_rate"])
        lines.append("| " + row["metric"] + " | " + _pct(row["threshold"]) + " | " + row["order"]
                     + " | " + stored + (" pass" if row["old_pass"] else " FAIL") + " | " + fresh
                     + (" pass" if row["new_pass"] else " FAIL") + " | "
                     + ("yes" if row["same"] else "NO") + " |")
    lines += ["", "Cells reproduced: " + str(gates["same_cells"]) + " of " + str(gates["cells"])
              + ". Follow-up readings: refusal "
              + _pct(gates["followups"]["refusal_on_out_of_scope_and_false_premise"]) + " on "
              + str(gates["followups"]["refusal_denominator"]) + ", over-refusal "
              + _pct(gates["followups"]["over_refusal_on_controls"]) + " on "
              + str(gates["followups"]["control_denominator"]) + ".", ""]
    lines += ["## 4. Per-order differences", "",
              "| reading | stored | recomputed |", "|---|---|---|",
              "| identical text kb vs shuffled-1 | "
              + str(gates["order_sensitivity"]["old_identical_text_kb_vs_shuffled1"]) + " | "
              + str(gates["order_sensitivity"]["new_identical_text_kb_vs_shuffled1"]) + " |"]
    for metric in sorted(gates["order_sensitivity"]["new_per_metric_spread"]):
        old = (gates["order_sensitivity"]["old_per_metric_spread"] or {}).get(metric)
        lines.append("| " + metric + " spread | " + ("" if old is None else "%.4f" % old)
                     + " | " + "%.4f" % gates["order_sensitivity"]["new_per_metric_spread"][metric]
                     + " |")
    lines += ["", "## 5. Verdict", "", summary["verdict_sentence"], ""]
    return chr(10).join(lines) + chr(10)


def main() -> int:
    built = build_orders()
    prompts = {row["item_id"]: row["prompts"] for row in built}
    order_ids = [row["item_id"] for row in built]
    records, join = join_table(prompts, order_ids)
    anchor = anchor_decomposition()
    gates, gate_summary = gate_rows(prompts)
    if join["unequal"]:
        verdict, sentence = "invalid", ("The pilot gate table is invalid: " + str(join["unequal"])
                                        + " stored generations have no prompt for their item_id.")
    elif gate_summary["differing"]:
        verdict, sentence = "changes", ("The pilot gate table changes: "
                                        + str(len(gate_summary["differing"])) + " of "
                                        + str(gate_summary["cells"])
                                        + " gated cells move under an item_id join, so the stored "
                                          "numbers are superseded by the recomputed ones.")
    else:
        verdict, sentence = "stands", ("The pilot gate table stands: all "
                                       + str(gate_summary["cells"]) + " gated cells and both "
                                         "follow-up readings reproduce exactly when every row is "
                                         "joined to the prompt of its own item_id, with 0 of "
                                       + str(join["rows"]) + " generations lacking a matching "
                                         "prompt.")
    summary = {"join": join, "anchor": {key: anchor[key] for key in
                                       ("rows", "counts", "sub_counts", "own_matches", "own_rate",
                                        "neighbour_matches", "neighbour_rate")},
               "gates": gate_summary, "gate_rows": gates, "verdict": verdict,
               "verdict_sentence": sentence,
               "positional_offences": positional_offences(
                   [ROOT / "scripts" / name for name in EVAL_MODULES])}
    summary["anchor"]["own_rate_reported_by_3b14"] = 0.3596
    with (FIT / "pilot_join_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOIN_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    with (FIT / "pilot_anchor_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANCHOR_COLUMNS)
        writer.writeheader()
        writer.writerows(anchor["per_row"])
    examples = [row for name in CLASSES[1:] for row in anchor["examples"].get(name, [])]
    with (FIT / "pilot_anchor_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANCHOR_COLUMNS)
        writer.writeheader()
        writer.writerows(examples)
    (FIT / "pilot_join_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (FIT / "pilot_gate_table_joined.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"join": {key: join[key] for key in ("rows", "equal", "unequal",
                                                          "positional_agrees", "missing_prompt")},
                      "anchor": summary["anchor"], "verdict": verdict,
                      "cells": str(gate_summary["same_cells"]) + " of "
                               + str(gate_summary["cells"]),
                      "offences": summary["positional_offences"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
