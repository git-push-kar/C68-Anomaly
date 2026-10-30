"""Reconcile the teacher rejection tables, audit direction rules, and size the drop.

Replay only: every number here is computed from the cache and the event files on
disk. No model is called, so this can be re-run after a filter edit to see what
the edit would do to the same answers.

Why a separate script: the two tables in the full-run report were produced by an
ad-hoc snippet that counted *codes* and printed them under an ``items`` heading.
Codes and items are not the same quantity -- one answer can fail two rules -- so
the totals did not reconcile with the cache. The definitions are stated here once
and asserted, so a table that does not reconcile fails the run instead of being
copied into a report.

Sections (``--section``):
    reconcile   non-overlapping rejection tables, first pass and final
    direction   KB initiating variable vs the trend stored in the evidence
    replay      what a proposed direction-rule change would recover or invalidate
    bias        dropped vs accepted items on detector strength, delay, |z|, variant
    policy      train class counts with the dropped items excluded
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.benchmark import filters as filters_module  # noqa: E402
from llm.benchmark.filters import check_answer  # noqa: E402
from llm.benchmark.teacher import parse_teacher_json  # noqa: E402
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from scripts.build_benchmark_dataset import (  # noqa: E402
    evidence_of,
    load_records,
)
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402
from scripts.tags import canonical_tag  # noqa: E402

DATASET = Path("outputs/benchmark_v2/dataset")
EVENTS = Path("outputs/benchmark_v2/events")
OUT = Path("outputs/analysis/phase2b_wrapup")

#: The faults whose KB "initiating" variable is a manipulated valve position
#: moved by a sticking disturbance. Kept here as a *proposal* target only; the
#: filter itself is unchanged.
VALVE_STICKING = (14, 15)


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset() -> Dict[str, List[Dict]]:
    return {split: load_jsonl(DATASET / f"{split}.jsonl")
            for split in ("train", "val", "test")}


def final_cache(filter_prefix: Optional[str] = None) -> Dict[str, Dict]:
    """One record per item: the last line written under a matching filter hash."""
    out: Dict[str, Dict] = {}
    for record in load_jsonl(DATASET / "teacher_cache.jsonl"):
        if filter_prefix and not str(record.get("filter_sha256") or "").startswith(filter_prefix):
            continue
        out[record["item_id"]] = record
    return out


def _code_type(code: str) -> str:
    return str(code).replace("retry:", "").split(":")[0]


def _family_label(family: Optional[str]) -> str:
    if not family:
        return "unknown"
    if family == "open_set":
        return "OPEN_SET"
    if family == "normal":
        return "NORMAL"
    if family.startswith("fault_"):
        return f"FAULT {family.split('_', 1)[1]}"
    return family


# -- 1. reconciliation ---------------------------------------------------------


def reconcile(items: Sequence[Dict], cache: Dict[str, Dict]) -> Dict:
    """Two non-overlapping tables: first pass and final, counted in items.

    Attribution rules, stated so the tables cannot drift:
    * one item, one family -- the family comes from the cache record, which
      labels an open-set episode ``open_set`` even though its event belongs to a
      fault; the episode is *not* also counted under that fault;
    * ``items`` counts items; ``codes`` counts rule firings and may exceed items;
    * ``primary`` counts one item under its first code only, so it sums to items.
    """
    by_item = {item["item_id"]: item for item in items}
    rows: Dict[str, Dict] = defaultdict(lambda: {
        "items": 0, "first_rejected": 0, "retried": 0, "recovered": 0,
        "dropped": 0, "dropped_no_retry": 0, "codes_first": Counter(),
        "codes_final": Counter(), "primary_first": Counter(), "primary_final": Counter(),
    })
    for item_id, record in cache.items():
        item = by_item.get(item_id)
        if item is None:
            continue
        row = rows[_family_label(record.get("family"))]
        row["items"] += 1
        first = [str(code) for code in (record.get("first_codes") or [])]
        final = [str(code) for code in (record.get("codes") or [])]
        if first:
            row["first_rejected"] += 1
            row["primary_first"][_code_type(first[0])] += 1
        if int(record.get("attempts") or 1) > 1:
            row["retried"] += 1
        for code in first:
            row["codes_first"][_code_type(code)] += 1
        if record.get("accepted"):
            if int(record.get("attempts") or 1) > 1:
                row["recovered"] += 1
        else:
            row["dropped"] += 1
            retry = [code for code in final if str(code).startswith("retry:")]
            for code in retry:
                row["codes_final"][_code_type(code)] += 1
            if retry:
                row["primary_final"][_code_type(retry[0])] += 1
            else:
                # A dropped item with no retry code was never re-asked (a single
                # rejection, not two). Counted so the tables still add up.
                row["dropped_no_retry"] += 1

    table = {name: {
        "items": row["items"],
        "first_rejected": row["first_rejected"],
        "retried": row["retried"],
        "recovered": row["recovered"],
        "dropped": row["dropped"],
        "dropped_no_retry": row["dropped_no_retry"],
        "codes_first": dict(row["codes_first"]),
        "codes_final": dict(row["codes_final"]),
        "primary_first": dict(row["primary_first"]),
        "primary_final": dict(row["primary_final"]),
    } for name, row in rows.items()}

    totals = {
        "items": sum(row["items"] for row in table.values()),
        "first_rejected_items": sum(row["first_rejected"] for row in table.values()),
        "first_codes": sum(sum(row["codes_first"].values()) for row in table.values()),
        "retried_items": sum(row["retried"] for row in table.values()),
        "recovered": sum(row["recovered"] for row in table.values()),
        "dropped": sum(row["dropped"] for row in table.values()),
        "dropped_no_retry": sum(row["dropped_no_retry"] for row in table.values()),
        "dropped_codes": sum(sum(row["codes_final"].values()) for row in table.values()),
    }
    totals["accepted"] = totals["items"] - totals["dropped"]
    totals["items_with_two_first_codes"] = totals["first_codes"] - totals["first_rejected_items"]
    totals["items_with_two_final_codes"] = (
        totals["dropped_codes"] - (totals["dropped"] - totals["dropped_no_retry"]))

    # Invariants: a table that breaks these must not be published.
    assert sum(row["primary_first"].get(k, 0) for row in table.values() for k in row["primary_first"]) \
        == totals["first_rejected_items"], "primary first-pass attribution does not sum to items"
    assert sum(row["primary_final"].get(k, 0) for row in table.values() for k in row["primary_final"]) \
        == totals["dropped"] - totals["dropped_no_retry"], \
        "primary final attribution does not sum to dropped items that were retried"
    assert totals["accepted"] + totals["dropped"] == totals["items"]
    assert totals["recovered"] + totals["dropped"] - totals["dropped_no_retry"] \
        == totals["retried_items"], "retried items must be recovered or dropped"
    return {"rows": table, "totals": totals}


# -- 2. direction audit --------------------------------------------------------


def _initiating(number: int) -> Dict:
    raw = (FAULT_KNOWLEDGE.get(int(number)) or {}).get("initiating")
    if isinstance(raw, str):
        import ast
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    return {}


def stored_trend(evidence: Dict, tag: str) -> Optional[str]:
    """Trend the evidence stores for ``tag``, or None when it was not ranked."""
    for sensor in (evidence.get("top_anomalous_sensors") or []):
        if not isinstance(sensor, dict):
            continue
        if canonical_tag(sensor.get("name")) == tag or canonical_tag(sensor.get("display_name")) == tag:
            trend = str(sensor.get("trend") or "").lower()
            if trend in ("increasing", "decreasing", "stable"):
                return trend
    return None


def direction_audit(items: Sequence[Dict], records: Dict[str, Dict]) -> Dict:
    """Per fault: the KB direction, and how often the evidence agrees with it."""
    by_fault: Dict[int, List[Dict]] = defaultdict(list)
    for item in items:
        fault = item.get("fault")
        if fault is None:
            continue
        record = records.get(item.get("event_id"))
        if record is None:
            continue
        by_fault[int(fault)].append({"item": item, "record": record})

    rows: List[Dict] = []
    for number in sorted(FAULT_KNOWLEDGE):
        entry = FAULT_KNOWLEDGE[number]
        initiating = _initiating(number)
        sensor = initiating.get("sensor")
        direction = str(initiating.get("direction") or "").lower()
        tag = canonical_tag(sensor) if sensor else None
        trends = Counter()
        for row in by_fault.get(int(number), []):
            trends[stored_trend(evidence_of(row["record"]), tag) if tag else None] += 1
        agree = trends["increasing"] if direction == "increase" else (
            trends["decreasing"] if direction == "decrease" else 0)
        observed = trends["increasing"] + trends["decreasing"]
        rows.append({
            "fault": int(number),
            "name": entry.get("name"),
            "kb_initiating": sensor,
            "kb_tag": tag,
            "kb_direction": direction,
            "tag_kind": "valve_position" if str(tag or "").startswith("XMV_") else "measurement",
            "train_events": len(by_fault.get(int(number), [])),
            "trend_counts": {str(k): v for k, v in trends.items()},
            "agree": agree,
            "observed": observed,
            "agreement_rate": round(agree / observed, 4) if observed else None,
            "absent": trends[None],
        })
    return {"faults": rows}


def direction_examples(items: Sequence[Dict], records: Dict[str, Dict],
                       cache: Dict[str, Dict], faults: Iterable[int], limit: int = 5) -> List[Dict]:
    """First-attempt rejected answers for the given faults, with stored trends."""
    wanted = {int(f) for f in faults}
    out: List[Dict] = []
    for item in items:
        if item.get("fault") not in wanted:
            continue
        record = cache.get(item["item_id"])
        if not record or not record.get("first_codes"):
            continue
        if not any(code.startswith("direction_conflict") for code in record["first_codes"]):
            continue
        event = records.get(item["event_id"])
        if event is None:
            continue
        initiating = _initiating(int(item["fault"]))
        tag = canonical_tag(initiating.get("sensor"))
        evidence = evidence_of(event)
        answer, _err = parse_teacher_json(record.get("raw_text") or "")
        out.append({
            "item_id": item["item_id"],
            "fault": int(item["fault"]),
            "kb_tag": tag,
            "kb_direction": initiating.get("direction"),
            "stored_trend": stored_trend(evidence, tag),
            "codes": record["first_codes"],
            "raw_excerpt": (str((answer or {}).get("physical_reasoning") or record.get("raw_text") or ""))[:420],
        })
        if len([row for row in out if row["fault"] == int(item["fault"])]) >= limit:
            if all(len([row for row in out if row["fault"] == f]) >= limit for f in wanted):
                break
    return out


# -- 3. replay of a proposed change -------------------------------------------


def replay_direction_change(items: Sequence[Dict], records: Dict[str, Dict],
                            cache: Dict[str, Dict], unconstrained: Sequence[int]) -> Dict:
    """What dropping the direction rule for ``unconstrained`` faults would do.

    Relaxation only, so no accepted item can be invalidated; the number is still
    computed rather than assumed. The re-check runs against the *original* prompt
    and the stored raw text, exactly as the live filter would.
    """
    unconstrained = tuple(int(f) for f in unconstrained)
    original = filters_module._direction_conflict
    recovered: List[str] = []
    invalidated: List[str] = []
    changed: List[str] = []

    def relaxed(text: str, number: Optional[int]) -> Optional[str]:
        if number is not None and int(number) in unconstrained:
            return None
        return original(text, number)

    filters_module._direction_conflict = relaxed  # type: ignore[assignment]
    try:
        for item in items:
            record = cache.get(item["item_id"])
            if not record:
                continue
            event = records.get(item["event_id"])
            if event is None:
                continue
            evidence = evidence_of(event)
            allowed = build_allowed_set(
                evidence, extra_faults=[event["fault"]] if event.get("fault") else [])
            answer, parse_error = parse_teacher_json(record.get("raw_text") or "")
            codes = check_answer(answer, item["target"], evidence, allowed,
                                 parse_error=parse_error, prompt=item.get("prompt"),
                                 menu_ids=item.get("menu_ids"))
            if record.get("accepted"):
                if codes:
                    invalidated.append(item["item_id"])
                    changed.append(item["item_id"])
            elif not codes:
                recovered.append(item["item_id"])
                changed.append(item["item_id"])
    finally:
        filters_module._direction_conflict = original  # type: ignore[assignment]

    # A stricter proposal -- flipping the KB direction instead of dropping it --
    # would keep the rule and change only its sign. Count what that invalidates.
    flipped_invalid: List[str] = []
    for item in items:
        record = cache.get(item["item_id"])
        if not record or not record.get("accepted"):
            continue
        if item.get("fault") not in unconstrained:
            continue
        event = records.get(item["event_id"])
        if event is None:
            continue
        evidence = evidence_of(event)
        allowed = build_allowed_set(
            evidence, extra_faults=[event["fault"]] if event.get("fault") else [])
        answer, parse_error = parse_teacher_json(record.get("raw_text") or "")
        flipped = dict(item["target"])
        codes = _codes_with_flipped_kb(item, answer, evidence, allowed, parse_error)
        if codes:
            flipped_invalid.append(item["item_id"])

    return {
        "unconstrained_faults": list(unconstrained),
        "recovered": len(recovered),
        "recovered_ids": recovered,
        "invalidated": len(invalidated),
        "invalidated_ids": invalidated,
        "changed": len(changed),
        "flipped_invalidated": len(flipped_invalid),
        "flipped_invalidated_ids": flipped_invalid,
    }


def replay_matrix(items: Sequence[Dict], records: Dict[str, Dict], cache: Dict[str, Dict]) -> List[Dict]:
    """Recovery/invalidation for each candidate scope of the relaxation.

    The choices are not equivalent: 14/15 are the valve-sticking faults whose KB
    direction describes a flow the tag does not measure, 7 shares that valve
    nomenclature, and the wider sets test whether the rule is wrong everywhere.
    """
    sets = (
        ("faults 14, 15 only", VALVE_STICKING),
        ("faults 7, 14, 15", (7, 14, 15)),
        ("every valve-position initiator", (1, 4, 5, 6, 7, 11, 12, 14, 15)),
        ("no direction rule at all", tuple(range(1, 23))),
    )
    rows: List[Dict] = []
    for name, faults in sets:
        result = replay_direction_change(items, records, cache, faults)
        recovered: Counter = Counter()
        for item_id in result["recovered_ids"]:
            parts = item_id.split("::")[0].split("-")
            recovered[parts[2] if len(parts) > 2 else "?"] += 1
        rows.append({
            "scope": name,
            "faults": list(faults),
            "recovered": result["recovered"],
            "invalidated": result["invalidated"],
            "flipped_invalidated": result["flipped_invalidated"],
            "recovered_by_fault": dict(sorted(recovered.items())),
        })
    return rows


def _codes_with_flipped_kb(item: Dict, answer: Optional[Dict], evidence: Dict,
                           allowed, parse_error: Optional[str]) -> List[str]:
    """Re-check with the KB direction inverted, without editing the KB."""
    number = item.get("fault")
    entry = FAULT_KNOWLEDGE.get(int(number)) if number is not None else None
    if not entry:
        return []
    original = entry.get("initiating")
    import copy
    flipped = copy.deepcopy(original)
    target = flipped[0] if isinstance(flipped, list) and flipped else flipped
    if not isinstance(target, dict):
        return []
    target["direction"] = "increase" if str(target.get("direction")).lower() == "decrease" else "decrease"
    entry_copy = dict(entry)
    entry_copy["initiating"] = [target] if isinstance(original, list) else target
    saved = FAULT_KNOWLEDGE[int(number)]
    FAULT_KNOWLEDGE[int(number)] = entry_copy
    try:
        return check_answer(answer, item["target"], evidence, allowed,
                            parse_error=parse_error, prompt=item.get("prompt"),
                            menu_ids=item.get("menu_ids"))
    finally:
        FAULT_KNOWLEDGE[int(number)] = saved


# -- 4. drop bias --------------------------------------------------------------


def _detector_ratio(evidence: Dict) -> Optional[float]:
    ratios: List[float] = []
    det = evidence.get("detector_evidence") or {}
    lstm = det.get("lstm_ae") or {}
    if lstm.get("threshold"):
        ratios.append(float(lstm.get("score") or 0.0) / float(lstm["threshold"]))
    dynamic = det.get("dynamic") or {}
    if dynamic.get("threshold"):
        ratios.append(float(dynamic.get("score") or 0.0) / float(dynamic["threshold"]))
    return max(ratios) if ratios else None


def _top_z(evidence: Dict) -> Optional[float]:
    values = [abs(float(sensor.get("z_score")))
              for sensor in (evidence.get("top_anomalous_sensors") or [])
              if isinstance(sensor, dict) and isinstance(sensor.get("z_score"), (int, float))]
    return max(values) if values else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(value) for value in values if value is not None]
    return round(statistics.median(kept), 4) if kept else None


def drop_bias(items: Sequence[Dict], records: Dict[str, Dict], cache: Dict[str, Dict]) -> Dict:
    """Compare dropped against accepted train items, per family and overall."""
    groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"accepted": [], "dropped": []})
    for item in items:
        record = cache.get(item["item_id"])
        if not record:
            continue
        event = records.get(item["event_id"])
        if event is None:
            continue
        evidence = evidence_of(event)
        row = {
            "ratio": _detector_ratio(evidence),
            "delay": event.get("event_open_delay"),
            "z": _top_z(evidence),
            "removed": item.get("variant") == "removed",
            "oracle": bool(item.get("oracle")),
        }
        key = _family_label(record.get("family"))
        groups[key]["accepted" if record.get("accepted") else "dropped"].append(row)
        groups["ALL"]["accepted" if record.get("accepted") else "dropped"].append(row)

    def summarise(rows: Sequence[Dict]) -> Dict:
        return {
            "n": len(rows),
            "median_ratio": _median(row["ratio"] for row in rows),
            "median_delay": _median(row["delay"] for row in rows),
            "median_abs_z": _median(row["z"] for row in rows),
            "removed_share": round(sum(1 for row in rows if row["removed"]) / len(rows), 4) if rows else None,
            "oracle_share": round(sum(1 for row in rows if row["oracle"]) / len(rows), 4) if rows else None,
        }

    return {name: {"accepted": summarise(rows["accepted"]),
                   "dropped": summarise(rows["dropped"])}
            for name, rows in sorted(groups.items())}


# -- 5. training-set policy ----------------------------------------------------


def train_policy(items: Sequence[Dict], cache: Dict[str, Dict]) -> Dict:
    """Class counts with dropped items removed, and the unclassified share."""
    counts = Counter()
    faulty = 0
    for item in items:
        record = cache.get(item["item_id"])
        if record is not None and not record.get("accepted"):
            continue
        label = str((item.get("target") or {}).get("fault_id") or "")
        if item.get("variant") == "removed":
            counts["UNCLASSIFIED_OPEN_SET"] += 1
        elif label == "UNCLASSIFIED_OPEN_SET":
            counts["UNCLASSIFIED_OPEN_SET"] += 1
        elif label == "NORMAL_TRANSIENT":
            counts["NORMAL_TRANSIENT"] += 1
        else:
            counts["classified"] += 1
        if item.get("fault") is not None:
            faulty += 1
    total = sum(counts.values())
    unclassified = counts["UNCLASSIFIED_OPEN_SET"]
    classified = counts["classified"]
    return {
        "items": total,
        "counts": dict(counts),
        "unclassified_share_of_all": round(unclassified / total, 4) if total else None,
        "unclassified_share_of_faulty": round(unclassified / (classified + unclassified), 4)
        if (classified + unclassified) else None,
        "faulty_items": faulty,
    }


# -- reporting -----------------------------------------------------------------


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _format_codes(codes: Dict[str, int]) -> str:
    if not codes:
        return "-"
    return ", ".join(f"{name} {count}" for name, count in
                     sorted(codes.items(), key=lambda item: (-item[1], item[0])))


def render_reconcile(result: Dict) -> str:
    rows = sorted(result["rows"].items(), key=lambda kv: (-kv[1]["first_rejected"], kv[0]))
    first = _md_table(
        ["family", "items", "rejected 1st pass", "retried", "recovered", "dropped",
         "1st-pass codes (occurrences)", "1st-pass codes by item (primary)"],
        [[name, row["items"], row["first_rejected"], row["retried"], row["recovered"],
          row["dropped"], _format_codes(row["codes_first"]), _format_codes(row["primary_first"])]
         for name, row in rows])
    final = _md_table(
        ["family", "dropped items", "retry codes (occurrences)", "retry codes by item (primary)"],
        [[name, row["dropped"], _format_codes(row["codes_final"]),
          _format_codes(row["primary_final"])]
         for name, row in rows if row["dropped"]])
    totals = result["totals"]
    text = [
        f"Items: {totals['items']}  accepted: {totals['accepted']}  dropped: {totals['dropped']}",
        f"First pass: {totals['first_rejected_items']} items rejected, "
        f"{totals['first_codes']} code occurrences "
        f"({totals['items_with_two_first_codes']} items failed two rules).",
        f"Retries: {totals['retried_items']} items, {totals['recovered']} recovered, "
        f"{totals['dropped']} dropped.",
        f"Final: {totals['dropped']} dropped items "
        f"({totals['dropped_no_retry']} never retried), {totals['dropped_codes']} retry "
        f"code occurrences ({totals['items_with_two_final_codes']} items failed two "
        f"rules on the retry).", "", "### First pass", "", first, "", "### Final attempt", "", final,
    ]
    return "\n".join(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default="all",
                        choices=["all", "reconcile", "direction", "replay", "bias", "policy"])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    train = dataset["train"]
    cache = final_cache()
    records = {record["event_id"]: record
               for records in load_records(EVENTS).values() for record in records}

    payload: Dict[str, object] = {}
    markdown: List[str] = ["# Teacher rejection reconciliation and direction audit", ""]

    if args.section in ("all", "reconcile"):
        result = reconcile(train, cache)
        payload["reconcile"] = result
        markdown += ["## 1. Rejection tables, reconciled", "", render_reconcile(result), ""]
        (out / "reconcile.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    if args.section in ("all", "direction"):
        audit = direction_audit(train, records)
        payload["direction"] = audit
        rows = [[row["fault"], row["name"], row["kb_initiating"], row["kb_direction"],
                 row["tag_kind"], row["agreement_rate"], row["observed"], row["absent"]]
                for row in audit["faults"] if row["train_events"]]
        markdown += ["## 2. KB direction vs stored evidence trend", "",
                     "Agreement is over events where the initiating tag was ranked in the "
                     "evidence at all; `absent` counts the events where it was not.",
                     "", _md_table(["fault", "KB name", "initiating", "KB dir", "kind",
                                    "agreement", "observed", "absent"], rows), ""]
        examples = direction_examples(train, records, cache, VALVE_STICKING)
        payload["direction_examples"] = examples
        for example in examples:
            markdown += [f"- **{example['item_id']}** fault {example['fault']}: "
                         f"{example['kb_tag']} KB={example['kb_direction']} "
                         f"stored={example['stored_trend']} codes={example['codes']}",
                         f"  > {example['raw_excerpt']}"]
        (out / "direction.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")

    if args.section in ("all", "replay"):
        replay = replay_direction_change(train, records, cache, VALVE_STICKING)
        payload["replay"] = replay
        markdown += ["", "## 3. Replay of the proposed rule change", "",
                     _md_table(["unconstrained faults", "recovered", "invalidated",
                                "changed total", "flipped-direction invalidates"],
                               [[replay["unconstrained_faults"], replay["recovered"],
                                 replay["invalidated"], replay["changed"],
                                 replay["flipped_invalidated"]]]), ""]
        matrix = replay_matrix(train, records, cache)
        payload["replay_matrix"] = matrix
        markdown += ["", "### Scope of the relaxation", "",
                     _md_table(["scope", "recovered", "invalidated",
                                "flipping the KB sign instead would invalidate",
                                "recovered by fault"],
                               [[row["scope"], row["recovered"], row["invalidated"],
                                 row["flipped_invalidated"],
                                 _format_codes(row["recovered_by_fault"])] for row in matrix]), ""]
        (out / "replay.json").write_text(
            json.dumps({"proposed": replay, "matrix": matrix}, indent=1), encoding="utf-8")

    if args.section in ("all", "bias"):
        bias = drop_bias(train, records, cache)
        payload["bias"] = bias
        rows = [[name, rows["accepted"]["n"], rows["dropped"]["n"],
                 rows["accepted"]["median_ratio"], rows["dropped"]["median_ratio"],
                 rows["accepted"]["median_delay"], rows["dropped"]["median_delay"],
                 rows["accepted"]["median_abs_z"], rows["dropped"]["median_abs_z"],
                 rows["accepted"]["removed_share"], rows["dropped"]["removed_share"]]
                for name, rows in bias.items()]
        markdown += ["", "## 4. Drop bias", "",
                     _md_table(["family", "acc n", "drop n", "acc ratio", "drop ratio",
                                "acc delay", "drop delay", "acc |z|", "drop |z|",
                                "acc removed", "drop removed"], rows), ""]
        (out / "bias.json").write_text(json.dumps(bias, indent=1), encoding="utf-8")

    if args.section in ("all", "policy"):
        policy = train_policy(train, cache)
        payload["policy"] = policy
        markdown += ["", "## 5. Training set with dropped items removed", "",
                     _md_table(["class", "items"],
                               [[name, count] for name, count in sorted(policy["counts"].items())]
                               + [["total", policy["items"]],
                                  ["unclassified share of faulty",
                                   policy["unclassified_share_of_faulty"]],
                                  ["unclassified share of all",
                                   policy["unclassified_share_of_all"]]]), ""]
        (out / "policy.json").write_text(json.dumps(policy, indent=1), encoding="utf-8")

    (out / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
