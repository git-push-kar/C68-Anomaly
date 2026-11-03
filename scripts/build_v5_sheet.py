"""Build the v5 A/B sheet (PROTOCOL 26.3): both sides one item, one layout, no classified row.

Ten open-set items on a menu, ten menu-removed, twenty NORMAL_TRANSIENT. Each row holds the pilot
adapter's stored answer for that item and the deterministic explainer's answer to the same prompt,
rendered into the identical `Diagnosis / Reasoning / Narrative` layout so the source cannot be read
off the format.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.ab_sheet_checks import FIT, Sources, format_features, v4_render  # noqa: E402
from scripts.build_prose_dataset import load_jsonl  # noqa: E402
from scripts.prose_pilot_prompts import EXPORT  # noqa: E402

COLUMNS = ["item_id", "text_A", "text_B", "correct_A", "grounded_A", "consistent_A", "useful_A",
           "correct_B", "grounded_B", "consistent_B", "useful_B", "preference", "notes"]
KEY_COLUMNS = ["item_id", "family", "diagnosis", "verdict", "A", "B", "seed"]
SCORES = COLUMNS[3:]
STRATA = {"abstention_menu": 10, "abstention_removed": 10, "normal": 20}
FAMILY = {"abstention_menu": "open_set", "abstention_removed": "removed", "normal": "normal"}
STOCK = (re.compile(r"^The verdict is [A-Z_0-9]+\.\s*"), re.compile(r"^Evidence for this event:\s*"))


def normalise(text: str) -> str:
    """Drop labels only one generator writes, so neither side carries its signature."""
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    body = re.sub(r"\bDifferential:[^.]*\.\s*", "", body)
    body = re.sub(r"\bRecommended:\s*", "Next step: ", body)
    for pattern in STOCK:
        body = pattern.sub("", body)
    return body.strip()


def adapter_parts(text: str):
    body = str(text or "").strip()
    start, end = body.find("{"), body.rfind("}")
    parsed = json.loads(body[start:end + 1])
    return str(parsed.get("physical_reasoning") or ""), str(parsed.get("narrative") or "")


def layout(diagnosis: str, reasoning: str, narrative: str) -> str:
    return ("Diagnosis: " + diagnosis + chr(10) + chr(10) + "Reasoning: " + normalise(reasoning)
            + chr(10) + chr(10) + "Narrative: " + normalise(narrative))


def stratum_of(row) -> str:
    gold, verdict = str(row["gold_fault_id"]), str(row["gold_verdict"])
    if gold == "UNCLASSIFIED_OPEN_SET":
        return "abstention_menu" if str(row["variant"]) == "menu" else "abstention_removed"
    if gold == "NORMAL_TRANSIENT" and verdict == "no_fault":
        return "normal"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(FIT))
    args = parser.parse_args()

    sources = Sources()
    rng = random.Random(args.seed)
    chosen = []
    for stratum, count in STRATA.items():
        pool = sorted((row for row in sources.items.values() if stratum_of(row) == stratum),
                      key=lambda row: row["item_id"])
        if len(pool) < count:
            raise SystemExit("not enough " + stratum + " items: " + str(len(pool)))
        chosen.extend((stratum, row) for row in rng.sample(pool, count))

    candidates = []
    for stratum, row in chosen:
        item_id = row["item_id"]
        item = sources.val.get(item_id) or {}
        tag, z = sources.sensor_of(item_id)
        answer = sources.fallback_answer[item_id]
        reasoning_a, narrative_a = adapter_parts(sources.adapter_text(item_id))
        text_a = layout(str(row["gold_fault_id"]), reasoning_a, narrative_a)
        text_b = layout(str(row["gold_fault_id"]), answer.get("physical_reasoning") or "",
                        answer.get("narrative") or "")
        assert sources.lookup(sources.adapter_text(item_id))[0] == item_id, (
            "adapter generation does not belong to " + item_id)
        assert sources.lookup(v4_render(answer))[0] == item_id, (
            "fallback answer does not belong to " + item_id)
        assert tag and sources.sensor_of(item_id) == (tag, z)
        candidates.append({"item_id": item_id, "stratum": stratum, "family": FAMILY[stratum],
                           "diagnosis": str(row["gold_fault_id"]),
                           "verdict": str(row["gold_verdict"]), "tag": tag, "z": z,
                           "text_adapter": text_a, "text_fallback": text_b})

    rows, key = build_rows(candidates, seed=args.seed)
    provenance = {entry["item_id"]: {"corpus_item_id": entry["corpus_item_id"],
                                     "adapter_item": entry["corpus_item_id"],
                                     "fallback_item": entry["corpus_item_id"],
                                     "tag": entry["tag"], "z": entry["z"]}
                  for entry in key}
    report = identity_report(rows, key)
    if not report["passed"]:
        raise SystemExit("format or balance check failed: " + json.dumps(report)[:800])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "audit_ab_100_v5.csv", COLUMNS,
          [{column: ("" if column in SCORES else row.get(column, "")) for column in COLUMNS}
           for row in rows])
    write(out / "audit_ab_100_v5_key.csv", KEY_COLUMNS, key)
    meta = {"sheet": "audit_ab_100_v5.csv", "key": "audit_ab_100_v5_key.csv",
            "rows": len(rows), "seed": args.seed,
            "strata": dict(Counter(entry["stratum"] for entry in candidates)),
            "verdicts": dict(Counter(entry["verdict"] for entry in candidates)),
            "adapter_on_A_by_stratum": report["per_stratum"],
            "format_identity": {k: v for k, v in report.items()
                                if k not in ("per_stratum", "passed")},
            "source": "raw_pilot_kb.jsonl (pilot) vs fallback_v31 on the same prompt",
            "columns_visible": COLUMNS}
    (out / "audit_ab_100_v5_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "audit_ab_100_v5_provenance.json").write_text(json.dumps(provenance, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


def build_rows(candidates, seed: int):
    rng = random.Random(seed)
    rows, key = [], []
    for stratum in STRATA:
        members = sorted([c for c in candidates if c["stratum"] == stratum],
                         key=lambda c: c["item_id"])
        flags = [True] * (len(members) // 2) + [False] * (len(members) // 2)
        rng.shuffle(flags)
        for member, adapter_on_a in zip(members, flags):
            row = {"item_id": "", "text_A": member["text_adapter"] if adapter_on_a
                   else member["text_fallback"],
                   "text_B": member["text_fallback"] if adapter_on_a
                   else member["text_adapter"]}
            rows.append((row, member, adapter_on_a))
    rng.shuffle(rows)
    for position, (row, member, adapter_on_a) in enumerate(rows, start=1):
        row["item_id"] = "v5-" + str(position).zfill(2)
        key.append({"item_id": row["item_id"], "family": member["family"],
                    "diagnosis": member["diagnosis"], "verdict": member["verdict"],
                    "A": "adapter" if adapter_on_a else "fallback",
                    "B": "fallback" if adapter_on_a else "adapter", "seed": seed,
                    "corpus_item_id": member["item_id"], "tag": member["tag"],
                    "z": member["z"]})
    return [row for row, _member, _adapter in rows], key


def identity_report(rows, key) -> dict:
    """Balance inside every stratum, and the same layout on both sides of every row."""
    sides = {entry["item_id"]: entry["A"] for entry in key}
    features = {entry["item_id"]: entry["family"] for entry in key}
    per_stratum, mismatched, leaks, passed = {}, 0, 0, True
    for stratum in STRATA:
        members = [row for row in rows if features[row["item_id"]] == FAMILY[stratum]]
        adapter_on_a = sum(1 for row in members if sides[row["item_id"]] == "adapter")
        per_stratum[stratum] = {"rows": len(members), "adapter_on_A": adapter_on_a}
        passed = passed and adapter_on_a * 2 == len(members)
    for row in rows:
        left, right = format_features(row["text_A"]), format_features(row["text_B"])
        if left != right:
            mismatched += 1
        leaks += len(left["leak_labels"]) + len(right["leak_labels"])
    passed = passed and mismatched == 0 and leaks == 0
    for row in rows:
        for column in ("text_A", "text_B"):
            if "Diagnosis: " not in row[column] or "Narrative: " not in row[column]:
                passed = False
    return {"passed": bool(passed), "rows": len(rows), "per_stratum": per_stratum,
            "rows_whose_two_sides_differ_in_layout": mismatched,
            "generator_labels_found": leaks,
            "identical_layout_both_sides": mismatched == 0}


def write(path: Path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
