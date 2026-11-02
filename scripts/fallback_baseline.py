"""What the deterministic explainer writes, measured on all 834 validation rows.

§19.2 makes the report path a choice between two prose sources, and the second one has never been
measured: `llm/hybrid/generate.py::fallback_answer` builds an answer from the knowledge base and the
stored evidence, with no model involved. This script runs it on every validation row with the
**classifier-plus-menu-rule diagnosis supplied**, and reports what it can and cannot write:

* schema validity — the five keys, their types, and whether `ranked_hypotheses` starts with the
  supplied diagnosis;
* grounding — every number and every tag in its prose has to appear in that item's evidence block,
  which is the property the fallback is supposed to have *by construction*;
* width — mean words per field, and how much of the text is a template (identical across rows);
* what it cannot write — the per-item differential (which alternatives were excluded and why) and
  an item-specific narrative.

It also builds the blinded A/B sheet §19.2 will be decided on: the same 100 items as
`audit_hybrid_100.csv`, adapter prose and fallback prose assigned to A/B in a seeded order with the
key kept in a separate file.

CPU only. No model, no generation, no test component.

Usage::

    python scripts/fallback_baseline.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.hybrid.generate import fallback_answer  # noqa: E402
from scripts.tags import iter_tags  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"
SEED = 42
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
SENSOR_LINE = re.compile(r"^- (.+?) \[((?:XMEAS|XMV)_\d+), [^\]]*\]: z=(-?[\d.]+), deviation (-?[\d.]+)%")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evidence_from_prompt(prompt: str) -> Dict:
    """The minimal evidence dict the fallback reads, parsed from the prompt it will be judged on.

    `fallback_answer` quotes the top sensors and nothing else, so this rebuilds exactly that: the
    first two sensor lines, name and z-score, straight from the page. Parsing rather than reading a
    stored event keeps the grounding claim honest — the fallback can only quote what a reader of the
    prompt can see.
    """
    sensors = []
    for line in str(prompt or "").splitlines():
        match = SENSOR_LINE.match(line)
        if match:
            sensors.append({"display_name": match.group(1), "name": match.group(1),
                            "z_score": float(match.group(3)),
                            "deviation_percent": float(match.group(4))})
    return {"top_anomalous_sensors": sensors}


def grounding(answer: Dict, prompt: str) -> Dict:
    """Numbers and tags in the answer that the evidence block does not contain."""
    text = " ".join(str(answer.get(key) or "") for key in
                    ("physical_reasoning", "narrative", "fault_id", "verdict"))
    bad_numbers = [number for number in NUMBER.findall(text) if number not in prompt]
    bad_tags = [tag for tag in iter_tags(text) if tag not in prompt]
    return {"numbers_not_in_prompt": sorted(set(bad_numbers)),
            "tags_not_in_prompt": sorted(set(bad_tags)),
            "violations": len(set(bad_numbers)) + len(set(bad_tags))}


def schema_ok(answer: Dict, menu_ids: Sequence[str]) -> Dict:
    keys = ("fault_id", "verdict", "ranked_hypotheses", "physical_reasoning", "narrative")
    missing = [key for key in keys if key not in answer]
    types_ok = (isinstance(answer.get("fault_id"), str)
                and isinstance(answer.get("verdict"), str)
                and isinstance(answer.get("ranked_hypotheses"), list)
                and all(isinstance(item, str) for item in answer.get("ranked_hypotheses") or [])
                and isinstance(answer.get("physical_reasoning"), str)
                and isinstance(answer.get("narrative"), str))
    ranked = answer.get("ranked_hypotheses") or []
    off_menu = [label for label in ranked if label not in set(menu_ids or [])
                and label not in ("NORMAL_TRANSIENT", "UNCLASSIFIED_OPEN_SET")]
    return {"valid": not missing and types_ok,
            "missing": missing, "types_ok": types_ok,
            "ranked_starts_with_diagnosis": bool(ranked) and ranked[0] == answer.get("fault_id"),
            "ranked_off_menu": off_menu}


def words(text: str) -> int:
    return len(str(text or "").split())


def adapter_answer(row: Dict) -> Dict:
    """The adapter's structured answer from a generation row (dict repr, or the prose field)."""
    raw = row.get("answer")
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return {"fault_id": row.get("diagnosis"), "verdict": row.get("verdict", ""),
            "physical_reasoning": row.get("prose") or "",
            "narrative": "", "ranked_hypotheses": []}


def render(item: Dict, answer: Dict) -> str:
    """One presentation format for both sources, so A/B differs only in the writing."""
    ranked = ", ".join(str(label) for label in (answer.get("ranked_hypotheses") or [])) or "—"
    return (f"Diagnosis: {answer.get('fault_id')} (verdict {answer.get('verdict') or '—'})\n"
            f"Ranked: {ranked}\n"
            f"Reasoning: {str(answer.get('physical_reasoning') or '').strip()}\n"
            f"Narrative: {str(answer.get('narrative') or '').strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    predicted = load_jsonl(HYBRID / "crossfit_val_rows.jsonl")
    generated = {row["item_id"]: row for row in load_jsonl(HYBRID / "generation_rows.jsonl")}

    answers: Dict[str, Dict] = {}
    rows: List[Dict] = []
    for row in predicted:
        item = val.get(row["item_id"])
        if item is None:
            continue
        evidence = evidence_from_prompt(item["prompt"])
        answer = fallback_answer(row["pred"], row.get("verdict") or "classified",
                                 item.get("menu_ids") or [], evidence)
        answers[row["item_id"]] = answer
        check = grounding(answer, item["prompt"])
        schema = schema_ok(answer, item.get("menu_ids") or [])
        rows.append({"item_id": row["item_id"], "diagnosis": row["pred"],
                     "variant": item.get("variant"), "unobservable": bool(item.get("unobservable")),
                     "schema": schema, "grounding": check,
                     "words": {key: words(answer.get(key)) for key in
                               ("physical_reasoning", "narrative")},
                     "ranked": answer.get("ranked_hypotheses")})

    total = len(rows)
    per_field = {
        field: round(sum(row["words"][field] for row in rows) / total, 1) if total else 0
        for field in ("physical_reasoning", "narrative")}
    templates = {
        "narrative_prefixed_with_diagnosis": sum(
            1 for row in rows
            if str(answers[row["item_id"]]["narrative"]).startswith("Diagnosis ")),
        "physical_reasoning_starts_with_evidence": sum(
            1 for row in rows if str(answers[row["item_id"]]["physical_reasoning"]).startswith(
                "Detector and sensor evidence")),
    }

    report = {
        "rows": total,
        "schema_valid": sum(1 for row in rows if row["schema"]["valid"]),
        "ranked_starts_with_diagnosis": sum(
            1 for row in rows if row["schema"]["ranked_starts_with_diagnosis"]),
        "ranked_off_menu_rows": sum(1 for row in rows if row["schema"]["ranked_off_menu"]),
        "grounding_violations": sum(1 for row in rows if row["grounding"]["violations"]),
        "violations_by_kind": {
            "numbers": sum(len(row["grounding"]["numbers_not_in_prompt"]) for row in rows),
            "tags": sum(len(row["grounding"]["tags_not_in_prompt"]) for row in rows),
        },
        "examples_of_numbers_not_in_prompt": sorted({
            number for row in rows for number in row["grounding"]["numbers_not_in_prompt"]})[:15],
        "mean_words": per_field,
        "templates": templates,
        "cannot_write": {
            "differential": "no per-item statement of which alternatives were excluded and why; "
                            "the ranked list is the KB's lookalike list, not a judgement about "
                            "this evidence",
            "narrative": "one template with the KB's recommended_action substituted; it names no "
                         "sensor, no onset and no trend",
            "mechanism_shape": "the KB's mechanism text is the same for every event of a fault, so "
                               "two events of FAULT_07 get byte-identical reasoning",
        },
        "per_diagnosis": {},
    }
    by_diagnosis: Dict[str, int] = {}
    for row in rows:
        by_diagnosis[str(row["diagnosis"])] = by_diagnosis.get(str(row["diagnosis"]), 0) + 1
    report["per_diagnosis"] = dict(sorted(by_diagnosis.items()))

    # ---------------------------------------------------------------- #
    # 10 side-by-side examples: 3 abstentions, 2 normal, 5 faults
    # ---------------------------------------------------------------- #
    def pick(predicate, count: int, exclude=()) -> List[Dict]:
        chosen = []
        for row in rows:
            if len(chosen) >= count:
                break
            if row["item_id"] in exclude or not predicate(row):
                continue
            if row["item_id"] not in generated:
                continue
            chosen.append(row)
        return chosen

    chosen: List[Dict] = []
    chosen += pick(lambda row: row["diagnosis"] == "UNCLASSIFIED_OPEN_SET", 3)
    chosen += pick(lambda row: row["diagnosis"] == "NORMAL_TRANSIENT", 2,
                   exclude={row["item_id"] for row in chosen})
    chosen += pick(lambda row: str(row["diagnosis"]).startswith("FAULT"), 5,
                   exclude={row["item_id"] for row in chosen})
    examples = []
    for row in chosen:
        item = val[row["item_id"]]
        adapter = adapter_answer(generated[row["item_id"]])
        examples.append({
            "item_id": row["item_id"], "diagnosis": row["diagnosis"],
            "family": generated[row["item_id"]].get("family"),
            "adapter": {"ranked_hypotheses": adapter.get("ranked_hypotheses"),
                        "physical_reasoning": str(adapter.get("physical_reasoning"))[:700],
                        "narrative": str(adapter.get("narrative"))[:400]},
            "fallback": answers[row["item_id"]],
            "item_sensors": [line.strip() for line in item["prompt"].splitlines()
                             if SENSOR_LINE.match(line)][:3],
        })

    # ---------------------------------------------------------------- #
    # the blinded A/B sheet
    # ---------------------------------------------------------------- #
    audit = list(csv.DictReader((HYBRID / "audit_hybrid_100.csv").open(encoding="utf-8")))
    rng = random.Random(args.seed)
    sheet_rows, key_rows = [], []
    for entry in audit:
        item_id = entry["item_id"]
        item = val.get(item_id)
        if item is None:
            continue
        diagnosis = entry.get("forced_diagnosis") or "UNCLASSIFIED_OPEN_SET"
        if item_id in answers:
            fallback = answers[item_id]
        else:
            evidence = evidence_from_prompt(item["prompt"])
            fallback = fallback_answer(diagnosis, "classified",
                                       item.get("menu_ids") or [], evidence)
        adapter = adapter_answer(generated.get(item_id, {}))
        adapter_text = render(item, {"fault_id": diagnosis, "verdict": "classified",
                                     "ranked_hypotheses": adapter.get("ranked_hypotheses"),
                                     "physical_reasoning": entry.get("prose")
                                     or adapter.get("physical_reasoning"),
                                     "narrative": adapter.get("narrative")})
        fallback_text = render(item, fallback)
        adapter_is_a = rng.random() < 0.5
        sheet_rows.append({
            "item_id": item_id, "family": entry.get("family"), "diagnosis": diagnosis,
            "text_A": adapter_text if adapter_is_a else fallback_text,
            "text_B": fallback_text if adapter_is_a else adapter_text,
            "correct_A": "", "grounded_A": "", "consistent_A": "", "useful_A": "",
            "correct_B": "", "grounded_B": "", "consistent_B": "", "useful_B": "",
            "preference": "", "notes": "",
        })
        key_rows.append({"item_id": item_id, "A": "adapter" if adapter_is_a else "fallback",
                         "B": "fallback" if adapter_is_a else "adapter", "seed": args.seed})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / "audit_ab_100.csv"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sheet_rows)
    key = out / "audit_ab_100_key.csv"
    with key.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    report["ab_sheet"] = {"rows": len(sheet_rows), "sheet": str(sheet.name),
                          "key": str(key.name), "seed": args.seed,
                          "adapter_as_A": sum(1 for row in key_rows if row["A"] == "adapter")}
    report["examples"] = examples
    (out / "fallback_baseline.json").write_text(json.dumps(report, indent=2, default=str),
                                                encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("rows", "schema_valid", "ranked_starts_with_diagnosis",
                       "grounding_violations", "violations_by_kind", "mean_words", "templates",
                       "ab_sheet")}, indent=2, default=str))
    print(f"written: {out / 'fallback_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
