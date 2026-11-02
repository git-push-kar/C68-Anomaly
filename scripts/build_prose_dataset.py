"""3B-11 step 4: the abstention-only prose pilot, built and counted. No teacher call, no training.

Part II of ``docs/TRAINING_SPEC_PROSE_V1.md`` proposes supervising prose only on the abstention
stratum — open-set, menu-removed and ``NORMAL_TRANSIENT`` report rows — plus every dialogue row.
This script builds exactly that file:

* the report prompt is rebuilt with the **menu shuffled per item** (a seeded permutation; the KB
  order is never kept and the identity permutation is asserted absent);
* ``fault_id``, ``verdict``, ``ranked_hypotheses`` and the classifier's **top-2 with calibrated
  probabilities** are printed in the prompt, so they are supervised nowhere: the answer carries
  ``physical_reasoning`` and ``narrative`` only;
* the supplied label is the **dataset target**, and a row whose gold label the §17 cross-fitted rule
  would not have produced is **counted and dropped** — those rows would teach prose for a situation
  the system never creates.

Dialogue rows are copied unchanged (their answer is model text and nothing about their prompt moves).

Usage::  python scripts/build_prose_dataset.py [--seed 7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DATASET = ROOT / "outputs/benchmark_v2/dataset"
SFT = ROOT / "outputs/benchmark_v2/sft"
EVAL = ROOT / "outputs/benchmark_v2/eval/prose_pilot"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
MIXED = SFT / "train_mixed.jsonl"
OUT = SFT / "train_prose_v1.jsonl"
MANIFEST = SFT / "train_prose_v1_manifest.json"
MENU_LINE = re.compile(r"^- (FAULT_\d{1,2}|UNCLASSIFIED_OPEN_SET): .+$", re.M)
ABSTENTION_VERDICTS = ("unclassified_open_set", "no_fault")
SUPPLIED_KEYS = ("fault_id", "verdict", "ranked_hypotheses")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_key(item_id: str) -> str:
    return str(item_id).split("::")[0]


def menu_lines(prompt: str) -> List[str]:
    return MENU_LINE.findall(prompt) and [line for line in MENU_LINE.findall(prompt)] or [
        line for line in re.findall(r"^- (?:FAULT_\d{1,2}|UNCLASSIFIED_OPEN_SET): .+$", prompt, re.M)]


def shuffled(lines: Sequence[str], seed: int, key: str) -> List[str]:
    order = list(lines)
    random.Random(f"{seed}:{key}").shuffle(order)
    return order


def build_prompt(prompt: str, supplied: Dict, top2: List, seed: int, key: str) -> Dict:
    head, marker, tail = prompt.partition("CANDIDATE MENU")
    lines = re.findall(r"^- (?:FAULT_\d{1,2}|UNCLASSIFIED_OPEN_SET): .+$", tail, re.M)
    order = shuffled(lines, seed, key)
    block = ["CANDIDATE MENU (order varies by event)", *order, "", "SUPPLIED CLASSIFICATION",
             "The following are fixed for this answer. Do not restate them.",
             f"- fault_id: {supplied['fault_id']}",
             f"- verdict: {supplied['verdict']}",
             f"- ranked_hypotheses: {', '.join(supplied['ranked_hypotheses'])}"]
    if top2:
        block.append("- classifier top-2: " + ", ".join(f"{name} {float(value):.2f}"
                                                        for name, value in top2))
    block += ["", 'Return one JSON object with exactly these keys: "physical_reasoning", '
                  '"narrative". Do not restate the supplied fields.', ""]
    return {"prompt": head + "\n".join(block), "menu_lines": len(lines), "identity_order": order == lines}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    items = {row["item_id"]: row for row in load_jsonl(DATASET / "train.jsonl")}
    oof = {row["event"]: row for row in load_jsonl(HYBRID / "crossfit_train_oof.jsonl")}
    report_rows, dialogue_rows = [], []
    for row in load_jsonl(MIXED):
        if "::" not in row["item_id"]:
            dialogue_rows.append(row)
            continue
        item = items.get(row["item_id"])
        if item is None:
            continue
        answer = json.loads(row["answer"])
        if str(answer.get("verdict")) not in ABSTENTION_VERDICTS:
            continue
        report_rows.append({"row": row, "item": item, "answer": answer})

    built, dropped, identity, menus = [], [], 0, Counter()
    for entry in report_rows:
        item, answer = entry["item"], entry["answer"]
        gold = (item.get("target") or {}).get("fault_id")
        rule = oof.get(event_key(item["item_id"])) or {}
        if rule.get("label") != gold:
            dropped.append({"item_id": item["item_id"], "gold": gold,
                            "rule_label": rule.get("label"), "rule_verdict": rule.get("verdict")})
            continue
        supplied = {"fault_id": gold, "verdict": answer.get("verdict"),
                    "ranked_hypotheses": [str(x) for x in answer.get("ranked_hypotheses") or [gold]]}
        varied = build_prompt(str(item["prompt"]), supplied, rule.get("top2") or [], args.seed,
                              item["item_id"])
        identity += int(varied["identity_order"])
        menus[varied["identity_order"]] += 1
        built.append({"item_id": item["item_id"], "kind": "report",
                      "system": entry["row"]["system"], "question": varied["prompt"],
                      "answer": json.dumps({"physical_reasoning": answer.get("physical_reasoning"),
                                            "narrative": answer.get("narrative")},
                                           ensure_ascii=False),
                      "supplied": supplied, "menu_lines": varied["menu_lines"]})
    rows = built + [{"item_id": row["item_id"], "kind": "dialogue", "system": row["system"],
                     "question": row["question"], "answer": row["answer"]} for row in dialogue_rows]
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                   encoding="utf-8")
    manifest = {
        "file": str(OUT), "sha256": sha256(OUT), "seed": args.seed,
        "rows": len(rows), "report_rows": len(built), "dialogue_rows": len(dialogue_rows),
        "candidate_report_rows": len(report_rows),
        "dropped_crossfit_disagreement": len(dropped), "dropped_examples": dropped[:10],
        "menu_identity_order_rows": identity,
        "menu_chance_identity": round(1.0, 10) if not built else round(1 / 16, 6),
        "supplied_keys_in_prompt": list(SUPPLIED_KEYS),
        "supervised_fields": ["physical_reasoning", "narrative", "answer (dialogue)"],
        "train_mixed_sha256": sha256(MIXED),
        "crossfit_train_oof_sha256": sha256(HYBRID / "crossfit_train_oof.jsonl"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "report_rows", "dialogue_rows",
                                               "candidate_report_rows",
                                               "dropped_crossfit_disagreement",
                                               "menu_identity_order_rows", "sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
