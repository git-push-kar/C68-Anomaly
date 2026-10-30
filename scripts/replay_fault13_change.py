"""Replay the proposed fault-13 KB direction change against the teacher cache.

The proposal is to flip fault 13's KB initiating direction from ``decrease`` to
``increase`` (XMEAS_7 / Reactor_Pressure rises in 102 of 102 train events) and to
extend ``_direction_conflict``'s vocabulary to gerunds in the same edit.

Nothing here is applied: the KB entry and the word lists are patched in memory,
the cached raw answers are re-judged twice, and the difference is reported. The
point is to price the change before anyone makes it -- for the prose, and (see
``--impact``) for the prompts, which is the part that decided the question.

Usage::

    python scripts/replay_fault13_change.py impact      # prompt/hash blast radius
    python scripts/replay_fault13_change.py replay      # prose validity change
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.benchmark import filters as F  # noqa: E402
from llm.benchmark.menu import CLASSIFIABLE_FAULTS, build_menu, signature_for  # noqa: E402
from llm.benchmark.prompt_v2 import render_prompt, teacher_instruction  # noqa: E402
from llm.benchmark.teacher import build_teacher_prompt, parse_teacher_json  # noqa: E402
from llm.response.allowed_set import build_allowed_set  # noqa: E402
from scripts.build_benchmark_dataset import evidence_of, load_records  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402

DATASET = ROOT / "outputs" / "benchmark_v2" / "dataset"
EVENTS = ROOT / "outputs" / "benchmark_v2" / "events"
FAULT = 13

#: The gerunds the proposal would add to the rejection vocabulary.
GERUNDS_UP = ("increasing", "rising")
GERUNDS_DOWN = ("decreasing", "falling")


def _menu_line(fault: int) -> str:
    return f"FAULT_{fault:02d}: {signature_for(fault)}"


def impact() -> dict:
    """How much text and how many hashes a direction edit to one KB entry moves."""
    old_line = _menu_line(FAULT)
    entry = FAULT_KNOWLEDGE[FAULT]
    original = json.loads(json.dumps(entry, default=str))
    # The teacher prompt is built from the same menu through ``teacher_menu``,
    # which rewrites only valve-position lines. Rendering it before the patch is
    # what shows whether a measurement's line is carried over unchanged -- it is,
    # so the teacher reads the edited claim too.
    record = json.loads((EVENTS / "faulty" / "train" / "f01_r001.json")
                        .read_text(encoding="utf-8"))
    evidence = evidence_of(record)
    target = {"fault_id": "FAULT_01", "truth_fault_number": 1}
    teacher_old = build_teacher_prompt(evidence, build_menu(), target)
    raw = entry["initiating"]
    patched = [dict(item) for item in raw]
    patched[0]["direction"] = "increase"
    FAULT_KNOWLEDGE[FAULT]["initiating"] = patched
    try:
        new_line = _menu_line(FAULT)
        teacher_new = build_teacher_prompt(evidence, build_menu(), target)
    finally:
        FAULT_KNOWLEDGE[FAULT] = original

    counts = {}
    prompts_hit = 0
    total = 0
    for split in ("train", "val", "test"):
        path = DATASET / f"{split}.jsonl"
        n = hit = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                n += 1
                if old_line in row["prompt"]:
                    hit += 1
        counts[split] = {"items": n, "prompts_with_line": hit}
        prompts_hit += hit
        total += n

    # Which items do NOT carry the line: the fault-13 open-set twins, where the
    # entry was removed on purpose.
    missing = {}
    for split in ("train", "val", "test"):
        with (DATASET / f"{split}.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        missing[split] = {
            "without_line": sum(1 for r in rows if old_line not in r["prompt"]),
            "of_which_fault13_removed": sum(
                1 for r in rows
                if old_line not in r["prompt"]
                and r.get("variant") == "removed"
                and (r.get("fault") == FAULT)),
        }
    return {
        "old_line": old_line,
        "new_line": new_line,
        "line_changes": old_line != new_line,
        "line_carried_into_teacher_prompt": old_line in teacher_old,
        "teacher_prompt_changes": teacher_old != teacher_new,
        "prompts_containing_line": prompts_hit,
        "items_total": total,
        "per_split": counts,
        "items_without_line": missing,
        "student_prompt_chars_delta": 0,
        "menu_entries_in_menu": len(CLASSIFIABLE_FAULTS) + 1,
        "student_prompt_sha_changes_for_items_without_line": False,
    }


def _load_items(split: str = "train") -> list:
    with (DATASET / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _load_cache() -> dict:
    cache = {}
    with (DATASET / "teacher_cache.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cache[row["item_id"]] = row
    return cache


def _judge(items, records, cache) -> dict:
    """item_id -> rejection codes, judged with whatever the filters are right now."""
    verdicts = {}
    for item in items:
        entry = cache.get(item["item_id"])
        if not entry:
            continue
        record = records.get(item["event_id"])
        if record is None:
            continue
        answer, parse_error = parse_teacher_json(entry.get("raw_text") or "")
        evidence = evidence_of(record)
        allowed = build_allowed_set(evidence,
                                    extra_faults=[record["fault"]] if record.get("fault") else [])
        codes = F.check_answer(answer, item["target"], evidence, allowed,
                               parse_error=parse_error, prompt=item.get("prompt"),
                               menu_ids=item.get("menu_ids"))
        verdicts[item["item_id"]] = {"codes": codes, "answer": answer, "evidence": evidence}
    return verdicts


def _with_patch(kb_direction: str | None, gerunds: bool, items, flat, cache) -> dict:
    """Judge the cache under one hypothetical rule regime, then restore it."""
    entry = FAULT_KNOWLEDGE[FAULT]
    original = json.loads(json.dumps(entry, default=str))
    old_up, old_down = F.INCREASE_WORDS, F.DECREASE_WORDS
    if kb_direction is not None:
        patched = [dict(item) for item in original["initiating"]]
        patched[0]["direction"] = kb_direction
        FAULT_KNOWLEDGE[FAULT]["initiating"] = patched
    if gerunds:
        F.INCREASE_WORDS = tuple(old_up) + GERUNDS_UP
        F.DECREASE_WORDS = tuple(old_down) + GERUNDS_DOWN
    try:
        return _judge(items, flat, cache)
    finally:
        FAULT_KNOWLEDGE[FAULT] = original
        F.INCREASE_WORDS, F.DECREASE_WORDS = old_up, old_down


def _offending_window(answer: dict, tag: str = "XMEAS_7") -> str:
    """The text ``_direction_conflict`` actually reads: 60 chars after the tag."""
    import re as _re

    whole = " ".join(str(answer.get(key) or "") for key in ("physical_reasoning", "narrative"))
    body = whole.lower()
    for match in _re.finditer(_re.escape(tag.lower()), body):
        window = whole[match.end():match.end() + 60]
        low = window.lower()
        if any(word in low for word in F.DECREASE_WORDS) or any(
                word in low for word in F.INCREASE_WORDS):
            return window.strip()
    return ""


def replay() -> dict:
    records = load_records(EVENTS)
    flat = {record["event_id"]: record for split in records.values() for record in split}
    items = _load_items("train")
    cache = _load_cache()
    fault_ids = {item["item_id"] for item in items if item.get("fault") == FAULT}

    baseline = _judge(items, flat, cache)
    accepted_fault = [k for k, v in baseline.items() if not v["codes"] and k in fault_ids]

    vocab_only = _with_patch(None, True, items, flat, cache)
    flip_only = _with_patch("increase", False, items, flat, cache)
    flip_and_vocab = _with_patch("increase", True, items, flat, cache)

    def newly_rejected(verdicts) -> list:
        return [k for k in accepted_fault if verdicts.get(k, {}).get("codes")]

    detail = []
    for item_id in newly_rejected(flip_and_vocab):
        answer = flip_and_vocab[item_id]["answer"] or {}
        detail.append({
            "item_id": item_id,
            "window": _offending_window(answer),
            "codes": flip_and_vocab[item_id]["codes"],
            "narrative": answer.get("narrative"),
        })

    # What the existing flag reports today, recomputed on the accepted prose: the
    # evidence the flip is based on. A flag needs a sentence naming exactly one
    # ranked sensor, so it can be smaller than the rejection count.
    flagged = []
    for item in items:
        if item["item_id"] not in accepted_fault:
            continue
        record = flat.get(item["event_id"])
        if record is None:
            continue
        evidence = evidence_of(record)
        allowed = build_allowed_set(evidence, extra_faults=[FAULT])
        prose = cache[item["item_id"]].get("prose") or {}
        for flag in F.direction_inconsistent(prose, evidence, allowed):
            if flag.get("sensor") == "XMEAS_7":
                flagged.append({"item_id": item["item_id"], **flag})

    return {
        "fault_items_in_train": len(fault_ids),
        "cached": sum(1 for item in items if item["item_id"] in fault_ids
                      and item["item_id"] in cache),
        "accepted_now": len(accepted_fault),
        "scenarios": {
            "vocabulary_only": {
                "newly_rejected": len(newly_rejected(vocab_only)),
                "note": "gerunds added while the KB still says decrease: this is "
                        "the damage the vocabulary edit does on its own",
            },
            "kb_flip_only": {"newly_rejected": len(newly_rejected(flip_only))},
            "kb_flip_and_vocabulary": {
                "newly_rejected": len(newly_rejected(flip_and_vocab)),
                "still_accepted": len(accepted_fault) - len(newly_rejected(flip_and_vocab)),
            },
        },
        "invalidated_detail": detail,
        "flags_today": flagged,
        "flag_count": len(flagged),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("impact", "replay", "both"))
    args = parser.parse_args()
    out = {}
    if args.what in ("impact", "both"):
        out["impact"] = impact()
        print(json.dumps(out["impact"], indent=2, ensure_ascii=False))
    if args.what in ("replay", "both"):
        out["replay"] = replay()
        print(json.dumps(out["replay"], indent=2, ensure_ascii=False))
    path = ROOT / "outputs" / "evaluation" / "fault13_change_impact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
