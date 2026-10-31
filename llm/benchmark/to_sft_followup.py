"""Export the follow-up dialogues as SFT rows, and mix them with the report task.

The trainer takes one assistant answer per row: it masks everything up to the
assistant header and learns the rest. So a multi-turn dialogue becomes **one row
per accepted assistant turn**, with the earlier turns rendered into the question as
a transcript. That keeps the existing trainer contract exactly as it is -- no
change to ``llm/dataset.py``, no custom collator -- at the cost of re-sending the
earlier turns as prompt tokens on each row, which the measured sequence budget
(§11 of PROTOCOL) can afford.

The answer is ``{"answer": ...}``: the same key the follow-up teacher was asked to
produce, and the only part of a follow-up that is model text. Tables, citations,
suggestions and the clarifying question are built from the store and the router
when the answer is rendered.

The mix is **two report rows to one dialogue row**, at the row level, because rows
are what the optimizer counts. Whole dialogues are taken, never half of one.

Usage::

    python -m llm.benchmark.to_sft_followup --split train
    python -m llm.benchmark.to_sft_followup --mix
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.followup.leak import leak_violations  # noqa: E402
from llm.followup.teacher import SYSTEM_PROMPT_FOLLOWUP  # noqa: E402

FOLLOWUP_DIR = Path("outputs/benchmark_v2/followup")
SFT_DIR = Path("outputs/benchmark_v2/sft")

#: Words that must not reach the student through a follow-up row. The leak
#: patterns (event ids, split names, run numbers, out-of-menu fault numbers) come
#: from ``llm.followup.leak`` so this path uses the same definition the build does;
#: these two are specific to training input, because the teacher was told never to
#: state them and the student should not be taught to expect them.
FORBIDDEN_IN_QUESTION = ("severity", "confidence")

#: Rows per case in the dialogue half of the 2:1 mix, out of 959 slots.
#:
#: Replaces the tier fill of PROTOCOL §12.1, which exhausted the declining and
#: comparison cases before the ordinary ones and left 17 rows in total for the
#: four explanatory cases -- the largest of them, ``valve_actuator``, contributing
#: two. A per-case target keeps every case present at a size worth training on and
#: still spends most slots on the turns a refusal or a comparison has to get right.
#: Whole dialogues only, so a case may land a few rows above or below its target.
CASE_TARGETS: Dict[str, int] = {
    "out_of_scope": 114,
    "false_premise": 120,
    "clarification_reply": 80,
    "repeated_question": 80,
    "ambiguous_question": 100,
    "hypothesis_comparison": 120,
    "valve_actuator": 140,
    "causal_why": 100,
    "mechanism": 80,
    "template_sample": 25,
}

#: The 959 slots those targets add up to, at two report rows per dialogue row.
TARGET_ROWS = sum(CASE_TARGETS.values())


class FollowupExportError(RuntimeError):
    """A row would teach the model something the runtime does not give it."""


def load_dialogues(split: str = "train") -> List[Dict]:
    path = FOLLOWUP_DIR / f"{split}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _transcript(turns: Sequence[Dict]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"Operator: {turn.get('question')}")
        lines.append(f"Assistant: {turn.get('answer')}")
    return "\n".join(lines)


def question_for(dialogue: Dict, index: int) -> str:
    """The student-facing question: incident context, transcript, then the ask."""
    turns = dialogue.get("turns") or []
    parts = ["INCIDENT CONTEXT", str(dialogue.get("context") or "").strip(), ""]
    if index:
        parts += ["CONVERSATION SO FAR", _transcript(turns[:index]), ""]
    parts += ["QUESTION", str(turns[index].get("question") or "").strip()]
    return "\n".join(parts)


def permitted_faults(dialogue: Dict) -> List[int]:
    """Fault numbers the assembled question may name.

    The context is a checked artifact and carries the incident's own differential,
    so every number in it is legitimate. What the check has to catch is a number
    that is *not* in the incident: on an open-set dialogue the true fault must not
    appear anywhere, because naming it would hand the student the label the item
    withholds.
    """
    from llm.response.validators import fault_ids_in

    context = str(dialogue.get("context") or "")
    permitted = set(fault_ids_in(context))
    fault = dialogue.get("fault")
    if fault is not None:
        permitted.add(int(fault))
    if dialogue.get("open_set") and fault is not None:
        permitted.discard(int(fault))
    return sorted(permitted)


def rows_for_dialogue(dialogue: Dict) -> List[Dict]:
    """One row per accepted assistant turn; a rejected turn ends the dialogue."""
    rows: List[Dict] = []
    permitted = permitted_faults(dialogue)
    for index, turn in enumerate(dialogue.get("turns") or []):
        if not turn.get("accepted"):
            break
        answer = str(turn.get("answer") or "").strip()
        if not answer:
            break
        question = question_for(dialogue, index)
        lowered = question.lower()
        banned = [fragment for fragment in FORBIDDEN_IN_QUESTION if fragment in lowered]
        if banned:
            raise FollowupExportError(
                f"{dialogue.get('dialogue_id')} turn {index}: question contains {banned}")
        leaks = leak_violations(question, permitted_faults=permitted, allow_opaque=True)
        if leaks:
            raise FollowupExportError(
                f"{dialogue.get('dialogue_id')} turn {index}: leaks {leaks[:2]}")
        rows.append({
            "item_id": f"{dialogue.get('dialogue_id')}#{index}",
            "system": SYSTEM_PROMPT_FOLLOWUP,
            "question": question,
            "answer": json.dumps({"answer": answer}, ensure_ascii=False),
        })
    return rows


def export(split: str = "train") -> List[Dict]:
    rows: List[Dict] = []
    for dialogue in load_dialogues(split):
        rows.extend(rows_for_dialogue(dialogue))
    return rows


def mix(report_rows: Sequence[Dict], dialogue_rows: Sequence[Dict], ratio: int = 2,
        seed: int = 42, case_of: Optional[Dict[str, str]] = None,
        targets: Optional[Dict[str, int]] = None,
        tolerance: int = 3) -> Dict:
    """Interleave two report rows per dialogue row, keeping the whole of each.

    ``ratio`` is report rows per dialogue **row**. Dialogues are taken whole, so
    the achieved ratio is reported rather than assumed.

    With ``case_of`` (dialogue id -> case type) each case is filled to its own
    target from ``CASE_TARGETS``, in a seeded order, and the total is balanced back
    to the slot count with the cases that have the most rows to give. A whole
    dialogue is the unit, so a case lands within one dialogue's length of its
    target; ``by_case`` reports the exact rows. Without ``case_of`` the selection
    falls back to a seeded shuffle, which is what the older call sites and their
    tests expect.
    """
    budget = len(report_rows) // ratio
    rng = random.Random(seed)
    by_dialogue: Dict[str, List[Dict]] = {}
    for row in dialogue_rows:
        key = str(row["item_id"]).split("#")[0]
        by_dialogue.setdefault(key, []).append(row)
    wanted = dict(targets or CASE_TARGETS)
    per_case_picks: Dict[str, List[str]] = {}
    ordered: Dict[str, List[str]] = {}
    pointer: Dict[str, int] = {}
    available: Counter = Counter()
    if case_of:
        buckets: Dict[str, List[str]] = {}
        for key in by_dialogue:
            case = case_of.get(key, "unknown")
            buckets.setdefault(case, []).append(key)
        def seeded(keys: List[str]) -> List[str]:
            return sorted(keys, key=lambda key: hashlib.sha256(
                f"{seed}|{key}".encode("utf-8")).hexdigest())

        for case, keys in buckets.items():
            ordered[case] = seeded(keys)
            available[case] = sum(len(by_dialogue[key]) for key in keys)
            pointer[case] = 0
            per_case_picks[case] = []
        for case, target in wanted.items():
            if target > available.get(case, 0):
                raise ValueError(
                    f"case {case}: target {target} exceeds the {available.get(case, 0)} "
                    "rows the corpus can give")
            taken = 0
            keys = ordered.get(case, [])
            while pointer[case] < len(keys) and taken < target:
                key = keys[pointer[case]]
                per_case_picks[case].append(key)
                pointer[case] += 1
                taken += len(by_dialogue[key])

        def rows_of(case: str) -> int:
            return sum(len(by_dialogue[key]) for key in per_case_picks[case])

        # Whole dialogues round, so the total lands a few rows off the budget.
        # The correction is taken from -- or given to -- the cases furthest from
        # their own target, never from a case that is already at it.
        while sum(rows_of(case) for case in per_case_picks) > budget + tolerance:
            movable = [case for case in per_case_picks if len(per_case_picks[case]) > 1]
            if not movable:
                break
            case = max(movable, key=lambda name: rows_of(name) - wanted.get(name, 0))
            per_case_picks[case].pop()
        while sum(rows_of(case) for case in per_case_picks) < budget - tolerance:
            movable = [case for case in ordered if pointer[case] < len(ordered[case])]
            if not movable:
                break
            case = max(movable, key=lambda name: available[name] - rows_of(name))
            per_case_picks[case].append(ordered[case][pointer[case]])
            pointer[case] += 1
        ids = [key for case in sorted(per_case_picks) for key in per_case_picks[case]]
        for case in per_case_picks:
            available.setdefault(case, 0)
    else:
        ids = sorted(by_dialogue)
        rng.shuffle(ids)
    chosen: List[Dict] = []
    used: List[str] = []
    for key in ids:
        if case_of and len(chosen) + len(by_dialogue[key]) > budget + tolerance:
            continue
        if not case_of and len(chosen) + len(by_dialogue[key]) > budget:
            continue
        chosen.extend(by_dialogue[key])
        used.append(key)
    combined: List[Dict] = []
    report_queue = list(report_rows)
    dialogue_queue = list(chosen)
    dialogue_rows = len(chosen)
    while report_queue or dialogue_queue:
        for _ in range(ratio):
            if report_queue:
                combined.append(report_queue.pop(0))
        if dialogue_queue:
            combined.append(dialogue_queue.pop(0))
    if not case_of:
        available = Counter()
        for key, rows_here in by_dialogue.items():
            available["unknown"] += len(rows_here)
    taken = Counter()
    for key in used:
        taken[(case_of or {}).get(key, "unknown")] += len(by_dialogue[key])
    cases = sorted(available)
    return {
        "rows": combined,
        "report_rows": len(report_rows),
        "dialogue_rows": dialogue_rows,
        "dialogues_used": len(used),
        "dialogue_budget": budget,
        "achieved_ratio": (round(len(report_rows) / dialogue_rows, 3) if dialogue_rows else None),
        "target_ratio": ratio,
        "case_targets": {case: wanted.get(case, 0) for case in cases},
        "target_rows": sum(wanted.get(case, 0) for case in cases) if case_of else None,
        "target_deviation": (dialogue_rows - sum(wanted.get(case, 0) for case in cases))
        if case_of else None,
        "by_case": {
            case: {"target": wanted.get(case, 0), "rows_taken": taken[case],
                   "rows_available": available[case]}
            for case in cases
        },
        "slots_unfilled": budget - dialogue_rows,
    }


def _load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mix", action="store_true",
                        help="also write the combined 2:1 file")
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(SFT_DIR))
    args = parser.parse_args()
    out = Path(args.out)

    rows = export(args.split)
    if rows:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"followup_{args.split}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"followup_{args.split}: {len(rows)} rows -> {path}")
    else:
        print(f"followup_{args.split}: no accepted dialogues")

    if args.mix:
        report_path = out / "train.jsonl"
        if not report_path.exists():
            raise SystemExit(f"no report rows at {report_path}; run to_sft first")
        report_rows = _load_jsonl(report_path)
        case_of = {str(d.get("dialogue_id")): str(d.get("case_type"))
                   for d in load_dialogues(args.split)}
        mixed = mix(report_rows, rows, ratio=args.ratio, seed=args.seed, case_of=case_of)
        path = out / "train_mixed.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in mixed.pop("rows"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # The file is build output and not tracked, so its identity has to be
        # written down here: a hash that only appears in a report cannot be
        # re-checked against the file the trainer actually reads.
        mixed["train_mixed_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        mixed["report_train_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        dialogue_path = out / f"followup_{args.split}.jsonl"
        if dialogue_path.exists():
            mixed["dialogue_rows_file_sha256"] = hashlib.sha256(
                dialogue_path.read_bytes()).hexdigest()
        (out / "mix_summary.json").write_text(json.dumps(mixed, indent=2), encoding="utf-8")
        print(json.dumps(mixed, indent=2))
        print(f"mixed -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
