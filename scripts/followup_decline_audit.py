"""How often the *training* follow-up prose already declines an in-scope question.

The 179-turn validation sample showed refusals on 53 % of re-asks and 73 % of
`causal_why`, which reads like a model defect until you look at what it was taught.
If the accepted teacher prose declines those questions too, the model is imitating
its data, and the fix belongs upstream in the corpus rather than in a prompt.

The decline test is the project's own (`llm.followup.filters.declines`), not a new
word list, so this cannot drift from the rule that accepted the answer in the first
place. A second, wider test catches the "the context does not mention …" family the
refusal filter deliberately allows, because that phrasing is how an answer declines
without using the refusal vocabulary.

Usage::

    python scripts/followup_decline_audit.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.followup.filters import declines  # noqa: E402

TRAIN = ROOT / "outputs/benchmark_v2/followup/train.jsonl"
VAL_SAMPLE = ROOT / "outputs/benchmark_v2/eval/followup_sanity_200.json"
CASES = ("causal_why", "mechanism", "valve_actuator", "repeated_question",
         "hypothesis_comparison")
#: The dialogue corpus names two cases differently from the validation sample
#: (``repeat`` / ``template``), so the comparison has to normalise before it can
#: put the two rates side by side.
ALIASES = {"repeat": "repeated_question", "template": "template_sample"}

#: "The context does not mention X" is a decline the refusal filter tolerates: the
#: sentence form is informative, so it is not caught by a refusal vocabulary that
#: looks for "I cannot" / "not available". Matched on the answer text only.
ABSENT = re.compile(
    r"(?:does not|doesn't|do not|cannot|can't|no longer|is not|isn't)\s+"
    r"(?:mention|provide|contain|include|record|show|specify|list|state)",
    re.IGNORECASE,
)
NOT_IN = re.compile(r"not (?:present|available|stored|captured|in the (?:context|event|store))",
                    re.IGNORECASE)


def answer_text(turn: Dict) -> str:
    body = turn.get("answer") or ""
    if isinstance(body, dict):
        return str(body.get("answer") or "")
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return str(parsed.get("answer") or body)
    except (TypeError, ValueError):
        pass
    return str(body)


def classify(text: str) -> Dict[str, bool]:
    return {"refusal": declines(text),
            "absent": bool(ABSENT.search(text) or NOT_IN.search(text))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(
        ROOT / "outputs/analysis/phase3a/followup_decline_audit.json"))
    args = parser.parse_args()

    dialogues = [json.loads(line) for line in
                 TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_case: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"turns": 0, "refusal": 0, "absent": 0, "either": 0, "neither": 0})
    examples: List[Dict] = []
    seen_answers = set()
    for dialogue in dialogues:
        for turn in dialogue.get("turns") or []:
            if not turn.get("accepted"):
                continue
            case = str(turn.get("case_type") or dialogue.get("case_type"))
            case = ALIASES.get(case, case)
            if case not in CASES:
                continue
            text = answer_text(turn)
            flags = classify(text)
            bucket = per_case[case]
            bucket["turns"] += 1
            bucket["refusal"] += int(flags["refusal"])
            bucket["absent"] += int(flags["absent"])
            bucket["either"] += int(flags["refusal"] or flags["absent"])
            bucket["neither"] += int(not flags["refusal"] and not flags["absent"])
            key = text[:120]
            if (flags["refusal"] or flags["absent"]) and key not in seen_answers \
                    and len(examples) < 10:
                seen_answers.add(key)
                examples.append({"case_type": case, "answer": text[:400]})

    table = {case: {"turns": value["turns"],
                    "decline_rate": round(value["either"] / max(value["turns"], 1), 4),
                    "refusal_vocabulary_rate": round(value["refusal"] / max(value["turns"], 1), 4),
                    "context_absent_rate": round(value["absent"] / max(value["turns"], 1), 4),
                    "answered_rate": round(value["neither"] / max(value["turns"], 1), 4)}
             for case, value in sorted(per_case.items())}

    validation = {}
    if VAL_SAMPLE.exists():
        payload = json.loads(VAL_SAMPLE.read_text(encoding="utf-8"))
        for row in payload.get("rows") or []:
            case = row.get("kind")
            if case not in CASES:
                continue
            entry = validation.setdefault(case, {"n": 0, "refused": 0})
            entry["n"] += 1
            entry["refused"] += int(bool(row.get("refused")))

    result = {"cases": list(CASES), "train": table,
              "validation_sample": {case: {"n": value["n"],
                                           "refusal_rate": round(value["refused"]
                                                                 / max(value["n"], 1), 4)}
                                    for case, value in sorted(validation.items())},
              "comparison": {case: {"train_decline_rate": table.get(case, {}).get("decline_rate"),
                                    "validation_refusal_rate": (validation.get(case) or {}).get("refused", 0)
                                    / max((validation.get(case) or {}).get("n", 1), 1)}
                             for case in CASES},
              "examples": examples}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"train": result["train"],
                      "validation_sample": result["validation_sample"]}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
