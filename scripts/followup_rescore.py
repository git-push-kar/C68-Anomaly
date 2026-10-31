"""Is a refusal the right answer, and does the model refuse the right things?

The 179-turn validation sample refuses 40-73 % of the prose cases. That number is
uninterpretable on its own: refusing an out-of-scope question is correct, refusing a
question the store can answer is over-refusal, and answering a question the store
cannot answer is a fabrication. The router already decides this -- it routes the
question to an intent and asks the store for a key, and if no key resolves it cannot
answer deterministically -- so ``expected_decline`` is a property of the *question
and the event*, not of the model.

Two readings are reported, because they answer different questions:

``store``   ``expected_decline`` = no store key resolves. This is the literal reading
            of "the key resolves or not", and it is the right test for a *lookup*
            question. For a prose question (why / mechanism / how does the valve
            behave) no key ever resolves, so the store reading calls the case
            unanswerable and every prose answer becomes "under-refusal". That is a
            property of the reading, not of the model, and the table shows it.
``router``  ``expected_decline`` = the router's own verdict is ``refuse``. This is the
            reading that separates in-domain from out-of-domain, so it is the one the
            adapter's refusal rate is compared against.

What this adds beyond the router's own verdict is the per-case table: agreement,
over-refusal (a key resolved and the model declined anyway) and under-refusal, on the
same 179 turns the adapter was scored on, plus the same fraction measured on the
training corpus -- because a behaviour the data taught is not a bug in the scorer.

No model is called: the router's deterministic path resolves keys and returns text.

Usage::

    python scripts/followup_rescore.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_followup_dataset import EVENTS  # noqa: E402
from build_benchmark_dataset import load_records  # noqa: E402
from llm.followup.filters import declines  # noqa: E402
from llm.response.router import LOOKUP_INTENTS, ResponseRouter  # noqa: E402

FOLLOWUP_VAL = ROOT / "outputs/benchmark_v2/followup/val.jsonl"
FOLLOWUP_TRAIN = ROOT / "outputs/benchmark_v2/followup/train.jsonl"
SANITY = ROOT / "outputs/benchmark_v2/eval/followup_sanity_200.json"
TRAIN_MIXED = ROOT / "outputs/benchmark_v2/sft/train_mixed.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/followup_rescore.json"

QUESTION_MARKER = re.compile(r"\n\s*QUESTION\s*\n", re.IGNORECASE)
ABSENT = re.compile(
    r"(?:does not|doesn't|do not|cannot|can't|no longer|is not|isn't)\s+"
    r"(?:mention|provide|contain|include|record|show|specify|list|state)",
    re.IGNORECASE,
)
NOT_IN = re.compile(r"not (?:present|available|stored|captured|in the (?:context|event|store))",
                    re.IGNORECASE)
ALIASES = {"repeat": "repeated_question", "template": "template_sample"}
MIXED_DIALOGUE = re.compile(r"^(FD-train-\d+)#(\d+)$")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def bare_question(text: str) -> str:
    parts = QUESTION_MARKER.split(text or "")
    return (parts[-1] if parts else text or "").strip()


def event_index() -> Dict[str, Dict]:
    """``source|fault|run`` -> the stored record, so a turn can find its event."""
    index: Dict[str, Dict] = {}
    for _split, records in load_records(EVENTS).items():
        for record in records:
            key = f"{record.get('source_file')}|{record.get('fault')}|{record.get('simulationRun')}"
            index[key] = record
    return index


def declared_decline(text: str) -> Dict[str, bool]:
    return {"refusal_vocabulary": bool(declines(text)),
            "context_absent": bool(ABSENT.search(text) or NOT_IN.search(text))}


def _tally(entries: Sequence[Dict], key: str) -> Dict:
    """Agreement, over- and under-refusal for one reading of ``expected_decline``."""
    table = defaultdict(lambda: {"turns": 0, "expected_decline": 0, "refused": 0,
                                 "agree": 0, "over_refusal": 0, "under_refusal": 0})
    for entry in entries:
        bucket = table[entry["kind"]]
        expected = bool(entry[key])
        refused = bool(entry["adapter_refused"])
        bucket["turns"] += 1
        bucket["expected_decline"] += int(expected)
        bucket["refused"] += int(refused)
        bucket["agree"] += int(expected == refused)
        bucket["over_refusal"] += int(refused and not expected)
        bucket["under_refusal"] += int(expected and not refused)
    out = {}
    for kind, value in sorted(table.items()):
        answerable = value["turns"] - value["expected_decline"]
        out[kind] = {
            **value,
            "agreement_rate": round(value["agree"] / max(value["turns"], 1), 4),
            "expected_decline_rate": round(value["expected_decline"] / max(value["turns"], 1), 4),
            "adapter_refusal_rate": round(value["refused"] / max(value["turns"], 1), 4),
            "over_refusal_rate": round(value["over_refusal"] / answerable, 4) if answerable else None,
            "under_refusal_rate": round(value["under_refusal"]
                                        / max(value["expected_decline"], 1), 4)
            if value["expected_decline"] else None,
        }
    return out


def rescore(rows: Sequence[Dict], dialogues: Dict[str, Dict], records: Dict[str, Dict]) -> Dict:
    router = ResponseRouter(enable_proposed=True)
    entries = []
    for row in rows:
        dialogue = dialogues.get(row.get("dialogue_id")) or {}
        keys = dialogue.get("keys") or {}
        record = records.get(f"{keys.get('source_file')}|{keys.get('fault')}"
                             f"|{keys.get('simulationRun')}") or {}
        event = record.get("event") or {}
        question = bare_question(row.get("question") or "")
        route = router.route(event, {}, question)
        entries.append({
            "kind": row.get("kind"),
            "dialogue_id": row.get("dialogue_id"),
            "question": question,
            "intent": route.primary.value if route.primary else None,
            "answerability": route.answerability,
            "store_resolves": route.deterministic_answer is not None,
            "expected_decline_store": route.deterministic_answer is None,
            "expected_decline_router": route.answerability == "refuse",
            "deterministic_answer": (route.deterministic_answer or "")[:160] or None,
            "adapter_refused": bool(row.get("refused")),
        })
    store = _tally(entries, "expected_decline_store")
    verdict = _tally(entries, "expected_decline_router")
    coverage = {}
    for kind in sorted({entry["kind"] for entry in entries}):
        subset = [entry for entry in entries if entry["kind"] == kind]
        resolved = [entry for entry in subset if entry["store_resolves"]]
        coverage[kind] = {
            "turns": len(subset),
            "store_resolves": len(resolved),
            "store_coverage": round(len(resolved) / max(len(subset), 1), 4),
            "refusal_rate_on_resolved": round(
                sum(1 for entry in resolved if entry["adapter_refused"]) / len(resolved), 4)
            if resolved else None,
            "refusal_rate_on_unresolved": round(
                sum(1 for entry in subset
                    if not entry["store_resolves"] and entry["adapter_refused"])
                / max(len(subset) - len(resolved), 1), 4)
            if len(subset) - len(resolved) else None,
        }
    def total(table: Dict) -> Dict:
        keys = ("turns", "expected_decline", "refused", "agree", "over_refusal", "under_refusal")
        summed = {key: sum(entry[key] for entry in table.values()) for key in keys}
        summed["agreement_rate"] = round(summed["agree"] / max(summed["turns"], 1), 4)
        return summed
    return {"rows": entries, "by_kind_store": store, "by_kind_router": verdict,
            "store_coverage": coverage,
            "total_store": total(store), "total_router": total(verdict)}


def training_decline(mixed: Sequence[Dict], dialogues: Dict[str, Dict]) -> Dict:
    """The same fraction on the dialogue turns that are actually in the mix.

    The mixed file is one row per assistant turn (``FD-train-00967#2``), so the case
    type comes from the dialogue it belongs to, and the text from the exported row --
    the same string the trainer sees.
    """
    per_case = defaultdict(lambda: {"turns": 0, "refusal_vocabulary": 0, "context_absent": 0,
                                    "either": 0})
    for row in mixed:
        match = MIXED_DIALOGUE.match(str(row.get("item_id") or ""))
        if not match:
            continue
        dialogue = dialogues.get(match.group(1)) or {}
        index = int(match.group(2))
        turns = [turn for turn in (dialogue.get("turns") or []) if turn.get("role") == "assistant"]
        case = str(dialogue.get("case_type") or "unknown")
        if index < len(turns) and turns[index].get("case_type"):
            case = str(turns[index]["case_type"])
        case = ALIASES.get(case, case)
        flags = declared_decline(str(row.get("answer") or ""))
        bucket = per_case[case]
        bucket["turns"] += 1
        bucket["refusal_vocabulary"] += int(flags["refusal_vocabulary"])
        bucket["context_absent"] += int(flags["context_absent"])
        bucket["either"] += int(flags["refusal_vocabulary"] or flags["context_absent"])
    return {case: {**value,
                   "decline_fraction": round(value["either"] / max(value["turns"], 1), 4),
                   "refusal_vocabulary_fraction": round(value["refusal_vocabulary"]
                                                        / max(value["turns"], 1), 4)}
            for case, value in sorted(per_case.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    dialogues = {row["dialogue_id"]: row for row in load_jsonl(FOLLOWUP_VAL)}
    train_dialogues = {row["dialogue_id"]: row for row in load_jsonl(FOLLOWUP_TRAIN)}
    rows = json.loads(SANITY.read_text(encoding="utf-8"))["rows"]
    rescored = rescore(rows, dialogues, event_index())
    train = training_decline(load_jsonl(TRAIN_MIXED), train_dialogues)

    comparison = {}
    for kind in sorted(set(rescored["by_kind_router"]) | set(train)):
        entry = rescored["by_kind_router"].get(kind) or {}
        store = rescored["by_kind_store"].get(kind) or {}
        comparison[kind] = {
            "turns": entry.get("turns"),
            "store_coverage": (rescored["store_coverage"].get(kind) or {}).get("store_coverage"),
            "expected_decline_router": entry.get("expected_decline_rate"),
            "expected_decline_store": store.get("expected_decline_rate"),
            "adapter_refusal_rate": entry.get("adapter_refusal_rate"),
            "over_refusal_rate_router": entry.get("over_refusal_rate"),
            "under_refusal_rate_router": entry.get("under_refusal_rate"),
            "train_decline_fraction": (train.get(kind) or {}).get("decline_fraction"),
            "train_turns": (train.get(kind) or {}).get("turns"),
        }
    result = {"total_store": rescored["total_store"],
              "total_router": rescored["total_router"],
              "by_kind_store": rescored["by_kind_store"],
              "by_kind_router": rescored["by_kind_router"],
              "store_coverage": rescored["store_coverage"],
              "train_mixed_decline": train, "comparison": comparison,
              "rows": rescored["rows"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"total_router": result["total_router"],
                      "total_store": result["total_store"],
                      "comparison": comparison}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
