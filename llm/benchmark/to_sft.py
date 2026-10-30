"""Export the benchmark items as ``{system, question, answer}`` SFT rows.

The trainer (``llm/train_adapter.py``) reads rows of ``question``/``answer`` with an
optional ``system`` and masks everything up to the assistant turn. It also falls
back to ``llm.dataset.SYSTEM_PROMPT`` when a row has no system, and that constant
still contains the fault->severity rubric -- the exact hint the benchmark removed.
So this exporter always writes an explicit system, and refuses to write a row whose
system or question carries the rubric or the subsystem hint. A missing system can
therefore never silently become the old prompt.

The answer is the report body as JSON text: the deterministic target fields plus
the teacher's two prose fields. ``severity`` and ``confidence`` are deliberately
absent -- severity is computed in code from detector statistics, and confidence is
not emitted at all until a calibrator exists.

Usage:
    python -m llm.benchmark.to_sft --split train
    python -m llm.benchmark.to_sft --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.benchmark.prompt_v2 import SYSTEM_PROMPT_V2  # noqa: E402

DATASET = Path("outputs/benchmark_v2/dataset")
OUT_DIR = Path("outputs/benchmark_v2/sft")

#: Keys the student emits, in order. No ``severity``, no ``confidence``.
ANSWER_KEYS = (
    "fault_id", "verdict", "abstain", "primary_subsystem", "ranked_hypotheses",
    "mitigation", "table_selection", "suggested_followups", "clarifying_question",
    "physical_reasoning", "narrative",
)

#: Strings that must not reach the trainer through the system prompt or question.
#: ``severity`` covers the old rubric's every mention; the rest name the removed
#: hint explicitly.
FORBIDDEN_IN_PROMPT = (
    "severity",
    "severity rubric",
    "affected subsystem",
    "candidate affected subsystem",
    "fault 15-type",
    "for fault 15",
)


class ExportGuardError(RuntimeError):
    """A row would teach the model something the evaluation does not give it."""


def answer_body(item: Dict) -> Dict:
    """The report body for one item: deterministic fields + teacher prose."""
    target = item.get("target") or {}
    prose = item.get("prose") or {}
    body = {
        "fault_id": target.get("fault_id"),
        "verdict": target.get("verdict"),
        "abstain": bool(target.get("abstain")),
        "primary_subsystem": target.get("primary_subsystem"),
        "ranked_hypotheses": [hyp.get("fault_id")
                              for hyp in (target.get("ranked_hypotheses") or [])],
        "mitigation": target.get("mitigation"),
        "table_selection": target.get("table_selection"),
        "suggested_followups": list(target.get("suggested_followups") or []),
        "clarifying_question": target.get("clarifying_question"),
        "physical_reasoning": prose.get("physical_reasoning"),
        "narrative": prose.get("narrative"),
    }
    missing = [key for key in ANSWER_KEYS if body.get(key) in (None, "", [])
               and key not in ("clarifying_question", "abstain")]
    if missing:
        raise ExportGuardError(f"answer body missing {missing} for {item.get('item_id')}")
    return {key: body[key] for key in ANSWER_KEYS}


def to_row(item: Dict) -> Dict:
    """One SFT row, with every guard applied before it can be written."""
    question = str(item.get("prompt") or "")
    if not question:
        raise ExportGuardError(f"no question for {item.get('item_id')}")
    system = SYSTEM_PROMPT_V2
    if not system:
        raise ExportGuardError("SYSTEM_PROMPT_V2 is empty; refusing to fall back")
    lowered = (system + "\n" + question).lower()
    for banned in FORBIDDEN_IN_PROMPT:
        if banned in lowered:
            raise ExportGuardError(
                f"{item.get('item_id')}: system/question contains {banned!r}")
    body = answer_body(item)
    if body["abstain"] and body["fault_id"] != "UNCLASSIFIED_OPEN_SET":
        raise ExportGuardError(
            f"{item.get('item_id')}: abstain row labelled {body['fault_id']}")
    answer = json.dumps(body, ensure_ascii=False)
    return {
        "item_id": item.get("item_id"),
        "system": system,
        "question": question,
        "answer": answer,
    }


def export(items: Iterable[Dict]) -> List[Dict]:
    """Rows for items that have prose; an item without prose is not a report task."""
    rows: List[Dict] = []
    for item in items:
        if not item.get("prose"):
            continue
        rows.append(to_row(item))
    return rows


def load_split(split: str) -> List[Dict]:
    path = DATASET / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_split(split: str, out_dir: Path = OUT_DIR) -> Path:
    rows = export(load_split(split))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--all", action="store_true", help="every split")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    splits = ["train", "val", "test"] if args.all or not args.split else [args.split]
    for split in splits:
        path = write_split(split, Path(args.out))
        rows = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        print(f"{split:5s} {rows:5d} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
