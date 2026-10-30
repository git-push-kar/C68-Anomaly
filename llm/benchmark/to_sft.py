"""Export the benchmark items as ``{system, question, answer}`` SFT rows.

The trainer (``llm/train_adapter.py``) reads rows of ``question``/``answer`` with an
optional ``system`` and masks everything up to the assistant turn. It also falls
back to ``llm.dataset.SYSTEM_PROMPT`` when a row has no system, and that constant
still contains the fault->severity rubric -- the exact hint the benchmark removed.
So this exporter always writes an explicit system, and refuses to write a row whose
system or question carries the rubric or the subsystem hint. A missing system can
therefore never silently become the old prompt.

The answer is the **slim** report body as JSON text: the label, the verdict, the
ranked hypothesis ids, and the teacher's two prose fields. Subsystem, mitigation,
table choice, suggested follow-ups and the clarifying question are removed: they
are derived in code at render time (``llm.benchmark.hydrate``), so encoding them
in the target only gave the model a chance to disagree with the KB and cost answer
tokens on every row.

``severity`` and ``confidence`` stay absent for the same reason they always were --
severity is computed from detector statistics, and confidence is not emitted until
a calibrator exists.

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
SPLITS = Path("data/splits_v2.json")

#: Protocol B folds, by the name used in the split file.
FOLDS = ("fold_1_cooling_loop_holdout", "fold_2_valve_sticking_family")

#: Keys the student emits, in order. No ``severity``, no ``confidence``, and no
#: field the render path can derive from the KB or the router.
ANSWER_KEYS = (
    "fault_id", "verdict", "ranked_hypotheses", "physical_reasoning", "narrative",
)

#: Fields the full target still carries for evaluation but the model must not
#: emit. Listed so a guard can fail if one reappears in the answer.
DERIVED_AT_RENDER = (
    "abstain", "primary_subsystem", "mitigation", "table_selection",
    "suggested_followups", "clarifying_question",
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
    """The slim report body for one item: label, verdict, ranking, teacher prose."""
    target = item.get("target") or {}
    prose = item.get("prose") or {}
    body = {
        "fault_id": target.get("fault_id"),
        "verdict": target.get("verdict"),
        "ranked_hypotheses": [hyp.get("fault_id")
                              for hyp in (target.get("ranked_hypotheses") or [])],
        "physical_reasoning": prose.get("physical_reasoning"),
        "narrative": prose.get("narrative"),
    }
    # A NORMAL_TRANSIENT item has no fault candidates at all, so an empty
    # hypothesis list is the honest answer there.
    optional = {"ranked_hypotheses"}
    missing = [key for key in ANSWER_KEYS
               if key not in optional and body.get(key) in (None, "", [])]
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
    # A derived field must not creep back into the answer: if it does, the model is
    # being trained to recite something the render path overwrites.
    leaked = [key for key in DERIVED_AT_RENDER if key in body]
    if leaked:
        raise ExportGuardError(
            f"{item.get('item_id')}: answer carries render-derived field(s) {leaked}")
    # Two verdicts abstain: the open-set disturbance and the normal excursion.
    # Anything else claiming to abstain is labelled inconsistently. ``abstain`` is
    # no longer a field, so the check reads the verdict.
    abstaining = body["verdict"] in ("unclassified_open_set", "no_fault")
    if abstaining and body["fault_id"] not in ("UNCLASSIFIED_OPEN_SET", "NORMAL_TRANSIENT"):
        raise ExportGuardError(
            f"{item.get('item_id')}: abstaining verdict labelled {body['fault_id']}")
    if not abstaining and body["fault_id"] in ("UNCLASSIFIED_OPEN_SET", "NORMAL_TRANSIENT"):
        raise ExportGuardError(
            f"{item.get('item_id')}: non-abstaining verdict with {body['fault_id']}")
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


def _fault_number(value: object) -> Optional[int]:
    text = str(value or "")
    if text.startswith(("FAULT_", "SYNTHETIC_")):
        try:
            return int(text.split("_")[-1])
        except ValueError:
            return None
    return None


def fold_faults(fold: str) -> List[int]:
    """The faults a fold holds out, read from the split file rather than typed here."""
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    block = splits["protocols"]["B_leave_faults_out"]["folds"][fold]
    return [int(number) for number in block["held_out_faults"]]


def fold_rows(fold: str) -> List[Dict]:
    """Train rows for one Protocol B fold: no held-out fault, no episode from one.

    An open-set episode derived from a held-out fault has that fault's menu entry
    removed, so its prompt still describes the held-out mechanism -- keeping it
    would leak the fold's task into training through the evidence rather than
    through the label. Both are dropped here, and the asserts below make the
    claim checkable instead of asserted in prose.
    """
    held = {int(number) for number in fold_faults(fold)}
    rows: List[Dict] = []
    for item in load_split("train"):
        fault = item.get("fault")
        if fault is not None and int(fault) in held:
            continue
        if item.get("target", {}).get("truth_fault_number") in held:
            continue
        if not item.get("prose"):
            continue
        row = to_row(item)
        body = json.loads(row["answer"])
        # A kept fault's KB lookalikes can include a held-out one (fault 4's
        # differential names fault 15). The menu still lists every class, so
        # nothing is hidden from the model -- but a training *answer* must not
        # teach it to rank a class the fold is evaluating.
        body["ranked_hypotheses"] = [
            value for value in body["ranked_hypotheses"]
            if _fault_number(value) not in held]
        row["answer"] = json.dumps(body, ensure_ascii=False)
        assert fault is None or int(fault) not in held, f"{fold}: held-out label kept"
        assert not [value for value in body["ranked_hypotheses"]
                    if _fault_number(value) in held], f"{fold}: held-out ranking kept"
        rows.append(row)
    return rows


def write_split(split: str, out_dir: Path = OUT_DIR) -> Optional[Path]:
    """Write one split, or nothing when it has no teacher prose.

    ``val``/``test`` are evaluation sets: they carry deterministic targets and no
    teacher prose, so a training row built from them would have empty prose. They
    are not written at all rather than written with a blank answer -- an empty
    file that looks like a training split is worse than a missing one.
    """
    rows = export(load_split(split))
    if not rows:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_fold(fold: str, out_dir: Path = OUT_DIR) -> Path:
    rows = fold_rows(fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"train_{fold}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--all", action="store_true", help="every split")
    parser.add_argument("--fold", default=None, choices=list(FOLDS) + ["all"],
                        help="also write the Protocol B train file(s)")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    splits = ["train", "val", "test"] if args.all or not args.split else [args.split]
    for split in splits:
        path = write_split(split, Path(args.out))
        if path is None:
            print(f"{split:5s} skipped: no teacher prose (evaluation split)")
            continue
        rows = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        print(f"{split:5s} {rows:5d} rows -> {path}")
    folds = list(FOLDS) if args.fold == "all" else ([args.fold] if args.fold else [])
    for fold in folds:
        path = write_fold(fold, Path(args.out))
        rows = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        held = ", ".join(str(number) for number in fold_faults(fold))
        print(f"fold   {rows:5d} rows -> {path} (held out {held})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
