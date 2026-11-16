"""Score a fresh adapter decode against the stored baseline predictions.

Two questions, kept apart:

* **accuracy** — how often the JSON `fault_id` equals the item's gold label, on the same 497 items the stored
  baseline covers;
* **reproducibility** — how often the fresh label equals the label in the stored file, item by item. This is the
  one that answers whether `preds_adapter_val503.jsonl` can be regenerated at all.

A missing or unparseable `fault_id` counts as wrong for accuracy and as a disagreement for reproducibility; the
counts are printed so a parse problem cannot hide as a model error.

    python scripts/score_adapter_decode.py --decode .../preds_adapter_val497.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DECODE = ROOT / "outputs/benchmark_v2/eval/preds_adapter_val497.jsonl"
DEFAULT_STORED = (ROOT / "outputs/analysis/phase3a/hybrid/vocabfix/evidence_classifier"
                  / "preds_adapter_val503.jsonl")
DEFAULT_VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"

FAULT = re.compile(r"FAULT[ _-]?(\d{1,2})", re.IGNORECASE)


def parse_label(raw: str) -> str | None:
    """The adapter's answer as the benchmark spells it, or ``None`` when the reply is not a diagnosis."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    value = payload.get("fault_id")
    if value is None:
        return None
    text = str(value).strip()
    if "UNCLASSIFIED" in text.upper():
        return "UNCLASSIFIED_OPEN_SET"
    match = FAULT.search(text)
    if match:
        return f"FAULT_{int(match.group(1)):02d}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", type=pathlib.Path, default=DEFAULT_DECODE)
    parser.add_argument("--stored", type=pathlib.Path, default=DEFAULT_STORED)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    decoded = [json.loads(line) for line in args.decode.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    stored = {row["item_id"]: row for row in
              (json.loads(line) for line in args.stored.read_text(encoding="utf-8").splitlines())}

    compared = [row for row in decoded if row["item_id"] in stored]
    labels = {row["item_id"]: parse_label(row["raw"]) for row in compared}
    unparsed = [item for item, label in labels.items() if label is None]

    accuracy = sum(1 for row in compared if labels[row["item_id"]] == stored[row["item_id"]]["gold"])
    agreement = sum(1 for row in compared if labels[row["item_id"]] == stored[row["item_id"]]["pred"])
    stored_accuracy = sum(1 for row in compared if stored[row["item_id"]]["pred"]
                          == stored[row["item_id"]]["gold"])

    outcome = {
        "decoded_rows": len(decoded),
        "stored_rows": len(stored),
        "compared": len(compared),
        "unparsed": len(unparsed),
        "unparsed_examples": unparsed[:5],
        "accuracy": round(accuracy / max(len(compared), 1), 4),
        "stored_accuracy_on_same_items": round(stored_accuracy / max(len(compared), 1), 4),
        "label_agreement_with_stored": round(agreement / max(len(compared), 1), 4),
        "coverage_of_stored": round(len(compared) / max(len(stored), 1), 4),
    }
    print(json.dumps(outcome, indent=2))
    if args.out:
        args.out.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
