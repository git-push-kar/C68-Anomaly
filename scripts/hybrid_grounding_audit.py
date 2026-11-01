"""Adjudicate every grounding flag, for the hybrid batch and for the run-1 reference.

`evaluate_v2.grounding_violations` is deliberately strict — it names every tag the evidence
does not contain and every number that does not appear in the store — which is the right
default for a rejection rule and the wrong instrument for *counting* fabrications. On the
hybrid batch its 27 flags turn out to be mostly three mechanical artefacts:

``inside_tag``      the digits are part of a sensor name (``Component_F_Stream9``, ``XMEAS_34``)
``substring``       the match is a piece of a longer number the prose also writes (``0.33``
                    inside ``0.333``)
``rounding``        the value is a stored percentage written to fewer digits (``223`` for
                    ``222.7``)
``genuine``         none of the above: the number is not in the store and is not a rounding

Both batches are adjudicated with the same function so the comparison in the report is like
for like. Nothing here changes a model output or a filter; it classifies an instrument's
flags.

Usage::

    python scripts/hybrid_grounding_audit.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
RUN1 = ROOT / "outputs/benchmark_v2/eval/generation_512.json"
TAG_LIKE = re.compile(r"[A-Za-z_][A-Za-z_]*\d")


def prompts() -> Dict[str, str]:
    rows = [json.loads(line) for line in VAL.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return {row["item_id"]: row["prompt"] for row in rows}


def adjudicate(text: str, number: str, stored: Sequence[float]) -> str:
    value = float(number)
    for match in re.finditer(re.escape(number), text):
        before = text[:match.start()]
        # digits attached to letters are a name, not a quantity
        if before and (before[-1].isalnum() or before[-1] == "_"):
            return "inside_tag"
        after = text[match.end():match.end() + 1]
        # a longer number the prose also writes: 0.33 in 0.333, 22 in 222.7
        if after.isdigit() or (after == "." and text[match.end() + 1:match.end() + 2].isdigit()):
            return "substring"
        break
    scaled = [abs(value - other) for other in list(stored) + [s * 100 for s in stored]]
    if any(gap <= max(1.0, 0.02 * max(abs(value), 1e-9)) for gap in scaled):
        return "rounding"
    return "genuine"


def audit(items: Iterable[Dict], prompts_by_id: Dict[str, str]) -> Dict:
    counts: Counter = Counter()
    rows_with_genuine: List[Dict] = []
    flags = 0
    for item in items:
        text = item.get("prose") or item.get("answer") or ""
        prompt = prompts_by_id.get(item["item_id"], "")
        stored = [float(value) for value in ev.NUMBER_PATTERN.findall(prompt)]
        tags = {tag.upper() for tag in ev.TAG_PATTERN.findall(prompt)}
        found = ev.grounding_violations(
            text, sorted(tags),
            [float(value) for value in ev.NUMBER_PATTERN.findall(prompt)])
        genuine = []
        for code in found:
            kind, _, payload = code.partition(":")
            if kind == "tag_not_in_evidence":
                counts["tag_genuine"] += 1
                genuine.append(code)
            else:
                verdict = adjudicate(text, payload, stored)
                counts[f"number_{verdict}"] += 1
                if verdict == "genuine":
                    genuine.append(code)
        flags += len(found)
        if genuine:
            rows_with_genuine.append({"item_id": item["item_id"], "codes": genuine})
    return {"items": len(list(items)) if isinstance(items, list) else None,
            "flags": flags, "by_kind": dict(counts),
            "rows_with_a_genuine_flag": rows_with_genuine,
            "rows": len(rows_with_genuine)}


def main() -> int:
    prompts_by_id = prompts()
    hybrid = [json.loads(line) for line in
              (OUT / "generation_rows.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    main_rows = [row for row in hybrid if row["kind"] == "main"]
    cf_rows = [row for row in hybrid if row["kind"] == "counterfactual"]
    run1 = json.loads(RUN1.read_text(encoding="utf-8"))["rows"]

    outcome = {
        "hybrid_main": audit(main_rows, prompts_by_id),
        "hybrid_counterfactual": audit(cf_rows, prompts_by_id),
        "run1_reference_50": audit([{**row, "prose": row.get("prose") or ""} for row in run1],
                                   prompts_by_id),
        "note": ("`flags` counts what evaluate_v2 rejects; `by_kind` says how many of those are "
                 "mechanical artefacts of a strict matcher. Run-1 rows come from "
                 "outputs/benchmark_v2/eval/generation_512.json (50 items, step 500)."),
    }
    for block in ("hybrid_main", "hybrid_counterfactual", "run1_reference_50"):
        outcome[block]["items"] = len(
            main_rows if block == "hybrid_main" else
            cf_rows if block == "hybrid_counterfactual" else run1)
    (OUT / "grounding_audit.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps({key: {name: outcome[key][name] for name in
                            ("items", "flags", "by_kind", "rows")}
                      for key in ("hybrid_main", "hybrid_counterfactual", "run1_reference_50")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
