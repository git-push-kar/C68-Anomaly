"""D33: score the confirmatory pass on the test split as a B-versus-A contrast.

Both conditions are scored by the same instrument that produced the validation cell table
(`scripts/pilot_rechecks.measure`, wrapped by `normalise_flags` in `scripts/overnight_run.py`):

* **B** is the adapter's generated prose from `raw_testkb_kb.jsonl`.
* **A** is rebuilt by `fallback_v31` from the same rows, with the candidate probabilities carried by
  `test.jsonl`; on validation A was built from crossfit predictions, so this is the one construction
  difference and it is declared in the output.

No gate, threshold or selection rule is applied here: the script reports rates, counts, and a McNemar test
on the primary (numeric defect) contrast. Deciding what the numbers mean is left to the report.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fallback_v2 import parse_store  # noqa: E402
from scripts.fallback_v3 import load_overlay  # noqa: E402
from scripts.fallback_v31 import fallback_v31  # noqa: E402
from scripts.overnight_run import normalise_flags  # noqa: E402
from scripts.pilot_rechecks import measure  # noqa: E402
from scripts.prose_pilot_prompts import parse_prose  # noqa: E402

CHECKS = ("numeric", "grounding", "contradiction", "tag_role")
ROWS = ROOT / "outputs/benchmark_v2/eval/prose_test/rows_test.jsonl"
RAW = ROOT / "outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl"
PROBS = ROOT / "outputs/analysis/phase3a/hybrid/crossfit_test_rows.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/d33"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def probs_of(item: dict) -> dict:
    raw = item.get("probs") or {}
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = {}
    return {str(k): float(v) for k, v in raw.items() if v is not None}


def mcnemar(b: int, c: int) -> float:
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = load_jsonl(pathlib.Path(args.rows))
    raws = {r["item_id"]: r for r in load_jsonl(pathlib.Path(args.raw))}
    probs = {item["item_id"]: probs_of(item) for item in load_jsonl(PROBS)}
    if not probs:
        raise SystemExit("no candidate probabilities: run scripts/crossfit_test_rows.py first")
    overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {cond: Counter() for cond in ("A", "B")}
    anchor = {cond: Counter() for cond in ("A", "B")}
    per_item = []
    missing = 0
    for row in rows:
        item = row["item_id"]
        if item not in raws:
            missing += 1
            continue
        if item not in probs:
            raise SystemExit("missing test probabilities for %s" % item)
        prompt = row["prompts"]["kb"]
        store = parse_store(prompt)
        answer = fallback_v31(str(row["gold_fault_id"]), str(row["gold_verdict"]),
                              row.get("menu_ids") or [], store, probs.get(item, {}), overlay)
        a_text = (str(answer.get("physical_reasoning") or "") + " "
                  + str(answer.get("narrative") or "")).strip()
        parsed = parse_prose(raws[item]["text"])
        b_text = parsed["prose"] if parsed["schema_valid"] else str(raws[item]["text"])
        flags = {}
        for cond, text in (("A", a_text), ("B", b_text)):
            res = normalise_flags(measure(text, prompt, row["gold_fault_id"],
                                          row["gold_verdict"], store))
            flag_counts = {c: len(res[c]) for c in CHECKS}
            if isinstance(res["anchor"], dict):
                anchor[cond][str(res["anchor"].get("status"))] += 1
            for c, n in flag_counts.items():
                counts[cond][c] += 1 if n else 0
            counts[cond]["rows"] += 1
            flags[cond] = flag_counts
        per_item.append({"item_id": item, "A": flags["A"], "B": flags["B"]})

    total = counts["A"]["rows"]
    summary = {"rows": total, "missing_generations": missing,
               "construction_note": ("A rebuilt with test.jsonl probabilities; on validation A used crossfit "
                                     "predictions (declared deviation)"),
               "counts": {c: {k: counts[c][k] for k in CHECKS} for c in ("A", "B")},
               "rates": {c: {k: round(counts[c][k] / max(total, 1), 4) for k in CHECKS}
                         for c in ("A", "B")},
               "anchor_status": {c: dict(anchor[c]) for c in ("A", "B")}}
    b_only = sum(1 for r in per_item if r["B"]["numeric"] and not r["A"]["numeric"])
    a_only = sum(1 for r in per_item if r["A"]["numeric"] and not r["B"]["numeric"])
    summary["numeric_mcnemar"] = {"b_only_B_defect": b_only, "c_only_A_defect": a_only,
                                  "p": round(mcnemar(b_only, a_only), 8)}
    (out / "test_contrast.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
