"""Assemble the four-instrument tables, the freeze input, and the McNemar tests.

Reads what ``instrument_check.py`` wrote and produces one JSON the report quotes, so
none of the numbers are transcribed by hand. It also does the two comparisons the
check asks for that need paired predictions: adapter-decode against the tabular
reference on the same 497 items, and each instrument against the others.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/instrument_check"
CLASSIFIER = ROOT / "outputs/analysis/phase3a/evidence_classifier"
DEST = ROOT / "outputs/analysis/phase3a/instrument_report.json"
INSTRUMENTS = ("mean", "joint", "hierarchical", "decode")


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table() -> Dict:
    table = load_json(OUT / "checkpoint_table_instruments.json") or []
    rows = []
    for entry in table:
        row = {"step": entry["step"]}
        for instrument in INSTRUMENTS:
            data = entry.get(instrument) or {}
            row[instrument] = {key: data.get(key) for key in
                               ("accuracy", "top3", "macro_f1", "seconds_per_item",
                                "fault15_recall", "normal_recall")}
            row[instrument]["abstention"] = (data.get("abstention") or {})
        rows.append(row)
    selections = {}
    for instrument in INSTRUMENTS:
        candidates = [row for row in rows if row[instrument]["macro_f1"] is not None]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: (row[instrument]["macro_f1"],
                                                row[instrument]["accuracy"], row["step"]))
        selections[instrument] = {"step": best["step"],
                                  "macro_f1": best[instrument]["macro_f1"],
                                  "accuracy": best[instrument]["accuracy"]}
    return {"rows": rows, "selection": selections}


def fullval(step: str) -> Dict:
    data = load_json(OUT / f"fullval_{step}.json")
    if not data:
        return {"present": False}
    out = {"present": True}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        out[key] = {"items": value.get("items"), "accuracy": value.get("accuracy"),
                    "top3": value.get("top3"), "macro_f1": value.get("macro_f1"),
                    "abstention": value.get("abstention"),
                    "per_fault": value.get("per_fault")}
    return out


def classifier_mcnemar() -> Dict:
    """Adapter-decode vs GBDT and MLP on the 497 items all three answer."""
    adapter = load_jsonl(OUT / "preds_decode_step300_full.jsonl")
    if not adapter:
        return {"present": False}
    gbdt = load_jsonl(CLASSIFIER / "preds_gbdt_val503.jsonl")
    mlp = load_jsonl(CLASSIFIER / "preds_mlp_val503.jsonl")
    adapter = [row for row in adapter if not row.get("unobservable")]
    common = ({row["item_id"] for row in adapter}
              & {row["item_id"] for row in gbdt}
              & {row["item_id"] for row in mlp})
    adapter = [row for row in adapter if row["item_id"] in common]
    gbdt = [row for row in gbdt if row["item_id"] in common]
    mlp = [row for row in mlp if row["item_id"] in common]

    def summary(rows, name):
        report = ev.report(rows, name)
        return {"items": report["items"], "accuracy": report["accuracy"],
                "macro_f1": report["macro_f1"], "top3": report["top3"],
                "errors": round((1 - report["accuracy"]) * report["items"]),
                "per_class": {label: {"precision": info["precision"],
                                      "recall": info["recall"], "f1": info["f1"],
                                      "support": info["support"]}
                              for label, info in report["per_fault"].items()}}

    result = {"present": True, "items": len(common),
              "adapter": summary(adapter, "adapter:decode"),
              "gbdt": summary(gbdt, "gbdt"),
              "mlp": summary(mlp, "mlp"),
              "mcnemar_adapter_vs_gbdt": ev.mcnemar(adapter, gbdt),
              "mcnemar_adapter_vs_mlp": ev.mcnemar(adapter, mlp)}
    # Per-class difference, which says where the classifier wins rather than how often.
    diffs = {}
    for label in sorted(result["adapter"]["per_class"]):
        a = result["adapter"]["per_class"][label]
        g = result["gbdt"]["per_class"].get(label, {})
        diffs[label] = {"adapter_f1": a["f1"], "gbdt_f1": g.get("f1"),
                        "delta_f1": round((g.get("f1") or 0) - (a["f1"] or 0), 4),
                        "support": a["support"]}
    result["per_class_delta"] = diffs
    return result


def instrument_agreement(step: str = "step300") -> Dict:
    """Pairwise McNemar between instruments at one checkpoint."""
    rows = {instrument: load_jsonl(OUT / f"preds_{instrument}_{step}.jsonl")
            for instrument in INSTRUMENTS}
    out = {}
    for first in INSTRUMENTS:
        for second in INSTRUMENTS:
            if first >= second or not rows[first] or not rows[second]:
                continue
            out[f"{first}_vs_{second}"] = ev.mcnemar(rows[first], rows[second])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEST))
    args = parser.parse_args()
    result = {
        "table": table(),
        "fullval_step300": fullval("300"),
        "fullval_step540": fullval("540"),
        "classifier_vs_adapter": classifier_mcnemar(),
        "instrument_agreement_step300": instrument_agreement("step300"),
        "classifier_summary": load_json(CLASSIFIER / "summary.json"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["table"]["selection"], indent=1))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
