"""The three-way comparison, on identical rows: adapter alone, classifier alone, hybrid.

Only the *diagnosis* is compared here, and by construction the hybrid's diagnosis is the
classifier's plus the menu rule -- so the hybrid column is identical to the classifier
column and is carried for readability, not as a second result. What the hybrid changes is
the explanation, measured by `scripts/hybrid_faithfulness.py`.

Reported per stratum, because the aggregate hides the thing that matters: 325 of the 822
observable rows are the open-set twins the adapter answers correctly by abstaining, and 125
more are open-set episodes that appear on a menu. A single accuracy over all of them says
more about the protocol's balance than about either model.

Every input is a saved prediction file, so a re-run of this script cannot disagree with the
tables in the report.

Usage::

    python scripts/hybrid_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
ADAPTER_PREDS = ROOT / "outputs/analysis/phase3a/instrument_check/preds_decode_step500_full.jsonl"
CLASSIFIER_PREDS = OUT / "preds_protocolA_val.jsonl"
SURROGATE = OUT / "surrogate.json"


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


STRATA = {
    "observable": lambda row: not row.get("unobservable"),
    "menu_present": lambda row: not row.get("unobservable") and row.get("variant") == "menu",
    "menu_removed": lambda row: not row.get("unobservable") and row.get("variant") != "menu",
    "fault_named": lambda row: (not row.get("unobservable") and row.get("variant") == "menu"
                               and str(row.get("gold", "")).startswith("FAULT")),
    "normal": lambda row: not row.get("unobservable")
                          and row.get("gold") == "NORMAL_TRANSIENT",
    "open_set_on_menu": lambda row: (not row.get("unobservable")
                                     and row.get("variant") == "menu"
                                     and row.get("gold") == "UNCLASSIFIED_OPEN_SET"),
    "unobservable": lambda row: bool(row.get("unobservable")),
}


def metrics(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {"items": 0}
    report = ev.report(rows, label="")
    return {
        "items": report["items"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "top3": report["top3"],
        "abstention": report["abstention"],
        "ece": report["ece"],
        "brier": report["brier"],
        "accuracy_ci": report["accuracy_ci"],
    }


def per_class(rows: Sequence[Dict]) -> Dict:
    return ev.per_class_prf(rows)


def main() -> int:
    classifier = {row["item_id"]: row for row in load_jsonl(CLASSIFIER_PREDS)}
    adapter = {row["item_id"]: row for row in load_jsonl(ADAPTER_PREDS)}
    shared = [item_id for item_id in classifier if item_id in adapter]

    result: Dict[str, Dict] = {}
    for name, keep in STRATA.items():
        ids = [item_id for item_id in shared if keep(classifier[item_id])]
        left = [adapter[item_id] for item_id in ids]
        right = [classifier[item_id] for item_id in ids]
        result[name] = {
            "adapter": metrics(left),
            "classifier_menu_rule": metrics(right),
            "mcnemar": ev.mcnemar(left, right) if ids else {},
            "per_class_adapter": per_class(left),
            "per_class_classifier": per_class(right),
        }

    outcome = {
        "rows": len(shared),
        "note": ("Hybrid diagnosis == classifier + menu rule; the hybrid column is the same "
                 "numbers, reported once."),
        "threshold": json.loads((OUT / "protocolA/threshold.json").read_text(
            encoding="utf-8")),
        "strata": result,
        "surrogate": json.loads(SURROGATE.read_text(encoding="utf-8"))
        if SURROGATE.exists() else {},
        "menu_present_classifier_only": metrics([classifier[item_id] for item_id in shared
                                                 if classifier[item_id]["variant"] == "menu"
                                                 and not classifier[item_id]["unobservable"]]),
    }
    (OUT / "comparison.json").write_text(json.dumps(outcome, indent=2, default=str),
                                         encoding="utf-8")

    print(f"{'stratum':<18}{'n':>5}  {'adapter':>8} {'cls+rule':>9}  {'adapter_f1':>10} "
          f"{'cls_f1':>8}  {'abst_prec':>9} {'abst_rec':>8}")
    for name in STRATA:
        block = result[name]
        a, c = block["adapter"], block["classifier_menu_rule"]
        if not c.get("items"):
            continue
        print(f"{name:<18}{c['items']:>5}  {a.get('accuracy'):>8} {c.get('accuracy'):>9}  "
              f"{a.get('macro_f1'):>10} {c.get('macro_f1'):>8}  "
              f"{c['abstention'].get('abstention_precision'):>9} "
              f"{c['abstention'].get('abstention_recall'):>8}")
    print()
    print(json.dumps({name: result[name]["mcnemar"] for name in STRATA}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
