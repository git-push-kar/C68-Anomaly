"""The tag-kind rule's recall, measured on the 21 it was written after — and on 60 it was not.

§17 counted 21 "tag-kind mislabels" in the 242 main-batch answers with a 40-character window.
3B-7 hand-read all 21 and reproduced the window; the attribution rule in
`llm/hybrid/tag_kind.py` finds **none** of them, because none is a mislabel. That leaves the rule
with a real problem the 3B-7 report named but did not measure: **it was written with those 21 in
view**, so any score it gets on them measures its author, not the rule.

This script therefore reports three things, and keeps them apart:

1. **the 21, exported** — tag, quantity word, the old rule's verdict, the new rule's verdict, and a
   blank human column, so the judgement is the reader's rather than mine;
2. **30 synthetic positives and 30 matched negatives**, written *after* the rule, in the four
   syntaxes that matter (copula, kind-noun, appositive, passive) and matched one-for-one, so recall
   and specificity are measured on sentences the rule was not tuned on;
3. **both variants' counts on the three stored answer sets** (242 main, 58 counterfactual, 21
   retry), so the permissive variant's cost is visible where the rule would actually be applied.

The **permissive, appositive-aware variant** adds two predicates the strict rule deliberately does
not have: apposition (`XMV_45, a temperature reading, ...`) and passive predication (`a temperature
is reported by XMV_45`). It is measured, **not adopted** — §20.8 requires a dated note before any
adoption, and the negative set shows what it costs.

CPU only, no model, no generation.

Usage::

    python scripts/validator_recall.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.hybrid import tag_kind as tk  # noqa: E402
from scripts.tag_kind_audit import old_window  # noqa: E402

HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"

#: 30 synthetic positives: a valve tag carrying a measurement quantity, or a measurement tag
#: carrying a valve quantity, across four syntaxes.
POSITIVES: Sequence[Tuple[str, str, str]] = (
    ("copula", "XMV_45", "XMV_45 is a reactor pressure reading of 2.7 bar."),
    ("copula", "XMV_51", "XMV_51 is a jacket temperature signal."),
    ("copula", "XMEAS_7", "XMEAS_7 shows a valve position of 62%."),
    ("copula", "XMEAS_18", "XMEAS_18 reads a valve opening near half stroke."),
    ("copula", "XMV_52", "XMV_52 is a condenser flow measurement."),
    ("copula", "XMV_50", "XMV_50 reads a steam temperature."),
    ("copula", "XMEAS_19", "XMEAS_19 indicates a valve position that is steady."),
    ("copula", "XMV_44", "XMV_44 represents a feed flow measurement."),
    ("copula", "XMEAS_9", "XMEAS_9 is a valve stroke value."),
    ("copula", "XMV_42", "XMV_42 behaves as a composition sensor."),
    ("kind_noun", "XMV_45", "the pressure reading of XMV_45 drifts."),
    ("kind_noun", "XMV_51", "the temperature signal of XMV_51 is flat."),
    ("kind_noun", "XMEAS_7", "the valve position of XMEAS_7 saturates."),
    ("kind_noun", "XMV_52", "the level measurement of XMV_52 falls."),
    ("kind_noun", "XMV_44", "the composition value of XMV_44 shifts."),
    ("kind_noun", "XMEAS_19", "the valve stem of XMEAS_19 is steady."),
    ("kind_noun", "XMV_50", "the flow reading from XMV_50 increases."),
    ("kind_noun", "XMEAS_11", "the actuator value of XMEAS_11 is fixed."),
    ("appositive", "XMV_45", "XMV_45, a temperature reading, drifts upward."),
    ("appositive", "XMV_52", "XMV_52, the condenser flow measurement, falls."),
    ("appositive", "XMEAS_7", "XMEAS_7, a valve position, saturates."),
    ("appositive", "XMV_51", "XMV_51, a pressure signal, stays high."),
    ("appositive", "XMEAS_19", "XMEAS_19, the valve stroke value, is pinned."),
    ("appositive", "XMV_44", "XMV_44, a composition measurement, swings."),
    ("passive", "XMV_45", "a reactor pressure is reported by XMV_45."),
    ("passive", "XMV_51", "a jacket temperature is measured by XMV_51."),
    ("passive", "XMEAS_7", "a valve position is shown by XMEAS_7."),
    ("passive", "XMV_52", "a condenser flow is registered by XMV_52."),
    ("passive", "XMEAS_18", "a valve stroke is indicated by XMEAS_18."),
    ("passive", "XMV_42", "a composition value is supplied by XMV_42."),
)

#: 30 matched negatives: the same four syntaxes, with the quantity the tag's own kind carries, or
#: a guard the rule is supposed to honour (negation, associative `for`, another sensor's name).
NEGATIVES: Sequence[Tuple[str, str, str]] = (
    ("copula", "XMV_45", "XMV_45 is a valve position that keeps opening."),
    ("copula", "XMV_51", "XMV_51 is a manipulated variable."),
    ("copula", "XMEAS_7", "XMEAS_7 is a pressure measurement."),
    ("copula", "XMEAS_18", "XMEAS_18 reads a temperature of 120.4 C."),
    ("copula", "XMV_52", "XMV_52 is not a pressure reading."),
    ("copula", "XMV_50", "XMV_50 is a valve position, not a temperature."),
    ("copula", "XMEAS_19", "XMEAS_19 is a flow measurement."),
    ("copula", "XMV_44", "XMV_44 is a valve position signal."),
    ("copula", "XMEAS_9", "XMEAS_9 is a temperature sensor."),
    ("copula", "XMV_42", "XMV_42 indicates a manipulated variable."),
    ("kind_noun", "XMV_45", "the valve position of XMV_45 drifts."),
    ("kind_noun", "XMEAS_9", "the temperature reading of XMEAS_9 falls."),
    ("kind_noun", "XMEAS_7", "the pressure signal from XMEAS_7 is flat."),
    ("kind_noun", "XMV_52", "the flow measurement for XMV_52 drops."),
    ("kind_noun", "XMV_51", "the valve stem of XMV_51 is steady."),
    ("kind_noun", "XMEAS_23", "the composition value of XMEAS_23 shifts."),
    ("kind_noun", "XMEAS_19", "the flow reading of XMEAS_19 increases."),
    ("kind_noun", "XMV_42", "the actuator value of XMV_42 is fixed."),
    ("appositive", "XMV_45", "XMV_45, a valve position, keeps opening."),
    ("appositive", "XMV_52", "XMV_52, the condenser cooling water flow valve, falls."),
    ("appositive", "XMEAS_7", "XMEAS_7, a pressure measurement, saturates."),
    ("appositive", "XMV_51", "XMV_51, a manipulated variable, stays high."),
    ("appositive", "XMEAS_19", "XMEAS_19, a flow measurement, is pinned."),
    ("appositive", "XMV_44", "XMV_44, the A feed flow valve, swings."),
    ("passive", "XMV_45", "a valve position is reported by XMV_45."),
    ("passive", "XMEAS_9", "a temperature is measured by XMEAS_9."),
    ("passive", "XMEAS_7", "a pressure is shown by XMEAS_7."),
    ("passive", "XMEAS_19", "a condenser flow is registered by XMEAS_19."),
    ("passive", "XMV_51", "a valve stroke is indicated by XMV_51."),
    ("passive", "XMEAS_23", "a composition value is supplied by XMEAS_23."),
)


def _quantities(kind: str) -> List[str]:
    if kind == "valve_position":
        return [word for words in tk.MEASUREMENT_QUANTITIES.values() for word in words]
    return list(tk.VALVE_QUANTITIES)


def violations_permissive(text: str) -> List[Dict]:
    """The strict rule, plus apposition and passive predication at sentence scope.

    Apposition: the quantity sits within four words after the tag (`XMV_45, a temperature
    reading, ...`). Passive: it sits before the tag with `by`/`from` between them (`a temperature
    is reported by XMV_45`). Both are the cases the strict clause-scoped rule cannot see, and both
    are exactly where the permissive variant buys recall with precision.
    """
    found: List[Dict] = list(tk.violations(text))
    for sentence in tk._SENTENCE.split(str(text or "")):
        if not sentence.strip():
            continue
        masked = tk.mask(sentence)
        lowered = masked.lower()
        for match in tk._TAG.finditer(masked):
            tag = tk.canonical(match.group(0))
            kind = tk.axis(tag)
            after = lowered[match.end():match.end() + 90]
            before = lowered[max(0, match.start() - 90):match.start()]
            for word in _quantities(kind):
                low = word.lower()
                at = after.find(low)
                appositive = at >= 0 and len(after[:at].split()) <= 4
                passive = re.search(rf"\b{re.escape(low)}\b[^.;]{{0,40}}\b(?:by|from)\b[^.;]*$",
                                    before) is not None
                if not (appositive or passive):
                    continue
                entry = {"tag": tag, "axis": kind,
                         "quantity": ("temperature" if low in
                                      tk.MEASUREMENT_QUANTITIES.get("temperature", ()) else
                                      next((name for name, words in tk.MEASUREMENT_QUANTITIES.items()
                                            if low in words), low)),
                         "rule": "permissive appositive/passive", "clause": sentence.strip()}
                if not any(other["tag"] == entry["tag"] and other["quantity"] == entry["quantity"]
                           for other in found):
                    found.append(entry)
    return found


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score(cases: Sequence[Tuple[str, str, str]], rule) -> Dict:
    """Recall or specificity over a labelled set, with the misses listed by syntax."""
    hits, misses = 0, []
    for syntax, tag, sentence in cases:
        found = bool(rule(sentence))
        if found:
            hits += 1
        else:
            misses.append({"syntax": syntax, "tag": tag, "sentence": sentence})
    by_syntax: Dict[str, Dict[str, int]] = {}
    for syntax, _tag, sentence in cases:
        bucket = by_syntax.setdefault(syntax, {"n": 0, "flagged": 0})
        bucket["n"] += 1
        bucket["flagged"] += int(bool(rule(sentence)))
    return {"n": len(cases), "flagged": hits, "rate": round(hits / len(cases), 4),
            "by_syntax": by_syntax, "misses": misses}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    rows = load_jsonl(HYBRID / "generation_rows.jsonl")
    main_rows = [row for row in rows if row.get("kind") == "main"]
    counter = [row for row in rows if row.get("kind") == "counterfactual"]
    retry = json.loads((HYBRID / "valve_flow_retry.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- the 21, exported
    exported = []
    for row in main_rows:
        labels = row.get("valve_flow_mislabels") or []
        if not labels:
            continue
        label = str(labels[0])
        tag = label.split()[0]
        reproduced = old_window(row.get("prose") or "", tag) or {}
        strict = tk.violations(row.get("prose") or "")
        exported.append({
            "item_id": row.get("item_id"),
            "tag": tk.canonical(tag),
            "quantity_word": reproduced.get("old_word") or "",
            "old_rule_verdict": "flagged" if reproduced.get("old_word") else "not flagged",
            "new_rule_verdict": ("violation: " + "; ".join(
                f"{entry['tag']} as {entry['quantity']}" for entry in strict)) if strict
            else "no violation",
            "old_rule_window": reproduced.get("window") or "",
            "context": reproduced.get("context") or "",
            "human_verdict": "",
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / "validator_recall_21.csv"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0].keys()))
        writer.writeheader()
        writer.writerows(exported)

    # ---------------------------------------------------------------- synthetic recall
    strict_pos = score(POSITIVES, lambda text: bool(tk.violations(text)))
    strict_neg = score(NEGATIVES, lambda text: bool(tk.violations(text)))
    perm_pos = score(POSITIVES, lambda text: bool(violations_permissive(text)))
    perm_neg = score(NEGATIVES, lambda text: bool(violations_permissive(text)))

    # ---------------------------------------------------------------- counts on the three sets
    def audit(subset: List[Dict], field: str = "prose") -> Dict:
        strict_rows = [row for row in subset if tk.violations(row.get(field) or "")]
        perm_rows = [row for row in subset if violations_permissive(row.get(field) or "")]
        return {"rows": len(subset), "strict_flagged_rows": len(strict_rows),
                "permissive_flagged_rows": len(perm_rows),
                "extra_rows_from_permissive": sorted(
                    {row.get("item_id") for row in perm_rows}
                    - {row.get("item_id") for row in strict_rows})}

    counts = {
        "main_242": audit(main_rows),
        "counterfactual_58": audit(counter),
        "retry_21": {
            "rows": len(retry["rows"]),
            "strict_flagged_rows": sum(1 for row in retry["rows"]
                                       if tk.violations(row.get("prose_after") or "")),
            "permissive_flagged_rows": sum(1 for row in retry["rows"]
                                           if violations_permissive(row.get("prose_after") or "")),
            "note": "counted on prose_after, the text the retry wrote",
        },
    }

    report = {
        "note": "the strict rule was written after hand-reading the 21, so its score on them is "
                "not a measurement of the rule; the synthetic sets were written after the rule",
        "exported_21": {"file": str(sheet), "rows": len(exported),
                        "found_by_strict_rule": sum(1 for entry in exported
                                                    if entry["new_rule_verdict"] != "no violation"),
                        "old_rule_flagged": sum(1 for entry in exported
                                                if entry["old_rule_verdict"] == "flagged")},
        "synthetic": {
            "strict": {"recall": strict_pos, "specificity": strict_neg},
            "permissive_appositive_aware": {"recall": perm_pos, "specificity": perm_neg},
        },
        "counts_on_stored_answers": counts,
        "verdict": {
            "strict": f"recall {strict_pos['flagged']}/30, specificity "
                      f"{30 - strict_neg['flagged']}/30",
            "permissive": f"recall {perm_pos['flagged']}/30, specificity "
                          f"{30 - perm_neg['flagged']}/30",
            "adopted": "strict — the permissive variant is measured, not adopted (§20.8)",
        },
    }
    (out / "validator_recall.json").write_text(json.dumps(report, indent=2, default=str),
                                               encoding="utf-8")
    print(json.dumps({
        "exported_21": report["exported_21"],
        "strict": {"recall": f"{strict_pos['flagged']}/30", "fp": f"{strict_neg['flagged']}/30",
                   "by_syntax": strict_pos["by_syntax"]},
        "permissive": {"recall": f"{perm_pos['flagged']}/30",
                       "fp": f"{perm_neg['flagged']}/30",
                       "by_syntax": perm_pos["by_syntax"],
                       "false_positives": [entry["sentence"] for entry in perm_neg["misses"]]},
        "counts": counts,
    }, indent=2, default=str))
    print(f"written: {out / 'validator_recall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
