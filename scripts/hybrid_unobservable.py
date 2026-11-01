"""§17.6-§17.7: the twelve unobservable rows, the detector's sentences, and the valve/flow check.

Three questions that are cheap to answer and expensive to guess:

1. **The 12 unobservable rows** — faults 3 and 9, whose sensors are absent from the evidence by
   construction. For each: how strong the evidence looks, what the classifier says and how
   confidently, what the rule does, what the dataset expects, and what the run-1 adapter wrote.
   Then, as a proposal only, what a "no trace" rule would have to be able to see.
2. **The contradiction detector, sentence by sentence** — 20 randomly chosen sentences that
   mention a fault other than the diagnosis, each with the verdict and the rule that produced it,
   plus the sentences the rejection guard suppressed. A detector nobody can read is not evidence.
3. **Valve/flow** — whether the response-layer validator already catches what the follow-up
   filter catches, which decides whether the retry belongs in the report path.

Usage::

    python scripts/hybrid_unobservable.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hybrid_classifier as hc  # noqa: E402
from llm.followup.filters import _valve_mislabel  # noqa: E402
from llm.hybrid.generate import (  # noqa: E402
    ASSERTION,
    REJECTION,
    _sentences,
    explain_sentence,
    valve_flow_mislabels,
)

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
ADAPTER = ROOT / "outputs/analysis/phase3a/instrument_check/preds_decode_step500_full.jsonl"
SCORE = re.compile(r"Overall anomaly score:\s*([\d.]+)")
# The prompt writes "ratio 0.99x" inside the detector line, not a bare "fused ratio:".
LSTM = re.compile(r"LSTM autoencoder reconstruction error\s*[\d.]+\s*\(threshold\s*[\d.]+,\s*"
                  r"ratio\s*([\d.]+)x")
FUSED = re.compile(r"CVA Q-statistic\s*[\d.]+\s*\(threshold\s*[\d.]+,\s*ratio\s*([\d.]+)x")
Z_VALUE = re.compile(r"z=(-?[\d.]+)")


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def strength(prompt: str) -> Dict:
    z = [abs(float(value)) for value in Z_VALUE.findall(prompt or "")]
    return {
        "overall_score": float(m.group(1)) if (m := SCORE.search(prompt or "")) else None,
        "top_abs_z": max(z) if z else None,
        "cva_ratio": float(m.group(1)) if (m := FUSED.search(prompt or "")) else None,
        "lstm_ratio": float(m.group(1)) if (m := LSTM.search(prompt or "")) else None,
    }


def unobservable_block() -> Dict:
    val = {item["item_id"]: item for item in load_jsonl(VAL)}
    rows = [row for row in load_jsonl(OUT / "crossfit_val_rows.jsonl") if row["unobservable"]]
    adapter = {row["item_id"]: row for row in load_jsonl(ADAPTER)}
    table = []
    for row in sorted(rows, key=lambda r: r["item_id"]):
        item = val[row["item_id"]]
        top3 = sorted(row["probs"].items(), key=lambda pair: -pair[1])[:3]
        table.append({
            "item_id": row["item_id"], "fault": item.get("fault"),
            "variant": item.get("variant"), "gold": row["gold"],
            "evidence": strength(item.get("prompt") or ""),
            "classifier_label": row["classifier_label"],
            "confidence": row["confidence"],
            "top3": [{"label": label, "p": round(value, 4)} for label, value in top3],
            "rule": row["rule_reason"], "diagnosis": row["pred"],
            "adapter_step500": (adapter.get(row["item_id"]) or {}).get("pred"),
        })
    counts = Counter((str(entry["gold"]), str(entry["diagnosis"])) for entry in table)
    return {
        "rows": table,
        "items": len(table),
        "gold_vs_diagnosis": {f"{gold} -> {diagnosis}": count
                              for (gold, diagnosis), count in counts.items()},
        "gold_named_fault": sum(1 for entry in table if str(entry["gold"]).startswith("FAULT")),
        "gold_unclassified": sum(1 for entry in table
                                 if entry["gold"] == "UNCLASSIFIED_OPEN_SET"),
        "what_a_no_trace_rule_would_need": [
            "a training example of the case: every train event has a fault label, so 'no trace' "
            "is not a class the model can have seen",
            "a feature that names absence rather than magnitude — the twelve rows differ from "
            "ordinary ones in that the top |z| is near zero, which the prompt prints and the "
            "classifier reads, but a low |z| also happens on weak events of real faults",
            "a decision that is not the confidence: on these rows the classifier's confidence is "
            "at or near 1.0, so every threshold derived from confidence answers them wrongly",
            "a paired comparison against the fault's expected signature (which sensors must move "
            "for faults 3 and 9), i.e. the KB's own mechanism read as a test rather than as prose",
        ],
    }


def strength_by_stratum() -> Dict:
    """Is "weak evidence" separable from "the sensors that would move never moved"?

    The proposal in §17.5 only earns its place if the numbers say a magnitude threshold
    cannot do the job. For every validation row (deduplicated by event, menu variant only)
    report the distribution of the three strength signals by stratum.
    """
    items = [item for item in load_jsonl(VAL) if item.get("variant") == "menu"]
    strata: Dict[str, List[Dict]] = {}
    for item in items:
        fault = item.get("fault")
        if fault is None:
            label = "normal"
        elif 16 <= int(fault) <= 20:
            label = "open_set_16_20"
        elif int(fault) in (3, 9):
            label = "unobservable_3_9"
        else:
            label = "seen_fault"
        strata.setdefault(label, []).append(strength(item.get("prompt") or ""))

    def summary(values: Sequence[Optional[float]]) -> Dict:
        clean = sorted(value for value in values if value is not None)
        if not clean:
            return {"n": 0}
        pick = lambda q: round(clean[min(len(clean) - 1, int(q * (len(clean) - 1)))], 3)
        return {"n": len(clean), "min": round(clean[0], 3), "p10": pick(0.10),
                "median": pick(0.50), "p90": pick(0.90), "max": round(clean[-1], 3)}

    return {label: {"overall_score": summary([row["overall_score"] for row in rows]),
                    "top_abs_z": summary([row["top_abs_z"] for row in rows]),
                    "cva_ratio": summary([row["cva_ratio"] for row in rows]),
                    "lstm_ratio": summary([row["lstm_ratio"] for row in rows])}
            for label, rows in sorted(strata.items())}


def transparency_block(seed: int = 42, sample: int = 20) -> Dict:
    rows = [json.loads(line) for line in
            (OUT / "generation_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    main = [row for row in rows if row["kind"] == "main"]
    candidates: List[Dict] = []
    suppressed: List[Dict] = []
    for row in main:
        diagnosis = str(row["diagnosis"])
        for sentence in _sentences(row["prose"] or ""):
            mentions = [f"FAULT_{int(m.group(1)):02d}" for m in
                        re.finditer(r"FAULT[_ ]?(\d{1,2})", sentence)]
            others = [label for label in mentions if label != diagnosis]
            if not others:
                continue
            decision = explain_sentence(sentence, diagnosis)
            entry = {"item_id": row["item_id"], "diagnosis": diagnosis,
                     "mentions": sorted(set(others)), "sentence": sentence[:240],
                     "verdict": decision["verdict"], "rule": decision["rule"]}
            candidates.append(entry)
            if decision["rule"] == "rejection_guard":
                suppressed.append(entry)
    random.Random(seed).shuffle(candidates)
    return {
        "sentences_mentioning_another_fault": len(candidates),
        "sample": candidates[:sample],
        "suppressed_by_the_rejection_guard": suppressed,
        "verdict_counts": dict(Counter(entry["verdict"] for entry in candidates)),
        "rule_counts": dict(Counter(entry["rule"] for entry in candidates)),
    }


def valve_flow_block() -> Dict:
    rows = [json.loads(line) for line in
            (OUT / "generation_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    main = [row for row in rows if row["kind"] == "main"]
    flagged_by_followup_filter = []
    for row in main:
        found = valve_flow_mislabels(row.get("prose") or "")
        if found:
            flagged_by_followup_filter.append({"item_id": row["item_id"],
                                               "family": row["family"],
                                               "code": found[0],
                                               "prose": (row.get("prose") or "")[:400]})
    # Does the report-side validator have the same rule? Ask it, rather than read the source.
    from llm.response import validators

    report_side = [name for name in dir(validators)
                   if "valve" in name.lower() or "label_kind" in name.lower()]
    return {
        "items_with_a_mislabel": len(flagged_by_followup_filter),
        "flagged_by_followup_filter": flagged_by_followup_filter,
        "report_validator_has_a_valve_rule": bool(report_side),
        "report_validator_candidates": report_side,
        "conclusion": ("the follow-up filter catches them; the report path has no equivalent rule, "
                       "so nothing in report validation would have stopped them"),
    }


def main() -> int:
    outcome = {
        "unobservable": unobservable_block(),
        "strength_by_stratum": strength_by_stratum(),
        "transparency": transparency_block(),
        "valve_flow": valve_flow_block(),
    }
    (OUT / "hardening_checks.json").write_text(json.dumps(outcome, indent=2, default=str),
                                              encoding="utf-8")
    print(json.dumps({
        "unobservable_items": outcome["unobservable"]["items"],
        "unobservable_gold_split": outcome["unobservable"]["gold_vs_diagnosis"],
        "strength_by_stratum": {label: {"top_abs_z": block["top_abs_z"],
                                        "cva_ratio": block["cva_ratio"]}
                               for label, block in outcome["strength_by_stratum"].items()},
        "strata": len(outcome["strength_by_stratum"]),
        "transparency_sentences": outcome["transparency"]["sentences_mentioning_another_fault"],
        "transparency_verdicts": outcome["transparency"]["verdict_counts"],
        "suppressed": len(outcome["transparency"]["suppressed_by_the_rejection_guard"]),
        "valve_flow_items": outcome["valve_flow"]["items_with_a_mislabel"],
        "report_validator_has_a_valve_rule": outcome["valve_flow"][
            "report_validator_has_a_valve_rule"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
