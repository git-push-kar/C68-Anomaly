"""Forced-diagnosis generation: does the adapter's prose fit a diagnosis it did not choose?

The hybrid hands the adapter a completed answer prefix and asks it for three prose fields.
Accuracy is therefore no longer the question — the fault id is decided upstream — and the
question becomes *faithfulness*: given ``FAULT_13`` in the prefix, does the reasoning argue
for ``FAULT_13``, or for something else, or nothing?

Two batches are generated, inside one 300-item GPU budget (PROTOCOL §16.5):

``main``  a stratified validation subsample: menu-present fault items, menu-removed items,
          ``NORMAL_TRANSIENT`` items, open-set faults 16-20 on a menu, and the unobservable
          faults 3/9 with the abstention forced.
``counterfactual``  the same kind of item with a *wrong* diagnosis forced — the KB's own
          differential where one exists. A prose writer that rationalises whatever it is
          handed is a prose writer whose reasoning cannot be read as evidence.

Nothing here retrains, tunes, or repairs output: a contradiction is counted, and the answer
is kept as written.

Usage::

    python scripts/hybrid_faithfulness.py --dry-run          # selection only, no GPU
    python scripts/hybrid_faithfulness.py --limit 5          # timing probe
    python scripts/hybrid_faithfulness.py                    # the full 300
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402
from llm.benchmark.label_scorer import prompt_for  # noqa: E402
from llm.benchmark.menu import lookalikes  # noqa: E402
from llm.hybrid.generate import (  # noqa: E402
    contradictions,
    decode_text,
    forced_prefix,
    parse_slim,
    validate_slim,
    valve_flow_mislabels,
)
from llm.hybrid.menu_rule import UNCLASSIFIED, menu_rule  # noqa: E402

import hybrid_classifier as hc  # noqa: E402  (scripts/ is on sys.path)

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
CONFIG = ROOT / "configs/config_v2_protocolA_seed42.yaml"
ADAPTER = ROOT / "outputs/tep_rca_v2_protocolA_seed42/checkpoint-500/tep_rca_v2_protocolA_seed42"

#: The GPU budget. Main subsample + counterfactual must not exceed it.
TOTAL_BUDGET = 300
PLAN = {"fault_menu": 110, "removed": 50, "normal": 45, "open_set": 25, "unobservable": 12}
COUNTERFACTUAL = 58
CAP = 512
SCHEMA_PREFIXES = ("enum_", "fault_id_not_forced")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def family_of(item: Dict) -> str:
    if item.get("unobservable"):
        return "unobservable"
    if item.get("variant") != "menu":
        return "removed"
    fault = item.get("fault")
    if fault is None:
        return "normal"
    return "open_set" if 16 <= int(fault) <= 20 else "fault_menu"


def take(pool: Sequence[Dict], count: int, seed: int) -> List[Dict]:
    """Seeded, fault-stratified, deterministic."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for entry in pool:
        groups[str(entry["item"].get("fault"))].append(entry)
    for group in groups.values():
        group.sort(key=lambda entry: str(entry["row"]["item_id"]))
        random.Random(seed).shuffle(group)
    picked: List[Dict] = []
    order = sorted(groups, key=lambda name: (-len(groups[name]), name))
    cursor = Counter()
    while len(picked) < count:
        added = False
        for name in order:
            if cursor[name] >= len(groups[name]):
                continue
            picked.append(groups[name][cursor[name]])
            cursor[name] += 1
            added = True
            if len(picked) >= count:
                break
        if not added:
            break
    return picked


def select(entries: Sequence[Dict], seed: int = hc.SEED) -> Dict[str, List[Dict]]:
    pools: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        pools[family_of(entry["item"])].append(entry)
    chosen: Dict[str, List[Dict]] = {}
    for family, target in PLAN.items():
        pool = pools.get(family, [])
        # Classifier errors are never sampled away: they are the items the hybrid exists for.
        errors = [entry for entry in pool
                  if entry["row"]["classifier_label"] != entry["row"]["true_label"]]
        rest = [entry for entry in pool if entry not in errors]
        want = min(target, len(pool))
        picked = errors[:want]
        if len(picked) < want:
            picked += take(rest, want - len(picked), seed)
        chosen[family] = picked
    return chosen


def diagnosis_of(entry: Dict) -> Tuple[str, str]:
    """The diagnosis the prefix will carry, and why.

    Everything comes from the menu rule except the unobservable rows, which are forced to
    the abstention: their sensors are absent from the evidence by construction, so the
    honest answer is that the event cannot be classified, whatever the classifier's
    confidence says.
    """
    row = entry["row"]
    if entry["item"].get("unobservable"):
        return UNCLASSIFIED, "forced_abstention_unobservable"
    return str(row["pred"]), str(row["rule_reason"])


def counterfactual_of(entry: Dict, seed: int) -> Optional[str]:
    """A wrong diagnosis that a competent reader could plausibly have been given."""
    item = entry["item"]
    gold = entry["row"]["gold"]
    menu = [str(value) for value in (item.get("menu_ids") or [])]
    fault = item.get("fault")
    candidates: List[str] = []
    if fault is not None:
        for other in lookalikes(int(fault), limit=4):
            label = f"FAULT_{int(other):02d}"
            if label in menu and label != gold:
                candidates.append(label)
    alternatives = [label for label in menu if label != gold and label != UNCLASSIFIED]
    alternatives.sort()
    random.Random(seed).shuffle(alternatives)
    candidates += alternatives
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def load_model():
    from llm.benchmark.decode_bench import build_model

    model, tokenizer, config, applied = build_model(CONFIG, str(ADAPTER), merged=True)
    return model, tokenizer, applied


def labels_for(items: Sequence[Dict]) -> List[str]:
    allowed = {UNCLASSIFIED, "NORMAL_TRANSIENT"}
    for item in items:
        allowed |= {str(value) for value in (item.get("menu_ids") or [])}
    return sorted(allowed)


def answer_of(model, tokenizer, item: Dict, diagnosis: str, cap: int,
              allowed: Sequence[str]) -> Dict:
    prefix = forced_prefix(diagnosis)
    started = time.time()
    try:
        tokens = decode_text(model, tokenizer, prompt_for(item["prompt"]) + prefix, cap)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        failure = None
    except Exception as exc:  # a crashed generation is a datum, not a silent gap
        tokens, text, failure = [], "", f"{type(exc).__name__}: {exc}"
    seconds = round(time.time() - started, 2)
    answer = parse_slim(text, prefix) if not failure else None
    problems = validate_slim(answer, allowed, diagnosis) if not failure else ["not_json"]
    schema = [code for code in problems if not code.startswith(SCHEMA_PREFIXES)]
    enums = [code for code in problems if code.startswith(SCHEMA_PREFIXES)]
    prose = " ".join(str((answer or {}).get(field) or "") for field in
                     ("physical_reasoning", "narrative")).strip()
    return {
        "tokens": len(tokens),
        "seconds": seconds,
        "text": text,
        "answer": answer,
        "schema_errors": schema,
        "enum_errors": enums,
        "schema_valid": not schema,
        "enum_valid": not enums,
        "contradictions": contradictions(answer, diagnosis),
        "valve_flow_mislabels": valve_flow_mislabels(prose),
        "grounding_violations": ev.grounding_violations(
            prose, sorted({tag.upper() for tag in ev.TAG_PATTERN.findall(item["prompt"])}),
            [float(value) for value in ev.NUMBER_PATTERN.findall(item["prompt"])]),
        "failure": failure,
        "prose": prose,
    }


def summarise(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {"items": 0}
    codes: Counter = Counter()
    for row in rows:
        for found in row["contradictions"]:
            codes[found["code"]] += 1
    grounding: Counter = Counter()
    for row in rows:
        for code in row["grounding_violations"]:
            grounding[code.split(":")[0]] += 1
    n = len(rows)
    return {
        "items": n,
        "schema_valid": sum(1 for row in rows if row["schema_valid"]),
        "schema_valid_rate": round(sum(1 for row in rows if row["schema_valid"]) / n, 4),
        "enum_valid": sum(1 for row in rows if row["enum_valid"]),
        "complete": sum(1 for row in rows if row["schema_valid"] and row["enum_valid"]),
        "completion_rate": round(sum(1 for row in rows if row["schema_valid"]
                                       and row["enum_valid"]) / n, 4),
        "grounded_clean": sum(1 for row in rows if not row["grounding_violations"]),
        "grounded_clean_rate": round(
            sum(1 for row in rows if not row["grounding_violations"]) / n, 4),
        "grounding_violations": dict(sorted(grounding.items())),
        "valve_flow_mislabels": sum(1 for row in rows if row["valve_flow_mislabels"]),
        "contradiction_rows": sum(1 for row in rows if row["contradictions"]),
        "contradiction_rate": round(sum(1 for row in rows if row["contradictions"]) / n, 4),
        "contradictions": dict(sorted(codes.items())),
        "seconds_per_item": round(sum(row["seconds"] for row in rows) / n, 2),
        "tokens_mean": round(sum(row["tokens"] for row in rows) / n, 1),
        "failures": dict(Counter(row["failure"] for row in rows if row["failure"])),
    }


def write_audit(rows: Sequence[Dict], path: Path, seed: int = hc.SEED) -> int:
    """100 rows, stratified by family, scoring columns blank."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: str(row["item_id"]))
        random.Random(seed).shuffle(group)
    total = len(rows)
    quota = {family: max(1, round(100 * len(group) / total)) for family, group in groups.items()}
    picked = [row for family in sorted(quota) for row in groups[family][: quota[family]]]
    picked = picked[:100]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "family", "forced_diagnosis", "correct_diagnosis", "prose",
                         "correct", "grounded", "consistent_with_diagnosis", "useful", "notes"])
        for row in picked:
            truth = row["truth"]
            if row["kind"] == "counterfactual":
                known = "n"
            elif truth == UNCLASSIFIED or truth == "UNCLASSIFIED_OPEN_SET":
                known = "y" if row["diagnosis"] == UNCLASSIFIED else "n"
            else:
                known = "y" if row["diagnosis"] == truth else "n"
            writer.writerow([row["item_id"], row["family"], row["diagnosis"], known,
                             row["prose"].replace("\n", " "), "", "", "", "", ""])
    return len(picked)


# --------------------------------------------------------------------------- #
def rederive(row: Dict, allowed: Sequence[str], prompts: Dict[str, str]) -> Dict:
    """Recompute every derived field of one saved generation.

    The generation is the datum; the detector is an instrument. When an instrument is wrong —
    as the first version of the contradiction rules was, counting "FAULT_14 was rejected
    because …" as naming another fault as the cause — the honest repair is to re-read the
    saved text with the corrected rule, not to re-generate. Nothing here calls a model.
    """
    diagnosis = str(row["diagnosis"])
    prefix = forced_prefix(diagnosis)
    answer = parse_slim(row.get("text") or "", prefix)
    problems = validate_slim(answer, allowed, diagnosis)
    schema = [code for code in problems if not code.startswith(SCHEMA_PREFIXES)]
    enums = [code for code in problems if code.startswith(SCHEMA_PREFIXES)]
    prose = " ".join(str((answer or {}).get(field) or "") for field in
                     ("physical_reasoning", "narrative")).strip()
    prompt = prompts.get(row["item_id"], "")
    return {
        "answer": answer,
        "schema_errors": schema, "enum_errors": enums,
        "schema_valid": not schema, "enum_valid": not enums,
        "contradictions": contradictions(answer, diagnosis),
        "valve_flow_mislabels": valve_flow_mislabels(prose),
        "grounding_violations": ev.grounding_violations(
            prose, sorted({tag.upper() for tag in ev.TAG_PATTERN.findall(prompt)}),
            [float(value) for value in ev.NUMBER_PATTERN.findall(prompt)]),
        "prose": prose,
        "ranked_top": ((answer or {}).get("ranked_hypotheses") or [None])[0],
    }


def finalise(rows: Sequence[Dict], out: Path, meta: Dict) -> Dict:
    """Write the four outputs and the audit sheet from the rows as they stand."""
    main_rows = [row for row in rows if row["kind"] == "main"]
    cf_rows = [row for row in rows if row["kind"] == "counterfactual"]
    by_family = {family: summarise([row for row in main_rows if row["family"] == family])
                 for family in PLAN}
    anomalies = [row for row in rows if row.get("contradictions")]
    outcome = {
        **meta,
        "items": {"main": len(main_rows), "counterfactual": len(cf_rows)},
        "overall": summarise(main_rows), "by_family": by_family,
        "classified_only": summarise([row for row in main_rows
                                      if row["diagnosis"].startswith("FAULT")]),
        "abstained_only": summarise([row for row in main_rows
                                     if not row["diagnosis"].startswith("FAULT")]),
        "diagnosis_was_correct": {"main": sum(1 for row in main_rows
                                               if row["diagnosis"] == row["gold"]),
                                   "of": len(main_rows)},
        "contradiction_examples": [{"item_id": row["item_id"],
                                    "diagnosis": row["diagnosis"],
                                    "codes": [found["code"] for found in row["contradictions"]],
                                    "sentence": (row["contradictions"][0].get("sentence")
                                                 or row["contradictions"][0].get("detail")
                                                 or "")[:220]}
                                   for row in anomalies[:10]],
    }
    verdicts = []
    for row in cf_rows:
        codes = {found["code"] for found in row["contradictions"]}
        if not row["schema_valid"]:
            label = "unusable"
        elif "names_other_fault_as_cause" in codes or "hedges_diagnosis" in codes:
            label = "contradicted_or_hedged"
        elif "ranked_top_mismatch" in codes or "ranked_empty_when_classified" in codes:
            label = "ranking_disagrees"
        else:
            label = "rationalised"
        verdicts.append({**row, "cf_verdict": label})
    counterfactual = {
        "items": len(cf_rows),
        "verdicts": dict(Counter(row["cf_verdict"] for row in verdicts)),
        "rationalised_rate": round(
            sum(1 for row in verdicts if row["cf_verdict"] == "rationalised")
            / max(len(verdicts), 1), 4),
        "contradicted_rate": round(
            sum(1 for row in verdicts if row["cf_verdict"] == "contradicted_or_hedged")
            / max(len(verdicts), 1), 4),
        "summary": summarise(cf_rows),
        "examples": {
            "rationalised": [{"item_id": row["item_id"], "forced": row["diagnosis"],
                              "truth": row["truth"], "prose": row["prose"][:400]}
                             for row in verdicts if row["cf_verdict"] == "rationalised"][:10],
            "contradicted_or_hedged": [
                {"item_id": row["item_id"], "forced": row["diagnosis"], "truth": row["truth"],
                 "codes": [found["code"] for found in row["contradictions"]],
                 "prose": row["prose"][:400]}
                for row in verdicts if row["cf_verdict"] == "contradicted_or_hedged"][:10],
        },
    }
    (out / "generation_rows.jsonl").write_text(
        "\n".join(json.dumps(row, default=str) for row in rows), encoding="utf-8")
    (out / "faithfulness.json").write_text(json.dumps(outcome, indent=2, default=str),
                                           encoding="utf-8")
    (out / "counterfactual.json").write_text(json.dumps(counterfactual, indent=2, default=str),
                                             encoding="utf-8")
    outcome["audit_rows"] = write_audit(rows, out / "audit_hybrid_100.csv")
    return outcome, counterfactual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reanalyse", action="store_true",
                        help="re-read the saved generations with the current detectors")
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant", default="protocolA")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The classifier's validation predictions are read from the file `hybrid_classifier.py
    # evaluate` writes, so this harness needs only torch: the LightGBM stack lives in the
    # app environment and the GPU stack in venv_train, and they are not installed together.
    preds_path = hc.OUT / f"preds_{args.variant}_val.jsonl"
    if not preds_path.exists():
        raise SystemExit(f"missing {preds_path}; run `python scripts/hybrid_classifier.py "
                         f"evaluate` first")
    items = hc.load_jsonl(VAL)
    preds = {row["item_id"]: row for row in hc.load_jsonl(preds_path)}
    rule_rows = [preds[item["item_id"]] for item in items if item["item_id"] in preds]
    items = [item for item in items if item["item_id"] in preds]
    threshold = {"threshold": json.loads(
        (preds_path.parent / "protocolA" / "threshold.json").read_text(encoding="utf-8"))[
            "threshold"]}
    entries = [{"row": row, "item": item} for row, item in zip(rule_rows, items)]

    chosen = select(entries)
    plan = {family: len(group) for family, group in chosen.items()}
    if sum(plan.values()) + COUNTERFACTUAL > TOTAL_BUDGET:
        raise SystemExit(f"selection exceeds the {TOTAL_BUDGET}-item budget: {plan}")

    main_entries = [entry for family in chosen for entry in chosen[family]]
    cf_source = take([entry for entry in chosen["fault_menu"]], COUNTERFACTUAL, hc.SEED + 7)
    cf_entries = []
    for index, entry in enumerate(cf_source):
        forced = counterfactual_of(entry, hc.SEED + index)
        if forced:
            cf_entries.append({"row": entry["row"], "item": entry["item"], "forced": forced})

    print(json.dumps({"plan": plan, "main": len(main_entries),
                      "counterfactual": len(cf_entries),
                      "total": len(main_entries) + len(cf_entries),
                      "adapter": str(ADAPTER), "cap": args.cap}, indent=2), flush=True)

    allowed = labels_for(items)
    prompts = {item["item_id"]: item["prompt"] for item in items}
    if args.reanalyse:
        saved = [json.loads(line) for line in
                 (out / "generation_rows.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        for row in saved:
            row.update(rederive(row, allowed, prompts))
        outcome, counterfactual = finalise(saved, out, {
            "adapter": "merged", "adapter_path": str(ADAPTER), "cap": args.cap,
            "threshold": threshold["threshold"], "plan": plan, "reanalysed": True})
        print(json.dumps({"overall": outcome["overall"],
                          "counterfactual": {key: counterfactual[key] for key in
                                              ("items", "verdicts", "rationalised_rate",
                                               "contradicted_rate")},
                          "audit_rows": outcome["audit_rows"]}, indent=2, default=str))
        return 0
    if args.dry_run:
        (out / "faithfulness_selection.json").write_text(
            json.dumps({"plan": plan, "counterfactual": len(cf_entries),
                        "threshold": threshold["threshold"],
                        "families": dict(Counter(family_of(entry["item"])
                                                  for entry in main_entries))},
                       indent=2), encoding="utf-8")
        return 0

    model, tokenizer, applied = load_model()
    print(json.dumps({"model": applied}), flush=True)
    import torch

    if torch.cuda.is_available():
        print(json.dumps({"gpu": torch.cuda.get_device_name(0),
                          "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}),
              flush=True)

    rows: List[Dict] = []
    started = time.time()
    limit = args.limit or 0
    for index, entry in enumerate(main_entries + cf_entries):
        if limit and index >= limit:
            break
        is_cf = index >= len(main_entries)
        family = "counterfactual" if is_cf else family_of(entry["item"])
        if is_cf:
            diagnosis, reason = entry["forced"], "counterfactual"
        else:
            diagnosis, reason = diagnosis_of(entry)
        produced = answer_of(model, tokenizer, entry["item"], diagnosis, args.cap, allowed)
        truth = entry["row"]["true_label"]
        rows.append({
            "item_id": entry["row"]["item_id"], "kind": "counterfactual" if is_cf else "main",
            "family": family, "variant": entry["item"].get("variant"),
            "fault": entry["item"].get("fault"), "truth": truth,
            "gold": entry["row"]["gold"], "classifier_label": entry["row"]["classifier_label"],
            "confidence": entry["row"]["confidence"], "diagnosis": diagnosis,
            "diagnosis_reason": reason,
            "ranked_top": ((produced["answer"] or {}).get("ranked_hypotheses") or [None])[0],
            **{key: produced[key] for key in
               ("tokens", "seconds", "schema_valid", "enum_valid", "schema_errors",
                "enum_errors", "contradictions", "valve_flow_mislabels",
                "grounding_violations", "failure", "prose", "text")},
        })
        if (index + 1) % 10 == 0:
            elapsed = time.time() - started
            print(json.dumps({"done": index + 1, "of": len(main_entries) + len(cf_entries),
                              "seconds_per_item": round(elapsed / (index + 1), 2),
                              "eta_minutes": round(elapsed / (index + 1)
                                                   * (len(main_entries) + len(cf_entries)
                                                      - index - 1) / 60, 1)}), flush=True)

    outcome, counterfactual = finalise(rows, out, {
        "adapter": applied, "adapter_path": str(ADAPTER), "cap": args.cap,
        "threshold": threshold["threshold"], "plan": plan, "reanalysed": False})
    print(json.dumps({"overall": outcome["overall"],
                      "counterfactual": {key: counterfactual[key] for key in
                                         ("items", "verdicts", "rationalised_rate",
                                          "contradicted_rate")},
                      "audit_rows": outcome["audit_rows"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
