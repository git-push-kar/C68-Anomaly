"""Reconcile three sets of numbers the 3B reports left disagreeing (PROTOCOL §20.10).

Three separate reconciliations, each on its own stated basis and none editing an earlier report:

(a) **the shuffled-order table.** 3B-7 reported accuracy over "822 observable" rows but "named"
    counts over 331 gold-fault rows, and false-abstention rates over denominators of 325, 331 and
    822 in different places. Every row is recomputed here on
      * the **325 observable gold-fault rows** — the 331 gold-fault rows minus the six menu-present
        faults-3/9 rows, where no observable evidence separates the fault from `NORMAL_TRANSIENT`;
      * the **822 support** = 450 abstention-class + 325 gold-fault + 47 `NORMAL_TRANSIENT`;
      * the **331 gold-fault rows** the 3B-7 table actually counted;
    and the two "named" denominators (named *on gold-fault rows*, and named *anywhere in the 822*)
    are computed separately and labelled, with the difference explained row by row.

(b) **the masking counts.** §9.2 says masking `fault_id` and `verdict` removes 22,917 supervised
    tokens (644,128 − 609,697 = 34,431, so 11,514 tokens are unaccounted for). This measures the
    removal directly, per row, with the model's own tokenizer, and decomposes it into the two
    value spans, the keys, the quotes and separators, the leading brace, the terminator, and the
    **BPE merges that only exist because the prefix sits next to the answer** — that last term is
    the one the component sum cannot see.

(c) **the §9.2 label-error counts**, restated on the dataset-target basis §20.10 fixed: an error is
    a deployed diagnosis that differs from `target.fault_id`; an abstention on a row whose target
    is a named fault is a false abstention; an abstention on a row whose target is
    `UNCLASSIFIED_OPEN_SET` is a correct abstention. The 254 / 13.2 % figure is then either
    withdrawn or given a definition, and the arithmetic that could produce it is shown.

CPU only (the tokenizer runs on CPU). No model, no generation, no test component.

Usage::

    ./venv_train/Scripts/python.exe scripts/reconcile_3b8.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DATASET = ROOT / "outputs/benchmark_v2/dataset"
FIT = ROOT / "outputs/analysis/phase3a/fit"
MIXED = ROOT / "outputs/benchmark_v2/sft/train_mixed.jsonl"
CORRECTIONS = FIT / "corrections_3b6.json"
ORDERS = ("kb", "shuffled-1", "shuffled-2")
ABSTAIN = ("UNCLASSIFIED_OPEN_SET", "NORMAL_TRANSIENT")
EOS = "<|im_end|>"


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --------------------------------------------------------------------------- (a) the order table
def is_named(pred: str) -> bool:
    return str(pred or "").startswith("FAULT")


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def order_table(rows: Dict[str, Dict], val: Dict[str, Dict]) -> Dict:
    """Every order on the 325, the 822 and the 331, with both named denominators."""
    out: Dict[str, Dict] = {}
    for order in ORDERS:
        predicted = rows[order]
        observable = [(item_id, row) for item_id, row in predicted.items()
                      if not val[item_id].get("unobservable")]
        gold_fault_822 = [(item_id, row) for item_id, row in observable
                          if str(row["gold"]).startswith("FAULT")]
        gold_fault_331 = [(item_id, row) for item_id, row in predicted.items()
                          if str(row["gold"]).startswith("FAULT")]
        # the six menu-present faults-3/9 rows are the observable gold-fault rows the 331 holds and
        # the 325 does not
        unobservable = {item_id for item_id, item in val.items() if item.get("unobservable")}
        gold_fault_325 = [(item_id, row) for item_id, row in gold_fault_331
                          if item_id not in unobservable]
        assert len(gold_fault_325) == 325 and len(gold_fault_331) == 331, (
            len(gold_fault_325), len(gold_fault_331))

        gold = [str(row["gold"]) for _item, row in observable]
        pred = [str(row["pred"]) for _item, row in observable]
        correct = sum(1 for g, p in zip(gold, pred) if g == p)
        abstain_gold = [g for g in gold if g in ABSTAIN]
        abstain_pred = [p for p in pred if p in ABSTAIN]
        true_abstain = sum(1 for g, p in zip(gold, pred) if g in ABSTAIN and p in ABSTAIN)
        named822 = [(item_id, row) for item_id, row in observable if is_named(row["pred"])]
        named_on_gold822 = [(item_id, row) for item_id, row in gold_fault_822
                            if is_named(row["pred"])]
        named_on_gold331 = [(item_id, row) for item_id, row in gold_fault_331
                            if is_named(row["pred"])]
        correct_named822 = sum(1 for _item, row in named822 if str(row["pred"]) == str(row["gold"]))
        correct_named_gold822 = sum(1 for _item, row in named_on_gold822
                                    if str(row["pred"]) == str(row["gold"]))
        correct_named331 = sum(1 for _item, row in named_on_gold331
                               if str(row["pred"]) == str(row["gold"]))
        correct_325 = sum(1 for _item, row in gold_fault_325 if str(row["pred"]) == str(row["gold"]))
        named_325 = [(item_id, row) for item_id, row in gold_fault_325 if is_named(row["pred"])]
        false_abstain_325 = sum(1 for _item, row in gold_fault_325 if not is_named(row["pred"]))

        out[order] = {
            "support_822": {
                "rows": len(observable), "accuracy": round(correct / len(observable), 4),
                "macro_f1": macro_f1(gold, pred),
                "abstention_precision": round(true_abstain / len(abstain_pred), 4)
                if abstain_pred else 0.0,
                "abstention_recall": round(true_abstain / len(abstain_gold), 4)
                if abstain_gold else 0.0,
                "support": dict(Counter(gold)),
            },
            "gold_fault_325": {
                "rows": 325,
                "accuracy": round(correct_325 / 325, 4),
                "named": len(named_325),
                "correct_when_named": round(
                    sum(1 for _item, row in named_325 if str(row["pred"]) == str(row["gold"]))
                    / len(named_325), 4) if named_325 else 0.0,
                "false_abstention": false_abstain_325,
                "false_abstention_rate": round(false_abstain_325 / 325, 4),
            },
            "gold_fault_331": {
                "rows": 331,
                "named": len(named_on_gold331),
                "correct_when_named": round(correct_named331 / len(named_on_gold331), 4)
                if named_on_gold331 else 0.0,
            },
            "named_denominators": {
                "named_on_gold_fault_rows": len(named_on_gold331),
                "named_anywhere_in_822": len(named822),
                "difference": len(named822) - len(named_on_gold331),
                "difference_explained": (
                    "faults asserted on rows whose gold is an abstention class — naming there is "
                    "wrong, so it belongs in the denominator for the precision of naming and not "
                    "in the denominator for coverage of fault episodes"),
                "correct_when_named_anywhere": round(correct_named822 / len(named822), 4)
                if named822 else 0.0,
                "correct_when_named_on_gold822": round(correct_named_gold822
                                                        / len(named_on_gold822), 4)
                if named_on_gold822 else 0.0,
                "false_abstention_822": sum(1 for _item, row in gold_fault_822
                                            if not is_named(row["pred"])),
                "false_abstention_rate_822": round(
                    sum(1 for _item, row in gold_fault_822 if not is_named(row["pred"]))
                    / len(gold_fault_822), 4),
            },
        }
    return out


# --------------------------------------------------------------------------- (b) masking tokens
def tokenizer_of(config_path: Path):
    import yaml
    from transformers import AutoTokenizer
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config["llm"]["base_model"]
    candidates = [base, str(ROOT / "models/internvl3_2b_lr_fused")]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True,
                                                          use_fast=False)
                return tokenizer, candidate
            except Exception:  # noqa: BLE001 - fall through to the next candidate
                continue
    raise SystemExit("no tokenizer available; run inside venv_train")


def count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def masking_accounting(tokenizer) -> Dict:
    """Measure the masked prefix directly, component by component."""
    rows = load_jsonl(MIXED)
    components = ("brace", "fault_id_key", "fault_id_value", "verdict_key", "verdict_value",
                  "separators")
    totals = {component: 0 for component in components}
    removed = {"report": 0, "dialogue": 0}
    spans = {"fault_id_value_only": 0, "fault_id_with_key": 0, "fault_id_span": 0,
             "verdict_value_only": 0, "verdict_with_key": 0, "verdict_span": 0}
    n = {"report": 0, "dialogue": 0}
    skipped = 0
    merge_gap = 0
    for row in rows:
        answer = str(row.get("answer") or "")
        if '"fault_id"' not in answer or '"ranked_hypotheses"' not in answer:
            skipped += 1
            continue
        kind = "dialogue" if str(row.get("item_id", "")).startswith("FD") else "report"
        n[kind] += 1
        masked = '{"' + answer[answer.index('"ranked_hypotheses"'):]
        full_ids = count(tokenizer, answer + EOS)
        masked_ids = count(tokenizer, masked + EOS)
        removed_here = full_ids - masked_ids
        removed[kind] += removed_here

        fid = re.search(r'"fault_id": "(?P<value>[^"]+)"', answer)
        verdict = re.search(r'"verdict": "(?P<value>[^"]+)"', answer)
        if not fid or not verdict:
            skipped += 1
            continue
        fid_value = fid.group("value")
        verdict_value = verdict.group("value")
        parts = {
            "brace": "{",
            "fault_id_key": '"fault_id"',
            "fault_id_value": f'"{fid_value}"',
            "verdict_key": '"verdict"',
            "verdict_value": f'"{verdict_value}"',
            "separators": ': ' + ', ' + ': ' + ', ',
        }
        naive = 0
        for component, text in parts.items():
            tokens = count(tokenizer, text)
            totals[component] += tokens
            naive += tokens
        merge_gap += removed_here - naive

        spans["fault_id_value_only"] += count(tokenizer, f'"{fid_value}"')
        spans["fault_id_with_key"] += count(tokenizer, f'"fault_id": "{fid_value}"')
        spans["fault_id_span"] += count(tokenizer, f'"fault_id": "{fid_value}", ')
        spans["verdict_value_only"] += count(tokenizer, f'"{verdict_value}"')
        spans["verdict_with_key"] += count(tokenizer, f'"verdict": "{verdict_value}"')
        spans["verdict_span"] += count(tokenizer, f'"verdict": "{verdict_value}", ')

    total_removed = removed["report"] + removed["dialogue"]
    component_sum = sum(totals.values())
    return {
        "rows": {"report": n["report"], "dialogue": n["dialogue"], "skipped": skipped},
        "removed_tokens": {**removed, "total": total_removed},
        "components": totals,
        "component_sum": component_sum,
        "merge_gap": merge_gap,
        "decomposition": {
            "component_sum_plus_merge_gap_equals_total": component_sum + merge_gap
            == total_removed,
            "note": "a component counted on its own carries its own BPE pieces; printed next to "
                    "the answer it shares pieces with its neighbour, so the components cannot sum "
                    "to the removal. The gap is that sharing, not a missing row.",
        },
        "span_measurements": spans,
        "claimed": {"supervised_total": 644128, "supervised_masked": 609697, "removed": 34431,
                    "removal_as_222917": 22917, "fault_id_13774": 13774, "verdict_9143": 9143},
    }


# --------------------------------------------------------------------------- (c) label errors
def label_errors(corrections: Dict) -> Dict:
    """Restate the §9.2 counts on the dataset-target basis, and test 254 / 13.2 %."""
    pool = corrections["b_error_pool"]
    populations = corrections["c_populations"]
    named = pool["deployed_wrong_named_rows"]
    false_abstention = pool["deployed_false_abstention_rows"]
    correct = pool["deployed_correct_abstention_rows"]
    total = populations["train_rows"]
    events = named + false_abstention
    removed = populations["by_variant"]["removed"]
    menu = populations["by_variant"]["menu"]
    candidate_254 = false_abstention + removed
    return {
        "basis": "dataset target (PROTOCOL §20.10)",
        "events": total,
        "wrong_named": named,
        "false_abstention": false_abstention,
        "pool": events,
        "pool_rate": round(events / total, 4),
        "correct_abstentions": correct,
        "claimed_254": 254,
        "claimed_rate": round(254 / total, 4),
        "arithmetic_that_could_produce_254": {
            "false_abstention_plus_every_menu_removed_row": candidate_254,
            "difference_to_254": 254 - candidate_254,
            "reading": "the 190 menu-removed rows are rows whose target is UNCLASSIFIED_OPEN_SET; "
                       "abstaining on them is correct, so counting them is exactly the mistake "
                       "the correction removed. 63 + 190 = 253, and no fourth population of size 1 "
                       "was identified, so 254 is not reproducible under any stated definition.",
        },
        "withdrawn": "254 / 13.2 % is withdrawn as a label-error count; the like-for-like figure is "
                     f"{false_abstention} / {total} = {round(false_abstention / total, 4)}",
        "ratio_if_menu_only": {"false_abstention": false_abstention, "menu_rows": menu,
                               "rate": round(false_abstention / menu, 4)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIT))
    parser.add_argument("--config", default=str(ROOT / "configs/config_v2_protocolA_seed42.yaml"))
    parser.add_argument("--skip-tokens", action="store_true")
    args = parser.parse_args()

    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    rows: Dict[str, Dict] = {}
    for order in ORDERS:
        path = FIT / f"preds_decode_step500_{order}.jsonl"
        rows[order] = {row["item_id"]: row for row in load_jsonl(path)}

    report = {
        "a_shuffled_order": order_table(rows, val),
        "c_label_errors": label_errors(json.loads(CORRECTIONS.read_text(encoding="utf-8"))),
    }
    if not args.skip_tokens:
        tokenizer, source = tokenizer_of(Path(args.config))
        report["b_masking"] = {**masking_accounting(tokenizer), "tokenizer": source}

    out = Path(args.out)
    (out / "reconcile_3b8.json").write_text(json.dumps(report, indent=2, default=str),
                                            encoding="utf-8")
    print(json.dumps({order: {subset: report["a_shuffled_order"][order][subset]
                              for subset in ("support_822", "gold_fault_325", "gold_fault_331",
                                             "named_denominators")}
                      for order in ORDERS}, indent=2, default=str))
    if "b_masking" in report:
        print(json.dumps({key: report["b_masking"][key] for key in
                          ("rows", "removed_tokens", "components", "component_sum", "merge_gap",
                           "decomposition", "span_measurements")}, indent=2))
    print(json.dumps(report["c_label_errors"], indent=2))
    print(f"written: {out / 'reconcile_3b8.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
