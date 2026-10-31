"""Does the model read the evidence, the menu, or neither?

The closed-set task is a choice among sixteen menu lines, and a menu is a strong
prior: it names every fault family with its initiating variable and direction. A model
that has learned "pick the line that sounds like the event" and a model that has
learned "the answer is usually the fourth line" both score above chance on a
validation set whose menus are always in KB order.

Two interventions separate them, both cheap because only the prompt changes:

``swap``     give the item another event's evidence and keep its own menu. If the
             answer moves to the donor's fault, the evidence is being read. If it
             stays, the item's identity -- not the evidence -- is driving the answer.
``shuffle``  permute the menu lines of the same item. Position sensitivity shows up as
             a change of answer with no change of content.

Neither needs new data, and neither can be gamed by the model: the prompt is the only
thing that differs.

Usage::

    venv_train/Scripts/python.exe scripts/evidence_swap_probe.py --items 100
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
OUT = ROOT / "outputs/analysis/phase3a/evidence_swap"
MENU_MARKER = "CANDIDATE MENU (answer with one of these fault_id values):"
MENU_LINE = re.compile(r"^-\s+(?:FAULT_\d{2}|UNCLASSIFIED_OPEN_SET):\s", re.MULTILINE)


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def split_prompt(prompt: str) -> Dict[str, object]:
    """The evidence part and the menu lines, apart from each other."""
    index = prompt.find(MENU_MARKER)
    if index < 0:
        raise ValueError("prompt has no candidate menu")
    head = prompt[:index]
    tail = prompt[index:]
    lines = tail.splitlines()
    menu, rest = [], []
    for line in lines:
        if line.startswith("- ") and re.match(r"^-\s+(?:FAULT_\d{2}|UNCLASSIFIED_OPEN_SET):",
                                              line):
            menu.append(line)
        else:
            rest.append(line)
    return {"head": head, "menu": menu, "rest": "\n".join(rest)}


def rebuild(parts: Dict[str, object], menu: Sequence[str]) -> str:
    return str(parts["head"]) + "\n".join([MENU_MARKER, *menu]) + "\n" + str(parts["rest"])


def gold_of(item: Dict) -> str:
    return str((item.get("target") or {}).get("fault_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", default="500")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cap", type=int, default=16)
    args = parser.parse_args()

    import torch
    from peft import PeftModel

    from eval_protocolA import adapter_root, bare, load_base
    from instrument_check import decode_first_field

    OUT.mkdir(parents=True, exist_ok=True)
    items = [item for item in load_jsonl(VAL)
             if item.get("prompt") and item.get("variant") == "menu"
             and not item.get("unobservable")]
    rng = random.Random(args.seed)
    # Stratified by gold so a fault with nine items is not a tenth of the sample.
    by_gold: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        by_gold[gold_of(item)].append(item)
    picked: List[Dict] = []
    golds = sorted(by_gold)
    quota = max(1, args.items // max(len(golds), 1))
    for gold in golds:
        bucket = sorted(by_gold[gold], key=lambda row: str(row["item_id"]))
        rng.shuffle(bucket)
        picked.extend(bucket[:quota])
    picked = picked[: args.items]

    # Donors: a different event, different fault, same (full) menu.
    donors = []
    for item in picked:
        pool = [candidate for candidate in items
                if gold_of(candidate) != gold_of(item)
                and candidate.get("simulationRun") != item.get("simulationRun")]
        donors.append(rng.choice(pool))

    model, tokenizer, config, _applied = load_base(
        ROOT / "configs/config_v2_protocolA_seed42.yaml")
    path = adapter_root(ROOT / f"outputs/tep_rca_v2_protocolA_seed42/checkpoint-{args.step}")
    wrapped = PeftModel.from_pretrained(model, str(path), adapter_name="swap")
    wrapped.set_adapter("swap")
    base_model = bare(wrapped)
    dummy = int(config["llm"].get("dummy_image_size", 448))

    original, swapped, shuffled = [], [], []
    started = time.time()
    for index, (item, donor) in enumerate(zip(picked, donors), start=1):
        parts = split_prompt(item["prompt"])
        donor_parts = split_prompt(donor["prompt"])
        # Same menu (the menu is identical for every menu-variant item), evidence from
        # the donor: only the part above CANDIDATE MENU changes.
        swapped_prompt = rebuild({"head": donor_parts["head"], "rest": parts["rest"]},
                                 parts["menu"])
        order = list(parts["menu"])
        rng.shuffle(order)
        shuffled_prompt = rebuild(parts, order)
        original.append(item["prompt"])
        swapped.append(swapped_prompt)
        shuffled.append(shuffled_prompt)
        if index % 25 == 0:
            print(json.dumps({"prepared": index, "of": len(picked)}), flush=True)
    print(json.dumps({"decode": "original"}), flush=True)
    original_out = decode_first_field(base_model, tokenizer, original, dummy, args.batch, args.cap)
    print(json.dumps({"decode": "swapped"}), flush=True)
    swapped_out = decode_first_field(base_model, tokenizer, swapped, dummy, args.batch, args.cap)
    print(json.dumps({"decode": "shuffled"}), flush=True)
    shuffled_out = decode_first_field(base_model, tokenizer, shuffled, dummy, args.batch, args.cap)
    seconds = time.time() - started

    rows = []
    for item, donor, a, b, c in zip(picked, donors, original_out, swapped_out, shuffled_out):
        gold, donor_gold = gold_of(item), gold_of(donor)
        rows.append({
            "item_id": item["item_id"], "gold": gold, "donor_gold": donor_gold,
            "original": a["label"] or a["raw"], "swapped": b["label"] or b["raw"],
            "shuffled": c["label"] or c["raw"],
            "gold_menu_position": next((i for i, line in
                                        enumerate(split_prompt(item["prompt"])["menu"], start=1)
                                        if line.split(":")[0].strip("- ") == gold), None),
        })

    def rate(predicate) -> float:
        return round(sum(1 for row in rows if predicate(row)) / max(len(rows), 1), 4)

    summary = {
        "items": len(rows),
        "seconds": round(seconds, 1),
        "seconds_per_decode": round(seconds / max(3 * len(rows), 1), 3),
        "accuracy_original": rate(lambda row: row["original"] == row["gold"]),
        "accuracy_shuffled": rate(lambda row: row["shuffled"] == row["gold"]),
        "prediction_change_rate_shuffle": rate(lambda row: row["shuffled"] != row["original"]),
        "follows_evidence": rate(lambda row: row["swapped"] == row["donor_gold"]),
        "follows_original": rate(lambda row: row["swapped"] == row["gold"]),
        "follows_neither": rate(lambda row: row["swapped"] not in (row["gold"],
                                                                  row["donor_gold"])),
        "by_gold": {},
        "rows": rows,
    }
    for gold in sorted({row["gold"] for row in rows}):
        subset = [row for row in rows if row["gold"] == gold]
        summary["by_gold"][gold] = {
            "items": len(subset),
            "accuracy_original": round(sum(1 for row in subset
                                           if row["original"] == row["gold"]) / len(subset), 4),
            "follows_evidence": round(sum(1 for row in subset
                                          if row["swapped"] == row["donor_gold"]) / len(subset), 4),
        }
    (OUT / f"swap_step{args.step}.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("items", "accuracy_original", "accuracy_shuffled",
                       "prediction_change_rate_shuffle", "follows_evidence",
                       "follows_original", "follows_neither")}, indent=2))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
