"""Adapter-only decode in KB order and in two shuffled orders (PROTOCOL §19.4).

§18.5 required adapter numbers in more than one menu order; §19.4 fixed the material as KB order,
`--menu-seed 1` and `--menu-seed 2`. The classifier plus menu rule is order-invariant by
construction and tested as such, so only the adapter needs three runs — and the three runs must be
the *same model*, loaded once, or the comparison measures the loader as much as the order.

The instrument is §14's decode instrument, unchanged: greedy decoding of the `fault_id` value with a
16-token cap, carrying the chain-rule log-probability. It is not the label scorer, so the 3B-5
LM-direct path does not apply here — that path replaces *scoring*, and this is generation.

A wall-clock budget is enforced between orders: if the remaining orders are projected past it they
are skipped and reported as skipped, which is what the task asked for rather than a partial table
presented as complete.

GPU. Sector 5 only; nothing is trained and the adapter files are read-only.

Usage::

    ./venv_train/Scripts/python.exe scripts/menu_order_decode.py --budget-seconds 1080
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark.menu_order import menu_ids, reorder_menu  # noqa: E402

VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
CONFIG = ROOT / "configs/config_v2_protocolA_seed42.yaml"
ADAPTER = ROOT / "outputs/tep_rca_v2_protocolA_seed42"
OUT = ROOT / "outputs/analysis/phase3a/fit"
STEP = "500"
ORDERS = (("kb", None), ("shuffled-1", 1), ("shuffled-2", 2))


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gold_of(item: Dict) -> str:
    return str((item.get("target") or {}).get("fault_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--adapter", default=str(ADAPTER))
    parser.add_argument("--step", default=STEP)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--budget-seconds", type=float, default=1080.0)
    parser.add_argument("--orders", default="kb,shuffled-1,shuffled-2")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    import torch
    from peft import PeftModel

    from eval_protocolA import adapter_root, bare, load_base
    from instrument_check import decode_first_field, metrics

    started = time.time()
    items = load_jsonl(VAL)
    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    path = (adapter_root(Path(args.adapter) / f"checkpoint-{args.step}")
            if args.step != "final" else adapter_root(Path(args.adapter)))
    wrapped = PeftModel.from_pretrained(model, str(path), adapter_name="full")
    wrapped.set_adapter("full")
    base_model = bare(wrapped)
    load_seconds = round(time.time() - started, 1)

    wanted = [name for name in args.orders.split(",") if name]
    summary: Dict[str, Dict] = {
        "model": {"adapter": str(path), "step": args.step, "items": len(items),
                  "cap": args.cap, "decode_batch": args.batch,
                  "load_seconds": load_seconds,
                  "gpu": _gpu()},
        "orders": {},
    }

    for name, seed in ORDERS:
        if name not in wanted:
            continue
        elapsed = time.time() - started
        remaining = args.budget_seconds - elapsed
        projected = None if not summary["orders"] else \
            elapsed / max(1, len(summary["orders"]))  # seconds per completed order
        if projected is not None and remaining < projected * 0.9:
            summary["orders"][name] = {"skipped": True,
                                       "reason": f"budget: {remaining:.0f}s left, one order takes "
                                                 f"about {projected:.0f}s"}
            continue
        questions = [reorder_menu(item["prompt"], seed=seed or 0, item_id=str(item["item_id"]))
                     if seed is not None else item["prompt"] for item in items]
        moved = sum(1 for item, question in zip(items, questions)
                    if menu_ids(question) != menu_ids(item["prompt"]))
        decoded = decode_first_field(base_model, tokenizer, questions, dummy,
                                     args.batch, args.cap, via="language_model")
        rows = []
        for item, entry in zip(items, decoded):
            rows.append({"item_id": item["item_id"], "gold": gold_of(item),
                         "run": "|".join(str(item.get(key)) for key in
                                         ("source_file", "fault", "simulationRun")),
                         "unobservable": bool(item.get("unobservable")),
                         "variant": item.get("variant"),
                         "pred": entry["label"] or entry["raw"], "raw": entry["raw"],
                         "value": entry["chain_logp"], "decoded_tokens": entry["tokens"],
                         "seconds": entry["seconds"], "menu_order": name})
        observable = [row for row in rows if not row["unobservable"]]
        result = metrics(observable, f"decode:{name}")
        result["menu_entries_moved"] = {"items": len(items), "items_with_a_different_order": moved}
        result["seconds"] = round(sum(row["seconds"] for row in rows), 1)
        summary["orders"][name] = result
        (Path(args.out) / f"preds_decode_step{args.step}_{name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        print(json.dumps({"order": name, "accuracy": result["accuracy"],
                          "macro_f1": result["macro_f1"], "moved": moved,
                          "seconds": result["seconds"]}, default=str), flush=True)

    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["invariant_reference"] = {
        "source": "outputs/analysis/phase3a/hybrid/crossfit.json",
        "note": "the classifier plus menu rule cannot see the order (tested in "
                "tests/test_menu_order.py), so its row is the same for all three orders",
    }
    (Path(args.out) / f"menu_order_decode_step{args.step}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({name: {key: value.get(key) for key in ("accuracy", "macro_f1", "skipped")}
                      for name, value in summary["orders"].items()}, indent=1, default=str))
    print(f"written: {Path(args.out) / f'menu_order_decode_step{args.step}.json'}")
    del torch
    return 0


def _gpu() -> Dict:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = "unavailable"
    return {"at_start": out}


if __name__ == "__main__":
    raise SystemExit(main())
