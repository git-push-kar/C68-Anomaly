"""Score exactly the 50 generation items with the adapter that generated them.

The checkpoint sweep scores a 300-item stratified subsample and the full
validation set for the *selected* checkpoint. Generation, however, was produced by
the checkpoint that was selected under the previous (buggy) scorer, so comparing it
against the new selection would compare a decision made by one set of weights with a
decision made by another. This scores those same 50 items with the checkpoint that
actually wrote the prose, using the corrected per-token score.

Usage::

    venv_train/Scripts/python.exe scripts/score_generation_items.py \
        --adapter outputs/tep_rca_v2_protocolA_seed42/checkpoint-400/tep_rca_v2_protocolA_seed42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
GEN = ROOT / "outputs/benchmark_v2/eval/generation_512.json"
OUT = ROOT / "outputs/benchmark_v2/eval/preds_gen_scored.jsonl"
CONFIG = ROOT / "configs/config_v2_protocolA_seed42.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    from peft import PeftModel

    from eval_protocolA import adapter_root, gold_of, load_base, score

    generation = json.loads(GEN.read_text(encoding="utf-8"))
    wanted = {row["item_id"] for row in generation["rows"]}
    items = [json.loads(line) for line in VAL.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    items = [item for item in items if item["item_id"] in wanted]
    missing = wanted - {item["item_id"] for item in items}
    if missing:
        raise SystemExit(f"generation items not found in val: {sorted(missing)[:5]}")

    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    wrapped = PeftModel.from_pretrained(model, str(adapter_root(Path(args.adapter))),
                                        adapter_name="gen")
    wrapped.set_adapter("gen")
    from eval_protocolA import bare

    rows = score(bare(wrapped), tokenizer, items, dummy, args.batch, via="language_model")
    Path(args.out).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    correct = sum(1 for row in rows if row["pred"] == row["gold"])
    print(json.dumps({"items": len(rows), "accuracy": round(correct / len(rows), 4),
                      "seconds_per_item": round(
                          sum(row["seconds"] or 0 for row in rows) / len(rows), 3)},
                     indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
