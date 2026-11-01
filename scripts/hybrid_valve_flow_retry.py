"""§17.7: regenerate the 21 valve/flow mislabels once, with the distinction stated.

The mislabel is a wording contradiction the prompt itself refutes — every tag is printed with its
kind (`[XMV_51, valve position, manipulated variable]`) — so the cheapest repair is to say it once
more in the user turn and regenerate. Only the items that carried a mislabel are regenerated
(≤ 21), with a merged in-memory copy of the adapter; the adapter files are never written.

The GPU is used only if it is idle. "Idle" is measured, not assumed: the script refuses when any
other process holds more than 2 GiB, and says so instead of queueing behind a training job.

Usage::

    python scripts/hybrid_valve_flow_retry.py             # runs if the GPU is idle
    python scripts/hybrid_valve_flow_retry.py --dry-run   # decide, do not generate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark.label_scorer import prompt_for  # noqa: E402
from llm.hybrid.generate import (  # noqa: E402
    contradictions,
    decode_text,
    forced_prefix,
    parse_slim,
    retry_question,
    validate_slim,
    valve_flow_mislabels,
)

OUT = ROOT / "outputs/analysis/phase3a/hybrid"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
CONFIG = ROOT / "configs/config_v2_protocolA_seed42.yaml"
ADAPTER = ROOT / "outputs/tep_rca_v2_protocolA_seed42/checkpoint-500/tep_rca_v2_protocolA_seed42"
CAP = 512
IDLE_LIMIT_MIB = 2048


def gpu_state() -> Dict:
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        used, util, temperature = [part.strip() for part in raw.stdout.strip().split(",")]
        return {"available": True, "used_mib": int(float(used)), "util": int(float(util)),
                "temperature": int(float(temperature)), "raw": raw.stdout.strip()}
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=str(OUT / "valve_flow_retry.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in
            (OUT / "generation_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    main_rows = [row for row in rows if row["kind"] == "main"]
    val = {item["item_id"]: item for item in
           [json.loads(line) for line in VAL.read_text(encoding="utf-8").splitlines() if line.strip()]}
    targets = [row for row in main_rows if valve_flow_mislabels(row.get("prose") or "")]
    state = gpu_state()
    decision = {
        "items_with_a_mislabel": len(targets),
        "gpu": state,
        "idle": bool(state.get("available") and state.get("used_mib", 10**9) <= IDLE_LIMIT_MIB),
        "note": ("generation runs only when the GPU is idle; a busy card means the code and its "
                 "test are committed and the experiment is reported as skipped"),
    }
    if args.dry_run or not decision["idle"] or not targets:
        decision["generated"] = 0
        Path(args.out).write_text(json.dumps(decision, indent=2), encoding="utf-8")
        print(json.dumps(decision, indent=2))
        return 0

    from llm.benchmark.decode_bench import build_model

    model, tokenizer, config, applied = build_model(CONFIG, str(ADAPTER), merged=True)
    allowed = sorted({str(label) for item in val.values() for label in (item.get("menu_ids") or [])}
                     | {"UNCLASSIFIED_OPEN_SET", "NORMAL_TRANSIENT"})
    results: List[Dict] = []
    started = time.time()
    for index, row in enumerate(targets):
        item = val[row["item_id"]]
        diagnosis = str(row["diagnosis"])
        prefix = forced_prefix(diagnosis)
        before = row.get("prose") or ""
        question = retry_question(item["prompt"])
        tokens = decode_text(model, tokenizer, prompt_for(question) + prefix, CAP)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        answer = parse_slim(text, prefix)
        problems = validate_slim(answer, allowed, diagnosis)
        after = " ".join(str((answer or {}).get(field) or "") for field in
                         ("physical_reasoning", "narrative")).strip()
        results.append({
            "item_id": row["item_id"], "diagnosis": diagnosis,
            "mislabel_before": row["valve_flow_mislabels"],
            "mislabel_after": valve_flow_mislabels(after),
            "fixed": bool(row["valve_flow_mislabels"]) and not valve_flow_mislabels(after),
            "new_mislabel": bool(valve_flow_mislabels(after))
                             and not set(valve_flow_mislabels(after)) & set(row["valve_flow_mislabels"]),
            "schema_errors": problems,
            "contradictions": [found["code"] for found in contradictions(answer, diagnosis)],
            "tokens": len(tokens),
            "prose_after": after[:600],
        })
        print(json.dumps({"done": index + 1, "of": len(targets),
                          "fixed": sum(1 for entry in results if entry["fixed"]),
                          "seconds": round(time.time() - started, 1)}), flush=True)

    decision.update({
        "generated": len(results),
        "fixed": sum(1 for entry in results if entry["fixed"]),
        "still_mislabelled": sum(1 for entry in results if entry["mislabel_after"]),
        "new_mislabels": sum(1 for entry in results if entry["new_mislabel"]),
        "schema_ok": sum(1 for entry in results if not entry["schema_errors"]),
        "seconds_per_item": round((time.time() - started) / max(len(results), 1), 2),
        "adapter": applied, "rows": results,
    })
    Path(args.out).write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: decision[key] for key in
                      ("items_with_a_mislabel", "generated", "fixed", "still_mislabelled",
                       "new_mislabels", "schema_ok", "seconds_per_item")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
