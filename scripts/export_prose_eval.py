"""3B-11 step 4 (validation half): the prose pilot's evaluation exports, hashed.

Three prompt orders for exactly the validation abstention / `NORMAL_TRANSIENT` / unclassified
items — the prompt as built (KB order), and two seeded shuffles (seeds 1, 2) — so menu-order
sensitivity is measurable on the same items. Plus every follow-up turn whose parent is one of those
items. CPU only: nothing here calls a model.

Usage::  python scripts/export_prose_eval.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_prose_dataset import build_prompt, load_jsonl  # noqa: E402
from scripts.fallback_v31 import effective_verdict  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
FOLLOWUP = ROOT / "outputs/benchmark_v2/followup/val.jsonl"
OUT = ROOT / "outputs/benchmark_v2/eval/prose_pilot"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [row for row in load_jsonl(DATASET / "val.jsonl")
             if str((row.get("target") or {}).get("verdict")) != "classified"]
    # narrowed below to the items the §17 rule actually abstains on (the adapter's own stratum)
    predictions = {row["item_id"]: row for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    items = [item for item in items
             if effective_verdict(str((predictions.get(item["item_id"]) or {}).get("pred")),
                                  str((predictions.get(item["item_id"]) or {}).get("verdict")
                                      or "classified")) != "classified"]
    items = [item for item in load_jsonl(DATASET / "val.jsonl")
             if effective_verdict(str((predictions.get(item["item_id"]) or {}).get("pred")),
                                  str((predictions.get(item["item_id"]) or {}).get("verdict")
                                      or "classified")) != "classified"]
    written: Dict[str, Dict] = {}
    for name, seed in (("kb", None), ("shuffled-1", 1), ("shuffled-2", 2)):
        rows: List[Dict] = []
        for item in items:
            target = item.get("target") or {}
            prediction = predictions.get(item["item_id"]) or {}
            probs = prediction.get("probs") or {}
            if isinstance(probs, str):
                import ast
                probs = ast.literal_eval(probs)
            top2 = sorted(((str(k), float(v)) for k, v in probs.items()), key=lambda x: -x[1])[:2]
            supplied = {"fault_id": target.get("fault_id"), "verdict": target.get("verdict"),
                        "ranked_hypotheses": [entry.get("fault_id") for entry in
                                              target.get("ranked_hypotheses") or []]
                        or [target.get("fault_id")]}
            if seed is None:
                prompt, identity = str(item["prompt"]), True
            else:
                varied = build_prompt(str(item["prompt"]), supplied, top2, seed, item["item_id"])
                prompt, identity = varied["prompt"], varied["identity_order"]
            rows.append({"item_id": item["item_id"], "order": name, "prompt": prompt,
                         "gold_fault_id": supplied["fault_id"], "gold_verdict": supplied["verdict"],
                         "identity_order": identity, "variant": item.get("variant")})
        path = OUT / f"prose_eval_val_{name}.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                        encoding="utf-8")
        written[name] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}

    events = {item.get("event_id") for item in items}
    turns, flagged_turns = [], 0
    for dialogue in load_jsonl(FOLLOWUP):
        by_event = dialogue.get("event_key") in events
        flagged = bool(dialogue.get("open_set") or dialogue.get("normal"))
        flagged_turns += len(dialogue.get("turns") or []) if flagged else 0
        if not (by_event or flagged):
            continue
        for index, turn in enumerate(dialogue.get("turns") or []):
            turns.append({"dialogue_id": dialogue["dialogue_id"], "turn": index,
                          "case_type": turn.get("case_type") or dialogue.get("case_type"),
                          "question": turn.get("question") or turn.get("prompt"),
                          "answer": turn.get("answer")})
    follow_path = OUT / "prose_eval_followups.jsonl"
    follow_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in turns) + "\n",
                           encoding="utf-8")
    manifest = {"items": len(items), "orders": written,
                "followups": {"path": str(follow_path), "sha256": sha256(follow_path),
                              "turns": len(turns),
                              "turns_in_flagged_dialogues": flagged_turns,
                              "dialogues": len({row["dialogue_id"] for row in turns})}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
