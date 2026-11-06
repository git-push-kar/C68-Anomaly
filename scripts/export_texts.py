"""D29 step 0: export the five report conditions as a flat text corpus.

Writes `texts.jsonl` rows `{"item_id", "condition", "text"}` for every item and every
condition measured on the report surface (PROTOCOL 30.2): A, B, C, M (the missing cell),
D. Nothing is generated and no model is loaded: A/C/M/D are assembled from stored text
exactly as `overnight_run.stage_n4` assembles them, so the corpus and the N4 cell table
describe the same texts.

The export carries no gold label, no family and no side information: it is safe to hand
to `scripts/preference_sheet.py`, which assigns sides itself.

Example:
    python scripts/export_texts.py --out outputs/analysis/phase3a/fit/d29
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.build_prose_dataset import load_jsonl                        # noqa: E402
from scripts.fallback_v2 import parse_store                               # noqa: E402
from scripts.fallback_v3 import load_overlay                              # noqa: E402
from scripts.overnight_run import (CHECK_NAMES, _a_text, compose,          # noqa: E402
                                  normalise_flags)
from scripts.pilot_rechecks import measure                                # noqa: E402
from scripts.prose_pilot_prompts import EXPORT, build_orders, parse_prose  # noqa: E402

HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
CONDITIONS = ("A", "B", "C", "M", "D")
ITEMS = 520


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out_dir: Path) -> dict:
    built = build_orders()
    if len(built) != ITEMS:
        raise SystemExit("prompt rebuild produced %d items, not %d" % (len(built), ITEMS))
    probs = {row["item_id"]: row.get("probs")
             for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    overlay = load_overlay(ROOT / "configs/kb_overlay_v1.json")
    a_text = _a_text(built, probs, overlay)
    rows_b = {row["item_id"]: row for row in load_jsonl(EXPORT / "raw_pilot_kb.jsonl")}
    if set(rows_b) != {row["item_id"] for row in built}:
        raise SystemExit("raw_pilot_kb.jsonl and the prompt rebuild cover different items")

    corpus = []
    empty = []
    fallback_rows = 0
    for row in built:
        item = row["item_id"]
        prompt = row["prompts"]["kb"]
        store = parse_store(prompt)
        stored = rows_b[item]
        parsed = parse_prose(stored["text"])
        b_text = parsed["prose"] if parsed["schema_valid"] else str(stored["text"])
        a = a_text[item]
        c_text, _ = compose(a, b_text, store)      # A's numbers, B's mechanism
        m_text, _ = compose(b_text, a, store)      # B's numbers, A's mechanism
        flags = normalise_flags(measure(b_text, prompt, stored["gold_fault_id"],
                                       stored["gold_verdict"], store))
        use_a = any(flags[c] for c in CHECK_NAMES)
        fallback_rows += 1 if use_a else 0
        d_text = a if use_a else b_text
        for name, text in (("A", a), ("B", b_text), ("C", c_text),
                           ("M", m_text), ("D", d_text)):
            body = str(text or "").strip()
            corpus.append({"item_id": item, "condition": name, "text": body})
            if not body:
                empty.append(item + ":" + name)

    if len(corpus) != ITEMS * len(CONDITIONS):
        raise SystemExit("corpus is %d rows, expected %d"
                         % (len(corpus), ITEMS * len(CONDITIONS)))
    if empty:
        raise SystemExit("%d empty texts, first: %s" % (len(empty), empty[:5]))

    by_item = defaultdict(dict)
    for entry in corpus:
        by_item[entry["item_id"]][entry["condition"]] = entry["text"]
    short = [item for item, sides in by_item.items() if len(sides) != len(CONDITIONS)]
    if short:
        raise SystemExit("%d items miss a condition, first: %s" % (len(short), short[:5]))

    out_dir.mkdir(parents=True, exist_ok=True)
    texts_path = out_dir / "texts.jsonl"
    with texts_path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in corpus:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    lengths = defaultdict(list)
    for entry in corpus:
        lengths[entry["condition"]].append(len(entry["text"]))
    per_condition = {}
    for name in CONDITIONS:
        values = sorted(lengths[name])
        per_condition[name] = {"rows": len(values), "chars_min": values[0],
                               "chars_median": values[len(values) // 2],
                               "chars_max": values[-1],
                               "chars_total": sum(values)}
    payload = {
        "generated_by": "scripts/export_texts.py",
        "items": len(by_item),
        "conditions": list(CONDITIONS),
        "rows": len(corpus),
        "texts": str(texts_path.relative_to(ROOT)),
        "sha256_texts": sha(texts_path),
        "per_condition": per_condition,
        "stored_text_only": True,
        "gpu": {"A": 0, "B": 0, "C": 0, "M": 0, "D": 0,
                "note": "assembled from stored generations; no model loaded"},
        "D_fallback_rows": fallback_rows,
        "D_fallback_share": round(fallback_rows / len(by_item), 4),
        "carries_labels": False,
        "carries_side_assignment": False,
        "sources": {
            "prompts": "scripts.prose_pilot_prompts.build_orders (520 items, kb order)",
            "B": "outputs/benchmark_v2/eval/prose_pilot/raw_pilot_kb.jsonl",
            "A": "scripts.overnight_run._a_text over "
                 "outputs/analysis/phase3a/hybrid/crossfit_val_rows.jsonl",
            "C": "scripts.overnight_run.compose(A numbers, B mechanism)",
            "M": "scripts.overnight_run.compose(B numbers, A mechanism)",
            "D": "A when any of the four checks flags B, else B",
        },
        "hashes": {
            "scripts/export_texts.py": sha(ROOT / "scripts/export_texts.py"),
            "scripts/overnight_run.py": sha(ROOT / "scripts/overnight_run.py"),
            "scripts/pilot_rechecks.py": sha(ROOT / "scripts/pilot_rechecks.py"),
        },
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/analysis/phase3a/fit/d29")
    args = ap.parse_args()
    payload = build(Path(args.out) if Path(args.out).is_absolute()
                    else ROOT / args.out)
    print(json.dumps({k: payload[k] for k in ("items", "rows", "sha256_texts",
                                              "D_fallback_share")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
