"""Rebuild the blinded A/B sheet on 100 non-counterfactual items (PROTOCOL §20.7).

The sheet §19.2 is scored on has a defect the 3B-7 report disclosed but did not fix: 19 of its
100 rows are **counterfactual** — the wrong diagnosis is forced into the prefix and the adapter is
required to assert it. §18.5 already measured that behaviour (58 of 58 rationalised), and a prose
source cannot be failed for being wrong about a diagnosis it was ordered to write. Scoring those
rows against §19.2's "the prose's diagnosis claim agrees with the supplied diagnosis" mixes two
questions and lets a forced-wrong row dilute the grounding and preference counts.

So this rebuilds the sheet **before any row is scored**:

* the **81 non-counterfactual items** of `audit_ab_100.csv` are kept as they are;
* the 19 counterfactual rows are replaced by **19 seeded draws from the 242 main-batch answers**,
  family-stratified to the surviving mix by largest remainder;
* `text_A` / `text_B` are adapter forced-diagnosis prose and **fallback v2** (§20.6), in a seeded
  random assignment (`--seed 42`);
* the key is a separate file; every scoring column is blank;
* the old sheet and its key are **retained and marked superseded**, not edited.

CPU only, no model, no generation, no test component.

Usage::

    python scripts/make_ab_sheet_v2.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fallback_baseline import adapter_answer, load_jsonl, render  # noqa: E402
from scripts.fallback_v2 import fallback_v2, parse_store  # noqa: E402

DATASET = ROOT / "outputs/benchmark_v2/dataset"
HYBRID = ROOT / "outputs/analysis/phase3a/hybrid"
OUT = ROOT / "outputs/analysis/phase3a/fit"
SEED = 42
WANTED = 100
REPLACE = 19
SCORE_COLUMNS = ("correct_A", "grounded_A", "consistent_A", "useful_A",
                 "correct_B", "grounded_B", "consistent_B", "useful_B", "preference")


def probs_of(row: Dict) -> Dict[str, float]:
    raw = row.get("probs")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = {}
    return {str(label): float(value) for label, value in (raw or {}).items()}


def allocate(counts: Dict[str, int], total: int) -> Dict[str, int]:
    """Largest-remainder allocation of ``total`` draws to the surviving family mix."""
    weight = sum(counts.values()) or 1
    exact = {family: total * count / weight for family, count in counts.items()}
    base = {family: int(value) for family, value in exact.items()}
    short = total - sum(base.values())
    for family in sorted(counts, key=lambda key: (-(exact[key] - base[key]), key))[:short]:
        base[family] += 1
    return base


def adapter_texts() -> Dict[str, Dict]:
    """The adapter side, assembled once so the sheet and the defect count cannot disagree."""
    generated = {row["item_id"]: row for row in load_jsonl(HYBRID / "generation_rows.jsonl")}
    audit_prose = {row["item_id"]: row.get("prose") or ""
                   for row in csv.DictReader((HYBRID / "audit_hybrid_100.csv").open(encoding="utf-8"))}
    out: Dict[str, Dict] = {}
    for item_id, row in generated.items():
        # the stored `answer` is already a dict; `adapter_answer` parses the string form only and
        # would else fall through to an empty narrative, which is how a first pass reported
        # "400 of 400 narratives empty" on data where none is.
        raw = row.get("answer")
        answer = raw if isinstance(raw, dict) else adapter_answer(row)
        out[item_id] = {
            "family": row.get("family"),
            "kind": row.get("kind"),
            "ranked_hypotheses": answer.get("ranked_hypotheses"),
            "physical_reasoning": audit_prose.get(item_id) or answer.get("physical_reasoning"),
            "narrative": answer.get("narrative"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    val = {row["item_id"]: row for row in load_jsonl(DATASET / "val.jsonl")}
    predicted = {row["item_id"]: row for row in load_jsonl(HYBRID / "crossfit_val_rows.jsonl")}
    adapters = adapter_texts()

    old_sheet = list(csv.DictReader((OUT / "audit_ab_100.csv").open(encoding="utf-8")))
    kept = [row for row in old_sheet if row["family"] != "counterfactual"]
    dropped = [row["item_id"] for row in old_sheet if row["family"] == "counterfactual"]
    if len(dropped) != REPLACE:
        raise SystemExit(f"expected {REPLACE} counterfactual rows, found {len(dropped)}")

    mix: Dict[str, int] = {}
    for row in kept:
        mix[row["family"]] = mix.get(row["family"], 0) + 1
    allocation = allocate(mix, REPLACE)

    rng = random.Random(args.seed)
    taken = {row["item_id"] for row in kept} | set(dropped)
    draws: List[str] = []
    for family, want in sorted(allocation.items()):
        pool = sorted(item_id for item_id, entry in adapters.items()
                      if entry["kind"] == "main" and entry["family"] == family
                      and item_id not in taken and item_id in val)
        rng.shuffle(pool)
        if len(pool) < want:
            raise SystemExit(f"only {len(pool)} {family} draws available, want {want}")
        draws.extend(pool[:want])
        taken.update(pool[:want])
    if len(draws) != REPLACE:
        raise SystemExit(f"drew {len(draws)} replacements, want {REPLACE}")

    entries = [{"item_id": row["item_id"], "family": row["family"],
                "prose": row.get("prose"), "source": "kept"} for row in kept]
    entries += [{"item_id": item_id, "family": adapters[item_id]["family"], "prose": None,
                 "source": "draw"} for item_id in draws]
    if len(entries) != WANTED or len({entry["item_id"] for entry in entries}) != WANTED:
        raise SystemExit(f"{len(entries)} rows over {len({e['item_id'] for e in entries})} items")

    assign = random.Random(args.seed)
    sheet_rows, key_rows = [], []
    for entry in entries:
        item_id = entry["item_id"]
        item = val[item_id]
        prediction = predicted.get(item_id) or {}
        diagnosis = str(prediction.get("pred") or "UNCLASSIFIED_OPEN_SET")
        verdict = str(prediction.get("verdict") or "classified")
        adapter = adapters[item_id]
        store = parse_store(item["prompt"])

        adapter_text = render(item, {
            "fault_id": diagnosis, "verdict": verdict,
            "ranked_hypotheses": adapter["ranked_hypotheses"],
            "physical_reasoning": entry["prose"] or adapter["physical_reasoning"],
            "narrative": adapter["narrative"],
        })
        fallback = fallback_v2(diagnosis, verdict, item.get("menu_ids") or [], store,
                               probs_of(prediction))
        fallback_text = render(item, fallback)

        adapter_is_a = assign.random() < 0.5
        sheet_rows.append({
            "item_id": item_id, "family": entry["family"], "diagnosis": diagnosis,
            "verdict": verdict, "source": entry["source"],
            "text_A": adapter_text if adapter_is_a else fallback_text,
            "text_B": fallback_text if adapter_is_a else adapter_text,
            **{column: "" for column in SCORE_COLUMNS}, "notes": "",
        })
        key_rows.append({"item_id": item_id, "A": "adapter" if adapter_is_a else "fallback",
                         "B": "fallback" if adapter_is_a else "adapter", "seed": args.seed})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / "audit_ab_100_v2.csv"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sheet_rows)
    key = out / "audit_ab_100_v2_key.csv"
    with key.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    superseded_note = out / "audit_ab_100_SUPERSEDED.txt"
    superseded_note.write_text(
        "SUPERSEDED 2026-11-01 by audit_ab_100_v2.csv and audit_ab_100_v2_key.csv.\n"
        "Reason: this sheet carries 19 counterfactual (forced-wrong) rows, which PROTOCOL §20.2\n"
        "excludes from the §19.2 prose verdict. The files are retained unchanged; see\n"
        "docs/reports/3b8_corrections.md and outputs/analysis/phase3a/fit/audit_ab_100_v2_meta.json.\n",
        encoding="utf-8")

    meta = {
        "rows": len(sheet_rows),
        "distinct_items": len({row["item_id"] for row in sheet_rows}),
        "seed": args.seed,
        "composition": {"kept_non_counterfactual": len(kept), "replaced_counterfactual": len(dropped),
                        "drawn_from_main_batch": len(draws)},
        "family_mix_kept": dict(sorted(mix.items())),
        "draw_allocation": dict(sorted(allocation.items())),
        "families_final": {family: sum(1 for row in sheet_rows if row["family"] == family)
                           for family in sorted({row["family"] for row in sheet_rows})},
        "adapter_as_A": sum(1 for row in key_rows if row["A"] == "adapter"),
        "scoring_columns_filled": {column: sum(1 for row in sheet_rows if row[column])
                                   for column in SCORE_COLUMNS},
        "replaced_items": sorted(dropped),
        "drawn_items": sorted(draws),
        "supersedes": ["audit_ab_100.csv", "audit_ab_100_key.csv"],
        "sheet": str(sheet), "key": str(key),
        "comparator": "fallback v2 (PROTOCOL §20.1, §20.6)",
    }
    (out / "audit_ab_100_v2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({key: meta[key] for key in
                      ("rows", "distinct_items", "composition", "family_mix_kept",
                       "draw_allocation", "families_final", "adapter_as_A",
                       "scoring_columns_filled")}, indent=2))
    print(f"written: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
