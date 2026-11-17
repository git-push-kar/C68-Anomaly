"""Copy every number source the paper cites into a committed evidence bundle.

The artefacts live under ``outputs/``, which is gitignored, so a reader who follows a citation in the paper cannot
fetch the file. This script copies the cited sources into ``docs/reports/evidence/files/``, hashes them, and writes
``index.json`` + ``README.md`` mapping each claim to the file that proves it.

    python scripts/bundle_evidence.py            # refresh the bundle
    python scripts/bundle_evidence.py --check    # verify hashes and that copies exist

Byte-for-byte copies: no number is edited, and ``--check`` fails if an origin moved on without a refresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "reports" / "evidence"
FILES = OUT / "files"
INDEX = OUT / "index.json"

#: (file name, origin, claims it supports, what a reader should look at)
SOURCES = [
    ("n4_cells.json", "outputs/overnight_2026-11-05/n4_cells.json", ["C1", "C2", "C6"],
     "Fit-set cell table: numeric/grounding/tag-role/anchor per cell, seam certification, D fallback rows, McNemar"),
    ("test_contrast.json", "outputs/analysis/phase3a/d33/test_contrast.json", ["C1", "C2"],
     "3,168-row confirmatory counts, rates and anchor decomposition"),
    ("classifier_contrasts.json", "outputs/analysis/phase3a/hybrid/vocabfix/classifier_contrasts.json", ["C3"],
     "Five-system diagnosis accuracy with 95 % CIs and every pairwise discordant count"),
    ("order_invariance.json", "outputs/overnight_2026-11-05/n5_invariance_dialogue.json", ["C1"],
     "368/520 identical defect profiles across prompt orders (0.7077)"),
    ("preference_result.json", "docs/judging/confirm/preference_2026-11-16_result.json", ["C4"],
     "360 pairs, three orderings, both rater sheets, exact binomial p per pairing"),
    ("preference_kappa.json", "docs/judging/confirm/preference_2026-11-16_kappa.json", ["C4"],
     "Cohen's kappa 0.5776, observed 0.7778, expected 0.4739, 80 disagreements listed"),
    ("selection_receipt.json", "docs/judging/confirm/selection_receipt_2026-11-16_final.json", ["C5"],
     "Rung-by-rung selector receipt: chosen A, decided_by_rung 5"),
    ("d_ceiling_decision.json", "docs/judging/confirm/d_ceiling_decision_2026-11-16.json", ["C6"],
     "Signed D decision: 'A with extra steps', 27.3 % pre-correction rate retained"),
    ("cases46_audit.csv", "docs/judging/confirm/cases46_audit.csv", ["instrument"],
     "46 hand-written contradictions; the checker agrees on all 46"),
]

#: Sources for the diagnosis row that are only supporting material (the row also needs a 520 KB prediction file).
SUPPORTING = [
    ("adapter_summary.json", "outputs/analysis/phase3a/hybrid/vocabfix/evidence_classifier/summary.json",
     ["C3", "adapter baseline"],
     "Adapter at checkpoint 300 on the 497 shared items: 414/497, abstention precision/recall"),
    ("preds_val_full_300.jsonl", "outputs/benchmark_v2/eval/preds_val_full_300.jsonl", ["adapter baseline"],
     "The checkpoint-300 predictions the summary is computed from (selected by macro-F1)"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict:
    FILES.mkdir(parents=True, exist_ok=True)
    entries, missing = [], []
    for name, rel, claims, note in SOURCES + SUPPORTING:
        origin = ROOT / rel
        if not origin.exists():
            missing.append(rel)
            continue
        copy = FILES / name
        if not copy.exists() or sha256(copy) != sha256(origin):
            shutil.copyfile(origin, copy)
        entries.append({
            "file": f"files/{name}",
            "origin": rel,
            "sha256": sha256(copy),
            "bytes": copy.stat().st_size,
            "claims": claims,
            "note": note,
            "role": "source" if (name, rel, claims, note) in SOURCES else "supporting",
        })
    index = {"generated": "2026-11-17", "root": "docs/reports/evidence", "entries": entries,
             "missing": missing}
    INDEX.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(render(index), encoding="utf-8")
    return index


def render(index: dict) -> str:
    lines = [
        "# Evidence bundle — every number source behind the paper",
        "",
        "Committed copies of the artefacts the paper cites. The originals live under `outputs/`, which is",
        "gitignored, so these copies are what a reader can actually fetch. Each file is byte-identical to its",
        "origin and is listed with its SHA-256 in `index.json`; `python scripts/bundle_evidence.py --check`",
        "re-verifies them.",
        "",
        "| claim | file | what it proves | SHA-256 |",
        "|---|---|---|---|",
    ]
    for entry in index["entries"]:
        claims = ", ".join(entry["claims"])
        lines.append(f"| {claims} | `{entry['file']}` | {entry['note']} | `{entry['sha256']}` |")
    lines += [
        "",
        "## How the claims map",
        "",
        "- **C1** numeric fidelity — `n4_cells.json` (fit set), `test_contrast.json` (test), "
        "`order_invariance.json` (kb vs shuffled orders).",
        "- **C2** anchor exactness — `n4_cells.json` summary `anchor_exact`, `test_contrast.json` "
        "`anchor_status`.",
        "- **C3** classifier parity — `classifier_contrasts.json`; adapter row from `adapter_summary.json`, "
        "whose predictions are `preds_val_full_300.jsonl`.",
        "- **C4** preference — `preference_result.json`, `preference_kappa.json`.",
        "- **C5** selection — `selection_receipt.json`.",
        "- **C6** D wrapper — `n4_cells.json` seam certification, `d_ceiling_decision.json`.",
        "- Instrument — `cases46_audit.csv` (46/46).",
        "",
        "Not in this bundle because they are already committed elsewhere: `docs/reports/figures/figure_data.json`",
        "(strata table and figure inputs), `docs/reports/figures/*.png`, `docs/reports/references.bib`.",
        "",
    ]
    return "\n".join(lines)


def check() -> int:
    if not INDEX.exists():
        print("no index.json to check; run without --check", file=sys.stderr)
        return 2
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    problems = []
    for entry in index["entries"]:
        path = ROOT / "docs/reports/evidence" / entry["file"]
        if not path.exists():
            problems.append(f"missing copy: {entry['file']}")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"hash drift in copy: {entry['file']}")
        origin = ROOT / entry["origin"]
        if origin.exists() and sha256(origin) != entry["sha256"]:
            problems.append(f"origin has changed since bundling: {entry['origin']}")
    print(f"{len(index['entries'])} entries checked")
    for problem in problems:
        print(" -", problem)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    index = build()
    print(f"wrote {INDEX.relative_to(ROOT)} with {len(index['entries'])} entries")
    if index["missing"]:
        print("missing origins:", ", ".join(index["missing"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
