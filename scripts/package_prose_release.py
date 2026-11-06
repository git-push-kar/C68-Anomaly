"""Package the prose release set: every artefact a reviewer needs, with hashes.

Writes `release_manifest.json` and `RELEASE_SET.md` into the out directory. The list is
explicit rather than globbed, so a missing file is reported as missing instead of quietly
shrinking the set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

ARTEFACTS = [
    "docs/HANDOVER_BENCHMARK_PLAN_V2.md",
    "docs/judging/confirm/signed.json",
    "outputs/benchmark_v2/eval/prose_pilot/raw_pilot_kb.jsonl",
    "outputs/benchmark_v2/eval/prose_pilot/raw_pilot_shuffled-1.jsonl",
    "outputs/benchmark_v2/eval/prose_pilot/raw_pilot_shuffled-2.jsonl",
    "outputs/benchmark_v2/eval/prose_pilot/prose_eval_followups.jsonl",
    "outputs/benchmark_v2/sft/train_prose_v1.jsonl",
    "outputs/benchmark_v2/sft/followup_train.jsonl",
    "outputs/overnight_2026-11-05/n1_calibration.json",
    "outputs/overnight_2026-11-05/n1_readings.json",
    "outputs/overnight_2026-11-05/n1_parity.json",
    "outputs/overnight_2026-11-05/n4_cells.json",
    "outputs/overnight_2026-11-05/n4_flags_per_item.csv",
    "outputs/overnight_2026-11-05/n5_invariance_dialogue.json",
    "outputs/overnight_2026-11-05/n5_examples.csv",
    "outputs/overnight_2026-11-05/prompts_520.json",
    "outputs/overnight_2026-11-05/selection_receipt.json",
    "scripts/cases46.py",
    "scripts/pilot_rechecks.py",
    "scripts/overnight_run.py",
    "scripts/select_condition.py",
    "scripts/preference_sheet.py",
    "scripts/preference_agreement.py",
    "scripts/mechanism_sheet.py",
    "configs/config_prose_pilot_seed42.yaml",
    "configs/config_prose_e_seed43.yaml",
    "configs/config_followup_f_seed42.yaml",
]


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs/overnight_2026-11-05"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    files, missing = [], []
    for rel in ARTEFACTS:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)})

    payload = {"root": str(ROOT), "count": len(files), "files": files, "missing": missing}
    (out / "release_manifest.json").write_text(json.dumps(payload, indent=2) + "\n",
                                               encoding="utf-8")
    lines = ["# Release set", "", "%d files hashed, %d missing." % (len(files), len(missing)), "",
             "| file | bytes | sha256 |", "|---|---|---|"]
    for item in files:
        lines.append("| `%s` | %d | `%s` |" % (item["path"], item["bytes"], item["sha256"][:24]))
    if missing:
        lines += ["", "## Missing", ""] + ["- `%s`" % m for m in missing]
    (out / "RELEASE_SET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("hashed %d, missing %d -> %s" % (len(files), len(missing), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
