"""Write a release manifest so every quoted number can be traced to a file and a command.

The manifest is a JSON document listing each artefact with its SHA-256, the command that rebuilds it, and the
claim identifier it supports. It refuses to include missing files, so a stale reproduction path fails loudly
instead of shipping a broken package.

Usage:  python scripts/package_release.py [--check]
        --check verifies hashes against the existing manifest without writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/reports/RELEASE_MANIFEST_2026-11-16.json"

ARTEFACTS = [
    ("cells", "outputs/overnight_2026-11-05/n4_cells.json",
     "python scripts/overnight_run.py --stage n4", "C1, C2, C7"),
    ("cells_preinstrument", "outputs/overnight_2026-11-05/n4_cells_preinstrument_2026-11-16.json",
     "(frozen copy; do not rebuild)", "C1 (superseded)"),
    ("selection_receipt", "docs/judging/confirm/selection_receipt_2026-11-16_corrected2.json",
     "python scripts/select_condition.py --emit-receipt", "C7"),
    ("test_contrast", "outputs/analysis/phase3a/d33/test_contrast.json",
     "python scripts/score_test_pass.py", "C1, C3, C4"),
    ("classifier_contrasts", "outputs/analysis/phase3a/p2/classifier_contrasts.json",
     "python scripts/evidence_classifier.py --contrasts", "C6"),
    ("crossfit_val", "outputs/analysis/phase3a/hybrid/crossfit_val_rows.jsonl",
     "python scripts/crossfit_val_rows.py", "C2"),
    ("crossfit_test", "outputs/analysis/phase3a/hybrid/crossfit_test_rows.jsonl",
     "python scripts/crossfit_test_rows.py", "C2"),
    ("figures", "docs/reports/figures/figure_data.json",
     "python scripts/make_figures.py", "figures 1-4"),
    ("claims", "docs/reports/PAPER_CLAIMS_2026-11-16.md",
     "(assertion document; edited by hand with a dated section per change)", "all"),
    ("methods", "docs/reports/PAPER_METHODS_METRICS_2026-11-16.md",
     "(assertion document)", "all"),
]


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    entries, missing = [], []
    for name, rel, command, claims in ARTEFACTS:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        entries.append({"name": name, "path": rel, "sha256": digest(path),
                        "bytes": path.stat().st_size, "rebuild": command, "claims": claims})
    return {"generated": "2026-11-16", "root": str(ROOT.name), "entry_count": len(entries),
            "missing": missing, "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build()
    if args.check:
        if not MANIFEST.exists():
            print("no manifest to check", file=sys.stderr)
            return 2
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        before = {e["name"]: e["sha256"] for e in old["entries"]}
        drift = [e["name"] for e in manifest["entries"] if before.get(e["name"]) != e["sha256"]]
        print(json.dumps({"entries": manifest["entry_count"], "missing": manifest["missing"],
                          "changed_since_manifest": drift}, indent=1))
        return 1 if drift or manifest["missing"] else 0
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST.relative_to(ROOT)), "entries": manifest["entry_count"],
                      "missing": manifest["missing"]}, indent=1))
    return 0 if not manifest["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
