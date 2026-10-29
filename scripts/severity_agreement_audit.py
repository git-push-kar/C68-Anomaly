"""B2: measure the proposed severity rule against the authoritative one.

The authoritative level is `evidence.process_limits.compute_deterministic_severity`,
which the production pipeline already runs and which wrote the `severity` column
of every stored event. `llm/response/severity.py` keeps its own rule only as a
cross-check. This script scores that cross-check on every stored event so a
decision to adopt it (or not) rests on a number instead of an impression.

Nothing is tuned here: both rules are evaluated exactly as written, and no
threshold in either is derived from this data.

    python scripts/severity_agreement_audit.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence.process_limits import compute_deterministic_severity  # noqa: E402
from llm.response import severity as sev  # noqa: E402
from utils import ensure_dir, save_json  # noqa: E402

DB = ROOT / "outputs" / "events.db"
OUT = ROOT / "outputs" / "evaluation" / "severity_agreement.json"
MD = ROOT / "outputs" / "evaluation" / "severity_agreement.md"

_ORDER = ["low", "medium", "high", "critical"]


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"event store not found: {DB}")
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "select event_id, fault_label, severity, event_json, report_json from events order by event_id"
    ).fetchall()

    agree = 0
    counts: Counter = Counter()
    gaps: Counter = Counter()
    matrix: Dict[str, Counter] = {}
    disagreements: List[Dict] = []
    stored_matches_authoritative = 0
    stored_block: Dict[str, Dict] = {}
    unknown_label = 0

    for row in rows:
        event = json.loads(row["event_json"] or "{}")
        event["fault_label"] = row["fault_label"]
        report = json.loads(row["report_json"] or "{}")
        event["report"] = report

        authority = str(compute_deterministic_severity(event.get("evidence") or {}) or "medium").lower()
        assessment = sev.assess(event, report=report)
        proposed = str(assessment.components.get("proposed_level", authority)).lower()

        stored = str(row["severity"] or "").lower()
        if stored == authority:
            stored_matches_authoritative += 1
        elif stored in _ORDER:
            stored_block.setdefault("confusion", {}).setdefault(authority, Counter())[stored] += 1
            stored_block.setdefault("events", []).append(
                {"event_id": row["event_id"], "stored": stored, "authoritative": authority}
            )
        if row["fault_label"] is None:
            unknown_label += 1

        same = authority == proposed
        agree += int(same)
        counts[(authority, proposed)] += 1
        gaps[_ORDER.index(proposed) - _ORDER.index(authority)] += 1
        matrix.setdefault(authority, Counter())[proposed] += 1
        if not same:
            disagreements.append(
                {
                    "event_id": row["event_id"],
                    "fault_label": row["fault_label"],
                    "stored": row["severity"],
                    "authoritative": authority,
                    "proposed": proposed,
                    "gap": _ORDER.index(proposed) - _ORDER.index(authority),
                    "root_cause": (report or {}).get("root_cause"),
                    "n_windows": event.get("n_windows"),
                    "triggered_by": ((event.get("evidence") or {}).get("detector_evidence") or {}).get("triggered_by"),
                }
            )

    n = len(rows)
    result = {
        "n_events": n,
        "authoritative_rule": sev.AUTHORITATIVE_RULE,
        "proposed_rule": sev.PROPOSED_RULE,
        "agreement_rate": agree / max(1, n),
        "n_agree": agree,
        "n_disagree": n - agree,
        "confusion": {a: dict(b) for a, b in sorted(matrix.items())},
        "gap_distribution": {str(k): v for k, v in sorted(gaps.items())},
        "stored_matches_authoritative": stored_matches_authoritative,
        "stored_vs_authoritative": {
            "n_differ": len(stored_block.get("events", [])),
            "confusion": {
                a: dict(c) for a, c in sorted((stored_block.get("confusion") or {}).items())
            },
            "events": stored_block.get("events", []),
            "note": "The stored column was written when the event was ingested; it is not "
                    "recomputed later, so a change to the production rule shows up here. "
                    "The response layer quotes the recomputed value and discloses the stored one.",
        },
        "events_without_fault_label": unknown_label,
        "disagreements": sorted(disagreements, key=lambda d: (d["authoritative"], d["proposed"], d["event_id"])),
    }
    ensure_dir(OUT.parent)
    save_json(OUT, result)

    lines = [
        "# Severity agreement audit (B2)",
        "",
        f"- events: **{n}**",
        f"- authoritative rule: `{sev.AUTHORITATIVE_RULE}`",
        f"- cross-check rule: `{sev.PROPOSED_RULE}`",
        f"- agreement: **{agree}/{n} = {result['agreement_rate']*100:.1f}%**",
        f"- the stored `severity` column equals the authoritative rule on "
        f"**{stored_matches_authoritative}/{n}** events "
        f"({n - stored_matches_authoritative} differ; disclosed in answers, never quoted as the level)",
        "",
        "## Confusion (rows = authoritative, cols = proposed)",
        "",
        "| authoritative \\ proposed | " + " | ".join(_ORDER) + " |",
        "|---|" + "---|" * len(_ORDER),
    ]
    for a in _ORDER:
        row = matrix.get(a, Counter())
        lines.append(f"| {a} | " + " | ".join(str(row.get(p, 0)) for p in _ORDER) + " |")
    lines += [
        "",
        f"Gap distribution (proposed minus authoritative): `{json.dumps(result['gap_distribution'])}`",
        "",
        "## Disagreement table",
        "",
        "| event_id | fault | stored | authoritative | proposed | gap | triggered_by |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in result["disagreements"]:
        lines.append(
            f"| {d['event_id']} | {d['fault_label'] if d['fault_label'] is not None else '-'} | "
            f"{d['stored']} | {d['authoritative']} | {d['proposed']} | {d['gap']:+d} | {d['triggered_by']} |"
        )
    MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"events: {n}")
    print(f"agreement: {agree}/{n} = {result['agreement_rate']*100:.1f}%")
    print(f"stored == authoritative: {stored_matches_authoritative}/{n}")
    print(f"gap distribution: {result['gap_distribution']}")
    for a in _ORDER:
        row = matrix.get(a, Counter())
        print(f"  {a:9s} -> " + "  ".join(f"{p}:{row.get(p,0)}" for p in _ORDER))
    print(f"wrote {OUT}\nwrote {MD}")


if __name__ == "__main__":
    main()
