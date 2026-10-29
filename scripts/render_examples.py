"""Render one report and one follow-up from the stored-event fixtures.

Regenerating this file (rather than pasting a transcript) is what keeps the
examples honest when the response layer changes: every cell shown here is
resolved through :class:`StoreAccess` at render time.

    python scripts/render_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.response import StoreAccess, build_report, lookup_answer, render_followup, render_report  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "response"
OUT = ROOT / "outputs" / "evaluation" / "phase2a_examples.md"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    event = load("classified")
    report, result = build_report(event["report"], event)
    if report is None:
        raise SystemExit(f"fixture report did not validate: {result.as_dict()}")
    store = StoreAccess(event, event["report"])
    doc = render_report(report, store)

    hit = lookup_answer(event, event["report"], "How serious is this?")
    if hit is None:
        raise SystemExit("the severity question did not route")
    follow = render_followup(hit.answer, store, context={"intent": hit.answer.intent.value}, detail=hit.detail)

    parts = [
        "## Rendered report",
        "",
        doc.markdown.strip(),
        "",
        "## Rendered follow-up",
        "",
        follow.markdown.strip(),
        "",
        "## Validation",
        "",
        f"- report validator: `{json.dumps(result.as_dict())}`",
        f"- severity authority: `{report.severity.components['authoritative_rule']}`",
        f"- cross-check level (not displayed): `{report.severity.components['proposed_level']}`"
        f" — agreement `{report.severity.components['severity_agreement']}`",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"severity level (quoted): {report.severity.level.value}")
    print(f"cross-check level:       {report.severity.components['proposed_level']}"
          f"  agree={report.severity.components['severity_agreement']}")


if __name__ == "__main__":
    main()
