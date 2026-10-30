#!/usr/bin/env python
"""Measure what the LIVE path sends to the model versus answers by template.

Part (a) of Phase 2C. For each of the 14 candidate intents in
``docs/FOLLOWUP_SPEC.md`` §2 this routes 50 realistic operator questions through

* the live dispatcher (``RCAInference._dispatch_with_intent``), which is the
  branch ``main.py:answer_followup`` calls first, and
* :class:`~llm.response.router.ResponseRouter`, once with the proposed intents
  off (the live configuration) and once with them on (the benchmark
  configuration).

and reports the resulting confusion table. A question the dispatcher does not
claim is the one the live path hands to the adapter -- that is the column the
follow-up dataset must be trained on.

It also measures the two existing question corpora, because their *format*
differs from what the router receives at run time:

* ``grounded_qa`` rows carry the telemetry preamble inline in ``question`` and
  end with ``Operator question: ...``; the live call site passes only the
  operator ask (``main.py:answer_followup(event_id, user_question)``).
* hand-written ``followup`` rows are operator asks already, so they show what
  fraction of natural follow-ups never matches a template.

Usage::

    python scripts/inspect_followup_routing.py            # write the artefact
    python scripts/inspect_followup_routing.py --no-corpus  # bank only (no arrow deps)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.followup.paraphrases import (  # noqa: E402
    FREE_FORM_LABELS,
    LABELS,
    REPLY_SHAPED,
    ROUTED_LABELS,
    probe_questions,
)
from llm.response.router import ResponseRouter  # noqa: E402
from llm.response.schemas import Intent  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "response"
OUT = ROOT / "outputs" / "evaluation"

#: Same word list the live numeric guardrail uses (``_guard_llm_answer``).
GUARDRAIL_MARKERS = (
    "score", "scores", "metric", "metrics", "threshold",
    "reconstruction error", "q-stat", "q stat",
    "evidence", "why is this an anomaly", "why was this flagged",
    "sensor value", "sensor values", "ideal value", "normal value",
    "baseline value", "measured value", "observed value",
)

OPERATOR_MARKER = "Operator question:"

BENCHMARK_ONLY = frozenset({"hypothesis_comparison", "out_of_scope_refusal", "clarification_reply"})


def family_of(label: str) -> str:
    """Which column of the brief this label answers: template, free-form, or benchmark."""
    if label in BENCHMARK_ONLY:
        return "benchmark_only"
    if label in FREE_FORM_LABELS:
        return "live_free_form"
    return "live_template"


def load_fixture(name: str) -> Dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def live_intent(event: Dict, report: Dict, question: str) -> Optional[str]:
    """The intent the live dispatcher reports, or ``None`` (falls to the model)."""
    from llm.inference import RCAInference

    hit = RCAInference._dispatch_with_intent(event, report, question)
    return hit[0] if hit is not None else None


def route(event: Dict, report: Dict, question: str, proposed: bool) -> Dict:
    router = ResponseRouter(enable_proposed=proposed)
    result = router.route(event, report, question, with_answer=False)
    return {
        "primary": result.primary.value if result.primary else None,
        "candidates": [c.value for c in result.candidates],
        "clarification": result.clarification.as_dict() if result.clarification else None,
    }


def guardrail_eligible(question: str) -> bool:
    q = (question or "").lower()
    return any(word in q for word in GUARDRAIL_MARKERS)


def measure_bank(event: Dict, report: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Route all 50 probe questions per label; return per-question and per-label rows."""
    per_question: List[Dict] = []
    for label in LABELS:
        for question in probe_questions(label):
            live = live_intent(event, report, question)
            proposed = route(event, report, question, proposed=True)
            live_cfg = route(event, report, question, proposed=False)
            per_question.append(
                {
                    "label": label,
                    "family": family_of(label),
                    "question": question,
                    "live_intent": live,
                    "live_route": "template" if live else "llm",
                    "bench_primary": proposed["primary"],
                    "bench_clarify": (proposed["clarification"] or {}).get("trigger"),
                    "live_cfg_candidates": len(live_cfg["candidates"]),
                    "guardrail_eligible": bool(live is None and guardrail_eligible(question)),
                }
            )
    rows = []
    for label in LABELS:
        items = [row for row in per_question if row["label"] == label]
        routed = Counter(row["live_intent"] or "(none: model)" for row in items)
        bench = Counter(row["bench_primary"] or "(none)" for row in items)
        rows.append(
            {
                "label": label,
                "family": family_of(label),
                "n": len(items),
                "routed": routed,
                "bench": bench,
                "template": sum(1 for row in items if row["live_route"] == "template"),
                "llm": sum(1 for row in items if row["live_route"] == "llm"),
                "clarify": sum(1 for row in items if row["bench_clarify"]),
                "guardrail_eligible": sum(1 for row in items if row["guardrail_eligible"]),
            }
        )
    return per_question, rows


def summarise_families(per_question: List[Dict]) -> List[Dict]:
    """Aggregate the three families: what the live path templates and what it sends on."""
    rows = []
    for family in ("live_template", "live_free_form", "benchmark_only"):
        items = [row for row in per_question if row["family"] == family]
        if not items:
            continue
        llm = sum(1 for row in items if row["live_route"] == "llm")
        rows.append(
            {
                "family": family,
                "labels": len({row["label"] for row in items}),
                "n": len(items),
                "template": len(items) - llm,
                "model": llm,
                "model_share": round(llm / len(items), 4),
            }
        )
    return rows


def measure_reply_shaped(event: Dict, report: Dict) -> Dict:
    counts = Counter()
    for question in REPLY_SHAPED:
        live = live_intent(event, report, question)
        counts[live or "(none: model)"] += 1
        counts["__clarify__"] += 1 if route(event, report, question, proposed=True)["clarification"] else 0
    return {"n": len(REPLY_SHAPED), "routed": dict(counts)}


def _split_operator_ask(text: str) -> Tuple[str, bool]:
    if OPERATOR_MARKER in (text or ""):
        return text.split(OPERATOR_MARKER, 1)[1].strip(), True
    return (text or "").strip(), False


def measure_corpus(event: Dict, report: Dict) -> List[Dict]:
    """Routing of the two existing corpora, measured both ways."""
    rows: List[Dict] = []
    for base in ("data/llm_1", "data/llm"):
        for split in ("train", "val", "test"):
            path = ROOT / base / split
            if not (path / "dataset_info.json").exists():
                continue
            try:
                from datasets import load_from_disk
            except Exception:  # pragma: no cover - optional dependency
                return rows
            ds = load_from_disk(str(path))
            kinds = ds["kind"] if "kind" in ds.features else ["?"] * len(ds)
            buckets: Dict[str, Counter] = {}
            preamble = Counter()
            for kind, question in zip(kinds, ds["question"]):
                if kind not in ("grounded_qa", "followup"):
                    continue
                ask, has_preamble = _split_operator_ask(question)
                preamble[kind] += int(has_preamble)
                buckets.setdefault(f"{kind}:ask", Counter())[live_intent(event, report, ask) or "(none: model)"] += 1
                if has_preamble:
                    buckets.setdefault(f"{kind}:full", Counter())[
                        live_intent(event, report, question) or "(none: model)"
                    ] += 1
            for key, counter in sorted(buckets.items()):
                total = sum(counter.values())
                rows.append(
                    {
                        "corpus": f"{base}/{split}",
                        "measure": key,
                        "n": total,
                        "llm": counter.get("(none: model)", 0),
                        "llm_share": round(counter.get("(none: model)", 0) / max(total, 1), 4),
                        "routed": dict(counter.most_common()),
                        "rows_with_telemetry_preamble": preamble.get(key.split(":")[0], 0),
                    }
                )
    return rows


def markdown_table(rows: List[Dict], columns: List[str]) -> List[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-corpus", action="store_true", help="skip the arrow corpora")
    args = ap.parse_args()

    event = load_fixture("classified")
    report = event.get("report") or {}

    per_question, rows = measure_bank(event, report)
    families = summarise_families(per_question)
    reply_shaped = measure_reply_shaped(event, report)
    corpus_rows = [] if args.no_corpus else measure_corpus(event, report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -- CSV: every routed question -------------------------------------
    csv_path = out / "followup_routing.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)

    # -- markdown --------------------------------------------------------
    routed_columns = [intent.value for intent in Intent if intent.value not in
                      ("hypothesis_comparison", "clarification_reply", "out_of_scope_refusal")]
    lines = [
        "# Follow-up routing table (live dispatcher vs template)",
        "",
        "Phase 2C inspection (a). Event fixture: `tests/fixtures/response/classified.json` "
        f"({event.get('event_id')}, leading cause `{report.get('root_cause')}`).",
        "Routing does not depend on the event; only the answer text and the "
        "clarification triggers do.",
        "",
        "`(none: model)` is the live path's LLM branch: the dispatcher claimed no "
        "intent, so `RCAInference.answer_followup` generates the answer.",
        "",
        "## Per intended intent (n = 50 each)",
        "",
    ]
    header = ["intended", "family", "template", "(none: model)", "clarify", "guardrail-eligible", "routes"]
    table_rows = []
    for row in rows:
        routes = ", ".join(f"{name} {count}" for name, count in row["routed"].most_common())
        table_rows.append(
            {
                "intended": f"`{row['label']}`",
                "family": row["family"],
                "template": row["template"],
                "(none: model)": row["llm"],
                "clarify": row["clarify"],
                "guardrail-eligible": row["guardrail_eligible"],
                "routes": routes,
            }
        )
    lines += markdown_table(table_rows, header)
    lines += ["", "## By family", ""]
    lines += markdown_table(
        [
            {
                k: v
                for k, v in row.items()
            }
            for row in families
        ],
        ["family", "labels", "n", "template", "model", "model_share"],
    )
    lines += [
        "",
        "`live_template` is the 11 intents the dispatcher claims; `live_free_form` is the "
        "three case types the dispatcher has no rule for, so the live path always sends "
        "them to the model; `benchmark_only` is the three proposed intents, off by default.",
        "",
    ]
    lines += ["", "## Benchmark configuration (proposed intents on)", ""]
    bench_rows = [
        {"intended": f"`{row['label']}`", "benchmark primary": ", ".join(
            f"{name} {count}" for name, count in row["bench"].most_common())}
        for row in rows
    ]
    lines += markdown_table(bench_rows, ["intended", "benchmark primary"])
    lines += [
        "",
        "## Reply-shaped statements (no clarification keyword)",
        "",
        f"n = {reply_shaped['n']}; clarification triggers fired on "
        f"{reply_shaped['routed'].get('__clarify__', 0)}. Routing: "
        + ", ".join(f"{k} {v}" for k, v in reply_shaped["routed"].items() if k != "__clarify__"),
        "",
        "These are answers to a pending clarifying question, so the keyword router "
        "cannot see them; the dataset resolves them from dialogue state instead "
        "(see `llm/followup/cases.py`).",
        "",
    ]
    if corpus_rows:
        lines += ["## Existing corpora, measured both ways", ""]
        corpus_table = [
            {
                "corpus": row["corpus"],
                "measured": row["measure"],
                "n": row["n"],
                "model": row["llm"],
                "model share": row["llm_share"],
                "routes": ", ".join(f"{k} {v}" for k, v in row["routed"].items()),
            }
            for row in corpus_rows
        ]
        lines += markdown_table(corpus_table, ["corpus", "measured", "n", "model", "model share", "routes"])
        lines += [
            "",
            "`...:full` is the stored `question` field routed verbatim (telemetry "
            "preamble included); `...:ask` is the text after `Operator question:`, "
            "which is what the live call site passes.",
            "",
        ]
    (out / "followup_routing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "followup_routing.json").write_text(
        json.dumps(
            {"per_label": rows, "families": families, "reply_shaped": reply_shaped,
             "corpus": corpus_rows},
            indent=2,
        ),
        encoding="utf-8",
    )

    for row in rows:
        print(f"{row['label']:<24} template {row['template']:>2}/50  model {row['llm']:>2}/50  clarify {row['clarify']:>2}")
    for row in families:
        print(f"{row['family']:<16} n={row['n']:<4} template={row['template']:<4} model={row['model']:<4} "
              f"model_share={row['model_share']}")
    print(f"wrote {out/'followup_routing.md'}, {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
