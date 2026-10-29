"""Render reports and follow-up answers to markdown + a UI block list.

Fidelity rule: **every data cell is a resolved citation key.** A builder asks
:class:`~llm.response.store_access.StoreAccess` for a key and prints what comes
back through a declared column format. Model text never enters a table.

Each column declares a format spec (``f4``, ``pct``, ``ratio``, ``alarm``, ...),
so a printed cell is a pure function of the stored value and the test can
recompute it. Tables expose ``raw`` (stored values) and ``keys`` alongside the
formatted ``rows``; cells with ``key=None`` are labels or row numbers, never data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm.response.schemas import FollowupAnswer, Report, TableID
from llm.response.store_access import StoreAccess, UnknownKeyError
from scripts.fault_ids import FaultID
from scripts.tags import canonical_tag, display_name

MAX_EVIDENCE_ROWS = 5
FILLER = "—"


def fmt(value: Any) -> str:
    """Generic formatter (significant digits; booleans read as detector states)."""
    if isinstance(value, bool):
        return "ALARMED" if value else "below threshold"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_cell(value: Any, spec: Optional[str], decimals: int = 2) -> str:
    """Format one cell from its declared column spec. Total function, never raises."""
    if value is None:
        return FILLER
    if spec in (None, "", "text"):
        return fmt(value)
    try:
        if spec == "sig":
            return fmt(value)
        if spec.startswith("f") and spec[1:].isdigit():
            return f"{float(value):.{int(spec[1:])}f}"
        if spec == "pct":
            return f"{float(value):.2f}%"
        if spec == "pct_signed":
            return f"{float(value):+.2f}%"
        if spec == "f2_signed":
            return f"{float(value):+.2f}"
        if spec == "ratio":
            return f"{float(value):.2f}x"
        if spec == "alarm":
            return "ALARMED" if bool(value) else "below threshold"
        if spec == "minutes":
            return f"{float(value):.1f} min"
        if spec == "int":
            return str(int(value))
    except (TypeError, ValueError):
        return FILLER
    return fmt(value)


@dataclass
class Table:
    id: TableID
    columns: List[str]
    formats: List[Optional[str]] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    keys: List[List[Optional[str]]] = field(default_factory=list)
    raw: List[List[Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.formats:
            self.formats = [None] * len(self.columns)
        assert len(self.formats) == len(self.columns), f"{self.id}: format/column count mismatch"

    def add(self, keys: Sequence[Optional[str]], raw: Sequence[Any]) -> None:
        assert len(keys) == len(self.columns) == len(raw), (
            f"{self.id}: row width {len(raw)} != {len(self.columns)} columns"
        )
        self.keys.append(list(keys))
        self.raw.append(list(raw))
        self.rows.append([format_cell(value, spec) for value, spec in zip(raw, self.formats)])

    def expected_rows(self) -> List[List[str]]:
        """Recompute every formatted cell from ``raw`` — used by the fidelity test."""
        return [[format_cell(value, spec) for value, spec in zip(row, self.formats)] for row in self.raw]

    def data_cells(self) -> List[tuple]:
        """(key, raw, cell, spec) for every cell that claims to come from the store."""
        out = []
        for index, row_keys in enumerate(self.keys):
            for col, key in enumerate(row_keys):
                if key is None:
                    continue
                out.append((key, self.raw[index][col], self.rows[index][col], self.formats[col]))
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id.value, "columns": list(self.columns), "rows": [list(r) for r in self.rows]}

    def markdown(self) -> str:
        header = "| " + " | ".join(self.columns) + " |"
        rule = "|" + "|".join("---" for _ in self.columns) + "|"
        body = [
            "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in self.rows
        ]
        return "\n".join([header, rule] + body)


@dataclass
class RenderedDoc:
    kind: str
    markdown: str
    ui: Dict[str, Any]
    tables: List[Table] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


def _cell(store: StoreAccess, key: str) -> Any:
    return store.resolve(key)


def _try(store: StoreAccess, key: str) -> Optional[Any]:
    try:
        return store.resolve(key)
    except UnknownKeyError:
        return None


# ----------------------------------------------------------------------------
# table builders
# ----------------------------------------------------------------------------
def build_detectors(store: StoreAccess) -> Table:
    table = Table(
        TableID.DETECTORS,
        ["Detector", "Score", "Threshold", "Margin", "Status"],
        [None, "f4", "f2", "ratio", "alarm"],
    )
    method = str(_try(store, "detector.dynamic.method") or "cva").upper()
    rows = [
        ("LSTM reconstruction error", "detector.lstm.score", "detector.lstm.threshold", "detector.lstm.ratio", "detector.lstm.alarmed"),
        (f"{method} Q-statistic", "detector.dynamic.score", "detector.dynamic.threshold", "detector.dynamic.ratio", "detector.dynamic.alarmed"),
    ]
    for label, score_key, thr_key, ratio_key, alarm_key in rows:
        score, thr = _try(store, score_key), _try(store, thr_key)
        if score is None or thr is None:
            continue
        table.add(
            [None, score_key, thr_key, ratio_key, alarm_key],
            [label, score, thr, _try(store, ratio_key), _try(store, alarm_key)],
        )
    overall = _try(store, "event.max_score")
    if overall is not None:
        table.add([None, "event.max_score", None, None, None], ["Overall event score", overall, FILLER, FILLER, FILLER])
    return table


def build_evidence(store: StoreAccess, tags: Optional[Sequence[str]] = None) -> Table:
    table = Table(
        TableID.EVIDENCE,
        ["Sensor", "Observed", "Baseline", "Dev %", "z", "Trend"],
        [None, "f4", "f4", "pct", "f2", None],
    )
    if tags is None:
        tags = store.known_tags()[:MAX_EVIDENCE_ROWS]
    for raw_tag in tags:
        tag = canonical_tag(raw_tag)
        if tag is None or tag not in store.known_tags():
            continue
        observed = _try(store, f"sensor.{tag}.observed")
        dev = _try(store, f"sensor.{tag}.deviation")
        if observed is None and dev is None:
            continue
        table.add(
            [
                None,
                f"sensor.{tag}.observed",
                f"sensor.{tag}.baseline",
                f"sensor.{tag}.deviation",
                f"sensor.{tag}.z",
                f"sensor.{tag}.trend",
            ],
            [
                display_name(tag) or tag,
                observed,
                _try(store, f"sensor.{tag}.baseline"),
                dev,
                _try(store, f"sensor.{tag}.z"),
                _try(store, f"sensor.{tag}.trend"),
            ],
        )
    return table


def build_timeline(store: StoreAccess, limit: int = 6) -> Table:
    table = Table(TableID.TIMELINE, ["#", "Sensor", "Event", "Minutes"], [None, None, None, "minutes"])
    for index in range(min(limit, len(store.timeline))):
        sensor = _try(store, f"timeline.{index}.sensor")
        event = _try(store, f"timeline.{index}.event")
        minutes = _try(store, f"timeline.{index}.minutes")
        # the display name is its own key when stored, so the cell stays store-derived
        display = _try(store, f"timeline.{index}.display_name")
        sensor_key = f"timeline.{index}.display_name" if display is not None else f"timeline.{index}.sensor"
        tag = canonical_tag(sensor) or sensor
        table.add(
            [None, sensor_key, f"timeline.{index}.event", f"timeline.{index}.minutes"],
            [str(index + 1), display if display is not None else (display_name(tag) or tag or sensor), event, minutes],
        )
    return table


def build_hypotheses(store: StoreAccess, hypotheses: Sequence) -> Table:
    """Labels from the ID space, names/subsystems from the KB.

    The model's confidence deliberately stays in prose: a reader must not be able
    to mistake a model score for a measured number.
    """
    table = Table(TableID.HYPOTHESES, ["Cause", "KB name", "Subsystem", "Supporting sensors"])
    for hypothesis in hypotheses:
        fault: FaultID = hypothesis.fault
        number = fault.value.split("_")[-1] if fault.value.startswith(("FAULT_", "SYNTHETIC_")) else None
        keys: List[Optional[str]] = [None]
        raw: List[Any] = [fault.value]
        if number is not None:
            keys += [f"kb.fault.{number}.name", f"kb.fault.{number}.subsystem"]
            raw += [_try(store, f"kb.fault.{number}.name") or FILLER, _try(store, f"kb.fault.{number}.subsystem") or FILLER]
        else:
            keys += [None, None]
            raw += ["unclassified open-set disturbance", _try(store, "subsystem.candidate") or FILLER]
        supported = [canonical_tag(t) for t in (hypothesis.supporting_tags or [])]
        supported = [t for t in supported if t]
        keys.append(None)
        raw.append(", ".join(display_name(t) or t for t in supported) if supported else FILLER)
        table.add(keys, raw)
    return table


def build_comparison(store: StoreAccess, hypotheses: Sequence) -> Table:
    """Two-cause comparison: this event's readings for each hypothesis's sensor."""
    table = Table(
        TableID.COMPARISON,
        ["Cause", "Initiating variable", "Observed", "Baseline", "Dev %"],
        [None, None, "f4", "f4", "pct"],
    )
    for hypothesis in list(hypotheses)[:2]:
        fault: FaultID = hypothesis.fault
        if not fault.value.startswith(("FAULT_", "SYNTHETIC_")):
            table.add([None] * 5, [fault.value, FILLER, FILLER, FILLER, FILLER])
            continue
        tag = next((canonical_tag(t) for t in (hypothesis.supporting_tags or []) if canonical_tag(t)), None)
        if tag is None:
            table.add([None] * 5, [fault.value, FILLER, FILLER, FILLER, FILLER])
            continue
        table.add(
            [None, f"sensor.{tag}.display_name", f"sensor.{tag}.observed", f"sensor.{tag}.baseline", f"sensor.{tag}.deviation"],
            [
                fault.value,
                display_name(tag) or tag,
                _try(store, f"sensor.{tag}.observed"),
                _try(store, f"sensor.{tag}.baseline"),
                _try(store, f"sensor.{tag}.deviation"),
            ],
        )
    return table


def build_actions(store: StoreAccess, fault: Optional[FaultID]) -> Table:
    """KB remediation text only; the model's own advice stays in prose."""
    table = Table(TableID.ACTIONS, ["Cause", "Subsystem", "KB recommended action"])
    if fault is None or not fault.value.startswith(("FAULT_", "SYNTHETIC_")):
        return table
    number = fault.value.split("_")[-1]
    action = _try(store, f"kb.fault.{number}.action")
    if action is None:
        return table
    table.add(
        [None, f"kb.fault.{number}.subsystem", f"kb.fault.{number}.action"],
        [fault.value, _try(store, f"kb.fault.{number}.subsystem"), action],
    )
    return table


def _table_for(table_id: TableID, store: StoreAccess, context: Dict) -> Optional[Table]:
    if table_id == TableID.NONE:
        return None
    if table_id == TableID.DETECTORS:
        return build_detectors(store)
    if table_id == TableID.EVIDENCE:
        return build_evidence(store, context.get("tags"))
    if table_id == TableID.TIMELINE:
        return build_timeline(store)
    if table_id == TableID.HYPOTHESES:
        return build_hypotheses(store, context.get("hypotheses") or [])
    if table_id == TableID.COMPARISON:
        return build_comparison(store, context.get("hypotheses") or [])
    if table_id == TableID.ACTIONS:
        return build_actions(store, context.get("fault"))
    return None


def _block_list(markdown_parts: List[str], tables: List[Table], citations: List[str], kind: str, title: str, meta: Dict) -> Dict:
    blocks: List[Dict[str, Any]] = []
    for part in markdown_parts:
        if part.strip():
            blocks.append({"kind": "markdown", "text": part})
    for table in tables:
        blocks.append({"kind": "table", **table.as_dict()})
    return {"type": kind, "title": title, "meta": meta, "blocks": blocks, "citations": citations}


# ----------------------------------------------------------------------------
# public renderers
# ----------------------------------------------------------------------------
def render_report(report: Report, store: StoreAccess, include_timeline: bool = True) -> RenderedDoc:
    severity = report.severity
    cause = report.root_cause.value
    kb_name = None
    if cause.startswith(("FAULT_", "SYNTHETIC_")):
        kb_name = _try(store, f"kb.fault.{cause.split('_')[-1]}.name")
    title = f"{report.event_id} — {kb_name or cause}"
    synthetic = cause.startswith("SYNTHETIC_")

    header = [
        f"## {title}" + (" _(synthetic)_" if synthetic else ""),
        f"**Severity:** {severity.level.value.capitalize()} (authoritative, computed in code)  ",
        f"**Subsystem:** {report.affected_subsystem}  ",
        f"**Confidence:** {report.confidence:.2f}",
        "",
        report.summary,
    ]
    body = ["### Evidence"]
    body += [f"- {item}" for item in report.evidence] or ["- (no evidence bullets supplied)"]
    body += ["", "### Reasoning", report.reasoning]

    tables: List[Table] = []
    table = _table_for(report.table, store, {"hypotheses": report.hypotheses, "fault": report.root_cause})
    if table is not None and table.rows:
        tables.append(table)
        body += ["", table.markdown()]
    if include_timeline and report.table != TableID.TIMELINE and store.timeline:
        timeline = build_timeline(store, limit=4)
        if timeline.rows:
            tables.append(timeline)
            body += ["", "### Timeline", timeline.markdown()]

    tail = ["", "### Recommended action", report.recommended_action, "", "### Uncertainty", report.uncertainty]
    if severity.rationale:
        tail += ["", f"_Severity basis:_ {severity.rationale}"]

    citations = list(dict.fromkeys(report.citations))
    md = "\n".join(header + [""] + body + tail)
    if citations:
        md += "\n\n_Citations:_ " + ", ".join(f"`{key}`" for key in citations)
    ui = _block_list(
        ["\n".join(header), "\n".join(body), "\n".join(tail)],
        tables,
        citations,
        "report",
        title,
        {
            "event_id": report.event_id,
            "severity": severity.level.value,
            "severity_score": severity.score,
            "synthetic": synthetic,
        },
    )
    return RenderedDoc("report", md, ui, tables, citations)


def render_followup(
    answer: FollowupAnswer,
    store: StoreAccess,
    context: Optional[Dict] = None,
    detail: Optional[str] = None,
) -> RenderedDoc:
    """Render a follow-up. ``detail`` carries the reused dispatcher template text."""
    context = context or {}
    title = f"Follow-up — {answer.intent.value}"

    head = [f"### {answer.intent.value.replace('_', ' ').capitalize()}"]
    head.append(answer.answer)

    tables: List[Table] = []
    table = _table_for(answer.table, store, context)
    if table is not None and table.rows:
        tables.append(table)
        head += ["", table.markdown()]

    if detail:
        head += ["", "---", "", detail.strip()]

    if answer.clarifying_question:
        head += ["", f"**Clarification needed:** {answer.clarifying_question}"]
    if answer.suggestions:
        head += ["", "**You might also ask:**"] + [f"- {s}" for s in answer.suggestions]

    citations = list(dict.fromkeys(answer.citations))
    md = "\n".join(head)
    if citations:
        md += "\n\n_Citations:_ " + ", ".join(f"`{key}`" for key in citations)
    ui = _block_list(
        ["\n".join(head)],
        tables,
        citations,
        "followup",
        title,
        {"intent": answer.intent.value, "clarifying": bool(answer.clarifying_question)},
    )
    return RenderedDoc("followup", md, ui, tables, citations)


__all__ = [
    "Table",
    "RenderedDoc",
    "render_report",
    "render_followup",
    "build_detectors",
    "build_evidence",
    "build_timeline",
    "build_hypotheses",
    "build_comparison",
    "build_actions",
    "fmt",
    "format_cell",
]
