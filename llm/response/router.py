"""Routing wrapper around the existing dispatcher.

Nothing here rewrites the live router: ``RCAInference._dispatch_with_intent``
still decides and still produces the answer for fact-lookup intents. This module
adds what the benchmark needs *around* that call:

* the full candidate list (the dispatcher answers with the first branch that
  matches, which hides near-misses), so ambiguity can be measured;
* an answerability verdict (deterministic / llm / clarify / refuse);
* clarifying-question triggers, each with the question it would ask;
* the three proposed intents from RESPONSE_SPEC §2.2, **behind a flag**.

To expose candidates without editing `llm/inference.py`, the live keyword gates
are mirrored here (``LIVE_GATES``) in the dispatcher's own priority order. A test
locks the mirror to the dispatcher's behaviour on live-format questions; if the
two drift apart, that test fails rather than the mirror silently winning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from llm.response.schemas import Intent

#: Intent priority exactly as in llm/inference.py:655-680.
PRIORITY: Tuple[Intent, ...] = (
    Intent.SENSOR_VALUES,
    Intent.EVIDENCE_WHY,
    Intent.SYSTEM_SENSORS,
    Intent.SENSOR_SCORES,
    Intent.DETECTOR_SEVERITY,
    Intent.DETECTOR_SCORES,
    Intent.SEVERITY,
    Intent.OVERVIEW,
    Intent.SENSORS_INSPECT,
    Intent.ACTION,
    Intent.OVERVIEW_FALLBACK,
)

#: Keyword tuples copied from llm/inference.py:655-680 (mirror, see module docstring).
LIVE_KEYWORDS: Dict[Intent, Tuple[str, ...]] = {
    Intent.SENSOR_VALUES: (
        "sensor value", "sensor values", "ideal value", "ideal values", "normal value",
        "normal values", "baseline value", "baseline values", "measured value",
        "measured values", "observed value", "observed values",
    ),
    Intent.EVIDENCE_WHY: (
        "evidence", "forces this to be an anomaly", "why is this an anomaly",
        "why was this flagged", "alarm trigger", "why this is an anomaly", "why anomaly",
    ),
    Intent.SYSTEM_SENSORS: (
        "which sensors are used", "sensors used in this system", "what sensors are in",
        "sensors in the system", "system sensors", "all sensors", "sensors used",
    ),
    Intent.SENSOR_SCORES: ("sensor score",),
    Intent.DETECTOR_SEVERITY: (),
    Intent.DETECTOR_SCORES: (
        "score", "scores", "metric", "metrics", "reconstruction error", "q-stat", "q stat",
        "q-score", "threshold",
    ),
    Intent.SEVERITY: (
        "severity", "how serious", "how dangerous", "critical", "alarm level", "risk level", "danger",
    ),
    Intent.OVERVIEW: (
        "which anomaly", "what is this anomaly", "what anomaly", "which fault", "what fault",
        "fault number", "fault type", "what happened", "what is this", "overview", "root cause",
        "cause", "explain anomaly", "explain this anomaly", "diagnose", "summary",
    ),
    Intent.SENSORS_INSPECT: ("sensor", "sensors", "inspect", "deviat", "which sensor"),
    Intent.ACTION: (
        "what should i do", "what to do", "action", "recommend", "how to fix", "provide a fix",
        "fix", "mitigat", "next step", "procedure",
    ),
    Intent.OVERVIEW_FALLBACK: ("why ", "why?", "explain", "diagnos"),
}

#: Proposed intents (RESPONSE_SPEC §2.2), only reachable with enable_proposed=True.
PROPOSED_KEYWORDS: Dict[Intent, Tuple[str, ...]] = {
    Intent.HYPOTHESIS_COMPARISON: (
        "distinguish", "differential", "lookalike", "how is this different",
        "how do we tell", "vs", "versus", "alternative cause", "alternative causes",
    ),
    Intent.CLARIFICATION_REPLY: (
        "what do you mean", "can you clarify", "clarify", "which one do you mean",
    ),
    Intent.OUT_OF_SCOPE_REFUSAL: (),
}

#: Intents whose answers are numeric facts and can be produced deterministically.
LOOKUP_INTENTS = frozenset(
    {
        Intent.SENSOR_VALUES,
        Intent.EVIDENCE_WHY,
        Intent.SYSTEM_SENSORS,
        Intent.SENSOR_SCORES,
        Intent.DETECTOR_SEVERITY,
        Intent.DETECTOR_SCORES,
        Intent.SEVERITY,
        Intent.OVERVIEW,
        Intent.SENSORS_INSPECT,
        Intent.ACTION,
        Intent.OVERVIEW_FALLBACK,
    }
)

#: Tokens that make a question in-domain even when no intent matches.
_DOMAIN_TOKENS = frozenset(
    {
        "tep", "plant", "process", "reactor", "valve", "sensor", "anomaly", "alarm", "fault",
        "incident", "event", "detector", "pressure", "temperature", "flow", "level", "feed",
        "condenser", "stripper", "separator", "compressor", "cooling", "pump", "loop",
        "diagnos", "control", "operator", "trend", "score", "threshold", "signal", "unit",
    }
)

_CONTENT_TOKEN = re.compile(r"[a-z0-9]+")


def content_tokens(question: str) -> List[str]:
    return [tok for tok in _CONTENT_TOKEN.findall((question or "").lower()) if len(tok) > 1]


def _matched(question: str, keywords: Sequence[str]) -> List[str]:
    return [kw for kw in keywords if kw in question]


@dataclass
class ClarificationRequest:
    trigger: str
    question: str
    options: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {"trigger": self.trigger, "question": self.question, "options": list(self.options)}


@dataclass
class RouteResult:
    question: str
    candidates: List[Intent]
    primary: Optional[Intent]
    answerability: str  # deterministic | llm | clarify | refuse
    deterministic_answer: Optional[str] = None
    clarification: Optional[ClarificationRequest] = None
    proposed: bool = False
    scores: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "question": self.question,
            "candidates": [c.value for c in self.candidates],
            "primary": self.primary.value if self.primary else None,
            "answerability": self.answerability,
            "clarification": self.clarification.as_dict() if self.clarification else None,
            "proposed": self.proposed,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


class ResponseRouter:
    """Live router + candidate scoring + answerability + clarification triggers."""

    def __init__(
        self,
        enable_proposed: bool = False,
        hypothesis_margin: float = 0.15,
        intent_margin: float = 0.15,
        max_candidates: int = 4,
    ) -> None:
        self.enable_proposed = bool(enable_proposed)
        self.hypothesis_margin = float(hypothesis_margin)
        self.intent_margin = float(intent_margin)
        self.max_candidates = int(max_candidates)

    # -- candidate scoring ------------------------------------------------
    def candidates(self, question: str) -> Tuple[List[Intent], Dict[str, float]]:
        q = (question or "").lower().strip()
        if not q:
            return [], {}
        has_system_sensors = any(kw in q for kw in LIVE_KEYWORDS[Intent.SYSTEM_SENSORS])
        has_sensor_values = not has_system_sensors and any(kw in q for kw in LIVE_KEYWORDS[Intent.SENSOR_VALUES])
        has_evidence_why = any(kw in q for kw in LIVE_KEYWORDS[Intent.EVIDENCE_WHY])
        has_sensor_scores = (
            not has_system_sensors
            and not has_sensor_values
            and (("sensor" in q and "score" in q) or "sensor score" in q)
        )
        has_detector_scores = (
            not has_sensor_scores
            and not has_system_sensors
            and not has_sensor_values
            and any(kw in q for kw in LIVE_KEYWORDS[Intent.DETECTOR_SCORES])
        )
        has_severity = any(kw in q for kw in LIVE_KEYWORDS[Intent.SEVERITY])
        has_sensors = (
            not has_system_sensors
            and not has_sensor_scores
            and not has_sensor_values
            and any(kw in q for kw in LIVE_KEYWORDS[Intent.SENSORS_INSPECT])
        )
        has_action = any(kw in q for kw in LIVE_KEYWORDS[Intent.ACTION])
        has_overview = any(kw in q for kw in LIVE_KEYWORDS[Intent.OVERVIEW])
        has_generic_why = any(kw in q for kw in LIVE_KEYWORDS[Intent.OVERVIEW_FALLBACK])

        gates: Dict[Intent, bool] = {
            Intent.SENSOR_VALUES: has_sensor_values,
            Intent.EVIDENCE_WHY: has_evidence_why,
            Intent.SYSTEM_SENSORS: has_system_sensors,
            Intent.SENSOR_SCORES: has_sensor_scores,
            Intent.DETECTOR_SEVERITY: has_detector_scores and has_severity,
            Intent.DETECTOR_SCORES: has_detector_scores,
            Intent.SEVERITY: has_severity,
            Intent.OVERVIEW: has_overview,
            Intent.SENSORS_INSPECT: has_sensors,
            Intent.ACTION: has_action,
            Intent.OVERVIEW_FALLBACK: has_generic_why,
        }

        matched: List[Intent] = []
        scores: Dict[str, float] = {}
        for intent in PRIORITY:
            hits = _matched(q, LIVE_KEYWORDS[intent])
            if intent is Intent.SENSOR_SCORES:
                hits = ["sensor+score"] if gates[intent] else []
            if intent is Intent.DETECTOR_SEVERITY:
                hits = _matched(q, LIVE_KEYWORDS[Intent.DETECTOR_SCORES]) + _matched(q, LIVE_KEYWORDS[Intent.SEVERITY]) if gates[intent] else []
            if not gates[intent]:
                scores[intent.value] = 0.0
                continue
            # intent score: matched keyword mass, normalised by the longest keyword
            weight = sum(len(h.split()) + len(h) / 20.0 for h in hits) or 1.0
            scores[intent.value] = float(weight)
            matched.append(intent)

        if self.enable_proposed:
            q_tokens = content_tokens(q)
            compare_hits = _matched(q, PROPOSED_KEYWORDS[Intent.HYPOTHESIS_COMPARISON])
            if compare_hits and (has_overview or len(q_tokens) >= 3):
                # inserted before `overview`, which is what makes it reachable
                weight = sum(len(h.split()) + len(h) / 20.0 for h in compare_hits)
                scores[Intent.HYPOTHESIS_COMPARISON.value] = float(weight)
                matched = [Intent.HYPOTHESIS_COMPARISON] + [m for m in matched if m is not Intent.HYPOTHESIS_COMPARISON]
            clarify_hits = _matched(q, PROPOSED_KEYWORDS[Intent.CLARIFICATION_REPLY])
            # Only an explicit clarification ask, or a stub too short to route at
            # all: a short *question* like "What happened?" already has an intent
            # and must not compete with a clarification reply.
            if clarify_hits or (not matched and len(q_tokens) < 3):
                scores[Intent.CLARIFICATION_REPLY.value] = 1.0 + float(len(clarify_hits))
                matched.append(Intent.CLARIFICATION_REPLY)
            if not matched:
                in_domain = any(tok in _DOMAIN_TOKENS for tok in q_tokens)
                if not in_domain:
                    scores[Intent.OUT_OF_SCOPE_REFUSAL.value] = 1.0
                    matched.append(Intent.OUT_OF_SCOPE_REFUSAL)

        return matched[: self.max_candidates], scores

    # -- clarification triggers -------------------------------------------
    def _hypotheses(self, report: Dict) -> List[Tuple[str, float]]:
        """Ranked hypotheses that carry a *measured* confidence.

        A missing confidence is not a tie. Phase 2B's targets deliberately leave
        ``confidence`` unset (it has to be calibrated from the adapter's own token
        probabilities), and defaulting it to 0.0 made every two-hypothesis report
        look like a 0.0-vs-0.0 dead heat -- which fired ``ambiguous_hypotheses`` on
        every question. Only a confidence that was actually recorded can produce
        a margin, so only those entries take part.
        """
        raw = report.get("hypotheses") if isinstance(report, dict) else None
        out: List[Tuple[str, float]] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if item.get("confidence") is None:
                    continue
                fault = item.get("fault") or item.get("root_cause") or ""
                fault = getattr(fault, "value", fault)
                try:
                    confidence = float(item.get("confidence"))
                except (TypeError, ValueError):
                    continue
                out.append((str(fault), confidence))
        return sorted(out, key=lambda pair: pair[1], reverse=True)

    def _trigger(
        self,
        candidates: Sequence[Intent],
        report: Dict,
        event: Dict,
        asked: Set[str],
    ) -> Optional[ClarificationRequest]:
        hypotheses = self._hypotheses(report)
        if len(hypotheses) >= 2:
            margin = hypotheses[0][1] - hypotheses[1][1]
            if margin < self.hypothesis_margin:
                name_a, name_b = hypotheses[0][0], hypotheses[1][0]
                request = ClarificationRequest(
                    trigger="ambiguous_hypotheses",
                    question=f"The top two causes are close ({name_a} vs {name_b}); which should I prioritise?",
                    options=[name_a, name_b],
                )
                if request.question not in asked:
                    return request

        root = str((report or {}).get("root_cause", "")).upper()
        if "UNCLASSIFIED" in root or root.startswith("UNCLASSIFIED_OPEN_SET"):
            request = ClarificationRequest(
                trigger="unclassified",
                question="The signature does not match a known fault; do you want the closest match or the raw evidence?",
                options=["closest match", "raw evidence"],
            )
            if request.question not in asked:
                return request

        if len(candidates) >= 2:
            scores = [self._score(c) for c in candidates[:2]]
            top, second = scores[0], scores[1]
            margin = (top - second) / max(top, 1e-9)
            if margin < self.intent_margin:
                request = ClarificationRequest(
                    trigger="ambiguous_question",
                    question=(
                        f"Did you mean the {candidates[0].value.replace('_', ' ')} view or the "
                        f"{candidates[1].value.replace('_', ' ')} view?"
                    ),
                    options=[candidates[0].value, candidates[1].value],
                )
                if request.question not in asked:
                    return request

        evidence = (event or {}).get("evidence") if isinstance((event or {}).get("evidence"), dict) else {}
        needs_context = candidates and candidates[0] in {
            Intent.SENSOR_VALUES,
            Intent.SENSOR_SCORES,
            Intent.SENSORS_INSPECT,
            Intent.ACTION,
        }
        if needs_context and not (evidence.get("top_anomalous_sensors") or evidence.get("detector_evidence")):
            request = ClarificationRequest(
                trigger="context_not_in_store",
                question="This event has no stored sensor or detector detail; should I answer from the report only?",
                options=["report only", "raw event"],
            )
            if request.question not in asked:
                return request
        return None

    def _score(self, intent: Intent) -> float:
        return self._last_scores.get(intent.value, 0.0)

    # -- main entry --------------------------------------------------------
    def route(
        self,
        event: Dict,
        report: Dict,
        question: str,
        asked: Optional[Sequence[str]] = None,
        with_answer: bool = True,
    ) -> RouteResult:
        candidates, scores = self.candidates(question)
        self._last_scores = scores
        primary = candidates[0] if candidates else None
        asked_set = set(asked or ())

        clarification = self._trigger(candidates, report, event, asked_set)

        deterministic: Optional[str] = None
        if with_answer and primary in LOOKUP_INTENTS:
            deterministic = self._live_answer(event, report, question)

        if clarification is not None:
            answerability = "clarify"
        elif primary is Intent.OUT_OF_SCOPE_REFUSAL:
            answerability = "refuse"
        elif deterministic is not None:
            answerability = "deterministic"
        else:
            answerability = "llm"

        return RouteResult(
            question=question,
            candidates=candidates,
            primary=primary,
            answerability=answerability,
            deterministic_answer=deterministic,
            clarification=clarification,
            proposed=bool(primary is not None and primary not in LOOKUP_INTENTS),
            scores=scores,
        )

    # -- live dispatcher (lazy import: torch/transformers are heavy) --------
    @staticmethod
    def _live_answer(event: Dict, report: Dict, question: str) -> Optional[str]:
        try:
            from llm.inference import RCAInference
        except Exception:  # pragma: no cover - environment without the model stack
            return None
        hit = RCAInference._dispatch_with_intent(event, report, question)
        return hit[1] if hit is not None else None

    @staticmethod
    def live_intent(event: Dict, report: Dict, question: str) -> Optional[str]:
        """The intent the live dispatcher itself reports (for equivalence tests)."""
        try:
            from llm.inference import RCAInference
        except Exception:  # pragma: no cover
            return None
        hit = RCAInference._dispatch_with_intent(event, report, question)
        return hit[0] if hit is not None else None


__all__ = [
    "ResponseRouter",
    "RouteResult",
    "ClarificationRequest",
    "PRIORITY",
    "LIVE_KEYWORDS",
    "PROPOSED_KEYWORDS",
    "LOOKUP_INTENTS",
]
