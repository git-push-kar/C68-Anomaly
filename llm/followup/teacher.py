"""The teacher prompt, the response cache, and a fake teacher for tests.

The teacher writes **one** string: ``answer`` (1-4 sentences). Intent, table,
citations, the clarifying question, the refusal decision and the suggested
follow-ups are already decided in ``llm/followup/cases.py`` and are shown to the
model as fixed facts it may use but not change.

Three reasons the prompt is shaped this way:

* every number the answer may quote is printed in the context, so "quote what
  you were shown" is a rule the filters can enforce exactly;
* the clarifying question is given verbatim and the answer must not restate it,
  which is how "answer first, ask at most once" stays checkable;
* an open-set incident never names a family, so an answer that names one has
  invented it.

``FakeTeacher`` is a real test double, not a stub: for each profile it writes an
answer of the intended shape (grounded, refusing, clarifying, repeating, ...) so
the filters and the end-to-end build are exercised without a model. It is the
default teacher; ``--teacher ollama`` is required for a real run and is never
used by the tests.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llm.followup.cases import label_for_fault, premise_sentence

SYSTEM_PROMPT_FOLLOWUP = (
    "You are a control-room diagnostic assistant for the Tennessee Eastman Process. "
    "An incident has already been detected and reported; an operator is now asking a "
    "follow-up question about it.\n\n"
    "RULES\n"
    "1. Answer the question first, in 1 to 4 sentences, in the voice of a senior "
    "process engineer. No preamble.\n"
    "2. Use only the incident context below. Quote a number only as it is printed "
    "there, and never introduce a variable, a number or a mechanism the context "
    "does not contain.\n"
    "3. Never state a severity level, a priority, or a numeric confidence: those "
    "are computed elsewhere.\n"
    "4. Never say 'as shown in the table below' or refer the reader to a table.\n"
    "5. If a clarifying question is supplied, it is already being asked in the "
    "interface: answer what you can first, do not repeat the question, and do not "
    "ask a question of your own.\n"
    "6. If the question asserts something the context contradicts or does not "
    "hold, say plainly that you cannot confirm it, then point to an answerable "
    "alternative from the list supplied.\n"
    "7. If the incident is unclassified, do not name a fault family as the cause; "
    "describe what the readings show and what should be checked on site.\n"
    "8. Do not mention identifiers, files, splits or runs.\n\n"
    'OUTPUT: one JSON object with exactly the key {"answer"}. No markdown fences.'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class TeacherPayload:
    """What the prompt is built from. Extra keys are test-only (fake teacher)."""

    context: str
    question: str
    intent: Optional[str] = None
    table: str = "none"
    citations: List[str] = field(default_factory=list)
    clarifying_question: Optional[str] = None
    refusal: Optional[Dict] = None
    premise: Optional[Dict] = None
    previous_turns: List[Dict[str, str]] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    values: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    open_set: bool = False
    repeat: bool = False
    turn_index: int = 0
    # test-only: the fake teacher needs to know what the *prompt* hides
    removed_fault_label: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "context": self.context,
            "question": self.question,
            "intent": self.intent,
            "table": self.table,
            "citations": list(self.citations),
            "clarifying_question": self.clarifying_question,
            "refusal": self.refusal,
            "previous_turns": list(self.previous_turns),
            "repeat": self.repeat,
        }


@dataclass
class TeacherResponse:
    model: str
    text: str
    seconds: float
    prompt_chars: int
    raw: Dict = field(default_factory=dict)


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_prompt(payload: TeacherPayload) -> str:
    """Render the user prompt: context, fixed decisions, history, question."""
    lines = [
        "INCIDENT CONTEXT (stored; the only facts you may use)",
        payload.context,
        "",
        "FIXED FOR THIS TURN (decided in code, do not change)",
        f"- routed intent: {payload.intent or 'none (free-form answer)'}",
        f"- table: {payload.table}",
        f"- citations: {', '.join(payload.citations) if payload.citations else 'none'}",
    ]
    if payload.clarifying_question:
        lines.append(f"- clarifying question shown to the operator: {payload.clarifying_question}")
    if payload.refusal and payload.refusal.get("required"):
        alternatives = "; ".join(
            f"{a.get('question')} ({a.get('intent')})" for a in payload.refusal.get("alternatives") or []
        )
        lines.append(f"- this question cannot be answered as asked: {payload.refusal.get('reason')}")
        lines.append(f"- answerable alternatives you may point to: {alternatives or 'none'}")
    if payload.repeat:
        lines.append(
            "- the operator is asking the same question again: answer it differently, "
            "add detail, and do not contradict the earlier reply"
        )
    if payload.open_set:
        lines.append("- the incident is unclassified: no listed fault family is established")
    if payload.previous_turns:
        lines += ["", "CONVERSATION SO FAR"]
        for turn in payload.previous_turns[-4:]:
            lines.append(f"operator: {turn.get('user', '')}")
            lines.append(f"assistant: {turn.get('assistant', '')}")
    lines += [
        "",
        "QUESTION",
        payload.question,
        "",
        "Write the answer now. 1-4 sentences, only from the context above, no "
        "severity and no confidence, no table reference, no identifiers.",
        'Return exactly: {"answer": "..."}',
    ]
    return "\n".join(lines)


def parse_answer_json(text: str) -> tuple:
    body = (text or "").strip()
    if not body:
        return None, "empty response"
    try:
        return json.loads(body), None
    except ValueError:
        match = _JSON_OBJECT.search(body)
        if not match:
            return None, "no JSON object in response"
        try:
            return json.loads(match.group(0)), None
        except ValueError as exc:
            return None, f"invalid JSON: {exc}"


def extract_answer(text: str) -> tuple:
    parsed, error = parse_answer_json(text)
    if error:
        return None, error
    if not isinstance(parsed, dict) or "answer" not in parsed:
        return None, "missing 'answer' key"
    return str(parsed.get("answer") or ""), None


# ---------------------------------------------------------------------------
class TeacherCache:
    """Append-only JSONL cache keyed by the prompt's hash.

    Keyed on the prompt text plus model/seed/temperature, so a changed rule,
    context or model is a different key and can never be served stale text.
    """

    def __init__(self, path: Path, model: str, seed: int = 42, temperature: float = 0.2):
        self.path = Path(path)
        self.model = model
        self.seed = int(seed)
        self.temperature = float(temperature)
        self._entries: Dict[str, Dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                self._entries[row.get("key")] = row
        self.hits = 0
        self.misses = 0

    def key_for(self, prompt: str) -> str:
        return sha256_text(f"{self.model}|{self.seed}|{self.temperature}|{prompt}")

    def get(self, prompt: str) -> Optional[str]:
        row = self._entries.get(self.key_for(prompt))
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return str(row.get("text") or "")

    def put(self, prompt: str, text: str, model: str, seconds: float) -> None:
        key = self.key_for(prompt)
        if key in self._entries:
            return
        row = {"key": key, "model": model, "seconds": round(seconds, 3), "text": text}
        self._entries[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
#: Plain-language name for an answerable alternative, so a refusal can offer
#: one without quoting a question (which would end in "?" and fail the
#: answer-is-not-a-question rule).
ALTERNATIVE_NOUNS = {
    "detector_scores": "the detector scores and their limits",
    "detector_severity": "the detector scores and the computed level",
    "severity": "the computed severity and its basis",
    "sensors_inspect": "the variables that moved and their directions",
    "sensor_values": "the stored values against their baseline",
    "sensor_scores": "the per-variable scores and trends",
    "action": "the recommended action for this incident",
    "overview": "the stored diagnosis and the affected area",
    "overview_fallback": "the stored diagnosis and the affected area",
    "evidence_why": "the readings that crossed their limits",
    "system_sensors": "the monitored variables for this plant",
}


class FakeTeacher:
    """Deterministic writer used by the tests and by ``--teacher fake``."""

    temperature = 0.0

    def __init__(self, profile: str = "good", model: str = "fake-followup-v1"):
        self.profile = profile
        self.model = model
        self.seed = 0
        self.calls = 0

    # -- sentence material -------------------------------------------------
    def _context_fact(self, payload: TeacherPayload) -> str:
        values = payload.values or {}
        labels = payload.labels or {}
        first = labels.get("first_sensor") or values.get("sensor") or "the leading variable"
        variant = payload.turn_index % 3
        if values.get("observed") and values.get("baseline"):
            if variant == 1:
                return (
                    f"the stored value for {first} moved from {values['baseline']} "
                    f"to {values['observed']}"
                )
            if variant == 2 and values.get("lstm") and values.get("dynamic"):
                return (
                    f"the reconstruction error is {values['lstm']} and the dynamic reading "
                    f"is {values['dynamic']}, while {first} sits at {values['observed']}"
                )
            return f"{first} is at {values['observed']} against a stored baseline of {values['baseline']}"
        if values.get("lstm"):
            return f"the reconstruction error is {values['lstm']}, as printed in the context"
        return f"the stored readings for {first} are in the context"

    def _case_sentence(self, payload: TeacherPayload) -> str:
        intent = payload.intent
        variant = payload.turn_index % 3
        if intent == "hypothesis_comparison":
            labels = payload.labels or {}
            return (
                f"the two candidates would separate on onset order, and here "
                f"{labels.get('first_sensor', 'the leading variable')} moved before the rest"
            )
        if payload.open_set and variant == 0:
            return "nothing in the stored signature establishes one listed family over another"
        if variant == 1:
            return (
                "what separates the candidates is which variable moved first and how "
                "quickly the loop settled afterwards"
            )
        if variant == 2:
            return (
                "the loop behaviour is what decides between an instrument problem and a "
                "genuine process disturbance"
            )
        if payload.table == "detectors":
            return "both limits are printed above, and the margin is what separates a real excursion from drift"
        return "the ordering of the moving variables is the part that carries the diagnosis"

    def _valve_sentence(self, payload: TeacherPayload) -> str:
        labels = payload.labels or {}
        valve = labels.get("valve") or "the cooling water valve"
        return (
            f"{valve} is a flow-control valve, so a sticky actuator shows up as a flow "
            "deviation and a slow oscillation rather than a step change"
        )

    @staticmethod
    def _join(sentences: Sequence[str]) -> str:
        """Join sentences, capitalising each and dropping doubles."""
        out: List[str] = []
        for sentence in sentences:
            text = (sentence or "").strip().rstrip(".")
            if not text:
                continue
            out.append(text[:1].upper() + text[1:])
        return ". ".join(out) + "."

    def _good(self, payload: TeacherPayload) -> str:
        if payload.refusal and payload.refusal.get("required"):
            premise = payload.refusal.get("premise") or {}
            alternatives = payload.refusal.get("alternatives") or []
            noun = "the stored readings"
            for alternative in alternatives:
                candidate = ALTERNATIVE_NOUNS.get(str(alternative.get("intent")))
                if candidate:
                    noun = candidate
                    break
            return self._join(
                [
                    f"I cannot confirm that: {premise_sentence(premise)}",
                    f"I can show you {noun} instead, which is stored for this incident",
                ]
            )
        if payload.intent == "clarification_reply":
            option = (payload.labels or {}).get("reply_option") or "the requested"
            return self._join(
                [
                    f"Taking the {option} view, {self._case_sentence(payload)}",
                    f"{self._context_fact(payload)}, and that is the reading the record holds",
                ]
            )
        if payload.clarifying_question:
            return self._join(
                [
                    f"The stored readings answer this directly: {self._context_fact(payload)}",
                    f"I can break it down either way, and {self._case_sentence(payload)}",
                ]
            )
        if payload.repeat:
            return self._join(
                [
                    f"To add to what I said: {self._case_sentence(payload)}",
                    f"{self._context_fact(payload)}, and that reading has not changed since the alarm",
                ]
            )
        if payload.intent == "hypothesis_comparison":
            return self._join(
                [
                    f"{self._case_sentence(payload)}",
                    f"{self._context_fact(payload)}, which fits the stored diagnosis and not the lookalike",
                ]
            )
        if payload.table == "evidence" or payload.intent is None:
            if payload.turn_index % 2 == 0:
                return self._join(
                    [
                        f"{self._valve_sentence(payload)}",
                        f"{self._context_fact(payload)}, and the loop settles once the actuator is freed",
                    ]
                )
            return self._join(
                [
                    f"{self._case_sentence(payload)}",
                    f"{self._context_fact(payload)}, so the next check is on site at that loop",
                ]
            )
        cause = (
            label_for_fault(payload.intent)
            if str(payload.intent or "").startswith(("FAULT_", "SYNTHETIC_"))
            else "the stored diagnosis"
        )
        return self._join(
            [
                f"{self._context_fact(payload)}",
                f"{self._case_sentence(payload)}, and {cause} is what the record supports",
            ]
        )

    # -- bad profiles used by filter tests ---------------------------------
    def _bad(self, payload: TeacherPayload) -> str:
        base = self._good(payload)
        if self.profile == "invented_number":
            return base + " The residual peaked at 12.345 in the third window."
        if self.profile == "severity":
            return base + " This is a high severity condition."
        if self.profile == "confidence":
            return base + " I am 87% confident in that reading."
        if self.profile == "table":
            return "The values are as shown in the table below."
        if self.profile == "question":
            return "Which view would you like me to expand?"
        if self.profile == "confirm":
            return "Yes, that is exactly right."
        if self.profile == "no_alternative":
            return "I cannot answer that from the incident."
        if self.profile == "repeat":
            previous = (payload.previous_turns or [{}])[-1].get("assistant") or base
            return previous
        if self.profile == "removed_fault":
            label = payload.removed_fault_label or "the removed family"
            return base + f" The evidence still points to {label}."
        if self.profile == "mislabel":
            return base + " XMV_52 temperature is the value to watch."
        if self.profile == "five_sentences":
            return base + " One. Two. Three."
        raise ValueError(f"unknown fake teacher profile {self.profile!r}")

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT_FOLLOWUP,
                 payload: Optional[TeacherPayload] = None) -> TeacherResponse:
        if payload is None:
            raise ValueError("FakeTeacher needs the payload, not only the prompt")
        self.calls += 1
        text = self._good(payload) if self.profile == "good" else self._bad(payload)
        text = text[:1].upper() + text[1:] if text else text
        encoded = json.dumps({"answer": text}, ensure_ascii=False)
        return TeacherResponse(
            model=self.model,
            text=encoded,
            seconds=0.0,
            prompt_chars=len(prompt),
            raw={"profile": self.profile},
        )

    def answer(self, payload: TeacherPayload, prompt: Optional[str] = None) -> tuple:
        prompt = prompt if prompt is not None else build_prompt(payload)
        response = self.generate(prompt, payload=payload)
        answer, error = extract_answer(response.text)
        return answer, response, error


# ---------------------------------------------------------------------------
class OllamaFollowupTeacher:
    """Real teacher: the Phase 2B Ollama client, asking for one prose string.

    Deliberately not an import-time dependency and never constructed by the
    tests or by the default build.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 temperature: float = 0.2, seed: int = 42):
        from llm.benchmark.teacher import OllamaTeacher

        self.model = model
        self.seed = int(seed)
        self.temperature = float(temperature)
        self._client = OllamaTeacher(model=model, host=host, temperature=temperature, seed=seed)

    def answer(self, payload: TeacherPayload, prompt: Optional[str] = None) -> tuple:
        prompt = prompt if prompt is not None else build_prompt(payload)
        response = self._client.generate(prompt, system=SYSTEM_PROMPT_FOLLOWUP)
        answer, error = extract_answer(response.text)
        return answer, response, error


__all__ = [
    "FakeTeacher",
    "OllamaFollowupTeacher",
    "SYSTEM_PROMPT_FOLLOWUP",
    "TeacherCache",
    "TeacherPayload",
    "TeacherResponse",
    "build_prompt",
    "extract_answer",
    "parse_answer_json",
    "sha256_text",
]
