"""Teacher step: two prose fields from a local model, nothing structural.

The teacher (a local Ollama model) is shown the event evidence, the candidate
menu, and the *already-decided* structured answer, and is asked for exactly two
strings: ``physical_reasoning`` and ``narrative``. Everything else -- label,
ranking, mitigation, table, follow-ups, the abstention decision -- was computed
deterministically in ``llm/benchmark/targets.py`` and is not open to revision
here. A teacher that returns a different label is rejected, not corrected.

Why ask at all: the structured fields are the part that must be exact, and the
prose is the part that benefits from a language model. Splitting the two means a
hallucinated sensor in the narrative cannot also corrupt the label.

Model tags are resolved against what the server actually has. A tag that is not
installed is a hard error naming the available tags: silently falling back to a
different model would change the teacher between runs and be invisible in the
data.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from llm.benchmark.kb_causes import distinctive_signatures
from llm.benchmark.menu import UNCLASSIFIED

#: Expected teacher. Verified against `/api/tags` before any call.
DEFAULT_MODEL = "qwen2.5:32b"
DEFAULT_HOST = "http://localhost:11434"

SYSTEM_PROMPT_TEACHER = (
    "You are a senior process engineer writing the narrative part of a "
    "diagnostic report for the Tennessee Eastman Process. The diagnosis has "
    "already been made by an independent analyser; your job is to explain it as "
    "a careful engineer would.\n\n"
    "RULES\n"
    "1. The root cause is GIVEN. Do not change it, do not hedge it into a "
    "different fault, and do not add alternatives that were not given.\n"
    "2. Ground every physical claim in the evidence supplied: name sensors only "
    "as the evidence names them, quote a number only if it appears verbatim in "
    "the evidence, and do not introduce a mechanism, sensor or value that is "
    "absent from it. Naming the disturbance itself is allowed even though it is "
    "not a measured sensor: use the wording given under DECIDED DIAGNOSIS, "
    "because a cause such as a cooling-water inlet temperature is inferred from "
    "its effects and has no tag. When the decided diagnosis is "
    f"{UNCLASSIFIED}, name no fault: say the signature matches none of the listed "
    "candidates and discuss only the nearest listed ones you rule out.\n"
    "2b. Every tag is labelled: an XMV is a valve position (a manipulated "
    "variable the controller moves) and an XMEAS is a process measurement. A "
    "valve position is not a flow, even when its name says \"flow\": the valve "
    "on the A/C feed can close while the measured feed flow rises, so never "
    "infer a flow's direction from a valve's direction, and say which of the two "
    "you mean. A menu line that reports a valve position's trend reports the "
    "position, not the flow: describe the trend the evidence shows and do not "
    "assert which way the flow is going.\n"
    "3. Never claim certainty. Say what the evidence suggests and what should be "
    "verified on site.\n"
    "4. Do not state a severity level, a priority, or a numeric risk score: that "
    "is computed elsewhere from detector statistics.\n"
    "5. 'physical_reasoning' is a step-by-step argument (3-6 sentences). "
    "'narrative' is 2-4 sentences addressed to the shift operator: what is "
    "happening and what to check first.\n\n"
    "OUTPUT: one JSON object with exactly the keys "
    '{"physical_reasoning", "narrative"}. No markdown fences.'
)

ANSWER_KEYS_TEACHER = ("physical_reasoning", "narrative")

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class TeacherUnavailable(RuntimeError):
    """Ollama is not reachable, or the requested model tag is not installed."""


@dataclass
class TeacherResponse:
    model: str
    text: str
    seconds: float
    prompt_chars: int
    raw: Dict = field(default_factory=dict)


def list_models(host: str = DEFAULT_HOST, timeout: float = 10.0) -> List[str]:
    """Model tags installed on the Ollama server."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise TeacherUnavailable(f"cannot reach Ollama at {host}: {exc}") from exc
    return sorted({m.get("name") for m in payload.get("models", []) if m.get("name")})


def resolve_model(preferred: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> str:
    """Return ``preferred`` only if the server really has it, else raise.

    ``qwen2.5:32b`` and ``qwen2.5:32b-instruct-q4_K_M`` are different teachers
    with different prose; accepting a nearest match would silently change the
    data, so this compares tags exactly and reports what is installed.
    """
    available = list_models(host)
    if preferred in available:
        return preferred
    base = preferred.split(":")[0]
    same_family = [tag for tag in available if tag.split(":")[0] == base]
    raise TeacherUnavailable(
        f"model tag {preferred!r} is not installed. available: {available}. "
        f"same family: {same_family or 'none'}"
    )


def license_text(model: str, host: str = DEFAULT_HOST, timeout: float = 20.0) -> Optional[str]:
    """The model's licence text as the *local runtime* reports it.

    Recorded only as a hash. The licence of a given tag is whatever the installed
    weights ship with; stating one from memory would be a claim about a file
    nobody checked, so this reads it from ``ollama show --license`` or
    ``/api/show`` and returns ``None`` when neither answers.
    """
    try:
        completed = subprocess.run(["ollama", "show", model, "--license"],
                                   capture_output=True, text=True, timeout=timeout)
        if completed.returncode == 0 and (completed.stdout or "").strip():
            return completed.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        body = json.dumps({"name": model}).encode("utf-8")
        request = urllib.request.Request(
            f"{host.rstrip('/')}/api/show", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        value = payload.get("license")
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value)
        return str(value) if value else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def license_hash(model: str, host: str = DEFAULT_HOST) -> Optional[str]:
    """SHA-256 of the licence text, or ``None`` when the runtime cannot report it."""
    text = license_text(model, host)
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def teacher_prompt_leaks(prompt: str, target: Dict,
                         menu_ids: Sequence[str] = ()) -> List[str]:
    """The removed fault named in a prompt written for an abstention item.

    An open-set episode removes one menu entry; faults 16-20 were never on it.
    In both cases the answer is UNCLASSIFIED_OPEN_SET, so a prompt that carries
    the true fault's name or number tells the teacher what to say and the
    filter -- if it trusted the same number -- what to accept. This check runs on
    every abstention prompt, before the call.
    """
    if str(target.get("fault_id") or "") != UNCLASSIFIED:
        return []
    number = target.get("truth_fault_number")
    if number is None:
        return []
    findings: List[str] = []
    flat = " ".join(str(prompt or "").lower().split())
    others = [int(str(value).split("_")[-1]) for value in menu_ids
              if str(value).startswith(("FAULT_", "SYNTHETIC_"))
              and str(value).split("_")[-1].isdigit()]
    for phrase in distinctive_signatures(int(number), others):
        if phrase and phrase in flat:
            findings.append(f"fault_name:{int(number)}")
            break
    if re.search(rf"\bfault[_ -]?0?{int(number)}\b", str(prompt or ""), re.IGNORECASE):
        findings.append(f"fault_number:{int(number)}")
    return findings


def _kb_initiating(number: int) -> Dict:
    """The KB's first initiating variable for a fault number."""
    import ast as _ast

    from scripts.fault_knowledge import FAULT_KNOWLEDGE

    raw = (FAULT_KNOWLEDGE.get(int(number)) or {}).get("initiating")
    if isinstance(raw, list):
        return raw[0] if raw and isinstance(raw[0], dict) else {}
    try:
        parsed = _ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return {}
    return parsed[0] if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) else {}


def _stored_trend(evidence: Dict, tag: str) -> Optional[str]:
    from scripts.tags import canonical_tag as _canonical

    for sensor in (evidence.get("top_anomalous_sensors") or []):
        if not isinstance(sensor, dict):
            continue
        if (_canonical(sensor.get("name")) == tag
                or _canonical(sensor.get("display_name")) == tag):
            trend = str(sensor.get("trend") or "").strip().lower()
            return trend or None
    return None


def teacher_menu(menu: Sequence[Dict[str, str]], evidence: Dict) -> List[Dict[str, str]]:
    """The menu as the *teacher* sees it: no flow direction on a valve position.

    The KB's ``initiating.direction`` for faults 1, 4, 5, 6, 7, 11, 12, 14 and 15
    describes a flow, but the tag it names is an XMV valve position. Rendering
    "A_C_Feed_Flow decreases first" invites the teacher to assert a direction the
    evidence contradicts in 120 of 120 fault-7 events, and the filter used to
    reject the correction. For those entries the teacher is shown the position's
    *stored* trend instead, or told the tag is absent, and is asked to describe
    what the evidence shows.

    This is teacher-side only: the student prompt keeps the published menu, so a
    student never sees a hint the evaluation does not give it.
    """
    from scripts.tags import canonical_tag

    out: List[Dict[str, str]] = []
    for entry in menu:
        fault_id = str(entry.get("fault_id") or "")
        match = re.fullmatch(r"FAULT_(\d{1,2})", fault_id)
        if not match:
            out.append(dict(entry))
            continue
        initiating = _kb_initiating(int(match.group(1)))
        tag = canonical_tag(initiating.get("sensor"))
        if not tag or not str(tag).upper().startswith("XMV_"):
            out.append(dict(entry))
            continue
        name = str(entry.get("signature") or "").split(" \u2014 ")[0]
        trend = _stored_trend(evidence, tag)
        if trend:
            signature = (f"{name} \u2014 valve position {tag}, trend in this event: "
                         f"{trend}; a valve position is not a flow")
        else:
            signature = (f"{name} \u2014 valve position {tag} is not in this event's "
                         f"evidence; do not infer a flow direction")
        out.append({**entry, "signature": signature})
    return out


def build_teacher_prompt(evidence: Dict, menu: Sequence[Dict[str, str]], target: Dict) -> str:
    """Evidence + menu + the decided answer, with a request for prose only."""
    from llm.benchmark.prompt_v2 import render_prompt

    from llm.benchmark.kb_causes import disturbance_phrase

    lines = [render_prompt(evidence, teacher_menu(menu, evidence), include_schema=False), ""]
    lines.append("DECIDED DIAGNOSIS (do not change):")
    lines.append(f"- {_label(target.get('fault_id'))}")
    # The cause in the KB's own words. Without it the teacher either omits the
    # mechanism or invents a name for it, and every invented name is a rejection.
    # Never on an abstention item: there the fault's own wording is the answer,
    # and stating it would turn the open-set item into a copying exercise.
    number = target.get("truth_fault_number")
    abstain = str(target.get("fault_id") or "") == UNCLASSIFIED
    if abstain:
        candidates = [_label(r.get("fault_id")) for r in (target.get("ranked_hypotheses") or [])]
        lines.append("- no listed candidate explains this signature: say so, and discuss "
                     "only the nearest listed candidates you rule out"
                     + (f" ({'; '.join(candidates)})" if candidates else ""))
    elif number:
        phrase = disturbance_phrase(int(number))
        if phrase:
            lines.append(f"- the disturbance: {phrase}")
    if target.get("primary_subsystem"):
        lines.append(f"- affected area: {str(target['primary_subsystem']).replace('_', ' ')}")
    ranked = target.get("ranked_hypotheses") or []
    if ranked and not abstain:
        lines.append("- considered and rejected: "
                     + "; ".join(_label(r.get("fault_id")) for r in ranked[1:]))
    # The stored action is deliberately NOT shown: it is taken from the knowledge
    # base and contains plant figures (valve numbers, pressure limits) that appear
    # nowhere in the student's evidence. A teacher quoting them would produce a
    # sentence the student cannot justify, and the number filter would be right to
    # reject it -- so the input must not offer them in the first place.
    lines += [
        "",
        "Write the two prose fields that justify this diagnosis from the evidence above.",
        "Quote a number only if it appears verbatim in the evidence, and reproduce "
        "it exactly as printed (write 206.82, never 206). If you cannot quote it "
        "exactly, describe the direction and relative size in words instead. Do not "
        "name a sensor that the evidence above does not list.",
        'Return exactly: {"physical_reasoning": "...", "narrative": "..."}',
    ]
    prompt = "\n".join(lines)
    leaks = teacher_prompt_leaks(prompt, target, [entry["fault_id"] for entry in menu])
    if leaks:
        raise ValueError(f"teacher prompt leaks the removed fault: {leaks}")
    return prompt


def _label(fault_id: object) -> str:
    """Human label for a fault id: the KB name, with the id in brackets.

    The internal bookkeeping strings (``kb_truth``, ``kb_differential``) are
    deliberately not shown: a teacher that echoes them puts them in the training
    prose, and they mean nothing to a reader.
    """
    text = str(fault_id or "")
    number = None
    if text.startswith(("FAULT_", "SYNTHETIC_")):
        try:
            number = int(text.split("_")[-1])
        except ValueError:
            number = None
    if number is None:
        return text
    from scripts.fault_knowledge import FAULT_KNOWLEDGE

    name = (FAULT_KNOWLEDGE.get(number) or {}).get("name")
    return f"{text} ({name})" if name else text


class OllamaTeacher:
    """Minimal Ollama client: one JSON answer per call, no retry storm."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        temperature: float = 0.2,
        seed: int = 42,
        num_predict: int = 600,
        timeout: float = 300.0,
        think: bool = False,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.num_predict = int(num_predict)
        self.timeout = float(timeout)
        # Qwen3-class models spend the whole ``num_predict`` budget on hidden
        # reasoning and then return an empty ``response``. Disabling thinking is
        # also what makes the timing usable: the hidden tokens were ~90% of it.
        self.think = bool(think)

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT_TEACHER,
                 seed: Optional[int] = None) -> TeacherResponse:
        request_body = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": "json",
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed if seed is None else int(seed),
                "num_predict": self.num_predict,
            },
        }
        started = time.time()
        payload = None
        # An older server may reject the ``think`` field outright; retrying
        # without it keeps the client usable there instead of failing the run.
        for attempt in (0, 1):
            body = json.dumps(request_body).encode("utf-8")
            request = urllib.request.Request(
                f"{self.host}/api/generate", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if attempt == 0:
                    request_body.pop("think", None)
                    continue
                raise TeacherUnavailable(f"Ollama generate failed: {exc}") from exc
            except (urllib.error.URLError, OSError) as exc:
                raise TeacherUnavailable(f"Ollama generate failed: {exc}") from exc
        assert payload is not None
        elapsed = time.time() - started
        return TeacherResponse(
            model=str(payload.get("model") or self.model),
            text=str(payload.get("response") or ""),
            seconds=elapsed,
            prompt_chars=len(prompt),
            raw={k: payload.get(k) for k in
                 ("done", "done_reason", "eval_count", "prompt_eval_count",
                  "total_duration", "eval_duration", "thinking")},
        )

    def answer(self, evidence: Dict, menu: Sequence[Dict[str, str]],
               target: Dict, extra_instruction: Optional[str] = None,
               seed: Optional[int] = None) -> tuple:
        """Return ``(parsed_dict_or_None, TeacherResponse, parse_error)``."""
        prompt = build_teacher_prompt(evidence, menu, target)
        if extra_instruction:
            prompt = f"{prompt}\n\n{extra_instruction.strip()}"
        response = self.generate(prompt, seed=seed)
        parsed, error = parse_teacher_json(response.text)
        return parsed, response, error


def parse_teacher_json(text: str) -> tuple:
    """Best-effort JSON extraction that still fails loudly on bad output."""
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


__all__ = [
    "ANSWER_KEYS_TEACHER",
    "DEFAULT_HOST",
    "DEFAULT_MODEL",
    "OllamaTeacher",
    "SYSTEM_PROMPT_TEACHER",
    "TeacherResponse",
    "TeacherUnavailable",
    "build_teacher_prompt",
    "license_hash",
    "license_text",
    "list_models",
    "parse_teacher_json",
    "resolve_model",
]
