"""Forced-label generation: the diagnosis is given, the adapter writes the prose.

The hybrid gives the adapter a completed prefix — ``{"fault_id": "FAULT_10", "verdict":
"classified", `` — and asks it to continue with the three fields that are genuinely prose.
The point is that the diagnosis can no longer be wrong *because of the adapter*: whatever it
writes afterwards, the answer's fault id was decided by the classifier and the menu rule.

That makes the interesting question faithfulness rather than accuracy: does the reasoning it
writes actually fit the diagnosis it was handed, or does it argue for something else? Three
contradiction signals are checked here, and they are checked on the *text*, because a
contradiction is a property of what was written:

``ranked_top_mismatch``   the answer's own ranking does not put the diagnosis first
``names_other_fault_as_cause``  a sentence with a causal cue names a different fault
``hedges_diagnosis``      a sentence says the evidence does not support / contradicts the
                          diagnosis

Nothing here rewrites the model's prose. A contradiction is recorded, and the answer is
otherwise kept: a run that quietly repaired contradictions would report a faith-fulness
number about the repair, not the model.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

UNCLASSIFIED = "UNCLASSIFIED_OPEN_SET"
NORMAL = "NORMAL_TRANSIENT"
SLIM_KEYS = ("fault_id", "verdict", "ranked_hypotheses", "physical_reasoning", "narrative")
VERDICT_BY_DIAGNOSIS = {UNCLASSIFIED: "unclassified_open_set", NORMAL: "no_fault"}

FAULT_REF = re.compile(r"\bFAULT[_ ]?(\d{1,2})\b", re.IGNORECASE)
CAUSAL_CUE = re.compile(
    r"\b(cause|causes|caused|explains?|explaining|consistent with|indicates?|suggests?|"
    r"points? to|driven by|due to|supports?|responsible|signature of|characteristic of)\b",
    re.IGNORECASE)
HEDGE = re.compile(
    r"\b(does not|doesn't|do not|cannot|can't|inconsistent with|contradicts?|"
    r"not consistent with|rules out|is not supported by)\b",
    re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def forced_prefix(diagnosis: str, verdict: Optional[str] = None) -> str:
    """The exact text the adapter must continue.

    ``verdict`` is derived from the diagnosis unless the caller overrides it, so the two
    fields can never disagree by construction.
    """
    verdict = verdict or VERDICT_BY_DIAGNOSIS.get(diagnosis, "classified")
    return json.dumps({"fault_id": diagnosis, "verdict": verdict}, ensure_ascii=False)[:-1] + ", "


def continuation_prompt(prompt: str, diagnosis: str, verdict: Optional[str] = None) -> str:
    """The model prompt, with the answer already started.

    ``prompt`` is the asked-to-answer chat prompt (it ends after the assistant header), so the
    prefix lands exactly where the answer would begin.
    """
    return prompt + forced_prefix(diagnosis, verdict)


def parse_slim(text: str, prefix: str = "") -> Optional[Dict]:
    """The answer dict from ``prefix + generated text``, or ``None`` if it is not JSON.

    The model continues a JSON object, so the object is ``prefix`` followed by its output.
    Trailing junk after the closing brace is tolerated; a missing brace is not.
    """
    body = (prefix + text) if prefix else text
    start = body.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(body[start:index + 1])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def validate_slim(answer: Optional[Dict], allowed_labels: Sequence[str],
                  diagnosis: Optional[str] = None) -> List[str]:
    """Schema and enum errors, as strings. Empty means valid."""
    if answer is None:
        return ["not_json"]
    problems = [f"missing_key:{key}" for key in SLIM_KEYS if key not in answer]
    if problems:
        return problems
    diagnosis = diagnosis or str(answer.get("fault_id"))
    allowed = set(allowed_labels)
    if str(answer.get("fault_id")) not in allowed:
        problems.append(f"enum_fault_id:{answer.get('fault_id')}")
    if str(answer.get("fault_id")) != diagnosis:
        problems.append(f"fault_id_not_forced:{answer.get('fault_id')}!={diagnosis}")
    if str(answer.get("verdict")) not in ("classified", "unclassified_open_set", "no_fault"):
        problems.append(f"enum_verdict:{answer.get('verdict')}")
    ranked = answer.get("ranked_hypotheses")
    if not isinstance(ranked, list) or not ranked:
        problems.append("ranked_hypotheses_not_list")
    else:
        unknown = [label for label in ranked if str(label) not in allowed]
        if unknown:
            problems.append(f"enum_ranked:{unknown[:3]}")
    for field in ("physical_reasoning", "narrative"):
        if not isinstance(answer.get(field), str) or not str(answer.get(field)).strip():
            problems.append(f"empty:{field}")
    return problems


def _sentences(text: str) -> List[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(text or "") if part.strip()]


def _spellings(diagnosis: str) -> List[str]:
    """How a diagnosis may be written in prose.

    The KB and the teacher both write ``FAULT_13``, but prose also says "FAULT 13" and
    "fault-13". The single-digit case is not a special case: ``FAULT_05`` and ``FAULT_13``
    differ only in the number, and an earlier version of this function dropped the zero and
    so watched for "FAULT 5" while the diagnosis was "FAULT 13" — which meant the hedge
    check never fired for faults 10-15.
    """
    match = re.search(r"(\d{1,2})\s*$", str(diagnosis))
    forms = [str(diagnosis)]
    if match:
        number = int(match.group(1))
        forms += [f"FAULT {number}", f"FAULT_{number}", f"FAULT-{number}"]
    return forms


def contradictions(answer: Optional[Dict], diagnosis: str) -> List[Dict]:
    """Every signal that the prose does not fit the diagnosis it was handed."""
    found: List[Dict] = []
    if not answer:
        return [{"code": "no_answer"}]
    ranked = answer.get("ranked_hypotheses")
    if isinstance(ranked, list) and ranked and str(ranked[0]) != str(diagnosis):
        found.append({"code": "ranked_top_mismatch", "detail": str(ranked[0])})
    prose = " ".join(str(answer.get(field) or "") for field in
                     ("physical_reasoning", "narrative"))
    for sentence in _sentences(prose):
        for match in FAULT_REF.finditer(sentence):
            label = f"FAULT_{int(match.group(1)):02d}"
            if label == str(diagnosis):
                continue
            if CAUSAL_CUE.search(sentence):
                found.append({"code": "names_other_fault_as_cause", "label": label,
                              "sentence": sentence[:220]})
                break
    forms = [form.lower() for form in _spellings(diagnosis)]
    for sentence in _sentences(prose):
        if HEDGE.search(sentence) and any(form in sentence.lower() for form in forms):
            found.append({"code": "hedges_diagnosis", "sentence": sentence[:220]})
    return found


def decode_text(model, tokenizer, prompt: str, max_new: int = 512) -> List[int]:
    """Greedy decode of ``prompt``, through the language model when it is reachable.

    Every text-only call through ``InternVLChatModel.forward`` extracts a ViT feature map for
    a dummy image and then discards it, because ``image_flags`` is zero. Forcing a diagnosis
    and continuing is text-only, so the vision tower is pure overhead: the parity check in
    PROTOCOL §15.5 measured the direct path 4.4x faster with identical probabilities. The
    wrapper path is kept as the fallback so this cannot fail on a model without
    ``language_model``.
    """
    import torch

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        from llm.benchmark.decode_bench import decode_kv

        return decode_kv(model, tokenizer, prompt, max_new, 448)

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    produced: List[int] = []
    with torch.no_grad():
        embeds = language_model.get_input_embeddings()(ids)
        out = language_model(inputs_embeds=embeds, attention_mask=mask, use_cache=True,
                             return_dict=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        for _ in range(max_new):
            token = int(torch.argmax(logits, dim=-1))
            if token in stops:
                break
            produced.append(token)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)],
                             dim=1)
            step = language_model(input_ids=torch.tensor([[token]], device=model.device),
                                  attention_mask=mask, past_key_values=past, use_cache=True,
                                  return_dict=True)
            past, logits = step.past_key_values, step.logits[:, -1, :]
    return produced


def valve_flow_mislabels(text: str) -> List[str]:
    """Tags described as the kind of variable they are not.

    Thin public wrapper over the Phase 2B follow-up rule: the prompt gives every tag its kind
    (``A_Feed_Flow [XMV_44, valve position, manipulated variable]``), so calling a valve
    position a pressure reading contradicts the context rather than wording it loosely.
    """
    from llm.followup.filters import _valve_mislabel

    found = _valve_mislabel(text or "")
    return [found] if found else []


def fallback_answer(diagnosis: str, verdict: str, menu_ids: Sequence[str],
                    evidence: Optional[Dict] = None) -> Dict:
    """A deterministic answer, for when generation fails validation.

    It is built from the knowledge base and the stored evidence, so it is grounded by
    construction: the numbers are copied from the event, the mechanism and the action come
    from ``scripts.fault_knowledge``, and the ranked list comes from the KB's own differential.
    No model is involved, so this cannot contradict the diagnosis.
    """
    from llm.benchmark.menu import lookalikes
    from scripts.fault_knowledge import FAULT_KNOWLEDGE

    number = None
    match = re.match(r"FAULT_(\d{1,2})", str(diagnosis))
    if match:
        number = int(match.group(1))
    ranked: List[str] = [str(diagnosis)]
    if number is not None:
        for other in lookalikes(number, limit=2):
            label = f"FAULT_{int(other):02d}"
            if label in set(menu_ids or []) and label not in ranked:
                ranked.append(label)
    entry = FAULT_KNOWLEDGE.get(number or -1) or {}
    evidence = evidence or {}
    sensors = [sensor for sensor in (evidence.get("top_anomalous_sensors") or [])
               if isinstance(sensor, dict)][:2]
    quotes = "; ".join(
        f"{sensor.get('display_name') or sensor.get('name')} z={sensor.get('z_score')}"
        for sensor in sensors) or "no single sensor dominates the evidence"
    mechanism = str(entry.get("physical_mechanism") or "the stored evidence is consistent with "
                    "this fault family and no other listed entry explains the pattern")
    action = str(entry.get("recommended_action") or "verify the affected loop on site")
    return {
        "fault_id": str(diagnosis),
        "verdict": str(verdict),
        "ranked_hypotheses": ranked,
        "physical_reasoning": f"Detector and sensor evidence: {quotes}. {mechanism}",
        "narrative": f"Diagnosis {diagnosis} is the best fit for this event. Recommended: {action}",
    }


__all__ = ["SLIM_KEYS", "forced_prefix", "continuation_prompt", "parse_slim", "validate_slim",
           "contradictions", "fallback_answer", "VERDICT_BY_DIAGNOSIS", "decode_text",
           "valve_flow_mislabels"]
