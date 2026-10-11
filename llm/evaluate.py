"""Evaluation of the InternVL3-2B / InternVL2-2B ``tep_rca`` adapter.

Independent of the anomaly detector. Runs the adapter over held-out fault
scenarios and scores:

  * root-cause accuracy          (grounded fault name match)
  * fault classification accuracy
  * evidence consistency         (mentions only sensors present in evidence)
  * reasoning quality            (heuristic: causal hedges / structure)
  * severity accuracy            (vs. reference severity)
  * recommendation quality       (actionable, non-empty)
  * hallucination / unsupported-claim rate
  * JSON validity rate

All metrics are heuristic but reproducible and logged to JSON.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional

from llm.dataset import SYSTEM_PROMPT, build_chat_prompt, format_evidence_question
from llm.inference import RCAInference, extract_json_object

logger = logging.getLogger(__name__)

CAUSAL_CLAIM_PATTERNS = [
    re.compile(r"\b(caused|proves?|definitely|certainly|guaranteed)\b", re.I),
    re.compile(r"\b(root cause is\b.*!|must be\b)", re.I),
]

HEDGE_TERMS = ["likely", "probably", "consistent with", "suggests", "supports",
               "may", "might", "candidate", "temporal evidence", "verify"]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _reference_for_fault(fault_id: int) -> Dict:
    from scripts.fault_knowledge import FAULT_KNOWLEDGE  # type: ignore

    return FAULT_KNOWLEDGE.get(fault_id, {})


def evaluate_adapter(
    inference: RCAInference,
    samples: List[Dict],
    top_k: int = 1,
) -> Dict:
    """Evaluate the adapter on structured fault samples.

    Args:
        inference: loaded RCAInference (InternVL2-2B + tep_rca).
        samples: list of {fault_id, fault_name, evidence, question, reference}
                 (see scripts/generate_llm_dataset.py).

    Returns:
        metrics dict + per-sample detail.
    """
    if not samples:
        raise ValueError("No evaluation samples provided.")

    stats = Counter()
    detail = []
    root_cause_hits = 0
    fault_hits = 0
    severity_hits = 0
    valid_json = 0
    hallucinated = 0
    hedged = 0
    evid_ok = 0
    rec_ok = 0

    for sample in samples:
        evidence = sample["evidence"]
        question = sample.get("question") or (
            format_evidence_question(evidence)
            .split("Determine the likely root cause")[0]
            + "Determine the likely root cause and explain the evidence."
        )
        prompt = build_chat_prompt(SYSTEM_PROMPT, [("user", question)])
        raw = inference._generate(prompt, max_new_tokens=768)
        parsed = extract_json_object(raw)
        if parsed is None:
            stats["invalid_json"] += 1
            detail.append({"fault_id": sample["fault_id"], "parsed": False,
                           "raw": raw[:500]})
            continue
        valid_json += 1
        answer = parsed

        ref_name = _normalize(sample.get("fault_name", ""))
        pred_rc = _normalize(str(answer.get("root_cause", "")))
        pred_subsys = _normalize(str(answer.get("affected_subsystem", "")))

        # root-cause accuracy (top_k by token overlap)
        if ref_name and (ref_name in pred_rc or _overlap(pred_rc, ref_name, top_k)):
            root_cause_hits += 1

        # fault classification accuracy: match evidence's sensor set direction
        reference = sample.get("reference") or _reference_for_fault(sample.get("fault_id"))
        fault_hits += _fault_class_match(answer, reference, evidence)

        # severity accuracy
        if sample.get("severity") and answer.get("severity") == sample["severity"]:
            severity_hits += 1

        # evidence consistency: answer only references sensors present in evidence.
        # Only unambiguous sensor names (underscored / X{MEAS,MV,...}N tokens) count,
        # so ordinary language can never trip the check.
        sensor_re = re.compile(r"(?:^|[\s\W])([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+|X(?:MEAS|MV|MAX|MIN|PV|CV)[0-9]+)(?:$|[\s\W])")
        mentioned = set(t.lower() for t in sensor_re.findall(str(answer)))
        evidence_sensors = {
            _normalize(s.get("display_name", "")).replace(" ", "_")
            for s in evidence.get("top_anomalous_sensors", [])
        }
        evidence_sensors.add(_normalize(str(evidence.get("candidate_subsystem", ""))).replace(" ", "_"))
        json_keys = {
            "summary", "root_cause", "affected_subsystem", "evidence", "reasoning",
            "reasoning_steps", "severity", "confidence", "recommended_action", "uncertainty",
            "anomaly_score", "event_id", "temporal_sequence", "first_onset_minutes",
            "top_anomalous_sensors", "display_name", "current_value",
            "baseline_value", "deviation_percent", "trend", "contribution",
            "candidate_subsystem", "candidate_subsystem_score", "pre_anomaly_context",
            "duration_minutes", "status", "relative_time_minutes", "z_score",
            "dev_range", "delay_min", "fault_id", "fault_name", "kind", "question",
            "answer", "reference", "name", "subsystem", "action", "limit_context",
        }
        foreign = mentioned - evidence_sensors - json_keys
        if foreign:
            hallucinated += 1
        else:
            evid_ok += 1

        # reasoning quality: hedged causal language present (reduces overclaim)
        reasoning = str(answer.get("reasoning", "")) + " " + " ".join(str(e) for e in answer.get("evidence", []))
        if any(term in reasoning.lower() for term in HEDGE_TERMS):
            hedged += 1
        elif any(p.search(reasoning) for p in CAUSAL_CLAIM_PATTERNS):
            stats["overclaiming"] += 1

        # recommendation quality
        action = str(answer.get("recommended_action", ""))
        if len(action) > 15 and not action.lower().startswith("unable"):
            rec_ok += 1

        detail.append({
            "fault_id": sample["fault_id"],
            "fault_name": sample.get("fault_name"),
            "root_cause": answer.get("root_cause"),
            "affected_subsystem": answer.get("affected_subsystem"),
            "severity": answer.get("severity"),
            "confidence": answer.get("confidence"),
            "parsed": True,
        })

    n = len(samples)
    metrics = {
        "n_samples": n,
        "root_cause_accuracy": root_cause_hits / n,
        "fault_classification_accuracy": fault_hits / n,
        "severity_accuracy": severity_hits / n if n else 0.0,
        "evidence_consistency": evid_ok / n,
        "hallucination_rate": hallucinated / n,
        "hedged_reasoning_rate": hedged / n,
        "recommendation_rate": rec_ok / n,
        "json_validity_rate": valid_json / n,
        "overclaiming_rate": stats["overclaiming"] / n,
    }
    return {"metrics": metrics, "detail": detail, "counts": dict(stats)}


def evaluate_grounding(
    inference: RCAInference,
    samples: List[Dict],
    max_new_tokens: int = 512,
) -> Dict:
    """Grounding eval for `grounded_qa` samples (Track 3).

    For each sample, generates a free-text answer from the sample's own
    question (which embeds the telemetry context) and checks:
      * number recall: LSTM score, CVA score, and >=1 sensor deviation/value
        from the evidence appear verbatim in the answer;
      * no vague boilerplate ('as shown in the table below');
      * no foreign sensor names (same token rule as evaluate_adapter).
    Returns metrics + per-sample detail. Free-text (no JSON parsing).
    """
    from llm.dataset import FOLLOWUP_SYSTEM_PROMPT  # noqa: E402

    grounded = [s for s in samples if s.get("kind") == "grounded_qa"]
    if not grounded:
        raise ValueError("No grounded_qa samples provided.")
    sensor_re = re.compile(r"(?:^|[\s\W])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|X(?:MEAS|MV|MAX|MIN|PV|CV)[0-9]+)(?:$|[\s\W])")
    num_re = re.compile(r"\d+\.\d+")
    n = len(grounded)
    num_hits = 0
    vague_hits = 0
    foreign_hits = 0
    detail = []
    for sample in grounded:
        evidence = sample["evidence"]
        system = sample.get("system") or FOLLOWUP_SYSTEM_PROMPT
        prompt = build_chat_prompt(system, [("user", sample["question"])])
        raw = inference._generate(prompt, max_new_tokens=max_new_tokens)
        # Recall vs the reference answer's own numbers (each intent quotes a
        # different subset: metrics -> scores, sensor_values -> val/baseline).
        ref_numbers = num_re.findall(str(sample.get("answer", "")))
        recalled = sum(1 for num in ref_numbers if num in raw)
        frac = (recalled / len(ref_numbers)) if ref_numbers else 1.0
        num_ok = frac >= 0.6
        num_hits += int(num_ok)
        vague = "as shown in the table below" in raw.lower() or "as shown below" in raw.lower()
        vague_hits += int(vague)
        sensors = evidence.get("top_anomalous_sensors", []) or []
        mentioned = set(t.lower() for t in sensor_re.findall(raw))
        # Allow multi-underscore names (regex only captures the first two
        # segments), raw tag names (XMEAS_N / XMV_N as shown in context),
        # and the subsystem name itself.
        allowed = set()
        for s in sensors:
            if not isinstance(s, dict):
                continue
            for key in ("display_name", "name"):
                val = s.get(key, "")
                if val:
                    allowed.add(_normalize(str(val)).replace(" ", "_"))
        allowed.add(_normalize(str(evidence.get("candidate_subsystem", ""))).replace(" ", "_"))
        foreign = {m for m in mentioned
                   if not any(a.startswith(m) or m.startswith(a) for a in allowed)}
        foreign_hits += int(bool(foreign))
        table_lines = sum(1 for ln in raw.splitlines() if ln.strip().startswith("|"))
        has_table = table_lines >= 3
        detail.append({
            "fault_id": sample.get("fault_id"),
            "intent": sample.get("intent"),
            "number_recall": f"{recalled}/{len(ref_numbers)}",
            "numbers_ok": num_ok,
            "vague": vague,
            "foreign_sensors": sorted(foreign),
            "has_table": has_table,
            "raw": raw[:800],
        })
    table_intents = {"metrics", "evidence_why", "sensor_values"}
    table_detail = [d for d in detail if d.get("intent") in table_intents]
    metrics = {
        "n_samples": n,
        "number_recall_rate": num_hits / n,
        "vague_boilerplate_rate": vague_hits / n,
        "foreign_sensor_rate": foreign_hits / n,
        "format_table_rate": (
            sum(1 for d in table_detail if d["has_table"]) / len(table_detail)
            if table_detail else 1.0
        ),
        "format_table_n": len(table_detail),
    }
    return {"metrics": metrics, "detail": detail}


def _overlap(pred: str, ref: str, top_k: int) -> bool:
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())
    if not ref_tokens:
        return False
    inter = pred_tokens & ref_tokens
    return len(inter) >= max(1, min(top_k, len(ref_tokens)))


def _fault_class_match(answer: Dict, reference: Dict, evidence: Dict) -> int:
    """1 if the answer names a subsystem/root cause consistent with reference."""
    if not reference:
        return 0
    ref_subsys = _normalize(str(reference.get("subsystem", "")))
    pred_subsys = _normalize(str(answer.get("affected_subsystem", "")))
    pred_rc = _normalize(str(answer.get("root_cause", "")))
    if ref_subsys and (ref_subsys in pred_subsys or ref_subsys in pred_rc):
        return 1
    # fall back: reference root cause name appears in the prediction
    ref_name = _normalize(str(reference.get("name", "")))
    if ref_name and (ref_name in pred_rc or _overlap(pred_rc, ref_name, 2)):
        return 1
    return 0