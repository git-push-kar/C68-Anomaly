"""Generate the supervised instruction dataset for InternVL2-2B.

Training target: sensor evidence -> likely root cause + reasoning (NOT a bare
fault_id -> fault_name mapping). Each sample contains the structured evidence,
a root-cause question, and the reference JSON answer. Follow-up-style samples
are generated as well so the adapter learns conversational continuation.

Splits are assigned by whole fault scenario (no leakage: a fault never appears
in both train and validation).

Usage:
    python scripts/generate_llm_dataset.py --samples-per-fault 8
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.dataset import format_evidence_question, render_answer_json  # noqa: E402
from scripts.fault_knowledge import FAULT_KNOWLEDGE  # noqa: E402
from utils import ensure_dir, load_config, save_json, set_seed  # noqa: E402

logger = logging.getLogger(__name__)


def _signed(direction: str, magnitude: float) -> float:
    return magnitude if direction == "increase" else -magnitude


def _trend(direction: str) -> str:
    return "increasing" if direction == "increase" else "decreasing"


def build_evidence(fault_id: int, rng: np.random.RandomState, counter: int) -> Dict:
    """Build a randomized structured event from the fault knowledge base."""
    kb = FAULT_KNOWLEDGE[fault_id]
    initiating = kb["initiating"][0]
    cascading = kb["cascading"]

    sensors = []
    temporal = []
    all_sensors = [initiating] + cascading
    if rng.rand() < 0.85:
        rng.shuffle(cascading)
        all_sensors = [initiating] + cascading

    contributions = rng.dirichlet(np.ones(len(all_sensors))).tolist()
    for i, spec in enumerate(all_sensors):
        dev = rng.uniform(*spec["dev_range"])
        sensors.append({
            "index": i,
            "name": spec["sensor"],
            "display_name": spec["sensor"],
            "error": float(rng.uniform(0.2, 1.0)),
            "current_value": round(float(rng.uniform(50, 120)), 2),
            "baseline_value": round(float(rng.uniform(50, 120)), 2),
            "deviation_percent": round(_signed(spec["direction"], dev), 1),
            "z_score": round(float(rng.uniform(2.0, 6.0)), 2),
            "trend": _trend(spec["direction"]),
            "contribution": round(float(contributions[i]), 3),
        })
        delay = 0.0 if i == 0 else float(rng.uniform(*spec["delay_min"]))
        temporal.append({
            "sensor": spec["sensor"],
            "display_name": spec["sensor"],
            "event": "decreased" if spec["direction"] == "decrease" else "increased",
            "relative_time_minutes": round(delay, 2),
        })

    # Synthetic but self-consistent detector evidence so
    # format_evidence_question() renders LSTM/CVA threshold lines and the
    # grounded-QA answers below can quote exact scores/thresholds.
    # Thresholds mirror production (0.6874 / 239.8); at least one detector
    # alarms since this is an anomaly event.
    l_th = 0.6874
    q_th = 239.8
    l_sc = round(float(rng.uniform(0.4, 2.5)), 4)
    # Critical/high faults alarm harder on CVA; mediums sometimes LSTM-only.
    if kb["severity"] in ("critical", "high"):
        q_sc = round(float(q_th * rng.uniform(1.5, 7.0)), 1)
    else:
        q_sc = round(float(q_th * rng.uniform(0.4, 3.0)), 1)
    if l_sc < l_th and q_sc < q_th:
        q_sc = round(float(q_th * rng.uniform(1.2, 3.0)), 1)
    dom_lag = int(rng.randint(0, 5))
    slow_drift = round(float(rng.uniform(0.1, 0.7)), 2)
    detector_evidence = {
        "triggered_by": ("both" if (l_sc >= l_th and q_sc >= q_th)
                         else ("lstm_ae" if l_sc >= l_th else "dynamic")),
        "change_type": ("mixed" if (l_sc >= l_th and q_sc >= q_th)
                        else ("level" if l_sc >= l_th else "dynamics")),
        "lstm_ae": {"score": l_sc, "threshold": l_th,
                    "alarmed": bool(l_sc >= l_th)},
        "dynamic": {"method": "cva",
                    "cva": {"score": q_sc, "threshold": q_th,
                            "is_anomalous": bool(q_sc >= q_th),
                            "alarmed": bool(q_sc >= q_th),
                            "dominant_lag": dom_lag,
                            "slow_drift_ratio": slow_drift}},
    }

    evidence = {
        "event_id": f"ANOM-T{counter:04d}",
        "anomaly_score": round(float(rng.uniform(1.2, 3.8)), 3),
        "severity": kb["severity"],
        "pre_anomaly_context": {
            "duration_minutes": round(float(rng.uniform(3, 8)), 1),
            "status": "normal",
        },
        "top_anomalous_sensors": sensors,
        "temporal_sequence": {"sequence": temporal, "first_onset_minutes": 0.0},
        "candidate_subsystem": kb["subsystem"],
        "candidate_subsystem_score": round(float(rng.uniform(0.6, 1.0)), 2),
        "reasoning_notes": [
            "Deviations measured against the normal-operation baseline.",
            "Onset ordering is temporal evidence; it does not prove causation.",
        ],
        "evidence_type": "model_derived_evidence",
        "uncertainty": "Correlation does not prove causation; verify on-site.",
        "detector_evidence": detector_evidence,
    }
    return evidence


def build_grounded_qa(evidence: Dict, kb: Dict, rng) -> List[Dict]:
    """Grounded follow-up QA: the QUESTION embeds the full telemetry context
    (unlike the legacy generic followups), and the ANSWER quotes exact
    numbers from that context so the adapter learns to copy scores/values
    instead of emitting boilerplate.

    Mirrors the 7 deterministic intents in llm/inference.py
    (_dispatch_with_intent) so the LLM path agrees with the router path.
    """
    from llm.dataset import FOLLOWUP_SYSTEM_PROMPT, format_evidence_question

    det = evidence.get("detector_evidence", {}) or {}
    lstm = det.get("lstm_ae", {}) or {}
    cva = ((det.get("dynamic", {}) or {}).get("cva", {}) or {})
    l_sc, l_th = float(lstm.get("score", 0.0)), float(lstm.get("threshold", 0.0))
    q_sc, q_th = float(cva.get("score", 0.0)), float(cva.get("threshold", 0.0))
    l_ratio = l_sc / max(l_th, 1e-9)
    q_ratio = q_sc / max(q_th, 1e-9)
    l_state = "ALARMED" if lstm.get("alarmed") else "Normal (below limit)"
    q_state = "ALARMED" if (cva.get("alarmed") or cva.get("is_anomalous")) else "Normal (below limit)"
    dom_lag = cva.get("dominant_lag", "N/A")
    slow_drift = float(cva.get("slow_drift_ratio", 0.0))
    sensors = evidence.get("top_anomalous_sensors", []) or []
    seq = (evidence.get("temporal_sequence", {}) or {}).get("sequence", []) or []
    order = ", then ".join(t.get("display_name", "") for t in seq[:3]) or "no clear ordering"
    first = (seq[0].get("display_name", "") if seq else "")
    top3 = ", ".join(s.get("display_name", "") for s in sensors[:3])
    ev_id = evidence.get("event_id", "UNKNOWN")
    sev = evidence.get("severity", kb.get("severity", "unknown"))
    sensor_lines = "; ".join(
        f"{s.get('display_name', '')} {float(s.get('deviation_percent', 0.0)):+.1f}% "
        f"(val={float(s.get('current_value', 0.0)):.3f}, normal={float(s.get('baseline_value', 0.0)):.3f}, "
        f"z={float(s.get('z_score', s.get('zScore', 0.0))):+.2f}, trend {s.get('trend', 'unknown')})"
        for s in sensors[:4]
    )

    pairs = [
        ("metrics",
         rng.choice(["What are the detector metrics and scores calling this an anomaly?",
                     "Give me the anomaly scores and thresholds."]),
         (f"Overall anomaly score {float(evidence.get('anomaly_score', 0.0)):.3f} for {ev_id}. "
          f"LSTM autoencoder reconstruction error = {l_sc:.4f} (threshold = {l_th:.4f}, "
          f"ratio = {l_ratio:.2f}x, {l_state}). "
          f"CVA dynamics Q-statistic = {q_sc:.1f} (threshold = {q_th:.1f}, "
          f"ratio = {q_ratio:.2f}x, {q_state}). "
          f"Dominant lag {dom_lag}, slow drift ratio {slow_drift:.2f}.")),
        ("evidence_why",
         rng.choice(["Provide the evidence for calling this an anomaly.",
                     "Why was this flagged as an anomaly? What is the evidence?"]),
         (f"Threshold exceedance: LSTM {l_sc:.4f} vs {l_th:.4f} ({l_state}); "
          f"CVA Q {q_sc:.1f} vs {q_th:.1f} ({q_state}). "
          f"Telemetry excursions: {sensor_lines}. "
          f"Diagnostic cause: {kb['name']} in subsystem {kb['subsystem']}.")),
        ("sensor_values",
         rng.choice(["What are the sensor values and ideal normal baselines?",
                     "Give me measured vs normal values for the deviating sensors."]),
         (f"For {ev_id}: {sensor_lines}.")),
        ("sensors_inspect",
         rng.choice(["Which sensors should I inspect?",
                     "Which sensors deviated most strongly?"]),
         (f"Inspect the sensors that deviated first and most strongly: {top3}. "
          f"Details: {sensor_lines}.")),
        ("overview_cause",
         rng.choice(["Explain the cause of this anomaly.",
                     "What is this anomaly and what is the likely root cause?"]),
         (f"{ev_id} is a {sev} anomaly in subsystem {kb['subsystem']}. "
          f"Likely root cause: {kb['name']}. {kb['reasoning']} "
          f"Temporal order: {order} (initiating: {first}). "
          f"Top sensors: {top3}. Recommended action: {kb['action']}")),
        ("severity",
         rng.choice(["How serious is this anomaly? What is the severity?",
                     "What severity was assessed and why?"]),
         (f"Severity is {sev}. This is based on the sensor deviation magnitudes "
          f"({sensor_lines}) and subsystem {kb['subsystem']}. {kb['action']}")),
        ("action",
         rng.choice(["What should I do to fix it?",
                     "What is the recommended action and mitigation?"]),
         (f"Direct action: {kb['action']} Target subsystem: {kb['subsystem']}. "
          f"Safety precaution: correlation does not prove causation; verify on site. "
          f"Severity: {sev}.")),
    ]
    context = format_evidence_question(evidence)
    out = []
    for intent, op_q, ans in pairs:
        out.append({
            "kind": "grounded_qa",
            "intent": intent,
            "question": context + "\n\nOperator question: " + op_q,
            "answer": ans,
            "system": FOLLOWUP_SYSTEM_PROMPT,
        })
    return out


def build_report_answer(evidence: Dict, kb: Dict) -> Dict:
    sensors = evidence["top_anomalous_sensors"]
    evidence_list = [
        f"{s['display_name']} changed by {s['deviation_percent']:+.1f}% "
        f"(trend {s['trend']})."
        for s in sensors[:3]
    ]
    return {
        "summary": (f"An anomaly was detected in the Tennessee Eastman Process with "
                    f"anomaly score {evidence['anomaly_score']:.2f}. The dominant deviations "
                    f"are in {', '.join(s['display_name'] for s in sensors[:3])}."),
        "root_cause": kb["name"],
        "affected_subsystem": kb["subsystem"],
        "evidence": evidence_list,
        "reasoning": kb["reasoning"],
        "severity": kb["severity"],
        "confidence": round(float(random.uniform(0.72, 0.94)), 2),
        "recommended_action": kb["action"],
        "uncertainty": ("The root cause is inferred from sensor evidence and temporal "
                        "order; on-site verification is required to confirm."),
    }


def build_followup_samples(evidence: Dict, kb: Dict) -> List[Dict]:
    """Conversational QA derived from the same evidence + knowledge."""
    subsystem = kb["subsystem"].replace("_", " ")
    q1 = f"Why do you think the {subsystem} is the most likely cause?"
    a1 = (f"{kb['reasoning']} The key point is the temporal ordering: the initiating variable "
          f"changed first and the downstream variables followed, which is consistent with a "
          f"{subsystem} abnormality rather than a downstream-only disturbance.")
    q2 = "How serious is this anomaly?"
    a2 = (f"The severity is assessed as {kb['severity']}. This is based on the magnitude of the "
          f"sensor deviations and the confidence of the evidence; monitor the affected variables "
          f"and follow the recommended action.")
    q3 = "Which sensors should I inspect?"
    a3 = (f"Inspect the sensors that deviated first and most strongly: "
          f"{', '.join(s['display_name'] for s in evidence['top_anomalous_sensors'][:3])}.")
    q4 = "Could the cause be something else?"
    a4 = ("Yes. The evidence supports the candidate cause but does not prove it. The "
          "uncertainty is that correlation and temporal order are supporting evidence, not "
          "proof. On-site inspection is needed to confirm the root cause.")
    return [
        {"kind": "followup", "question": q1, "answer": a1},
        {"kind": "followup", "question": q2, "answer": a2},
        {"kind": "followup", "question": q3, "answer": a3},
        {"kind": "followup", "question": q4, "answer": a4},
    ]


def build_severity_samples(evidence: Dict, kb: Dict) -> List[Dict]:
    """Corrective samples that force a rubric-grounded severity decision.

    The base report samples state severity as a fact; these ask the model to
    justify it from deviation magnitudes so critical/high faults are not
    downgraded to medium.
    """
    sensors = evidence["top_anomalous_sensors"]
    max_dev = max(s["deviation_percent"] for s in sensors) if sensors else 0.0
    max_dev = max(abs(v) for v in [max_dev])

    q = (
        "An anomaly was detected in the TEP. The strongest deviations are: "
        + "; ".join(
            f"{s['display_name']} {s['deviation_percent']:+.1f}% (trend {s['trend']})"
            for s in sensors[:3]
        )
        + f". The anomaly score is {evidence['anomaly_score']:.2f}. "
        "Use the severity rubric. What severity should this anomaly be assigned, "
        "and why? Respond with ONLY a JSON object using exactly these keys: "
        'severity, reasoning. severity must be one of "low", "medium", "high", "critical".'
    )
    reason = (
        f"The largest deviation magnitude is {max_dev:.0f}% and the temporal "
        f"sequence and affected subsystem ({kb['subsystem']}) indicate a "
        f"{kb['severity']} event."
    )
    return [
        {"kind": "severity", "question": q,
         "answer": json.dumps({"severity": kb["severity"], "reasoning": reason},
                              ensure_ascii=False)},
    ]


def build_discrimination_samples(
    evidence: Dict, kb: Dict, alt_kb: Dict
) -> List[Dict]:
    """Corrective samples that distinguish this fault from a same-subsystem peer.

    The base report samples name only the true cause. These present the true
    cause against a confusable peer (same subsystem) and ask which fits the
    evidence, teaching the model to rely on the initiating variable / cascade
    rather than the subsystem alone.
    """
    init = kb["initiating"][0]
    alt = alt_kb["initiating"][0] if alt_kb["initiating"] else None

    q = (
        "An anomaly was detected in the TEP. Top deviations: "
        + "; ".join(
            f"{s['display_name']} {s['deviation_percent']:+.1f}% "
            f"(trend: {s['trend']})"
            for s in evidence["top_anomalous_sensors"][:4]
        )
        + ". Temporal order: "
        + ", then ".join(
            tick["display_name"] for tick in evidence["temporal_sequence"]["sequence"][:3]
        )
        + ". Two candidate root causes are considered: "
        f"{kb['name']} vs {alt_kb['name']}. Which is more likely and why? "
        "Respond with ONLY a JSON object using exactly these keys: "
        "root_cause, reasoning. Distinguish using the initiating variable and "
        "the propagation pattern, not just the subsystem."
    )
    reason = (
        f"{kb['name']}: the initiating variable {init['sensor']} "
        f"{'decreased' if init['direction'] == 'decrease' else 'increased'} first, "
        f"consistent with {kb['reasoning']}. "
        f"Compared to {alt_kb['name']}, the evidence direction and cascade fit "
        f"{kb['name']} better"
        + (f" (the peer would deviate {alt['sensor']} instead)." if alt else ".")
    )
    return [
        {"kind": "discrimination", "question": q,
         "answer": json.dumps({"root_cause": kb["name"], "reasoning": reason},
                              ensure_ascii=False)},
    ]


def _same_subsystem_peer(fault_id: int, kb: Dict, fault_ids) -> Dict:
    """Pick a confusable peer sharing the subsystem (prefer sharing the
    initiating sensor)."""
    peers = [f for f in fault_ids if f != fault_id and FAULT_KNOWLEDGE[f]["subsystem"] == kb["subsystem"]]
    if not peers:
        peers = [f for f in fault_ids if f != fault_id]
    init_sensor = {s["sensor"] for s in kb["initiating"]}
    peers.sort(key=lambda f: len({s["sensor"] for s in FAULT_KNOWLEDGE[f]["initiating"]} & init_sensor), reverse=True)
    return FAULT_KNOWLEDGE[peers[0]]


def generate_dataset(config: Dict, samples_per_fault: int, seed: int) -> Dict:
    set_seed(seed)
    rng = np.random.RandomState(seed)
    splits_cfg = config["llm"]["training"]
    split_map = {
        "train": splits_cfg.get("train_faults", list(range(1, 23))),
        "val": splits_cfg.get("val_faults", [4, 14]),
        "test": splits_cfg.get("test_faults", [15, 21]),
    }

    records = {"train": [], "val": [], "test": []}
    counter = 0
    split_meta: Dict[str, List[int]] = {}

    for split, fault_ids in split_map.items():
        split_meta[split] = list(fault_ids)
        n = samples_per_fault if split == "train" else max(3, samples_per_fault // 2)
        for fault_id in fault_ids:
            kb = FAULT_KNOWLEDGE[fault_id]
            for _ in range(n):
                counter += 1
                evidence = build_evidence(fault_id, rng, counter)
                question = format_evidence_question(evidence)
                answer = build_report_answer(evidence, kb)
                records[split].append({
                    "kind": "report",
                    "intent": "report",
                    "fault_id": fault_id,
                    "fault_name": kb["name"],
                    "severity": kb["severity"],
                    "question": question,
                    "answer": render_answer_json(answer),
                    "system": "",
                    "evidence": evidence,
                    "reference": {
                        "name": kb["name"], "subsystem": kb["subsystem"],
                        "severity": kb["severity"],
                    },
                })
                for fq in build_followup_samples(evidence, kb):
                    records[split].append({
                        "kind": "followup",
                        "intent": "followup_generic",
                        "fault_id": fault_id,
                        "fault_name": kb["name"],
                        "severity": kb["severity"],
                        "question": fq["question"],
                        "answer": fq["answer"],
                        "system": "",
                        "evidence": evidence,
                        "reference": {
                            "name": kb["name"], "subsystem": kb["subsystem"],
                            "severity": kb["severity"],
                        },
                    })
                # Grounded QA (Track 3): question carries telemetry context,
                # answer quotes exact numbers. Added to every split so val /
                # test measure grounding, not just train.
                for gq in build_grounded_qa(evidence, kb, rng):
                    records[split].append({
                        "kind": "grounded_qa",
                        "intent": gq["intent"],
                        "fault_id": fault_id,
                        "fault_name": kb["name"],
                        "severity": kb["severity"],
                        "question": gq["question"],
                        "answer": gq["answer"],
                        "system": gq["system"],
                        "evidence": evidence,
                        "reference": {
                            "name": kb["name"], "subsystem": kb["subsystem"],
                            "severity": kb["severity"],
                        },
                    })
                if split == "train":
                    for ss in build_severity_samples(evidence, kb):
                        records[split].append({
                            "kind": "severity",
                            "intent": "severity",
                            "fault_id": fault_id,
                            "fault_name": kb["name"],
                            "severity": kb["severity"],
                            "question": ss["question"],
                            "answer": ss["answer"],
                            "system": "",
                            "evidence": evidence,
                            "reference": {
                                "name": kb["name"], "subsystem": kb["subsystem"],
                                "severity": kb["severity"],
                            },
                        })
                    for ds in build_discrimination_samples(evidence, kb,
                                                           _same_subsystem_peer(fault_id, kb, split_map["train"])):
                        records[split].append({
                            "kind": "discrimination",
                            "intent": "discrimination",
                            "fault_id": fault_id,
                            "fault_name": kb["name"],
                            "severity": kb["severity"],
                            "question": ds["question"],
                            "answer": ds["answer"],
                            "system": "",
                            "evidence": evidence,
                            "reference": {
                                "name": kb["name"], "subsystem": kb["subsystem"],
                                "severity": kb["severity"],
                            },
                        })
        logger.info("Split '%s': %d samples from faults %s",
                    split, len(records[split]), fault_ids)

    out_dir = ensure_dir(config["paths"]["llm_dataset_path"])
    try:
        from datasets import Dataset, DatasetDict

        ds = DatasetDict({k: Dataset.from_list(v) for k, v in records.items()})
        ds.save_to_disk(str(out_dir))
        mode = "hf_datasets"
    except ImportError:  # pragma: no cover - datasets optional
        for split, rows in records.items():
            with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        mode = "jsonl"
    save_json(out_dir / "split_metadata.json",
              {"splits": split_meta, "samples_per_fault": samples_per_fault,
               "seed": seed, "n_train": len(records["train"]),
               "n_val": len(records["val"]), "n_test": len(records["test"]),
               "format": mode})
    logger.info("LLM dataset saved to %s (train=%d, val=%d, test=%d)",
                out_dir, len(records["train"]), len(records["val"]), len(records["test"]))
    return {k: len(v) for k, v in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate InternVL2-2B TEP instruction dataset.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--samples-per-fault", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    config = load_config(args.config)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    generate_dataset(config, args.samples_per_fault, seed)


if __name__ == "__main__":
    main()