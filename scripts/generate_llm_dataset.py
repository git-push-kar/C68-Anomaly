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
    }
    return evidence


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
                    "fault_id": fault_id,
                    "fault_name": kb["name"],
                    "severity": kb["severity"],
                    "question": question,
                    "answer": render_answer_json(answer),
                    "evidence": evidence,
                    "reference": {
                        "name": kb["name"], "subsystem": kb["subsystem"],
                        "severity": kb["severity"],
                    },
                })
                for fq in build_followup_samples(evidence, kb):
                    records[split].append({
                        "kind": "followup",
                        "fault_id": fault_id,
                        "fault_name": kb["name"],
                        "severity": kb["severity"],
                        "question": fq["question"],
                        "answer": fq["answer"],
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
                            "fault_id": fault_id,
                            "fault_name": kb["name"],
                            "severity": kb["severity"],
                            "question": ss["question"],
                            "answer": ss["answer"],
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
                            "fault_id": fault_id,
                            "fault_name": kb["name"],
                            "severity": kb["severity"],
                            "question": ds["question"],
                            "answer": ds["answer"],
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