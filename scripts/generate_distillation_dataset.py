"""Generate a high-quality distillation dataset using a Teacher LLM on a 24GB VRAM GPU.

Supports:
1. Deep fault-specific and sensor-specific knowledge synthesis covering all 22 TEP faults.
2. Grounded physical telemetry aligned with verified 52-sensor steady-state baseline statistics.
3. Diverse operator QA intents: exact sensor values, detector evidence, physical mass/energy
   propagation, valve/actuator stiction diagnosis, differential diagnosis (distinguishing lookalikes),
   and process safety limit evaluations.
4. Support for both live Teacher LLMs (Qwen2.5-32B/14B-AWQ within 24GB VRAM) and deterministic
   high-fidelity physics generation.

Output:
    data/llm/train.jsonl
    data/llm/val.jsonl
    data/llm/test.jsonl
    data/llm/split_metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence.process_relationships import SENSOR_NAMES
from llm.dataset import (
    ANSWER_SCHEMA_HINT,
    EOS,
    FOLLOWUP_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_evidence_question,
    render_answer_json,
)
from scripts.fault_knowledge import FAULT_KNOWLEDGE
from utils import ensure_dir, load_config, load_json, save_json, set_seed

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Baseline Stats Loader for Realistic Engineering Values
# -----------------------------------------------------------------------------
def _load_baseline_lookup() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load real plant means and standard deviations for all 52 sensors."""
    stats_file = Path("outputs/preprocessing/baseline_stats.json")
    lookup_mean: Dict[str, float] = {}
    lookup_std: Dict[str, float] = {}

    if stats_file.exists():
        try:
            stats = load_json(stats_file)
            names = stats.get("feature_names", [])
            means = stats.get("mean", [])
            stds = stats.get("std", [])
            for raw_name, m, s in zip(names, means, stds):
                display = SENSOR_NAMES.get(raw_name, raw_name)
                lookup_mean[raw_name] = float(m)
                lookup_mean[display] = float(m)
                lookup_std[raw_name] = float(s)
                lookup_std[display] = float(s)
        except Exception as exc:
            logger.warning("Could not parse baseline_stats.json (%s); using defaults", exc)

    return lookup_mean, lookup_std


# -----------------------------------------------------------------------------
# Synthetic Telemetry Builder (Ground-Truth Calibrated)
# -----------------------------------------------------------------------------
def build_grounded_evidence(
    fault_id: int,
    rng: np.random.RandomState,
    counter: int,
    lookup_mean: Dict[str, float],
    lookup_std: Dict[str, float],
) -> Dict[str, Any]:
    """Build a realistic anomaly event with mathematically consistent telemetry."""
    from evidence.process_limits import get_sensor_limit_context

    kb = FAULT_KNOWLEDGE[fault_id]
    initiating = kb["initiating"][0]
    cascading = kb.get("cascading", [])

    all_specs = [initiating] + cascading
    if rng.rand() < 0.85:
        shuffled_casc = list(cascading)
        rng.shuffle(shuffled_casc)
        all_specs = [initiating] + shuffled_casc

    sensors = []
    temporal = []
    contributions = rng.dirichlet(np.ones(len(all_specs))).tolist()

    for i, spec in enumerate(all_specs):
        s_name = spec["sensor"]
        direction = spec["direction"]
        dev_mag = float(rng.uniform(*spec["dev_range"]))
        signed_dev = round(dev_mag if direction == "increase" else -dev_mag, 2)

        nominal_mean = lookup_mean.get(s_name, 50.0)
        nominal_std = lookup_std.get(s_name, max(nominal_mean * 0.05, 1.0))

        base_val = round(nominal_mean, 4)
        cur_val = round(base_val * (1.0 + signed_dev / 100.0), 4)
        z = round((cur_val - base_val) / max(nominal_std, 1e-6), 2)
        trend = "increasing" if signed_dev > 0 else "decreasing"

        lim = get_sensor_limit_context(s_name)
        tag = spec.get("tag", SENSOR_NAMES.get(s_name, s_name))

        sensors.append({
            "index": i,
            "name": s_name,
            "display_name": s_name,
            "tag": tag,
            "current_value": cur_val,
            "baseline_value": base_val,
            "deviation_percent": round(signed_dev, 2),
            "z_score": z,
            "trend": trend,
            "contribution": round(float(contributions[i]), 3),
            "limit_context": lim,
        })

        delay = 0.0 if i == 0 else round(float(rng.uniform(*spec.get("delay_min", (2.0, 6.0)))), 2)
        temporal.append({
            "sensor": s_name,
            "display_name": s_name,
            "tag": tag,
            "event": "increased" if signed_dev > 0 else "decreased",
            "relative_time_minutes": delay,
        })

    # Sort temporal order
    temporal.sort(key=lambda x: x["relative_time_minutes"])

    # Detector scores
    l_th = 0.6874
    q_th = 239.8
    if kb["severity"] in ("critical", "high"):
        l_sc = round(float(l_th * rng.uniform(1.2, 4.5)), 4)
        q_sc = round(float(q_th * rng.uniform(2.0, 15.0)), 1)
    else:
        l_sc = round(float(l_th * rng.uniform(0.6, 2.2)), 4)
        q_sc = round(float(q_th * rng.uniform(0.8, 3.5)), 1)

    if l_sc < l_th and q_sc < q_th:
        q_sc = round(float(q_th * rng.uniform(1.1, 2.5)), 1)

    dom_lag = int(rng.randint(1, 5))
    slow_drift = round(float(rng.uniform(0.25, 0.75)), 2)

    det_evidence = {
        "triggered_by": "both" if (l_sc >= l_th and q_sc >= q_th) else ("lstm_ae" if l_sc >= l_th else "cva"),
        "change_type": "mixed" if (l_sc >= l_th and q_sc >= q_th) else ("level" if l_sc >= l_th else "dynamics"),
        "lstm_ae": {
            "score": l_sc,
            "threshold": l_th,
            "alarmed": bool(l_sc >= l_th),
            "is_anomalous": bool(l_sc >= l_th),
        },
        "dynamic": {
            "method": "cva",
            "cva": {
                "score": q_sc,
                "threshold": q_th,
                "alarmed": bool(q_sc >= q_th),
                "is_anomalous": bool(q_sc >= q_th),
                "dominant_lag": dom_lag,
                "slow_drift_ratio": slow_drift,
            },
        },
    }

    max_z = max(abs(s["z_score"]) for s in sensors)
    overall_score = round(max(max_z / 3.0, q_sc / q_th, l_sc / l_th), 3)

    return {
        "event_id": f"ANOM-DIST{counter:04d}",
        "anomaly_score": overall_score,
        "severity": kb["severity"],
        "pre_anomaly_context": {
            "duration_minutes": round(float(rng.uniform(4.0, 10.0)), 1),
            "status": "normal",
        },
        "top_anomalous_sensors": sensors,
        "temporal_sequence": {
            "sequence": temporal,
            "first_onset_minutes": 0.0,
        },
        "candidate_subsystem": kb["subsystem"],
        "candidate_subsystem_score": round(float(rng.uniform(0.78, 0.98)), 2),
        "detector_evidence": det_evidence,
        "reasoning_notes": [
            "Deviations measured against verified 52-sensor steady-state baseline.",
            "Temporal order-of-arrival identifies initiating disturbance.",
        ],
        "evidence_type": "distilled_calibrated_evidence",
        "uncertainty": "Correlation and temporal order are supporting evidence; on-site inspection confirms.",
    }


# -----------------------------------------------------------------------------
# Deep Fault-Specific and Sensor-Specific QA and CoT Synthesis
# -----------------------------------------------------------------------------
def synthesize_reasoning_and_answers(evidence: Dict[str, Any], kb: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize complete, non-generic step-by-step physical reasoning (CoT) and structured answers."""
    sensors = evidence.get("top_anomalous_sensors", [])
    temporal = evidence.get("temporal_sequence", {}).get("sequence", [])
    det = evidence.get("detector_evidence", {})
    lstm = det.get("lstm_ae", {})
    cva = det.get("dynamic", {}).get("cva", {})

    init_sensor = sensors[0] if sensors else {}
    init_name = init_sensor.get("display_name", "Primary Sensor")
    init_tag = init_sensor.get("tag", init_name)
    init_dir = temporal[0]["event"] if temporal else "drifted"

    casc_sensors = sensors[1:4] if len(sensors) > 1 else []
    casc_names = ", ".join(s["display_name"] for s in casc_sensors) if casc_sensors else "downstream loops"
    casc_first_name = casc_sensors[0]["display_name"] if casc_sensors else "downstream process"

    equipment_list = kb.get("equipment", ["Process unit"])
    equipment_str = ", ".join(equipment_list)
    valves_list = kb.get("valves", ["Associated control valves"])
    valves_str = ", ".join(valves_list)

    top3_sensors = ", ".join(s["display_name"] for s in sensors[:3])
    sensor_detail_lines = "; ".join(
        f"{s['display_name']} ({s.get('tag', s['name'])}) [observed: {s['current_value']}, normal: {s['baseline_value']}, "
        f"dev: {s['deviation_percent']:+.1f}%, z: {s['z_score']:+.2f}, trend: {s['trend']}]"
        for s in sensors[:4]
    )

    l_sc, l_th = lstm.get("score", 0.0), lstm.get("threshold", 0.6874)
    q_sc, q_th = cva.get("score", 0.0), cva.get("threshold", 239.8)
    l_ratio = l_sc / max(l_th, 1e-9)
    q_ratio = q_sc / max(q_th, 1e-9)
    l_alarm = "ALARMED" if lstm.get("alarmed") else "Normal (below limit)"
    q_alarm = "ALARMED" if cva.get("alarmed") else "Normal (below limit)"

    # Safety limits context
    sev_upper = kb["severity"].upper()
    if kb["severity"] == "critical":
        severity_rationale = (
            f"Assessed as CRITICAL because unmitigated progression threatens emergency reactor trip or vessel overpressure. "
            f"Immediate operator intervention is required on {valves_str}."
        )
    elif kb["severity"] == "high":
        severity_rationale = (
            f"Assessed as HIGH due to major process disturbances, significant production rate impact, or large deviations "
            f"in critical cooling/feed loops ({init_name})."
        )
    elif kb["severity"] == "medium":
        severity_rationale = (
            f"Assessed as MEDIUM. Excursions in {init_name} perturb steady-state composition/thermal balances but remain "
            f"within regulatory control bounds without immediate trip hazard."
        )
    else:
        severity_rationale = "Assessed as LOW. Slow long-term drift without acute stability threat."

    # Comprehensive Chain-of-Thought
    reasoning_steps = (
        f"1. Temporal Sequence & Telemetry Onset: Telemetry for event {evidence['event_id']} demonstrates that {init_name} "
        f"({init_tag}) was the first variable to become anomalous (t=0.0 min, observed {init_sensor.get('current_value')} vs "
        f"nominal {init_sensor.get('baseline_value')}, dev {init_sensor.get('deviation_percent', 0.0):+.1f}%, "
        f"z={init_sensor.get('z_score', 0.0):+.2f}). Subsequent cascading occurred in {casc_names}, confirming {init_name} "
        f"as the root initiating perturbation rather than a downstream symptom.\n"
        f"2. Physical Propagation Mechanism: In the {kb['subsystem'].replace('_', ' ')} ({equipment_str}), {kb.get('physical_mechanism', kb['reasoning'])}\n"
        f"3. Actuator & Valve Analysis: Manipulated variables and actuators directly coupled to this loop include {valves_str}. "
        f"The telemetry signature is physically consistent with {kb['name'].lower()}.\n"
        f"4. Differential Diagnosis & Alternative Exclusion: {kb.get('differential_diagnosis', 'Independent loops remain normal.')}\n"
        f"5. Severity Assessment: {severity_rationale}"
    )

    # Structured Primary RCA Report
    report_dict = {
        "reasoning_steps": reasoning_steps,
        "summary": (
            f"Anomaly {evidence['event_id']} in {kb['subsystem'].replace('_', ' ')}: Initiating excursion at "
            f"{init_name} ({init_tag}, dev {init_sensor.get('deviation_percent', 0.0):+.1f}%, z={init_sensor.get('z_score', 0.0):+.2f}) "
            f"propagated to {top3_sensors}. Root cause identified as {kb['name']} ({equipment_str})."
        ),
        "root_cause": kb["name"],
        "affected_subsystem": kb["subsystem"],
        "equipment": equipment_list,
        "valves": valves_list,
        "evidence": [
            f"{s['display_name']} ({s.get('tag', s['name'])}): observed={s['current_value']} vs normal={s['baseline_value']} "
            f"({s['deviation_percent']:+.1f}%, z={s['z_score']:+.2f}, trend={s['trend']})"
            for s in sensors[:4]
        ],
        "reasoning": kb.get("physical_mechanism", kb["reasoning"]),
        "differential_diagnosis": kb.get("differential_diagnosis", ""),
        "severity": kb["severity"],
        "confidence": round(float(random.uniform(0.85, 0.97)), 2),
        "recommended_action": kb.get("recommended_action", kb["action"]),
        "uncertainty": "Order-of-arrival and dynamic detector signatures provide strong causal evidence; verify physical valve/transmitter on-site.",
    }

    # Sensor Table Markdown Generator
    def _render_sensor_table() -> str:
        lines = [
            f"| Sensor | Tag | Observed | Baseline | Dev% | z-score | Trend | Limit Context |",
            f"|---|---|---|---|---|---|---|---|",
        ]
        for s in sensors[:5]:
            lim_ctx = s.get("limit_context") or "Nominal operating envelope"
            lines.append(
                f"| {s['display_name']} | {s.get('tag', s['name'])} | {s['current_value']} | {s['baseline_value']} | "
                f"{s['deviation_percent']:+.2f}% | {s['z_score']:+.2f} | {s['trend']} | {lim_ctx} |"
            )
        return "\n".join(lines)

    # Detector Table Markdown Generator
    def _render_detector_table() -> str:
        return (
            f"| Detector | Metric / Score | Threshold | Margin | Status |\n"
            f"|---|---|---|---|---|\n"
            f"| Overall Anomaly Score | {evidence['anomaly_score']:.3f} | 1.000 | {evidence['anomaly_score']:.2f}x | **{'ALARMED' if evidence['anomaly_score'] >= 1.0 else 'ELEVATED'}** |\n"
            f"| LSTM Autoencoder | {l_sc:.4f} (reconstruction error) | {l_th:.4f} | {l_ratio:.2f}x | {l_alarm} |\n"
            f"| Dynamic CVA | {q_sc:.1f} (Q-statistic) | {q_th:.1f} | {q_ratio:.2f}x | {q_alarm} |\n"
            f"| Dynamics Lag Analysis | Dominant lag: {cva.get('dominant_lag')}, drift ratio: {float(cva.get('slow_drift_ratio', 0.0)):.2f} | n/a | n/a | Pattern: {det.get('change_type', 'dynamic')} |"
        )

    # Rich Grounded QA Pairs Covering Diverse Operator Questions
    grounded_qa = [
        # 1. Exact Sensor Values & Baseline (operator prompt: "what are the sensor values")
        {
            "intent": "sensor_values",
            "question": "What are the sensor values and ideal normal baselines?",
            "answer": (
                f"Observed telemetry and baseline comparisons for incident {evidence['event_id']}:\n\n"
                f"{_render_sensor_table()}\n\n"
                f"**Telemetry Summary:** Excursions initiated at {init_name} with a {init_sensor.get('deviation_percent', 0.0):+.2f}% deviation "
                f"(z = {init_sensor.get('z_score', 0.0):+.2f}), followed by downstream changes in {casc_names}."
            ),
        },
        # 2. Rephrased Sensor Values Question (matching user's literal prompt: "mention those sensors with their scores and the ideal values it should have to be normal")
        {
            "intent": "sensor_values",
            "question": "Mention those sensors with their scores and the ideal values it should have to be normal.",
            "answer": (
                f"Here are the affected sensors for event {evidence['event_id']}, their observed telemetry, and verified normal baselines:\n\n"
                f"{_render_sensor_table()}\n\n"
                f"**Key Findings:**\n"
                f"- Primary deviating tag: `{init_name}` ({init_tag}) observed at `{init_sensor.get('current_value')}` vs normal `{init_sensor.get('baseline_value')}`.\n"
                f"- Affected equipment: {equipment_str}.\n"
                f"- Associated control valves: {valves_str}."
            ),
        },
        # 3. Detector Evidence & Anomaly Scores (operator prompt: "what is the evidence which forces this to be an anomaly")
        {
            "intent": "detector_evidence",
            "question": "What is the evidence and detector scores that force this to be an anomaly?",
            "answer": (
                f"The statistical and unsupervised detector evidence establishing incident {evidence['event_id']} as an anomaly:\n\n"
                f"{_render_detector_table()}\n\n"
                f"**Diagnostic Justification:**\n"
                f"- Both LSTM spatial autoencoder ({l_ratio:.2f}x threshold) and CVA dynamic tracking ({q_ratio:.2f}x threshold) alarmed simultaneously.\n"
                f"- CVA dynamic tracking detected a {det.get('change_type', 'dynamic')} signature with dominant lag {cva.get('dominant_lag')}, confirming "
                f"that process correlation dynamics broke away from the steady-state 52-sensor plant baseline."
            ),
        },
        # 4. Sensor-Specific & Physical Relationship Question
        {
            "intent": "sensor_specific_why",
            "question": f"Why did {init_name} {init_dir} and how does that affect {casc_first_name}?",
            "answer": (
                f"**Direct Causal Coupling:**\n"
                f"Telemetry shows {init_name} ({init_tag}) was the first variable to deviate (at t=0.0 min, {init_dir} to "
                f"{init_sensor.get('current_value')} vs nominal {init_sensor.get('baseline_value')}, dev {init_sensor.get('deviation_percent', 0.0):+.1f}%).\n\n"
                f"**Physical Mechanism in {equipment_str}:**\n"
                f"{kb.get('physical_mechanism', kb['reasoning'])}\n\n"
                f"This thermodynamic coupling directly explains why {casc_first_name} responded shortly after {init_name}."
            ),
        },
        # 5. Valve & Actuator Stiction / Failure Analysis
        {
            "intent": "valve_actuator_status",
            "question": f"Which specific valves or actuators ({valves_str}) are involved and what is their failure mode?",
            "answer": (
                f"**Actuator and Valve Diagnostic Report:**\n"
                f"- **Implicated Valves:** {valves_str}\n"
                f"- **Equipment Unit:** {equipment_str}\n"
                f"- **Failure Mode:** {kb['name']}\n\n"
                f"**Engineering Analysis:**\n"
                f"{kb.get('physical_mechanism', kb['reasoning'])}\n"
                f"**Field Action:** {kb.get('recommended_action', kb['action'])}"
            ),
        },
        # 6. Differential Diagnosis / Lookalike Fault Discrimination
        {
            "intent": "differential_diagnosis",
            "question": f"How do we distinguish this event ({kb['name']}) from alternative causes or lookalike disturbances?",
            "answer": (
                f"**Differential Diagnosis for Event {evidence['event_id']}:**\n"
                f"{kb.get('differential_diagnosis', 'Independent subsystems show nominal behavior (< 2 sigma).')}\n\n"
                f"**Supporting Statistical Proof:**\n"
                f"- Initiating arrival order: {init_name} leads by several minutes over downstream variables ({casc_names}).\n"
                f"- Unsupervised detectors confirm multi-sigma deviation on {init_name} while unrelated plant sections remain normal."
            ),
        },
        # 7. Field Inspection & Maintenance Checklist
        {
            "intent": "sensors_inspect",
            "question": "Which sensors, valves, and equipment should I inspect first in the field?",
            "answer": (
                f"**Prioritized Inspection Checklist for {kb['name']}:**\n\n"
                f"1. **Primary Transmitter:** Inspect {init_name} ({init_tag}) — current reading is {init_sensor.get('current_value')} "
                f"(baseline {init_sensor.get('baseline_value')}, deviation {init_sensor.get('deviation_percent', 0.0):+.1f}%).\n"
                f"2. **Actuator & Control Valves:** Inspect {valves_str}. Check valve stem position, pneumatic air supply pressure, "
                f"and smart positioner I/P error codes.\n"
                f"3. **Process Equipment:** Inspect {equipment_str}. Check local pressure gauges and thermal indicators.\n"
                f"4. **Cascading Transmitters:** Verify secondary instrumentation on {top3_sensors}."
            ),
        },
        # 8. Severity, Safety Envelope & Hazard Risk
        {
            "intent": "severity_risk",
            "question": "What is the severity of this anomaly and what is the risk to plant safety?",
            "answer": (
                f"**Severity Level:** **{sev_upper}**\n\n"
                f"**Safety Context & Envelope Analysis:**\n"
                f"{severity_rationale}\n\n"
                f"**Observed Key Excursions:**\n"
                f"{sensor_detail_lines}\n\n"
                f"**Required Mitigation:** {kb.get('recommended_action', kb['action'])}"
            ),
        },
    ]

    return {
        "report": report_dict,
        "grounded_qa": grounded_qa,
    }


# -----------------------------------------------------------------------------
# Live Teacher LLM Interface (Qwen-70B/72B / 32B / 14B / API / Mock)
# -----------------------------------------------------------------------------
class TeacherLLM:
    """Wrapper to load and run Teacher models (Qwen-72B/70B, 32B, or API endpoints)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-72B-Instruct",
        load_in_4bit: bool = True,
        device: str = "cuda",
        max_gpu_memory: str = "22GiB",
        max_cpu_memory: str = "64GiB",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 600,
        no_fallback: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.no_fallback = no_fallback
        self.is_mock = False
        self.is_api = bool(api_base)

        if model_id == "mock":
            self.is_mock = True
            logger.info("Initialized High-Fidelity Deterministic Physics Teacher Engine (Zero VRAM).")
            return

        if self.is_api:
            logger.info("Configured API-based Teacher LLM (%s) at endpoint %s (timeout=%ds)",
                        model_id, api_base, self.timeout)
            self._preflight_check()
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Teacher LLM %s (4-bit=%s, max_gpu=%s, max_cpu=%s)...",
                    model_id, load_in_4bit, max_gpu_memory, max_cpu_memory)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "max_memory": {0: max_gpu_memory, "cpu": max_cpu_memory},
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                logger.warning("BitsAndBytesConfig failed (%s); trying standard weights", e)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        logger.info("Teacher LLM %s loaded successfully.", model_id)

    def _preflight_check(self) -> None:
        """Verify endpoint connectivity and perform an upfront warm-up ping to load model into RAM/VRAM."""
        import urllib.request
        import urllib.parse
        import urllib.error

        parsed = urllib.parse.urlparse(self.api_base)
        base_origin = f"{parsed.scheme}://{parsed.netloc}"
        is_ollama = "11434" in parsed.netloc or "ollama" in self.api_base.lower()

        if is_ollama:
            try:
                tags_url = f"{base_origin}/api/tags"
                req = urllib.request.Request(tags_url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    tags_data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in tags_data.get("models", [])]
                    logger.info("Ollama server online at %s. Detected installed models: %s", base_origin, models)
                    matched = any(self.model_id in m or m in self.model_id for m in models)
                    if not matched:
                        logger.warning(
                            "Requested model '%s' not explicitly found in Ollama models list %s. "
                            "If inference fails, verify your tag with 'ollama list' or run 'ollama pull %s'.",
                            self.model_id, models, self.model_id
                        )
            except Exception as e:
                logger.warning("Could not query Ollama tags at %s (%s). Proceeding to test chat endpoint...", base_origin, e)

        # Warm-up ping to trigger model loading into memory
        logger.info("[Pre-flight] Warming up teacher model '%s' at %s...", self.model_id, self.api_base)
        logger.info("[Pre-flight] Cold-loading a 70B/72B model typically takes 1-3 minutes; waiting up to %ds...", self.timeout)
        t0 = time.time()
        try:
            _ = self._call_api("Say 'Ready' in one word.", max_new_tokens=5, retries=1)
            elapsed = time.time() - t0
            logger.info("[Pre-flight] Teacher model '%s' successfully loaded and warmed up in %.1fs!", self.model_id, elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            logger.warning("[Pre-flight] Warm-up ping failed after %.1fs: %s", elapsed, e)
            if self.no_fallback:
                raise RuntimeError(
                    f"Teacher LLM warm-up failed for model '{self.model_id}' at {self.api_base}: {e}\n"
                    f"Make sure Ollama is running and responsive ('ollama run {self.model_id}')."
                ) from e
            logger.warning("Proceeding with generation (automatic retries will be attempted per event).")

    def _call_api(self, prompt: str, max_new_tokens: int = 512, retries: int = 2) -> str:
        """Call OpenAI-compatible chat completion endpoint with timeout, keep-alive, and retry logic."""
        import urllib.request
        import urllib.error

        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "keep_alive": "60m",  # Ollama extension: keep weights in memory for 1 hour
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_err: Optional[Exception] = None
        for attempt in range(1 + retries):
            try:
                req = urllib.request.Request(url, data=body_bytes, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"].strip()
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                if attempt < retries:
                    wait_sec = 5 * (attempt + 1)
                    logger.warning(
                        "Teacher API call failed on attempt %d/%d (%s). Retrying in %ds...",
                        attempt + 1, 1 + retries, e, wait_sec
                    )
                    time.sleep(wait_sec)
                else:
                    raise last_err
            except Exception as e:
                raise e
        if last_err:
            raise last_err
        raise RuntimeError("Teacher API call failed with unknown error.")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if self.is_mock:
            raise RuntimeError("Mock teacher does not run generate(); uses deterministic physics generator.")

        if self.is_api:
            return self._call_api(prompt, max_new_tokens=max_new_tokens, retries=2)

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip()


# -----------------------------------------------------------------------------
# Main Dataset Generation Orchestrator (with Resume & Flush Support)
# -----------------------------------------------------------------------------
def generate_distillation_dataset(
    config: Dict[str, Any],
    samples_per_fault: int = 12,
    seed: int = 42,
    teacher_model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    all_faults_in_train: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    no_fallback: Optional[bool] = None,
) -> Dict[str, int]:
    """Generate the complete distilled training, validation, and test datasets.
    
    Supports:
    - Qwen 70B/72B Teacher models (Ollama/vLLM API, local 4-bit + CPU offload, or deterministic physics).
    - Upfront pre-flight model check and warm-up ping to prevent cold-load timeouts.
    - Configurable inference timeouts (default 600s) and automatic retries.
    - Incremental resume: checks existing files and continues from the exact deficit.
    - Overwrite mode: cleanly wipes previous runs/fallbacks and regenerates from scratch.
    - Progressive disk flush: records are written and flushed immediately upon generation.
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)

    distill_cfg = config.get("distillation", {})
    t_model = teacher_model_name or distill_cfg.get("teacher_model", "Qwen/Qwen2.5-72B-Instruct")
    n_per_fault = samples_per_fault or distill_cfg.get("samples_per_fault", 12)
    max_gpu_mem = distill_cfg.get("max_gpu_memory", "22GiB")
    max_cpu_mem = distill_cfg.get("max_cpu_memory", "64GiB")
    load_4bit = bool(distill_cfg.get("load_in_4bit", True))
    endpoint = api_base or distill_cfg.get("api_base")
    key = api_key or distill_cfg.get("api_key")
    timeout_sec = timeout or distill_cfg.get("timeout", 600)
    no_fb = bool(no_fallback if no_fallback is not None else distill_cfg.get("no_fallback", False))

    lookup_mean, lookup_std = _load_baseline_lookup()
    out_dir = Path(output_dir or config["paths"]["llm_dataset_path"])
    ensure_dir(out_dir)

    # Handle overwrite mode
    if overwrite:
        resume = False
        for split_name in ("train", "val", "test"):
            fpath = out_dir / f"{split_name}.jsonl"
            if fpath.exists():
                fpath.unlink()
        meta_path = out_dir / "split_metadata.json"
        if meta_path.exists():
            meta_path.unlink()
        logger.info("Overwrite mode active: cleared existing jsonl dataset files in %s", out_dir)

    all_22_faults = list(range(1, 23))
    if all_faults_in_train:
        split_map = {
            "train": all_22_faults,
            "val": all_22_faults,
            "test": all_22_faults,
        }
    else:
        splits_cfg = config.get("llm", {}).get("training", {})
        split_map = {
            "train": splits_cfg.get("train_faults", all_22_faults),
            "val": splits_cfg.get("val_faults", [4, 14]),
            "test": splits_cfg.get("test_faults", [15, 21]),
        }

    # Inspect existing records if resume is enabled
    existing_events_by_fault: Dict[str, Dict[int, set]] = {"train": {}, "val": {}, "test": {}}
    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    fallback_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split in split_map.keys():
        file_path = out_dir / f"{split}.jsonl"
        existing_events_by_fault[split] = {fid: set() for fid in all_22_faults}
        if resume and file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    counts[split] += 1
                    try:
                        row = json.loads(line)
                        fid = row.get("fault_id")
                        ev_id = row.get("evidence", {}).get("event_id") or row.get("event_id")
                        if row.get("teacher_status") == "physics_baseline":
                            fallback_counts[split] += 1
                        if fid is not None and ev_id:
                            existing_events_by_fault[split].setdefault(int(fid), set()).add(ev_id)
                    except Exception:
                        pass
            logger.info("Resume check: '%s.jsonl' has %d existing samples across %d faults.",
                        split, counts[split], len([k for k, v in existing_events_by_fault[split].items() if v]))
            if fallback_counts[split] > 0 and t_model != "mock":
                logger.warning(
                    "Notice: '%s.jsonl' contains %d fallback baseline records. "
                    "To regenerate with pure 72B teacher outputs, pass --overwrite.",
                    split, fallback_counts[split]
                )

    teacher = TeacherLLM(
        model_id=t_model,
        load_in_4bit=load_4bit,
        max_gpu_memory=max_gpu_mem,
        max_cpu_memory=max_cpu_mem,
        api_base=endpoint,
        api_key=key,
        timeout=timeout_sec,
        no_fallback=no_fb,
    )

    logger.info("Beginning dataset distillation across 22 faults (target samples_per_fault=%d)...", n_per_fault)
    logger.info("Teacher engine: %s | Resume: %s | Timeout: %ds | Strict: %s",
                t_model, resume, timeout_sec, no_fb)

    counter = sum(counts.values())

    for split, fault_ids in split_map.items():
        file_path = out_dir / f"{split}.jsonl"
        mode = "a" if resume else "w"
        target_events_per_fault = n_per_fault if split == "train" else max(2, n_per_fault // 5)

        with open(file_path, mode, encoding="utf-8") as handle:
            for fault_id in fault_ids:
                if fault_id not in FAULT_KNOWLEDGE:
                    continue
                kb = FAULT_KNOWLEDGE[fault_id]
                existing_for_fault = len(existing_events_by_fault[split].get(fault_id, set()))
                needed = target_events_per_fault - existing_for_fault

                if needed <= 0:
                    logger.info("Split %s - Fault %02d (%s): already has %d/%d events. Skipping.",
                                split, fault_id, kb["name"], existing_for_fault, target_events_per_fault)
                    continue

                logger.info("Split %s - Fault %02d (%s): generating %d new events (existing %d)...",
                            split, fault_id, kb["name"], needed, existing_for_fault)

                for event_idx in range(needed):
                    counter += 1
                    evidence = build_grounded_evidence(fault_id, rng, counter, lookup_mean, lookup_std)
                    qa_data = synthesize_reasoning_and_answers(evidence, kb)

                    # 1. Primary RCA Report Sample with Detailed CoT
                    report_question = format_evidence_question(evidence)
                    teacher_status = "physics_baseline"
                    if not teacher.is_mock:
                        try:
                            teacher_prompt = f"{report_question}\n\nProvide the complete root-cause analysis report in JSON."
                            raw_llm_out = teacher.generate(teacher_prompt, max_new_tokens=600)
                            from llm.inference import extract_json_object, sanitize_answer
                            parsed_llm = extract_json_object(raw_llm_out)
                            if parsed_llm and "root_cause" in parsed_llm:
                                qa_data["report"] = sanitize_answer(parsed_llm, evidence=evidence)
                                teacher_status = "teacher_llm"
                            else:
                                logger.warning("Teacher output for %s did not contain valid JSON with 'root_cause'.",
                                               evidence["event_id"])
                        except Exception as e:
                            if no_fb:
                                raise RuntimeError(
                                    f"Teacher generation failed for event {evidence['event_id']} "
                                    f"(strict mode --no-fallback is active): {e}"
                                ) from e
                            logger.warning("Teacher generation failed for %s (%s); using deterministic baseline",
                                           evidence["event_id"], e)

                    report_answer_json = render_answer_json(qa_data["report"])

                    report_sample = {
                        "kind": "report",
                        "intent": "rca_diagnosis",
                        "fault_id": fault_id,
                        "fault_name": kb["name"],
                        "severity": kb["severity"],
                        "equipment": kb.get("equipment", []),
                        "valves": kb.get("valves", []),
                        "question": report_question,
                        "answer": report_answer_json,
                        "system": SYSTEM_PROMPT,
                        "evidence": evidence,
                        "teacher_model": t_model if teacher_status == "teacher_llm" else "physics_baseline",
                        "teacher_status": teacher_status,
                    }
                    handle.write(json.dumps(report_sample, ensure_ascii=False) + "\n")
                    counts[split] += 1

                    # 2. Grounded Follow-up QA Samples (Diverse operator questions)
                    for gq in qa_data["grounded_qa"]:
                        qa_sample = {
                            "kind": "grounded_qa",
                            "intent": gq["intent"],
                            "fault_id": fault_id,
                            "fault_name": kb["name"],
                            "severity": kb["severity"],
                            "equipment": kb.get("equipment", []),
                            "valves": kb.get("valves", []),
                            "question": report_question + "\n\nOperator question: " + gq["question"],
                            "answer": gq["answer"],
                            "system": FOLLOWUP_SYSTEM_PROMPT,
                            "evidence": evidence,
                        }
                        handle.write(json.dumps(qa_sample, ensure_ascii=False) + "\n")
                        counts[split] += 1

                    handle.flush()
                    existing_events_by_fault[split].setdefault(fault_id, set()).add(evidence["event_id"])

    # Save metadata
    meta = {
        "splits": split_map,
        "teacher_model": t_model,
        "all_faults_in_train": all_faults_in_train,
        "samples_per_fault": n_per_fault,
        "seed": seed,
        "counts": counts,
        "total_samples": sum(counts.values()),
        "format": "jsonl",
    }
    save_json(out_dir / "split_metadata.json", meta)
    logger.info("Distillation completed! Total samples in dataset: %d", meta["total_samples"])
    return counts


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Teacher Distillation Dataset (Qwen 70B/32B/Mock).")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--samples-per-fault", type=int, default=None, help="Events per fault (default from config: 12)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--teacher-model",
        default=None,
        help="Teacher model ('mock', or 'qwen2.5:72b', or 'Qwen/Qwen2.5-72B-Instruct')",
    )
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL (e.g. http://localhost:11434/v1)")
    parser.add_argument("--api-key", default=None, help="API key for teacher endpoint if required")
    parser.add_argument("--timeout", type=int, default=None, help="Inference timeout in seconds (default: 600)")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume from existing jsonl files on disk")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite and regenerate existing dataset files from scratch")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        default=None,
        help="Strict mode: raise error on teacher failure instead of falling back to baseline",
    )
    parser.add_argument("--output-dir", default=None, help="Custom output directory")
    parser.add_argument(
        "--all-faults-in-train",
        action="store_true",
        default=True,
        help="Include all 22 faults in training (default: True)",
    )
    parser.add_argument(
        "--holdout-faults",
        action="store_true",
        default=False,
        help="Use academic zero-shot holdout split (excludes faults 4, 14, 15, 21 from train)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    distill_cfg = config.get("distillation", {})

    samples = args.samples_per_fault if args.samples_per_fault is not None else distill_cfg.get("samples_per_fault", 12)
    seed = args.seed if args.seed is not None else distill_cfg.get("seed", 42)
    teacher = args.teacher_model or distill_cfg.get("teacher_model", "Qwen/Qwen2.5-72B-Instruct")
    timeout = args.timeout or distill_cfg.get("timeout", 600)
    no_fallback = args.no_fallback if args.no_fallback is not None else distill_cfg.get("no_fallback", False)

    if args.overwrite:
        resume_flag = False
    else:
        resume_flag = args.resume if args.resume is not None else distill_cfg.get("resume", False)

    use_all_faults = not args.holdout_faults

    generate_distillation_dataset(
        config=config,
        samples_per_fault=samples,
        seed=seed,
        teacher_model_name=teacher,
        output_dir=args.output_dir,
        all_faults_in_train=use_all_faults,
        resume=resume_flag,
        overwrite=args.overwrite,
        api_base=args.api_base,
        api_key=args.api_key,
        timeout=timeout,
        no_fallback=no_fallback,
    )


if __name__ == "__main__":
    main()

