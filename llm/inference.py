"""InternVL3-2B / InternVL2-2B + tep_rca inference: automatic reports and follow-up answers.

This module is the LLM-facing half of the pipeline. It receives structured
anomaly events (never raw sensor series) and returns structured reports or
conversational answers grounded in the event evidence and prior context.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from llm.adapter_loader import load_tep_adapter
from llm.dataset import (
    ANSWER_KEYS,
    EOS,
    SYSTEM_PROMPT,
    build_chat_prompt,
    format_evidence_question,
)
from utils import load_config

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = (
    "You are an industrial process anomaly diagnosis assistant. A new anomaly "
    "has automatically been detected. Analyze the provided evidence and "
    "generate a complete initial report WITHOUT waiting for the user to ask a "
    "question."
)


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering on a batch x vocab probability tensor."""
    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
    cumsum = sorted_probs.cumsum(dim=-1)
    keep = cumsum - sorted_probs <= top_p
    sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_json_object(text: str) -> Optional[Dict]:
    """Extract the first balanced JSON object from a model response."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    logger.warning("JSON decode failed: %s", exc)
                    return None
    return None


def sanitize_answer(answer: Dict, evidence: Optional[Dict] = None) -> Dict:
    """Fill in missing answer keys so reports always match the schema."""
    for key in ANSWER_KEYS:
        answer.setdefault(key, "" if key in ("summary", "root_cause",
                                             "affected_subsystem", "reasoning",
                                             "reasoning_steps",
                                             "recommended_action", "uncertainty",
                                             "evidence") else ("unknown" if key == "severity" else 0.0))
    if not isinstance(answer["evidence"], list):
        answer["evidence"] = [str(answer["evidence"])]
    if answer.get("confidence") is None or not (0.0 <= float(answer["confidence"]) <= 1.0):
        answer["confidence"] = 0.0
    if answer["severity"] not in ("low", "medium", "high", "critical"):
        answer["severity"] = "unknown"

    # Deterministic severity guardrail: If the model failed to output valid severity or if evidence provides deterministic rating
    if evidence and answer["severity"] in ("unknown", ""):
        from evidence.process_limits import compute_deterministic_severity
        answer["severity"] = evidence.get("severity") or compute_deterministic_severity(evidence)
    return answer


class RCAInference:
    """High-level LLM wrapper around (InternVL2-2B + tep_rca)."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.config = config or load_config()
        self.model = model
        self.tokenizer = tokenizer
        self._gen_config = self.config["llm"]["generation"]

    # ------------------------------------------------------------------
    @classmethod
    def from_adapter(
        cls,
        base_model: Optional[str] = None,
        adapter_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> "RCAInference":
        from transformers import AutoTokenizer

        config = config or load_config()
        llm = config["llm"]
        base = base_model or llm["base_model"]
        adapter = adapter_path or llm["adapter_dir"]
        model = load_tep_adapter(
            base,
            adapter,
            trust_remote_code=bool(llm.get("trust_remote_code", True)),
            use_flash_attn=bool(llm.get("use_flash_attn", False)),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base, trust_remote_code=True, use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model=model, tokenizer=tokenizer, config=config)

    # ------------------------------------------------------------------
    def _generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Greedy decode via repeated forward passes.

        The base InternVL2 remote code predates the KV-cache APIs used by the
        installed transformers, so ``model.generate`` is unreliable here. We
        instead run the plain forward pass (which the LoRA training path uses
        successfully) and append argmax tokens until EOS or the token budget.
        """
        eos_token_id = self.tokenizer.convert_tokens_to_ids(EOS)
        max_new = max_new_tokens or int(self._gen_config.get("max_new_tokens", 384))
        if do_sample is None:
            do_sample = bool(self._gen_config.get("do_sample", False))
        temp = float(self._gen_config.get("temperature", 0.7) if temperature is None else temperature)
        top_p = float(self._gen_config.get("top_p", 0.9) if top_p is None else top_p)
        top_k = int(self._gen_config.get("top_k", 50) if top_k is None else top_k)
        rep_penalty = float(self._gen_config.get("repetition_penalty", 1.0) if repetition_penalty is None else repetition_penalty)
        pad = self.tokenizer.pad_token_id or eos_token_id

        # Collect all valid stop token IDs (EOS, <|im_end|>, tokenizer eos, etc.)
        stop_token_ids = {eos_token_id, self.tokenizer.eos_token_id}
        for token_str in (EOS, "<|im_end|>", "</s>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and isinstance(tid, int) and tid > 0:
                stop_token_ids.add(tid)
        stop_token_ids.discard(None)

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        # Text-only InternVL2 forward needs the dummy image placeholders for prefill.
        dummy = torch.zeros(
            1, 3, int(self.config["llm"].get("dummy_image_size", 448)),
            int(self.config["llm"].get("dummy_image_size", 448)),
            dtype=torch.bfloat16, device=self.model.device,
        )
        image_flags = torch.zeros((input_ids.shape[0],), dtype=torch.long, device=self.model.device)

        generated: List[int] = []
        past_key_values = None
        use_cache = True
        next_id = None

        with torch.no_grad():
            for step in range(max_new):
                if step == 0:
                    # Step 0 (Prefill): evaluate prompt and initialize KV cache
                    try:
                        outputs = self.model(
                            pixel_values=dummy,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            use_cache=True,
                        )
                        past_key_values = getattr(outputs, "past_key_values", None)
                        logits = outputs.logits[:, -1, :]
                        if past_key_values is None:
                            use_cache = False
                    except Exception as exc:
                        logger.warning("KV caching prefill not supported (%s); using direct decode.", exc)
                        use_cache = False
                        logits = self.model(
                            pixel_values=dummy,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            use_cache=False,
                        ).logits[:, -1, :]
                else:
                    # Autoregressive generation steps
                    if use_cache and past_key_values is not None:
                        try:
                            # Fast path: pass only the 1 newly generated token with KV cache (O(1))
                            outputs = self.model(
                                pixel_values=None,
                                input_ids=next_id,
                                attention_mask=attention_mask,
                                image_flags=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            past_key_values = outputs.past_key_values
                            logits = outputs.logits[:, -1, :]
                        except Exception:
                            # Fallback if pixel_values=None is rejected by custom model architecture
                            try:
                                outputs = self.model(
                                    pixel_values=dummy,
                                    input_ids=next_id,
                                    attention_mask=attention_mask,
                                    image_flags=image_flags,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                )
                                past_key_values = outputs.past_key_values
                                logits = outputs.logits[:, -1, :]
                            except Exception as exc:
                                logger.warning("KV cache step failed (%s); reverting to full sequence.", exc)
                                use_cache = False
                                logits = self.model(
                                    pixel_values=dummy,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    image_flags=image_flags,
                                    use_cache=False,
                                ).logits[:, -1, :]
                    else:
                        logits = self.model(
                            pixel_values=dummy,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            use_cache=False,
                        ).logits[:, -1, :]

                if rep_penalty != 1.0 and generated:
                    for g in set(generated):
                        logits[:, g] /= rep_penalty

                if do_sample:
                    logits = logits / temp
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    if top_p < 1.0:
                        probs = _apply_top_p(probs, top_p)
                    if top_k > 0:
                        top_probs, _ = torch.topk(probs, min(top_k, probs.shape[-1]))
                        probs[probs < top_probs[:, -1:]] = 0.0
                        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    if top_k > 0:
                        top_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                        logits[logits < top_vals[:, -1:]] = -float("inf")
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                next_id = next_id.to(self.model.device)
                curr_token = int(next_id.item())
                if curr_token in stop_token_ids:
                    break

                generated.append(curr_token)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_id)], dim=-1
                )

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    # ------------------------------------------------------------------
    def generate_report(self, evidence: Dict) -> Tuple[Dict, str]:
        """Automatic initial report for an anomaly event.

        Returns:
            (report_dict, raw_text).
        """
        question = format_evidence_question(evidence)
        prompt = build_chat_prompt(REPORT_SYSTEM_PROMPT + " " + SYSTEM_PROMPT, [("user", question)])
        raw = self._generate(prompt, max_new_tokens=int(self._gen_config.get("max_new_tokens", 768)))
        parsed = extract_json_object(raw)
        report = sanitize_answer(parsed, evidence=evidence) if parsed else {
            "reasoning_steps": "Fallback deterministic diagnosis based on sensor limits.",
            "summary": raw,
            "root_cause": "unable_to_parse",
            "affected_subsystem": evidence.get("candidate_subsystem", "unknown"),
            "evidence": [],
            "reasoning": raw,
            "severity": evidence.get("severity", "unknown"),
            "confidence": 0.0,
            "recommended_action": "Review the raw evidence and re-run analysis.",
            "uncertainty": "The model response could not be parsed as JSON.",
        }
        return report, raw

    def answer_followup(
        self,
        event: Dict,
        previous_report: Dict,
        history: List[Dict],
        question: str,
    ) -> str:
        """Answer a user question about a specific detected anomaly.

        Provides immediate, 100% grounded answers for operational queries
        (scores, severity, sensors, root cause, actions) and uses the
        underlying base chat LLM (with LoRA disabled) for novel free-form queries.
        """
        # 1. Fast, exact operational intent dispatch
        intent_answer = self._dispatch_operator_intent(event, previous_report, question)
        if intent_answer is not None:
            return intent_answer

        # 2. Conversational LLM generation for free-form questions
        chat_system_prompt = (
            "You are an industrial process assistant for the Tennessee Eastman Process (TEP). "
            "An operator is asking questions about a specific detected anomaly event.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Answer the operator's question directly, accurately, and concisely based on the incident data provided.\n"
            "- If asked for the scores: clearly state the overall score, LSTM reconstruction score vs threshold, and CVA Q-statistic vs threshold.\n"
            "- If asked for the severity: state the exact assessed severity level (low, medium, high, or critical) and the reason why.\n"
            "- If asked for the root cause or subsystem: explain the physical mechanism and suspect components.\n"
            "- Do NOT repeat the initial diagnosis paragraph unless specifically asked to explain the entire event from scratch."
        )

        context = self._render_event_context(event, previous_report)
        ev_id = event.get("event_id", "UNKNOWN")
        rc = previous_report.get("root_cause", "Unknown")
        sev = previous_report.get("severity", "Unknown")

        messages: List[Tuple[str, str]] = [
            ("user", f"Here is the telemetry and diagnostic data for anomaly {ev_id}:\n\n{context}"),
            ("assistant", f"I have reviewed the diagnostic data for incident {ev_id}. Suspected cause: {rc}, Severity: {sev.upper()}. How can I assist you with this event?"),
        ]

        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                messages.append((role, str(content)))

        messages.append(("user", question))
        prompt = build_chat_prompt(chat_system_prompt, messages)

        # Disable overfitted LoRA adapter during chat if model is PeftModel
        # so base model's conversational intelligence is used
        if hasattr(self.model, "disable_adapter"):
            try:
                with self.model.disable_adapter():
                    return self._generate(
                        prompt,
                        max_new_tokens=int(self._gen_config.get("max_new_tokens", 384)),
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        repetition_penalty=1.15,
                    )
            except Exception as exc:
                logger.warning("Generation with disable_adapter failed (%s); using default generate.", exc)

        return self._generate(
            prompt,
            max_new_tokens=int(self._gen_config.get("max_new_tokens", 384)),
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15,
        )

    @classmethod
    def _dispatch_operator_intent(
        cls,
        event: Dict,
        report: Dict,
        question: str,
    ) -> Optional[str]:
        """Detect and immediately answer common operational control-room questions."""
        q = question.lower().strip()
        ev_id = event.get("event_id", "UNKNOWN")
        evidence = event.get("evidence", {}) or {}
        det_ev = evidence.get("detector_evidence", {}) or {}
        sev = str(report.get("severity") or evidence.get("severity", "unknown")).upper()
        score = float(event.get("max_anomaly_score", evidence.get("anomaly_score", 0.0)))
        root_cause = report.get("root_cause", "Process disturbance")
        subsystem = report.get("affected_subsystem") or evidence.get("candidate_subsystem", "unknown")
        action = report.get("recommended_action", "Inspect suspect equipment on site.")
        uncertainty = report.get("uncertainty", "On-site verification required to confirm causation.")

        # Keywords
        has_score = any(w in q for w in ("score", "scores", "metric", "metrics", "reconstruction error", "q-stat", "q stat", "q-score", "threshold"))
        has_severity = any(w in q for w in ("severity", "how serious", "how dangerous", "critical", "alarm level", "risk level", "danger"))
        has_sensors = any(w in q for w in ("sensor", "sensors", "inspect", "deviat", "which sensor"))
        has_action = any(w in q for w in ("what should i do", "what to do", "action", "recommend", "how to fix", "mitigat", "next step", "procedure"))
        has_overview = any(w in q for w in ("what is this anomaly", "what anomaly", "what happened", "what is this", "overview", "root cause", "explain anomaly", "explain this anomaly", "diagnose", "summary"))

        lstm = det_ev.get("lstm_ae") or {}
        l_sc = float(lstm.get("score", 0.0))
        l_th = float(lstm.get("threshold", 0.0))
        l_ratio = l_sc / max(l_th, 1e-9)
        l_alarm = "ALARMED (Active)" if lstm.get("is_anomalous") else "Normal (Below limit)"

        dyn = det_ev.get("dynamic") or {}
        cva = dyn.get("cva") or {}
        q_sc = float(cva.get("score", 0.0))
        q_th = float(cva.get("threshold", 0.0))
        q_ratio = q_sc / max(q_th, 1e-9)
        q_alarm = "ALARMED (Active)" if cva.get("is_anomalous") else "Normal (Below limit)"
        dom_lag = cva.get("dominant_lag", "N/A")
        slow_drift = float(cva.get("slow_drift_ratio", 0.0))

        # Intent 1: Both scores and severity requested
        if has_score and has_severity:
            lines = [
                f"### Telemetry Scores & Severity for Incident {ev_id}",
                f"- **Assessed Severity:** **{sev}** (Evaluated against TEP process safety limits)",
                f"- **Overall Anomaly Score:** `{score:.3f}`",
                "",
                "**Detector Breakdown:**",
                f"- **LSTM Autoencoder:** Reconstruction error = `{l_sc:.4f}` (Threshold: `{l_th:.4f}`, Margin: `{l_ratio:.2f}x` — {l_alarm})",
                f"- **CVA Dynamics (Q-Statistic):** Score = `{q_sc:.1f}` (Threshold: `{q_th:.1f}`, Margin: `{q_ratio:.2f}x` — {q_alarm})",
                f"  * CVA Lag Metrics: Dominant lag `{dom_lag}`, Slow drift ratio `{slow_drift:.2f}`",
                "",
                f"- **Suspected Root Cause:** {root_cause} (`{subsystem}`)",
            ]
            return "\n".join(lines)

        # Intent 2: Scores only
        if has_score:
            lines = [
                f"### Anomaly Detector Scores for Incident {ev_id}",
                f"- **Overall Anomaly Score:** `{score:.3f}`",
                "",
                "**Detector Metrics:**",
                f"- **LSTM Autoencoder:** Reconstruction error = `{l_sc:.4f}` (Threshold: `{l_th:.4f}`, Margin: `{l_ratio:.2f}x` — {l_alarm})",
                f"- **CVA Dynamics Q-Statistic:** Score = `{q_sc:.1f}` (Threshold: `{q_th:.1f}`, Margin: `{q_ratio:.2f}x` — {q_alarm})",
                f"- **CVA Lag Profile:** Dominant lag `{dom_lag}`, Slow drift ratio `{slow_drift:.2f}`",
            ]
            return "\n".join(lines)

        # Intent 3: Severity only
        if has_severity:
            sev_expl = {
                "CRITICAL": "Exceeds major safety thresholds (high risk of reactor trip, overpressure, or equipment shutdown).",
                "HIGH": "Significant process deviation approaching safety envelopes or severely impacting product specs.",
                "MEDIUM": "Moderate process disturbance; control loops are compensating but efficiency is degraded.",
                "LOW": "Minor drift within operational safety margins.",
            }.get(sev, "Evaluated against Tennessee Eastman Process safety limits.")
            lines = [
                f"### Severity Assessment for Incident {ev_id}",
                f"- **Assessed Severity Level:** **{sev}**",
                f"- **Safety Context:** {sev_expl}",
                f"- **Suspected Cause:** {root_cause} affecting `{subsystem}`.",
            ]
            return "\n".join(lines)

        # Intent 4: Overview / What is this anomaly
        if has_overview:
            trig = det_ev.get("triggered_by", "fused")
            ctype = det_ev.get("change_type", "unspecified")
            top_s = evidence.get("top_anomalous_sensors", [])[:3]
            sensor_str = ", ".join(s.get("display_name", s.get("name", "Sensor")) for s in top_s) or "No single sensor dominating"
            lines = [
                f"### Incident Diagnosis for {ev_id}",
                f"- **Root Cause:** **{root_cause}**",
                f"- **Affected Subsystem:** `{subsystem}`",
                f"- **Severity:** **{sev}**",
                f"- **Alarm Trigger:** Mode `{trig}` ({ctype} dynamic pattern)",
                f"- **Primary Deviating Sensors:** {sensor_str}",
                "",
                f"**Recommended Action:** {action}",
            ]
            return "\n".join(lines)

        # Intent 5: Sensors to inspect
        if has_sensors:
            top_sensors = evidence.get("top_anomalous_sensors", [])
            lines = [f"### Sensors to Inspect for Incident {ev_id}", ""]
            if top_sensors:
                for s in top_sensors:
                    name = s.get("display_name", s.get("name", "Unknown"))
                    dev = s.get("deviation_percent", 0.0)
                    z = s.get("z_score", s.get("zScore", 0.0))
                    trend = s.get("trend", "unknown")
                    lim = s.get("limit_context", "")
                    lim_text = f" -> **[Limit Warning: {lim}]**" if lim else ""
                    lines.append(f"- **{name}:** Deviation `{dev:+.1f}%`, z-score `{z:+.2f}`, Trend: `{trend}`{lim_text}")
            else:
                lines.append("No individual sensors showed severe statistical excursions; anomaly was detected via dynamic correlation.")
            return "\n".join(lines)

        # Intent 6: Recommended action / mitigation
        if has_action:
            lines = [
                f"### Recommended Actions for Incident {ev_id}",
                f"- **Direct Operator Action:** {action}",
                f"- **Subsystem Target:** Inspect all valves and instruments in `{subsystem}`.",
                f"- **Precaution:** {uncertainty}",
            ]
            return "\n".join(lines)

        return None

    @staticmethod
    def _render_event_context(event: Dict, previous_report: Dict) -> str:
        evidence = event.get("evidence", {}) or {}
        det_ev = evidence.get("detector_evidence", {}) or {}
        sev = previous_report.get("severity") or evidence.get("severity", "unknown")
        score = event.get("max_anomaly_score", evidence.get("anomaly_score", 0.0))

        lines = [
            f"Event ID: {event.get('event_id', 'UNKNOWN')}",
            f"Assessed Severity: {str(sev).upper()}",
            f"Overall Anomaly Score: {float(score):.3f}",
            f"Suspected Root Cause: {previous_report.get('root_cause', 'Unknown')}",
            f"Affected Subsystem: {previous_report.get('affected_subsystem', 'Unknown')}",
        ]

        # Detector scores breakdown
        if det_ev:
            lines.append("\nDetector Metrics & Scores:")
            trig = det_ev.get("triggered_by", "unspecified")
            ctype = det_ev.get("change_type", "unspecified")
            lines.append(f"- Alarm Trigger: {trig} mode ({ctype} dynamic pattern)")

            lstm = det_ev.get("lstm_ae") or {}
            if lstm and "score" in lstm:
                l_sc = float(lstm.get("score", 0.0))
                l_th = float(lstm.get("threshold", 0.0))
                l_ratio = l_sc / max(l_th, 1e-9)
                lines.append(f"- LSTM Autoencoder: reconstruction error = {l_sc:.4f} (threshold = {l_th:.4f}, ratio = {l_ratio:.2f}x, alarmed = {lstm.get('is_anomalous', False)})")

            dyn = det_ev.get("dynamic") or {}
            cva = dyn.get("cva") or {}
            if cva and "score" in cva:
                q_sc = float(cva.get("score", 0.0))
                q_th = float(cva.get("threshold", 0.0))
                q_ratio = q_sc / max(q_th, 1e-9)
                lines.append(f"- CVA Dynamics Q-Statistic: score = {q_sc:.1f} (threshold = {q_th:.1f}, ratio = {q_ratio:.2f}x, alarmed = {cva.get('is_anomalous', False)})")
                if cva.get("dominant_lag") is not None:
                    lines.append(f"  * CVA lag metrics: dominant lag = {cva.get('dominant_lag')}, slow drift ratio = {float(cva.get('slow_drift_ratio', 0.0)):.2f}")

        # Top sensor deviations
        top_sensors = evidence.get("top_anomalous_sensors", [])
        if top_sensors:
            lines.append("\nTop Deviating Sensors:")
            for s in top_sensors:
                name = s.get("display_name", s.get("name", "Unknown"))
                z = s.get("z_score", s.get("zScore", 0.0))
                dev = s.get("deviation_percent", 0.0)
                trend = s.get("trend", "unknown")
                lim = s.get("limit_context", "")
                lim_str = f" [Limit: {lim}]" if lim else ""
                lines.append(f"- {name}: deviation {dev:+.1f}%, z={z:+.2f}, trend {trend}{lim_str}")

        if previous_report.get("recommended_action"):
            lines.append(f"\nRecommended Action: {previous_report['recommended_action']}")

        return "\n".join(lines)