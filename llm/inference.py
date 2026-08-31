"""InternVL2-2B + tep_rca inference: automatic reports and follow-up answers.

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


def sanitize_answer(answer: Dict) -> Dict:
    """Fill in missing answer keys so reports always match the schema."""
    for key in ANSWER_KEYS:
        answer.setdefault(key, "" if key in ("summary", "root_cause",
                                             "affected_subsystem", "reasoning",
                                             "recommended_action", "uncertainty",
                                             "evidence") else ("unknown" if key == "severity" else 0.0))
    if not isinstance(answer["evidence"], list):
        answer["evidence"] = [str(answer["evidence"])]
    if answer.get("confidence") is None or not (0.0 <= float(answer["confidence"]) <= 1.0):
        answer["confidence"] = 0.0
    if answer["severity"] not in ("low", "medium", "high", "critical"):
        answer["severity"] = "unknown"
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
        max_new = max_new_tokens or int(self._gen_config.get("max_new_tokens", 768))
        do_sample = bool(self._gen_config.get("do_sample", False))
        temp = float(self._gen_config.get("temperature", 0.7))
        top_p = float(self._gen_config.get("top_p", 0.9))
        top_k = int(self._gen_config.get("top_k", 50))
        rep_penalty = float(self._gen_config.get("repetition_penalty", 1.0))
        pad = self.tokenizer.pad_token_id or eos_token_id

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        # Text-only InternVL2 forward needs the dummy image placeholders.
        dummy = torch.zeros(
            1, 3, int(self.config["llm"].get("dummy_image_size", 448)),
            int(self.config["llm"].get("dummy_image_size", 448)),
            dtype=torch.bfloat16, device=self.model.device,
        )
        image_flags = torch.zeros((input_ids.shape[0],), dtype=torch.long, device=self.model.device)

        generated: List[int] = []
        with torch.no_grad():
            for _ in range(max_new):
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
                if int(next_id.item()) == eos_token_id:
                    break
                generated.append(int(next_id.item()))
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
        report = sanitize_answer(parsed) if parsed else {
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

        The model receives the event evidence, the original report, the
        conversation history and the new question, so it reasons about THIS
        anomaly instead of starting a new diagnosis.
        """
        context = self._render_event_context(event, previous_report)
        messages: List[Tuple[str, str]] = [("user", context)]
        for msg in history:
            if msg.get("role") == "user" and msg.get("content"):
                messages.append(("user", str(msg["content"])))
            elif msg.get("role") == "assistant" and msg.get("content"):
                messages.append(("assistant", str(msg["content"])))
        messages.append(("user", question))
        prompt = build_chat_prompt(SYSTEM_PROMPT, messages)
        return self._generate(prompt, max_new_tokens=int(self._gen_config.get("max_new_tokens", 768)))

    @staticmethod
    def _render_event_context(event: Dict, previous_report: Dict) -> str:
        evidence = event.get("evidence", {})
        lines = [
            "The following anomaly was detected and analyzed earlier. Use it as "
            "context when answering the user's question about THIS anomaly.",
            "",
            f"Event ID: {event.get('event_id', 'UNKNOWN')}",
            f"Anomaly score: {event.get('max_anomaly_score', 0.0):.3f}",
            "Structured evidence:",
        ]
        if evidence:
            lines.append(json.dumps(evidence, ensure_ascii=False, indent=2))
        else:
            lines.append(str(event))
        lines.append("")
        lines.append("Previous automatic report:")
        lines.append(json.dumps(previous_report, ensure_ascii=False, indent=2))
        return "\n".join(lines)