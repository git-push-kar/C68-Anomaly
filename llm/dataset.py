"""Supervised instruction-tuning dataset for InternVL2-2B (text-only).

The LLM does NOT receive raw sensor time series. It receives structured anomaly
events. Training examples follow the InternVL2 "internlm2-chat" prompt format
with the assistant answer being a JSON object, so the model learns to produce
the same structured output at inference time.

Text-only approach
------------------
TEP is sensor-only and contains no images. InternVL2-2B's ``forward()`` expects
a ``pixel_values`` tensor (it always runs the ViT), but the official InternVL
training code handles image-free samples with a dummy zero image and
``image_flags=0`` so no vision embeddings are injected. We follow exactly that
supported pattern rather than inventing an unsupported API.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<IMG_CONTEXT>"
EOS = "<|im_end|>"

SYSTEM_PROMPT = (
    "You are an industrial process anomaly diagnosis assistant for the "
    "Tennessee Eastman Process (TEP). You analyze structured anomaly evidence "
    "produced by an unsupervised sensor model. You reason about likely root "
    "causes, affected subsystems, severity, and recommended actions. You never "
    "claim absolute causal certainty: correlation and temporal order are "
    "supporting evidence, not proof. Whenever evidence is insufficient you say "
    "so explicitly and recommend verification.\n"
    "Severity rubric: low = minor deviations, process stable; medium = moderate "
    "deviations contained by control; high = large deviations, significant "
    "operational impact; critical = deviations approaching safety limits "
    "(e.g. strong sustained rises in reactor temperature/pressure with large "
    "magnitude deviations). Use strong magnitudes and safety-relevant trends "
    "to justify high/critical, do not default to medium."
)

ANSWER_KEYS = [
    "summary", "root_cause", "affected_subsystem", "evidence",
    "reasoning", "severity", "confidence", "recommended_action", "uncertainty",
]

ANSWER_SCHEMA_HINT = (
    'Respond with ONLY a single JSON object using exactly these keys: '
    f'{", ".join(ANSWER_KEYS)}. severity must be one of '
    '"low", "medium", "high", "critical". confidence is a number between 0 and 1.'
)


def build_chat_prompt(
    system: str,
    messages: Sequence[Tuple[str, str]],
    assistant_turn: bool = True,
) -> str:
    """Build an internlm2-chat prompt with a custom system message.

    messages: list of (role, content) where role is "user" or "assistant".
    """
    parts = [f"<|im_start|>system\n{system}{EOS}"]
    for role, content in messages:
        if content:
            parts.append(f"<|im_start|>{role}\n{content}{EOS}")
        else:
            parts.append(f"<|im_start|>{role}\n")
    if assistant_turn and (not messages or messages[-1][1] is not None):
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def format_evidence_question(evidence: Dict) -> str:
    """Render a structured event into the text prompt shown to the model."""
    lines: List[str] = [
        "A new anomaly has been detected in the Tennessee Eastman Process.",
        "",
        f"Event ID: {evidence.get('event_id', 'UNKNOWN')}",
        f"Anomaly score: {evidence.get('anomaly_score', 0.0):.3f}",
        "",
        "Top sensor deviations:",
    ]
    sensors = evidence.get("top_anomalous_sensors", [])
    if sensors:
        for s in sensors:
            # Primary: stable z-score, secondary: percent for readability (bounded)
            z = s.get("z_score", s.get("zScore", 0.0))
            lines.append(
                f"- {s.get('display_name', s.get('name'))}: "
                f"z={z:+.2f} "
                f"(deviation: {s.get('deviation_percent', 0.0):+.1f}%, "
                f"trend: {s.get('trend', 'unknown')}, "
                f"contribution: {s.get('contribution', 0.0):.2f})"
            )
    else:
        lines.append("- (no strong sensor deviations detected)")
    lines.append("")

    seq = evidence.get("temporal_sequence", {}).get("sequence", [])
    if seq:
        lines.append("Temporal sequence:")
        for i, item in enumerate(seq, start=1):
            lines.append(
                f"{i}. {item.get('display_name')} became anomalous "
                f"{item.get('relative_time_minutes', 0):.1f} min after the first onset."
            )
    else:
        lines.append("Temporal sequence: (no clear onset ordering detected)")
    lines.append("")

    subsystem = evidence.get("candidate_subsystem", "unknown")
    lines.append(f"Candidate affected subsystem: {subsystem}")

    context = evidence.get("pre_anomaly_context", {})
    lines.append(
        f"Pre-anomaly context: {context.get('status', 'unknown')} "
        f"(baseline duration ~{context.get('duration_minutes', '?')} min)"
    )
    lines.append("")
    lines.append(
        "Determine the likely root cause and explain the evidence. "
        "Base your reasoning only on the evidence above. " + ANSWER_SCHEMA_HINT
    )
    return "\n".join(lines)


def render_answer_json(answer: Dict) -> str:
    """Serialize a structured answer dict to the exact JSON string trained on."""
    import json

    return json.dumps(answer, ensure_ascii=False)


class TEPInstructionDataset(torch.utils.data.Dataset):
    """Tokenized instruction samples for text-only InternVL2-2B SFT.

    Each sample yields: input_ids, attention_mask, labels (masked), a dummy
    ``pixel_values`` image and ``image_flags=0`` (official InternVL text-only
    pattern), so the model's ``forward()`` runs with no injected vision tokens.
    """

    def __init__(
        self,
        samples: Sequence[Dict],
        tokenizer: Any,
        max_length: int = 4096,
        dummy_image_size: int = 448,
        system_prompt: str = SYSTEM_PROMPT,
        pixel_dtype: Any = None,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.dummy_image_size = int(dummy_image_size)
        self.system_prompt = system_prompt
        self.pixel_dtype = pixel_dtype or torch.float32
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_parts(
        self, prompt: str, answer: str
    ) -> Tuple[List[int], List[int]]:
        """Tokenize prompt and answer separately and concatenate.

        The mask is applied from the assistant-turn header onward so only the
        answer tokens are learned (standard SFT masking).
        """
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, truncation=True,
            max_length=self.max_length,
        ).input_ids
        answer_ids = self.tokenizer(
            answer, add_special_tokens=False, truncation=True,
            max_length=self.max_length,
        ).input_ids
        budget = self.max_length - len(prompt_ids)
        answer_ids = answer_ids[:budget]
        return prompt_ids, answer_ids

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        question = sample["question"]
        answer = sample["answer"]
        system = sample.get("system", self.system_prompt)

        prompt = build_chat_prompt(system, [("user", question)])
        prompt_ids, answer_ids = self._tokenize_parts(prompt, answer + EOS)
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.zeros(
                1, 3, self.dummy_image_size, self.dummy_image_size,
                dtype=self.pixel_dtype,
            ),
            "image_flags": torch.tensor([0], dtype=torch.long),
        }


def _pad_batch(
    batch: List[Dict], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """Pad one collated batch to the longest sequence in it."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        ids, labs, attn = b["input_ids"], b["labels"], b["attention_mask"]
        pad = max_len - ids.shape[0]
        if pad > 0:
            ids = torch.cat([ids, torch.full((pad,), pad_token_id, dtype=ids.dtype)])
            labs = torch.cat([labs, torch.full((pad,), -100, dtype=labs.dtype)])
            attn = torch.cat([attn, torch.zeros(pad, dtype=attn.dtype)])
        out["input_ids"].append(ids)
        out["labels"].append(labs)
        out["attention_mask"].append(attn)
    out["input_ids"] = torch.stack(out["input_ids"])
    out["labels"] = torch.stack(out["labels"])
    out["attention_mask"] = torch.stack(out["attention_mask"])
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    out["image_flags"] = torch.stack([b["image_flags"] for b in batch])
    return out


class TEPCollator:
    """Picklable data collator (module-level class, safe for worker processes)."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return _pad_batch(batch, pad_token_id=self.pad_token_id)


def make_collator(pad_token_id: int):
    """Factory that injects the tokenizer pad id into the collator."""
    return TEPCollator(pad_token_id)
