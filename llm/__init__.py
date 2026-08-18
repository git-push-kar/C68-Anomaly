"""InternVL2-2B TEP root-cause-adapter (LoRA/QLoRA) components.

Everything here trains / loads ONE adapter named ``tep_rca``. The LSTM
autoencoder, scaler, evidence builder, event store and streaming engine remain
external modules and are never part of the adapter weights.
"""
from .dataset import (
    SYSTEM_PROMPT,
    TEPInstructionDataset,
    build_chat_prompt,
    format_evidence_question,
    render_answer_json,
)
from .model import (
    assert_no_adapter,
    build_lora_config,
    load_base_model,
    print_training_header,
    wrap_peft,
)
from .inference import RCAInference, extract_json_object

__all__ = [
    "SYSTEM_PROMPT",
    "TEPInstructionDataset",
    "build_chat_prompt",
    "format_evidence_question",
    "render_answer_json",
    "assert_no_adapter",
    "build_lora_config",
    "load_base_model",
    "print_training_header",
    "wrap_peft",
    "RCAInference",
    "extract_json_object",
]