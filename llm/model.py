"""InternVL2-2B loading + single LoRA/QLoRA adapter setup ("tep_rca").

Only the ORIGINAL pretrained InternVL2-2B is ever loaded as the base. The base
model stays frozen; only ``tep_rca`` parameters are trainable. Before training
the loader prints the required header and FAILS if any unexpected adapter is
found on the base model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from utils import count_parameters, gpu_memory_summary

logger = logging.getLogger(__name__)

ADAPTER_NAME = "tep_rca"

# InternVL2-2B's LLM backbone is InternLM2-Chat-1.8B (verified from
# config.json: llm_config.architectures == ["InternLM2ForCausalLM"]). These are
# the exact linear layer names that receive LoRA adapters.
# InternLM2 uses fused QKV (wqkv) and different MLP names (w1/w2/w3)
# compared to standard LLaMA (q/k/v_proj, gate/up/down_proj).
INTERNLM2_LORA_TARGETS = [
    "wqkv",   # fused Q/K/V projection (replaces q_proj, k_proj, v_proj)
    "wo",     # attention output projection (replaces o_proj)
    "w1",     # MLP gate projection (replaces gate_proj)
    "w2",     # MLP down projection (replaces down_proj)
    "w3",     # MLP up projection (replaces up_proj)
]

# Adapter markers used by assert_no_adapter (fail on unexpected adapters).
ADAPTER_MARKERS = ("lora_", "adalora_", "ia3_", "prefix_encoder", "prompt_encoder")


def assert_no_adapter(model: Any) -> None:
    """Raise RuntimeError if the base model already carries a PEFT adapter."""
    if getattr(model, "peft_config", None):
        raise RuntimeError(
            "Unexpected PEFT adapter detected on the base model "
            f"(peft_config keys: {list(model.peft_config.keys())}). "
            "Refusing to train on a pre-adapted model."
        )
    for name, module in model.named_modules():
        if any(marker in name for marker in ADAPTER_MARKERS):
            raise RuntimeError(
                f"Unexpected adapter module found on the base model: {name}"
            )
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            raise RuntimeError(f"LoRA parameters found on base module: {name}")
    logger.info("Adapter check passed: base model carries NO adapters.")


def print_training_header(config: Dict, model: Any) -> None:
    """Print the mandatory training preamble AFTER the LoRA wrap so the
    trainable counts reflect only the ``tep_rca`` adapter parameters."""
    total, trainable = count_parameters(model)
    pct = 100.0 * trainable / total if total else 0.0
    header = "\n" + "=" * 70 + "\n"
    header += f"Base model:\n{config['llm']['base_model']}\n\n"
    header += "Adapters loaded:\nNONE\n\n"
    header += f"Training adapter:\n{config['llm'].get('adapter_name', ADAPTER_NAME)}\n\n"
    header += f"Total parameters:\n{total:,}\n\n"
    header += f"Trainable parameters:\n{trainable:,}\n\n"
    header += f"Trainable percentage:\n{pct:.4f}%\n"
    header += "=" * 70
    print(header)
    logger.info("GPU: %s", gpu_memory_summary())


def _print_load_header(config: Dict, model: Any) -> None:
    total, _ = count_parameters(model)
    header = "\n" + "=" * 70 + "\n"
    header += f"Base model:\n{config['llm']['base_model']}\n\n"
    header += "Adapters loaded:\nNONE\n\n"
    header += f"Total (base) parameters:\n{total:,}\n"
    header += "=" * 70
    print(header)


def _make_bnb_config(config: Dict) -> Any:
    from transformers import BitsAndBytesConfig

    training = config["llm"]["training"]
    compute_dtype = torch.bfloat16 if training.get("bf16", False) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=training.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=training.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(
    config: Dict,
    use_4bit: Optional[bool] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Union[str, Dict, None] = "auto",
    eval_mode: bool = False,
) -> Any:
    """Load the ORIGINAL pretrained InternVL2-2B (frozen, no adapters).

    Args:
        config: merged configuration dict.
        use_4bit: override ``llm.training.use_4bit`` (QLoRA via bitsandbytes).
        torch_dtype: override compute dtype (defaults to bf16/fp16 per config).
        device_map: HF device_map; "auto" for a single GPU.
        eval_mode: if True, skip gradient-related setup (used by test_adapter).

    Returns:
        The ``InternVLChatModel`` with ``img_context_token_id`` set and, unless
        ``eval_mode``, gradient checkpointing / input-grad hooks prepared.
    """
    from transformers import AutoModel

    llm_cfg = config["llm"]
    training = llm_cfg["training"]
    base_path = llm_cfg["base_model"]
    use_4bit = bool(training.get("use_4bit", True)) if use_4bit is None else bool(use_4bit)

    # Fail fast if the configured base repo ships an adapter already.
    _check_base_dir_for_adapter(base_path)

    compute_dtype = torch_dtype or (
        torch.bfloat16 if training.get("bf16", False) else torch.float16
    )

    kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(llm_cfg.get("trust_remote_code", True)),
        "use_flash_attn": bool(llm_cfg.get("use_flash_attn", False)),
        "low_cpu_mem_usage": True,
        "device_map": device_map,
    }
    if use_4bit:
        kwargs["quantization_config"] = _make_bnb_config(config)
    else:
        kwargs["torch_dtype"] = compute_dtype

    logger.info("Loading base model %s (4-bit=%s, dtype=%s) ...",
                base_path, use_4bit, compute_dtype)
    model = AutoModel.from_pretrained(base_path, **kwargs)

    # Print the base-load preamble and fail if an adapter is already present.
    _print_load_header(config, model)
    assert_no_adapter(model)

    # Set the image-context token id: forward()/generate() require it. For
    # text-only data image_flags are 0 so no vision embeddings are injected.
    try:
        tokenizer = load_tokenizer(base_path)
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not set img_context_token_id: %s", exc)

    if not eval_mode:
        model.config.use_cache = False
        _enable_grad_checkpointing(model, bool(training.get("gradient_checkpointing", True)))
        if use_4bit:
            _prepare_kbit_model(model)
        else:
            model.enable_input_require_grads()
    return model


def _check_base_dir_for_adapter(base_path: str) -> None:
    """Fail if the local base-model directory contains adapter files."""
    path = Path(base_path)
    if not path.exists():
        return  # hub id / remote path: nothing local to inspect
    for candidate in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"):
        if (path / candidate).exists():
            raise RuntimeError(
                f"Unexpected adapter file '{candidate}' found in base model "
                f"directory {path}. Base must be the original InternVL2-2B."
            )


def _enable_grad_checkpointing(model: Any, enable: bool) -> None:
    if not enable:
        return
    try:
        model.gradient_checkpointing_enable()
    except (NotImplementedError, AttributeError):
        model.language_model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled.")


def _prepare_kbit_model(model: Any) -> None:
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    logger.info("Model prepared for k-bit (QLoRA) training.")


def build_lora_config(config: Dict) -> Any:
    """LoraConfig targeting only the InternLM2 attention/MLP projections."""
    from peft import LoraConfig, TaskType

    lora_cfg = config["llm"]["lora"]
    targets = list(lora_cfg.get("target_modules", INTERNLM2_LORA_TARGETS))
    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_path: str) -> Any:
    """Load the InternVL2-2B tokenizer robustly.

    ``AutoTokenizer.from_pretrained`` can return ``False`` (a bool) when
    the remote-code loading fails (e.g. sentencepiece version mismatch).
    This helper falls back to manually loading ``InternLM2Tokenizer`` from
    the cached remote code.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if tok is None or isinstance(tok, bool):
        # Auto-mapping failed — load the slow tokenizer from remote code.
        import importlib.util
        import os

        from huggingface_hub import try_to_load_from_cache

        snapshot_dir = Path(model_path) if Path(model_path).exists() else None
        if snapshot_dir is None:
            # Hub model — find the cached snapshot.
            cached = try_to_load_from_cache(model_path, "tokenization_internlm2.py")
            if cached is not None and Path(cached).exists():
                snapshot_dir = Path(cached).parent
        if snapshot_dir is not None:
            sys_path_saved = list(__import__("sys").path)
            try:
                __import__("sys").path.insert(0, str(snapshot_dir))
                mod = importlib.util.module_from_spec(
                    importlib.util.spec_from_file_location(
                        "tokenization_internlm2",
                        snapshot_dir / "tokenization_internlm2.py",
                    )
                )
                importlib.util.spec_from_file_location(
                    "tokenization_internlm2",
                    snapshot_dir / "tokenization_internlm2.py",
                ).loader.exec_module(mod)
                tok = mod.InternLM2Tokenizer(
                    str(snapshot_dir / "tokenizer.model")
                )
            finally:
                __import__("sys").path = sys_path_saved
        if tok is None or isinstance(tok, bool):
            raise RuntimeError(
                "Failed to load tokenizer for "
                f"{model_path}. Check sentencepiece version "
                "(`pip install sentencepiece==0.2.0`)."
            )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def wrap_peft(model: Any, config: Dict) -> Any:
    """Inject the ``tep_rca`` LoRA adapter into the frozen base model.

    Uses ``inject_adapter_in_model`` (rather than ``get_peft_model``) to add
    LoRA layers directly to the target modules without wrapping the model in
    ``PeftModelForCausalLM``.  The wrapper passes ``inputs_embeds`` to the
    forward call, which ``InternVLChatModel`` does not accept.
    """
    from peft import inject_adapter_in_model

    lora_config = build_lora_config(config)
    model = inject_adapter_in_model(lora_config, model, adapter_name=ADAPTER_NAME)
    return model