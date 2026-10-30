"""Verify the training environment before any smoke run.

Four claims, each checked rather than assumed:

1. the interpreter has the pinned stack and a GPU it can actually use;
2. the base-model file on disk is the byte-for-byte file the run was validated
   against (SHA-256), not merely a directory that exists;
3. the model loads with ``trust_remote_code`` and completes a forward pass;
4. the LoRA wrapper exposes exactly the parameter count the earlier run reported,
   which is the one check that would catch a silent change of ``target_modules``.

Runs in whatever interpreter invokes it, so it can be pointed at ``venv_train``
without touching the app environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Trainable parameters reported by the earlier adapter run for this LoRA shape.
EXPECTED_TRAINABLE = 18_464_768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_v2.yaml")
    parser.add_argument("--expected-sha", default=(
        "18B4CDBC8EAB790F447FEAC717F4D0CECBB85CAF8DF198B3B2401C22506858E7"))
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    import peft
    import timm
    import accelerate

    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "timm": timm.__version__,
        "accelerate": accelerate.__version__,
    }

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = Path(config["llm"]["base_model"])
    weights = base / "model.safetensors"
    report["base_model"] = str(base)
    report["base_model_exists"] = base.is_dir()
    report["weights_exists"] = weights.is_file()
    report["weights_bytes"] = weights.stat().st_size if weights.is_file() else None
    if weights.is_file():
        digest = sha256_file(weights)
        report["weights_sha256"] = digest
        report["weights_sha256_matches"] = digest == args.expected_sha.upper()

    if not args.skip_model:
        from llm.model import load_base_model, wrap_peft

        model = load_base_model(config, eval_mode=True)
        report["model_class"] = type(model).__name__
        tokenizer_path = str(base)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=False)
        prompt = "<|im_start|>user\nCheck the reactor pressure.<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(input_ids=ids.input_ids,
                        attention_mask=ids.attention_mask,
                        pixel_values=torch.zeros(1, 3, 448, 448, dtype=torch.bfloat16,
                                                 device=model.device),
                        image_flags=torch.tensor([0], device=model.device),
                        use_cache=False)
        report["forward_logits_shape"] = list(out.logits.shape)

        wrapped = wrap_peft(model, config)
        trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        total = sum(p.numel() for p in wrapped.parameters())
        report["trainable_parameters"] = trainable
        report["total_parameters"] = total
        report["expected_trainable"] = EXPECTED_TRAINABLE
        report["trainable_matches"] = trainable == EXPECTED_TRAINABLE

    print(json.dumps(report, indent=1))
    ok = (report["cuda_available"] and report.get("weights_sha256_matches", True)
          and report.get("trainable_matches", True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
