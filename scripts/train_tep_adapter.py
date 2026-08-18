"""Fine-tune InternVL2-2B (frozen) with the single ``tep_rca`` LoRA/QLoRA adapter.

Prints the mandatory training header (base model, adapters: NONE, training
adapter: tep_rca, parameter counts) and fails if an unexpected adapter exists.

Usage:
    python scripts/train_tep_adapter.py --config configs/config.yaml
    python scripts/train_tep_adapter.py --config configs/config.yaml --resume
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.train_adapter import train_tep_adapter  # noqa: E402
from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the InternVL2-2B tep_rca LoRA/QLoRA adapter."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint in the adapter dir.")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Train with FP16 LoRA instead of 4-bit QLoRA.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    if args.no_4bit:
        config["llm"]["training"]["use_4bit"] = False
        config["llm"]["training"]["optim"] = "adamw_torch"
    if args.epochs:
        config["llm"]["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["llm"]["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["llm"]["training"]["learning_rate"] = args.lr
    if args.lora_r:
        config["llm"]["lora"]["r"] = args.lora_r

    out = train_tep_adapter(config, resume=args.resume, output_dir=args.output_dir)
    print("\nTEP adapter saved to:", out)


if __name__ == "__main__":
    main()