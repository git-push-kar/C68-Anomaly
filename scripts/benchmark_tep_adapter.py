"""Benchmark the trained ``tep_rca`` adapter on the held-out test split.

Loads the report-kind samples from ``data/llm/test.jsonl`` (faults 15, 21),
runs the adapter over each, and scores with ``llm.evaluate.evaluate_adapter``
(root-cause accuracy, fault classification, severity, evidence consistency,
hedging, recommendation quality, hallucination, JSON validity).

Usage:
    python scripts/benchmark_tep_adapter.py --config configs/config.yaml [--top-k 3]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.evaluate import evaluate_adapter  # noqa: E402
from llm.inference import RCAInference  # noqa: E402
from utils import ensure_dir, load_config, save_json  # noqa: E402

logger = logging.getLogger(__name__)


def load_test_samples(dataset_dir: Path) -> list:
    samples = []
    for path in [dataset_dir / "test.jsonl", dataset_dir / "test"]:
        if Path(path).is_file():
            with open(path, encoding="utf-8") as handle:
                samples = [json.loads(line) for line in handle if line.strip()]
            break
    if not samples:
        from datasets import load_from_disk  # type: ignore

        samples = list(load_from_disk(str(dataset_dir / "test")))
    return [s for s in samples if s.get("kind") == "report"]


def generate_test_samples_fresh(config, n_per_fault: int, seed: int = 42) -> list:
    """Generate fresh report-kind test samples on the fly (faults 15,21).

    Uses same logic as scripts/generate_llm_dataset.py but without touching
    disk, so train/val are untouched.
    """
    import numpy as np

    from llm.dataset import format_evidence_question, render_answer_json
    from scripts.fault_knowledge import FAULT_KNOWLEDGE
    from scripts.generate_llm_dataset import build_evidence, build_report_answer

    test_faults = config["llm"]["training"].get("test_faults", [15, 21])
    rng = np.random.RandomState(seed)
    samples = []
    counter = 9000
    for fault_id in test_faults:
        kb = FAULT_KNOWLEDGE[fault_id]
        for _ in range(n_per_fault):
            counter += 1
            evidence = build_evidence(fault_id, rng, counter)
            question = format_evidence_question(evidence)
            answer = build_report_answer(evidence, kb)
            samples.append({
                "kind": "report",
                "fault_id": fault_id,
                "fault_name": kb["name"],
                "severity": kb["severity"],
                "question": question,
                "answer": render_answer_json(answer),
                "evidence": evidence,
                "reference": {"name": kb["name"], "subsystem": kb["subsystem"], "severity": kb["severity"]},
            })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the tep_rca adapter.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--large-test", type=int, default=0,
                        help="If >0, ignore dataset_dir and generate this many FRESH report samples per test fault (e.g. 40 -> 80 total for faults 15,21). More reliable than 8.")
    parser.add_argument("--large-test-seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    base = args.base_model or config["llm"]["base_model"]
    adapter = args.adapter or config["llm"]["adapter_dir"]
    dataset_dir = Path(args.dataset_dir or config["paths"]["llm_dataset_path"])

    if args.large_test and args.large_test > 0:
        samples = generate_test_samples_fresh(config, n_per_fault=args.large_test, seed=args.large_test_seed)
        logger.info("Generated %d FRESH report samples (%d per fault) for faults %s (seed=%d)",
                    len(samples), args.large_test, config["llm"]["training"].get("test_faults", [15, 21]), args.large_test_seed)
        tag = f"fresh_{args.large_test}perFault_seed{args.large_test_seed}"
    else:
        samples = load_test_samples(dataset_dir)
        logger.info("Loaded %d report samples from %s", len(samples), dataset_dir)
        tag = dataset_dir.name

    inference = RCAInference.from_adapter(base_model=base, adapter_path=adapter, config=config)

    result = evaluate_adapter(inference, samples, top_k=args.top_k)

    out = ensure_dir(Path(adapter)) / "evaluation.json"
    save_json(out, {
        "top_k": args.top_k,
        "samples": tag,
        "large_test_per_fault": args.large_test if args.large_test else None,
        **result,
    })
    # also save a dedicated copy for large-test runs
    if args.large_test:
        save_json(ensure_dir(Path(adapter)) / f"evaluation_large_{args.large_test}perFault.json", {
            "top_k": args.top_k, "samples": tag, **result,
        })
    for key, value in result["metrics"].items():
        logger.info("%-28s %.3f", key, value)
    print(f"\nSaved evaluation to {out}")


if __name__ == "__main__":
    main()