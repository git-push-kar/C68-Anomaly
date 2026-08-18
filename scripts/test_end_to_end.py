"""End-to-end test: continuous stream -> detection -> evidence -> report ->
follow-up question, using the full TEPApp pipeline.

Requires prepared data and a trained anomaly detector (see README steps 2-3).
The InternVL adapter is optional; without it deterministic reports are used and
a note is printed.

Usage:
    python scripts/test_end_to_end.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import TEPApp  # noqa: E402
from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end TEP RCA test.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None, help="Normal CSV to replay")
    parser.add_argument("--fault", default=None, help="Fault CSV to inject")
    parser.add_argument("--inject-at", type=int, default=800)
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM adapter")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    source = args.source or config["streaming"]["normal_source"]
    fault = args.fault or config["streaming"]["fault_source"]

    print("\n=== TEP RCA end-to-end test ===\n")
    t0 = time.time()
    app = TEPApp(config=config, enable_llm=not args.no_llm)
    print(f"App initialized in {time.time() - t0:.1f}s. LLM loaded: {app.rca is not None}")

    print("\nReplaying continuous sensor stream and injecting a fault...")
    events = app.run_stream_from_file(
        source_file=source,
        inject_fault_file=fault,
        inject_fault_at=args.inject_at,
    )
    print(f"Closed anomaly events: {len(events)}")

    if not events:
        print("\nNo anomaly events were detected. Check the fault onset and threshold.")
        return

    for ev in events:
        print("\n" + "=" * 70)
        print("!!! ANOMALY DETECTED !!!")
        print("=" * 70)
        report = ev["report"]
        evidence = ev["evidence"]
        print(f"Event ID        : {ev['event_id']}")
        print(f"Severity        : {evidence.get('severity')}")
        print(f"Anomaly score   : {ev['max_anomaly_score']:.3f}")
        print(f"Likely root cause (candidate): {report.get('root_cause')}")
        print(f"Affected subsystem           : {report.get('affected_subsystem')}")
        print("\nEvidence:")
        for line in report.get("evidence", []):
            print(f"  * {line}")
        print(f"\nReasoning:\n  {report.get('reasoning')}")
        print(f"\nRecommended action:\n  {report.get('recommended_action')}")
        print(f"Confidence: {report.get('confidence')} | Uncertainty: {report.get('uncertainty')}")

    event_id = events[0]["event_id"]
    question = "Why is the cooling system the most likely cause?"
    print(f"\n--- Follow-up question on {event_id} ---")
    print(f"Q: {question}")
    if app.rca is None:
        print("A: (no LLM loaded; deterministic fallback)")
    answer = app.answer_followup(event_id, question)
    print(f"A: {answer['answer']}\n")

    print("End-to-end test completed successfully.")


if __name__ == "__main__":
    main()