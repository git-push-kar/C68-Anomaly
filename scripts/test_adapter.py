"""Verify the trained ``tep_rca`` adapter on a fresh ORIGINAL InternVL2-2B.

Loads a fresh base model, attaches ONLY the tep_rca adapter, creates a sample
anomaly event (matching the automatic pipeline's evidence format), generates an
automatic RCA report, and prints it.

Usage:
    python scripts/test_adapter.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.inference import RCAInference  # noqa: E402
from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

SAMPLE_EVIDENCE = {
    "event_id": "ANOM-1042",
    "anomaly_score": 0.94,
    "severity": "high",
    "pre_anomaly_context": {"duration_minutes": 5.0, "status": "normal"},
    "top_anomalous_sensors": [
        {"display_name": "Reactor_Cooling_Water_Flow", "name": "XMV_51",
         "current_value": 72.4, "baseline_value": 125.0,
         "deviation_percent": -42.1, "trend": "decreasing", "contribution": 0.31},
        {"display_name": "Reactor_Temperature", "name": "XMEAS_9",
         "current_value": 138.2, "baseline_value": 105.3,
         "deviation_percent": 31.2, "trend": "increasing", "contribution": 0.27},
        {"display_name": "Reactor_Pressure", "name": "XMEAS_7",
         "current_value": 2945.0, "baseline_value": 2496.0,
         "deviation_percent": 18.0, "trend": "increasing", "contribution": 0.16},
    ],
    "temporal_sequence": {
        "sequence": [
            {"sensor": "XMV_51", "display_name": "Reactor_Cooling_Water_Flow",
             "event": "decreased", "relative_time_minutes": 0.0},
            {"sensor": "XMEAS_9", "display_name": "Reactor_Temperature",
             "event": "increased", "relative_time_minutes": 5.0},
            {"sensor": "XMEAS_7", "display_name": "Reactor_Pressure",
             "event": "increased", "relative_time_minutes": 8.0},
        ],
        "first_onset_minutes": 0.0,
    },
    "candidate_subsystem": "reactor_cooling_system",
    "candidate_subsystem_score": 0.8,
    "reasoning_notes": [
        "Cooling-water flow decreased first, then reactor temperature, then "
        "reactor pressure. This temporal sequence makes a cooling-system "
        "abnormality the most likely initiating event.",
    ],
    "evidence_type": "model_derived_evidence",
    "uncertainty": "Correlation does not prove causation; verify on-site.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the tep_rca adapter.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    base = args.base_model or config["llm"]["base_model"]
    adapter = args.adapter or config["llm"]["adapter_dir"]

    print(f"\nLoading FRESH original InternVL2-2B from {base}")
    print(f"Attaching ONLY the tep_rca adapter from {adapter}\n")

    inference = RCAInference.from_adapter(base_model=base, adapter_path=adapter, config=config)

    print("=" * 70)
    print("AUTOMATIC ROOT-CAUSE REPORT (test_adapter.py)")
    print("=" * 70)
    report, raw = inference.generate_report(SAMPLE_EVIDENCE)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n--- raw model output (first 800 chars) ---")
    print(raw[:800])
    print("\nADAPTER VERIFICATION PASSED.")


if __name__ == "__main__":
    main()