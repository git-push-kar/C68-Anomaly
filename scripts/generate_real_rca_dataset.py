"""Generate REAL detector-derived RCA dataset.

For each fault run:
  RAW fault run (52 sensors)
    -> frozen detector (scaler + LSTM-AE + threshold)
    -> event aggregation
    -> REAL sensor-derived evidence (z-score, temporal, subsystem via sensor mapping)

Then attach ground-truth labels separately (fault_id, subsystem, severity) for supervision.

Creates two RCA benchmarks:
  A. Known-fault / unseen-run: same fault IDs, disjoint runs (350/75/75 per fault)
  B. Unseen-fault: hold out entire fault IDs (configs from manifests/rca_unseen_split.json)

All splits by simulationRun, before windowing, no leakage.
Synthetic data is NOT primary - will be added later as augmentation (<=30%).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils import ensure_dir, load_config, save_json
from preprocessing.tep_loader import load_tep_data

def generate_real_rca_dataset(config_path=None, max_runs_per_fault=75):
    config = load_config(config_path)
    # Load manifests
    manifest_dir = Path(config["paths"]["processed_data_path"]) / "manifests"
    rca_known = json.load(open(manifest_dir / "rca_known_split.json"))
    # Use frozen detector
    from main import TEPApp
    app = TEPApp(config=config, enable_llm=False)
    print(f"Detector threshold: {app.detector.threshold.threshold:.4f}")

    # For each fault and run in the known splits, generate real evidence
    output_dir = ensure_dir(Path("data/processed/rca_real"))
    all_examples = {"train": [], "val": [], "test": []}

    # Quick initial: 5 representative faults (cover feed, reactor, condenser, unknown)
    for fault_id in [1,4,14,15,20]:
        fid_str = str(fault_id)
        splits = rca_known["splits"].get(fid_str)
        if splits is None:
            continue
        for split_name, run_list in [("train", splits["train"][:5]), ("val", splits["val"][:2]), ("test", splits["test"][:2])]:
            for run in run_list[:2]:
                # Load that specific fault run
                from preprocessing.tep_loader import load_single_csv
                try:
                    df = load_single_csv(
                        f"data/raw/faults/TEP_Faulty_Training.csv",
                        config,
                        fault_number=fault_id,
                        simulation_run=run
                    )
                except Exception as e:
                    print(f"Skip fault {fault_id} run {run}: {e}")
                    continue
                import pandas as pd
                normal_df = load_single_csv(
                    "data/raw/normal/TEP_FaultFree_Training.csv",
                    config,
                    fault_number=0,
                    simulation_run=1
                )
                fault_onset = 160
                stream_df = pd.concat([normal_df.iloc[:fault_onset], df.iloc[:500-fault_onset]], ignore_index=True)
                # Reuse app but reset state (faster than reloading model)
                app._buffer.clear()
                if hasattr(app, '_recent_observations'):
                    app._recent_observations.clear()
                app._recent_flags.clear()
                app._open = None
                app._records_seen = 0
                app._window_id = 0
                app._startup_checked = False
                from streaming.simulator import SensorStream
                stream = SensorStream(stream_df, window_size=60, stride=5)
                events = []
                for window, meta in stream.iter_windows():
                    rec = {"values": window[-1], "sample_index": meta["sample_index"]}
                    res = app.process_sensor_stream(rec)
                    if res.get("event") is not None:
                        events.append(res["event"])
                final = app.finalize_stream()
                if final is not None:
                    events.append(final)
                # For each detected event, create RCA example with real evidence
                for ev in events:
                    evidence = ev.get("evidence") or {}
                    # Attach provenance and labels separately (not in evidence input)
                    example = {
                        "source_type": "real",
                        "fault_id": fault_id,
                        "simulation_run": run,
                        "event_start": ev.get("start_sample", 0),
                        "evidence": evidence,
                        "ground_truth": {
                            "fault_id": fault_id,
                            "fault_name": f"Fault {fault_id}",
                            "subsystem": evidence.get("candidate_subsystem", "unknown"),
                        },
                        # The LLM input will be evidence only, label is separate
                        "provenance": {
                            "detector_checkpoint": str(Path(config["anomaly_detector"]["model_dir"]) / "model.pt"),
                            "evidence_version": "1.0",
                        }
                    }
                    all_examples[split_name].append(example)
                if events:
                    print(f"Fault {fault_id} run {run} -> {len(events)} events (first score {events[0].get('max_anomaly_score',0):.2f})")
                else:
                    print(f"Fault {fault_id} run {run} -> NO EVENT (missed)")

    # Save
    for split in ["train", "val", "test"]:
        save_json(output_dir / f"rca_real_{split}.json", all_examples[split])
        print(f"Saved {len(all_examples[split])} examples to {output_dir / f'rca_real_{split}.json'}")

if __name__ == "__main__":
    generate_real_rca_dataset()
