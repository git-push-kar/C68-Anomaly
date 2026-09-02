"""Generate real RCA datasets from frozen detector:
- detector_derived: actual detector events (what LLM will see)
- ground_truth_aligned: real fault onset windows (even when detector missed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from utils import ensure_dir, load_config, load_json, save_json, set_seed
from preprocessing.scaler import load_scaler
from preprocessing.windowing import to_windows
from evidence.process_relationships import suggest_subsystems

def main():
    config = load_config("configs/config_a5000.yaml")
    set_seed(42)
    # Load manifests for 350/75/75 splits
    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    rca_known = load_json(Path("data/processed/manifests/rca_known_split.json"))

    # Load detector and scaler
    from main import TEPApp
    app = TEPApp(config=config, enable_llm=False)
    print(f"Detector threshold: {app.detector.threshold.threshold:.4f}")

    # Pre-load data once
    print("Pre-loading data...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    baseline_mean = np.array(scaler.baseline.mean, dtype=np.float32)
    baseline_std = np.array(scaler.baseline.std, dtype=np.float32)
    sensor_cols = [c for c in fault_all.columns if c.startswith("xmeas") or c.startswith("xmv")]
    sensor_cols_ordered = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in [f"XMEAS_{i}" for i in range(1,42)] + [f"XMV_{i}" for i in range(1,12)]]
    sensor_cols_ordered = [c for c in sensor_cols_ordered if c in fault_all.columns]

    out_dir = ensure_dir(Path("outputs/llm_dataset_v2"))
    detector_derived = []
    ground_truth_aligned = []

    # For each fault 1..20, for each split, for each run in that split
    for fault_id in range(1, 21):
        train_runs = rca_known["splits"][str(fault_id)]["train"]
        # For this initial generation, use 10 train runs per fault to keep it fast (will expand to 350 later)
        for run in train_runs[:5]:
            df_run_full = fault_all[(fault_all["faultNumber"]==fault_id) & (fault_all["simulationRun"]==run)].sort_values("sample")
            if len(df_run_full) < 500:
                continue
            df_run = df_run_full[sensor_cols_ordered]
            # Ground truth aligned: take post-onset window at sample 160
            df_post = df_run_full[df_run_full["sample"] >= 160]
            # Use the first post-onset window's sensor values to create evidence
            # For ground truth, we take the mean post-onset values for top sensors
            sensor_means_post = df_post[sensor_cols_ordered].mean().values
            z_post = (sensor_means_post - baseline_mean) / (baseline_std + 1e-9)
            top_idx = np.argsort(np.abs(z_post))[::-1][:5]
            top_sensors_gt = []
            for idx in top_idx:
                # Map idx to display name via feature_names
                raw_name = f"XMEAS_{idx+1}" if idx < 41 else f"XMV_{idx-41+1}"
                from evidence.process_relationships import SENSOR_NAMES
                display = SENSOR_NAMES.get(raw_name, raw_name)
                z_val = float(z_post[idx])
                mean_val = float(sensor_means_post[idx])
                baseline_m = float(baseline_mean[idx])
                if abs(baseline_m) < 1e-3:
                    dev_pct = None
                else:
                    dev_pct = float((mean_val - baseline_m) / abs(baseline_m) * 100)
                    if abs(dev_pct) > 1000:
                        dev_pct = None
                top_sensors_gt.append({
                    "display_name": display,
                    "z_score": round(float(z_val), 2),
                    "deviation_percent": round(dev_pct, 1) if dev_pct is not None else None,
                    "direction": "increasing" if z_val > 0 else "decreasing",
                })
            # Candidate via sensor mapping
            top_display_gt = [s["display_name"] for s in top_sensors_gt[:3]]
            candidates_gt = suggest_subsystems(top_display_gt, top_k=2)
            candidate_gt = candidates_gt[0]["subsystem"] if candidates_gt else "unknown"
            from scripts.fault_knowledge import FAULT_KNOWLEDGE
            kb = FAULT_KNOWLEDGE[fault_id]
            # Ground truth aligned evidence
            evidence_gt = {
                "event_id": f"GT-{fault_id:02d}-{run:03d}",
                "anomaly_score": round(float(np.abs(z_post).max()), 2),
                "top_anomalous_sensors": [
                    {
                        "display_name": s["display_name"],
                        "z_score": s["z_score"],
                        "deviation_percent": s["deviation_percent"],
                        "direction": s["direction"],
                        "trend": "increasing" if s["z_score"] > 0 else "decreasing",
                        "contribution": round(float(abs(s["z_score"]) / sum(abs(x["z_score"]) for x in top_sensors_gt)), 3) if top_sensors_gt else 0,
                    } for s in top_sensors_gt
                ],
                "temporal_sequence": {"sequence": [{"display_name": s["display_name"], "relative_time_minutes": round(float(i*0.5), 1)} for i, s in enumerate(top_sensors_gt[:3])], "first_onset_minutes": 0.0},
                "candidate_subsystem": candidate_gt,
                "evidence_type": "ground_truth_aligned",
                "provenance": {"source_type": "ground_truth_aligned", "fault_id": fault_id, "simulation_run": run, "event_start": 160},
            }
            ground_truth_aligned.append({
                "fault_id": fault_id,
                "simulation_run": run,
                "evidence": evidence_gt,
                "ground_truth": {"fault_id": fault_id, "fault_name": kb["name"], "subsystem": kb["subsystem"]},
                "source_type": "ground_truth_aligned",
            })

            # Detector derived: run through detector to get actual events
            # Create stream as before: 160 normal + 340 fault, sensor cols only
            normal_run = 1
            df_normal_full = normal_all[normal_all["simulationRun"]==normal_run].sort_values("sample")
            df_normal = df_normal_full[sensor_cols_ordered]
            df_run_sensor = df_run_full[sensor_cols_ordered]
            stream_df = pd.concat([df_normal.iloc[:160], df_run_sensor.iloc[:340]], ignore_index=True)
            # Use TEPApp to get events
            # Reset app
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
            for ev in events:
                ev_dict = ev if isinstance(ev, dict) else ev.to_dict()
                evidence = ev_dict.get("evidence") or {}
                # Ensure z-score primary is present (it is from event_builder)
                # Add provenance
                evidence["provenance"] = {"source_type": "detector_derived", "fault_id": fault_id, "simulation_run": run}
                evidence["event_id"] = ev_dict.get("event_id", f"DET-{fault_id:02d}-{run:03d}")
                detector_derived.append({
                    "fault_id": fault_id,
                    "simulationRun": run,
                    "evidence": evidence,
                    "ground_truth": {"fault_id": fault_id, "fault_name": kb["name"], "subsystem": kb["subsystem"]},
                    "source_type": "detector_derived",
                    "max_score": ev_dict.get("max_anomaly_score", 0),
                })
            if events:
                print(f"Fault {fault_id} run {run}: {len(events)} detector events, GT z top {top_sensors_gt[0]['z_score']:.2f}")

    # Save
    with open(out_dir / "detector_derived_rca.jsonl", "w") as f:
        for ex in detector_derived:
            f.write(json.dumps(ex) + "\n")
    with open(out_dir / "ground_truth_aligned_rca.jsonl", "w") as f:
        for ex in ground_truth_aligned:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved detector_derived: {len(detector_derived)} examples")
    print(f"Saved ground_truth_aligned: {len(ground_truth_aligned)} examples")

    # Also save stats
    save_json(out_dir / "real_rca_stats.json", {
        "detector_derived": len(detector_derived),
        "ground_truth_aligned": len(ground_truth_aligned),
        "per_fault_detector": {str(fid): len([x for x in detector_derived if x["fault_id"]==fid]) for fid in range(1,21)},
        "per_fault_gt": {str(fid): len([x for x in ground_truth_aligned if x["fault_id"]==fid]) for fid in range(1,21)},
    })

if __name__ == "__main__":
    main()
