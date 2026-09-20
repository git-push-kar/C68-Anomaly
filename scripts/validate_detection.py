"""Validate anomaly detection on normal runs (no false alarms) and fault runs.

Each simulationRun is processed independently (fresh TEPApp) to match how
training windows were prepared (never spanning run boundaries).

Usage:
    python scripts/validate_detection.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import TEPApp
from utils import load_config

CANONICAL_NAMES = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(42, 53)]
RIETH_TO_CANONICAL = {f"xmeas_{i}": f"XMEAS_{i}" for i in range(1, 42)}
RIETH_TO_CANONICAL.update({f"xmv_{i}": f"XMV_{i + 41}" for i in range(1, 12)})

FAULT_NAMES = {
    1: "A Feed Loss (Step)",
    3: "D Feed Temp (Step)",
    4: "Reactor Cooling Water Inlet Temp (Step)",
    9: "D Feed Temp (Random Variation)",
    14: "Reactor Cooling Water Valve (Sticking)",
    15: "Condenser Cooling Water Valve (Sticking)",
    21: "Reactor Cooling Water Loss",
}


def _load_runs(csv_path, config, simulation_runs, fault_number=None):
    """Load per-run sensor DataFrames from a Rieth CSV. Returns {run_id: DataFrame}."""
    ds = config["dataset"]
    run_buffers = {}
    requested_set = set(simulation_runs)
    # Expected samples: 960 for testing, 500 for training
    expected_samples = 960 if "Testing" in str(csv_path) else 500
    for chunk in pd.read_csv(csv_path, header=0, delimiter=ds.get("delimiter", ","), chunksize=250_000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        col_map = {}
        for c in chunk.columns:
            low = c.lower()
            if low in RIETH_TO_CANONICAL:
                col_map[c] = RIETH_TO_CANONICAL[low]
        if col_map:
            chunk = chunk.rename(columns=col_map)
        if "simulationRun" in chunk.columns:
            chunk = chunk[chunk["simulationRun"].isin(simulation_runs)]
        if chunk.empty:
            continue
        if fault_number is not None and "faultNumber" in chunk.columns:
            chunk = chunk[chunk["faultNumber"] == fault_number]
            if chunk.empty:
                continue
        sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
        if "simulationRun" in chunk.columns:
            for run_id, group in chunk.groupby("simulationRun"):
                run_buffers.setdefault(int(run_id), []).append(group[sensor_cols])
        else:
            run_buffers.setdefault(0, []).append(chunk[sensor_cols])

        # Early exit once all requested simulation runs have reached expected samples
        if requested_set.issubset(run_buffers.keys()):
            if all(sum(len(b) for b in run_buffers[r]) >= expected_samples for r in requested_set):
                break
    return {rid: pd.concat(bufs, ignore_index=True) for rid, bufs in run_buffers.items()}


def _run_app(app, df):
    """Feed every row through process_sensor_stream, collect closed events."""
    events = []
    for _, row in df.iterrows():
        result = app.process_sensor_stream(row.to_numpy(dtype=np.float32))
        if result["anomaly_detected"]:
            events.append(result["event"])
    flushed = app.flush()
    if flushed:
        events.append(flushed)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["testing", "training"], default="testing", help="Dataset split (default: testing for unseen data)")
    parser.add_argument("--normal-csv", default=None, help="Explicit path to normal CSV")
    parser.add_argument("--fault-csv", default=None, help="Explicit path to fault CSV")
    parser.add_argument("--faults", type=int, nargs="+", default=[1, 3, 4, 9, 14, 15, 21], help="Fault numbers to test")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--normal-runs", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--fault-runs", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    split_cap = "Testing" if args.split == "testing" else "Training"
    normal_default = Path(config["paths"]["data_root"]) / "normal" / f"TEP_FaultFree_{split_cap}.csv"
    fault_default = Path(config["paths"]["data_root"]) / "faults" / f"TEP_Faulty_{split_cap}.csv"
    normal_csv = str(args.normal_csv or normal_default)
    fault_csv = str(args.fault_csv or fault_default)

    # Part 1: Normal validation
    print("=" * 70)
    print(f"PART 1: Normal validation on {args.split.upper()} data (zero false alarms target)")
    print(f"Source: {normal_csv}")
    print("=" * 70)
    normal_runs = _load_runs(normal_csv, config, simulation_runs=args.normal_runs)
    print(f"Loaded {len(normal_runs)} runs: {sorted(normal_runs.keys())}")
    total_events = 0
    t0 = time.time()
    for run_id in sorted(normal_runs):
        app = TEPApp(config=config, enable_llm=not args.no_llm)
        events = _run_app(app, normal_runs[run_id])
        total_events += len(events)
        if events:
            max_s = max(e.get("max_anomaly_score", 0) for e in events)
            print(f"  Run {run_id}: FALSE ALARM  ({len(events)} events, max_score={max_s:.4f})")
            for e in events:
                ev = e.get("evidence", {})
                top = [s.get("display_name", "?") for s in (ev.get("top_anomalous_sensors") or [])[:5]]
                print(f"    {e['event_id']} sev={ev.get('severity')} score={e['max_anomaly_score']:.4f} "
                      f"subsystem={e.get('report',{}).get('affected_subsystem','?')} top={top}")
        else:
            print(f"  Run {run_id}: clean (0 events)")
    threshold = app.detector.threshold.threshold
    print(f"\nThreshold: {threshold:.5f} | Time: {time.time()-t0:.1f}s")
    print(f"Result: {'PASS' if total_events == 0 else 'FAIL'} ({total_events} false alarms)")

    # Part 2: Fault validation
    print("\n" + "=" * 70)
    print(f"PART 2: Fault validation on {args.split.upper()} data ({args.faults})")
    print(f"Source: {fault_csv}")
    print("=" * 70)
    for fid in args.faults:
        fname = FAULT_NAMES.get(fid, f"Fault {fid}")
        print(f"\n--- Fault {fid}: {fname} ---")
        fault_runs = _load_runs(fault_csv, config, args.fault_runs, fault_number=fid)
        if not fault_runs:
            print(f"  NOT IN DATASET (fault {fid} not available)")
            continue
        print(f"  Runs: {sorted(fault_runs.keys())}")
        all_events = []
        t0 = time.time()
        for run_id in sorted(fault_runs):
            app = TEPApp(config=config, enable_llm=not args.no_llm)
            fe = _run_app(app, fault_runs[run_id])
            all_events.extend(fe)
            if fe:
                ms = max(e.get("max_anomaly_score", 0) for e in fe)
                print(f"  Run {run_id}: DETECTED ({len(fe)} events, max_score={ms:.4f})")
            else:
                print(f"  Run {run_id}: NOT detected")
        detected = len(all_events) > 0
        max_score = max((e.get("max_anomaly_score", 0) for e in all_events), default=0)
        subsystem = None
        top_sensors = []
        if all_events:
            rpt = all_events[0].get("report") or {}
            ev = all_events[0].get("evidence") or {}
            subsystem = rpt.get("affected_subsystem", "?")
            top_sensors = [s.get("display_name", "?") for s in (ev.get("top_anomalous_sensors") or [])[:5]]
        print(f"  detected={'Y' if detected else 'N'}  events={len(all_events)}  "
              f"max_score={max_score:.4f}  subsystem={subsystem}")
        if top_sensors:
            print(f"  top_sensors: {top_sensors}")
        print(f"  Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
