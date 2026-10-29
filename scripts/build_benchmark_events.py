"""Build benchmark events from the real pipeline at the frozen v2 operating point.

One event per run: the first event that CLOSES after the onset sample, exactly
the rule the live aggregator uses (open after `persistence` consecutive alarmed
windows, close after `min_separation` quiet windows or at `max_event_windows`).
Runs are taken as-is -- 960 samples, onset 160, no prepended fault-free data --
and streamed from TEP_Faulty_Testing.csv grouped by (faultNumber, simulationRun).

Why this is a separate script rather than TEPApp:
  * TEPApp reads the deployed thresholds from the artifact chain and uses the
    live confirmation count from config; this benchmark must run the FROZEN v2
    point (configs/detector_v2_operating_point.json, persistence 12, no
    smoothing).
  * TEPApp derives `start_sample` from a window counter that increments per
    record, not per window (main.py:224), so its sample indices are inflated.
    Here sample indices come from the window starts this script owns.
Aggregation and evidence are the live code's, not a reimplementation:
`segment_events` mirrors main.py's aggregator and
`evidence.event_builder.build_event` produces the evidence.

Outputs (outputs/benchmark_v2/events/):
    faulty/<split>/f{ff}_r{rrr}[_oracle].json   one event record per run
    normal/<split>_r{rrr}.json                  NORMAL_TRANSIENT candidates
    misses.jsonl | oracle.jsonl | normal.jsonl
    build_summary.json                          counts, delays, lengths, timings

    python scripts/build_benchmark_events.py --faults 1 2 --max-runs 3   # smoke
    python scripts/build_benchmark_events.py --faults 1 2 3              # batch
    python scripts/build_benchmark_events.py                             # all
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from anomaly_detection.dynamic.contributions import sensor_contributions  # noqa: E402
from anomaly_detection.dynamic.lag_builder import build_past_future  # noqa: E402
from evidence.event_builder import build_event  # noqa: E402
from preprocessing.scaler import load_scaler  # noqa: E402
from preprocessing.tep_loader import CANONICAL_NAMES  # noqa: E402
from preprocessing.windowing import to_windows  # noqa: E402
from utils import ensure_dir, get_fault_onset, load_config, save_json  # noqa: E402

from evaluate_detector_v2 import DetectorBundle, iter_runs, segment_events  # noqa: E402

logger = logging.getLogger("build_benchmark_events")

SPLIT_LABEL = "faulty_testing"
OUT = ROOT / "outputs" / "benchmark_v2" / "events"
FROZEN = ROOT / "configs" / "detector_v2_operating_point.json"

SPLIT_RUNS = {"train": (1, 300), "val": (301, 400), "test": (401, 500)}
TRAIN_RUNS_PER_FAULT = 120
VAL_RUNS_PER_FAULT = 25


class FrozenDetector:
    """LSTM + CVA Q + DPCA SPE at the frozen v2 thresholds, no smoothing."""

    def __init__(self, config: Dict, op: Dict) -> None:
        self.config = config
        self.window_size = int(config["windowing"]["window_size"])
        self.stride = int(config["windowing"]["stride"])
        self.onset = int(get_fault_onset(config, SPLIT_LABEL))
        self.thresholds = op["thresholds"]
        self.persistence = int(op["persistence_windows"])
        ev = config["events"]
        self.separate = int(ev["min_separation_windows"])
        self.max_event_windows = int(ev["max_event_windows"])
        self.bundle = DetectorBundle(
            config,
            {"lstm_ae": {"threshold": self.thresholds["lstm_ae"]},
             "dynamic": {"cva": {}, "dpca": {}}},
            device="cpu",
            window_size=self.window_size,
            stride=self.stride,
        )
        self.bundle.apply_thresholds(
            self.thresholds["lstm_ae"],
            {"cva_Q": self.thresholds["cva_Q"], "dpca_SPE": self.thresholds["dpca_SPE"]},
        )
        self.scaler = load_scaler(config["preprocessing"]["scaler_dir"])

    # -- scoring ---------------------------------------------------------
    def score_run(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        scaled = self.bundle.scale(raw)
        starts, lstm = self.bundle.score_lstm(scaled)
        dyn = self.bundle.score_dynamic(scaled, starts)
        fused = np.maximum(
            lstm / max(self.thresholds["lstm_ae"], 1e-9),
            dyn["cva_Q"] / max(self.thresholds["cva_Q"], 1e-9),
        )
        return {
            "starts": starts,
            "lstm": np.asarray(lstm, dtype=np.float64),
            "cva_Q": np.asarray(dyn["cva_Q"], dtype=np.float64),
            "fused": np.asarray(fused, dtype=np.float64),
        }

    def n_windows(self, n_samples: int = 960) -> int:
        return (n_samples - self.window_size) // self.stride + 1

    # -- evidence --------------------------------------------------------
    def detector_evidence(self, series: Dict[str, np.ndarray], idx_last: int) -> Dict:
        """Fused evidence at the event's last window.

        No CVA contribution vector: the v2 prompt no longer consumes it
        (candidate subsystem and the fault-number reasoning notes were removed),
        so computing it would cost a second pass for nothing.
        """
        lstm_s = float(series["lstm"][idx_last])
        cva_s = float(series["cva_Q"][idx_last])
        lstm_alarm = lstm_s > self.thresholds["lstm_ae"]
        cva_alarm = cva_s > self.thresholds["cva_Q"]
        if lstm_alarm and cva_alarm:
            triggered_by, change_type = "both", "mixed"
        elif cva_alarm:
            triggered_by, change_type = "dynamic", "dynamics"
        elif lstm_alarm:
            triggered_by, change_type = "lstm_ae", "level"
        else:
            triggered_by, change_type = "none", "none"
        return {
            "triggered_by": triggered_by,
            "change_type": change_type,
            "lstm_ae": {"score": lstm_s, "threshold": self.thresholds["lstm_ae"],
                        "alarmed": lstm_alarm},
            "dynamic": {
                "method": "cva",
                "score": cva_s,
                "threshold": self.thresholds["cva_Q"],
                "alarmed": cva_alarm,
                "cva": {
                    "score": cva_s,
                    "threshold": self.thresholds["cva_Q"],
                    "stats": {"Q": cva_s},
                    "thresholds": {"Q": self.thresholds["cva_Q"]},
                    "is_anomalous": cva_alarm,
                },
            },
        }

    def lag_attribution(self, scaled: np.ndarray, starts: Sequence[int],
                        open_idx: int, close_idx: int) -> Dict:
        """Per-variable CVA residual share and lag profile over the event."""
        cva = getattr(self.bundle, "cva", None)
        empty = {"dominant_lag": None, "slow_drift_ratio": None, "top_variables": []}
        if cva is None or not hasattr(cva, "n_past"):
            return empty
        acc = np.zeros(len(CANONICAL_NAMES), dtype=np.float64)
        profiles = []
        for idx in range(open_idx, close_idx + 1, 10):
            start = int(starts[idx])
            window = scaled[start:start + self.window_size]
            if window.shape[0] < self.window_size:
                continue
            past, _future, _t = build_past_future(window, cva.n_past, cva.n_future)
            if not len(past):
                continue
            residual = cva.score(past)["e"][-1]
            contribution, lag_profile = sensor_contributions(
                residual, cva.n_past, len(CANONICAL_NAMES)
            )
            acc += contribution
            profiles.append(np.sum(lag_profile ** 2, axis=1))
        if not profiles:
            return empty
        acc /= len(profiles)
        total = float(acc.sum())
        order = np.argsort(acc)[::-1][:5]
        lag_energy = np.mean(np.stack(profiles), axis=0)
        return {
            "dominant_lag": int(np.argmax(lag_energy)),
            "slow_drift_ratio": round(float(lag_energy[-2:].sum() / max(lag_energy.sum(), 1e-9)), 4),
            "top_variables": [
                {"variable": CANONICAL_NAMES[int(i)].strip(),
                 "share": round(float(acc[int(i)] / max(total, 1e-12)), 4)}
                for i in order
            ],
        }

    # -- one run -> one event record -------------------------------------
    def build(self, event_id: str, raw: np.ndarray, fault_label: Optional[int],
              window_block: Optional[Tuple[int, int]] = None) -> Dict:
        series = self.score_run(raw)
        starts = series["starts"]
        alarm = (series["fused"] > 1.0).astype(int)
        events = segment_events(alarm, starts, self.persistence, self.separate,
                                self.max_event_windows)
        end_samples = np.asarray(starts) + self.window_size - 1
        post = end_samples >= self.onset

        if window_block is not None:
            open_idx, close_idx = window_block
            forced = True
        else:
            chosen = next(((o, c) for o, c in events if post[o]), None)
            if chosen is None:
                # Why there is no event matters: an alarm that never confirms is a
                # detector miss, while a confirmed alarm whose event is still open
                # when the stream ends is the aggregator's silent drop (it needs
                # `separate` quiet windows to close and the run has run out).
                window_runs = np.convolve(alarm, np.ones(self.persistence, dtype=int), mode="valid")
                confirmed_anywhere = bool(window_runs.size and window_runs.max() >= self.persistence)
                return {
                    "found": False,
                    "n_events_in_run": len(events),
                    "reason": ("unclosed_at_run_end" if confirmed_anywhere else "detector_miss"),
                    "n_alarmed_windows": int(alarm.sum()),
                    "max_consecutive_alarms": int(window_runs.max()) if window_runs.size else 0,
                }
            open_idx, close_idx = chosen
            forced = False

        raw_windows = to_windows(raw.astype(np.float32), self.window_size, self.stride)
        scaled = self.bundle.scale(raw)
        scaled_windows = to_windows(scaled.astype(np.float32), self.window_size, self.stride)
        window_slice = slice(open_idx, close_idx + 1)
        _lstm_scores, per_sensor_errors = self.bundle.lstm.score_windows(
            scaled_windows[window_slice], already_scaled=True
        )

        evidence_ev = self.detector_evidence(series, close_idx)
        # Dynamics faults (valve sticking) barely move the level, so the LSTM
        # sensor ranking is flat for them; the CVA residual attribution is what
        # actually localizes them. Averaged over every 10th event window to keep
        # the cost per event small.
        evidence_ev["lag_attribution"] = self.lag_attribution(scaled, starts,
                                                              open_idx, close_idx)
        event = build_event(
            event_id=event_id,
            anomaly_scores=series["fused"][window_slice] * self.thresholds["lstm_ae"],
            per_sensor_errors=per_sensor_errors,
            event_windows=raw_windows[window_slice],
            baseline=self.scaler.baseline,
            feature_names=list(self.scaler.baseline.feature_names),
            threshold=self.thresholds["lstm_ae"],
            start_sample=int(starts[open_idx]),
            config=self.config,
            fault_label=fault_label,
            detector_evidence=evidence_ev,
        )

        first_alarm_idx = next(
            (i for i in range(open_idx + 1) if alarm[i] and post[i]), open_idx
        )
        open_sample = int(end_samples[open_idx])
        close_sample = int(end_samples[close_idx])
        return {
            "event_id": event_id,
            "found": True,
            "forced_windows": forced,
            "fault_label": fault_label,
            "n_windows": int(close_idx - open_idx + 1),
            "event_open_window": int(open_idx),
            "event_first_alarm_window": int(first_alarm_idx),
            "event_open_sample": open_sample,
            "event_first_alarm_sample": int(end_samples[first_alarm_idx]),
            "event_close_sample": close_sample,
            "injection_sample": self.onset,
            "event_open_delay": open_sample - self.onset,
            "first_alarm_delay": int(end_samples[first_alarm_idx]) - self.onset,
            "event_length_samples": close_sample - open_sample + self.stride,
            "triggered_by": evidence_ev["triggered_by"],
            "n_events_in_run": len(events),
            "event": event.to_dict(),
        }


def split_for_run(run: int) -> Optional[str]:
    for name, (lo, hi) in SPLIT_RUNS.items():
        if lo <= run <= hi:
            return name
    return None


def wanted(component: str, index_in_split: int) -> bool:
    if component == "train":
        return index_in_split < TRAIN_RUNS_PER_FAULT
    if component == "val":
        return index_in_split < VAL_RUNS_PER_FAULT
    return True


def dist(values: Sequence[float]) -> Dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p05": round(float(np.percentile(arr, 5)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
    }


def delay_by_subsystem(records: Sequence[Dict]) -> Dict[str, Dict]:
    from scripts.fault_knowledge import FAULT_KNOWLEDGE

    buckets: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        entry = FAULT_KNOWLEDGE.get(int(rec["fault"])) or {}
        buckets[str(entry.get("subsystem", "unknown"))].append(float(rec["event_open_delay"]))
    return {name: dist(values) for name, values in sorted(buckets.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config_a5000.yaml")
    ap.add_argument("--faults", type=int, nargs="*", default=None)
    ap.add_argument("--max-runs", type=int, default=None, help="smoke: runs per component")
    ap.add_argument("--skip-normal", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    op = json.loads(FROZEN.read_text(encoding="utf-8"))
    det = FrozenDetector(config, op)
    out = Path(args.out)
    ensure_dir(out / "faulty")

    faults = args.faults or list(range(1, 21))
    faulty_csv = Path(config["paths"]["fault_data_path"]) / "TEP_Faulty_Testing.csv"
    t0 = time.time()

    per_split: Dict[str, List[Dict]] = defaultdict(list)
    misses: List[Dict] = []

    # One pass over the faulty file for every requested fault: `iter_runs` reads
    # the whole 3.6 GB CSV, so per-fault calls would re-scan it 20 times.
    counters: Counter = Counter()
    processed: Counter = Counter()
    for f_num, run, raw in iter_runs(faulty_csv, faults=faults, run_range=(1, 500),
                                     samples_per_run=960):
        fault = int(f_num)
        if fault not in set(faults):
            continue
        component = split_for_run(run)
        if component is None:
            continue
        idx = counters[(fault, component)]
        counters[(fault, component)] += 1
        if not wanted(component, idx):
            continue
        if args.max_runs is not None and idx >= args.max_runs:
            continue
        processed[fault] += 1
        event_id = f"EVT-{component}-f{fault:02d}-r{run:03d}"
        rec = det.build(event_id, raw, fault)
        if rec["found"]:
            rec.update({"source_file": faulty_csv.name, "fault": fault,
                        "simulationRun": run, "split": component, "seed": 42})
            ensure_dir(out / "faulty" / component)
            save_json(out / "faulty" / component / f"f{fault:02d}_r{run:03d}.json", rec)
            per_split[component].append(rec)
        else:
            misses.append({"source_file": faulty_csv.name, "fault": fault,
                           "simulationRun": run, "split": component, "seed": 42,
                           "injection_sample": det.onset,
                           "n_events_in_run": rec["n_events_in_run"],
                           "reason": rec.get("reason", "unknown"),
                           "n_alarmed_windows": rec.get("n_alarmed_windows"),
                           "max_consecutive_alarms": rec.get("max_consecutive_alarms")})
        if sum(processed.values()) % 250 == 0:
            logger.info("  processed %d runs (%.0fs elapsed)", sum(processed.values()),
                        time.time() - t0)
    logger.info("faulty pass done: %d runs processed in %.0fs", sum(processed.values()),
                time.time() - t0)

    all_faulty = [r for v in per_split.values() for r in v]

    # ---- oracle windows for misses; offset from TRAIN median only -------
    train_opens = [r["event_open_sample"] for r in per_split["train"]]
    train_median_open = int(statistics.median(train_opens)) if train_opens else det.onset + 60
    logger.info("train median event-open sample %d from %d train events",
                train_median_open, len(train_opens))

    oracle: List[Dict] = []
    if not args.skip_oracle and misses:
        by_fault: Dict[int, set] = defaultdict(set)
        for m in misses:
            by_fault[int(m["fault"])].add(int(m["simulationRun"]))
        wanted_faults = sorted(by_fault)
        target_start = train_median_open - (det.window_size - 1)
        start = max(0, min(int(round(target_start / det.stride)), det.n_windows() - 1))
        block = (start, min(start + max(det.persistence, 1) - 1, det.n_windows() - 1))
        for fault, run, raw in iter_runs(faulty_csv, faults=wanted_faults,
                                         run_range=(1, 500), samples_per_run=960):
            fault = int(fault)
            if run not in by_fault.get(fault, set()):
                continue
            component = split_for_run(run) or "train"
            event_id = f"EVT-{component}-f{fault:02d}-r{run:03d}-oracle"
            rec = det.build(event_id, raw, fault, window_block=block)
            rec.update({"source_file": faulty_csv.name, "fault": fault,
                        "simulationRun": run, "split": component, "seed": 42,
                        "oracle": True, "oracle_offset_sample": train_median_open,
                        "oracle_offset_source": "median event-open sample over the TRAIN split only"})
            oracle.append(rec)
            ensure_dir(out / "faulty" / component)
            save_json(out / "faulty" / component / f"f{fault:02d}_r{run:03d}_oracle.json", rec)
        logger.info("oracle pass done: %d runs in %.0fs", len(oracle), time.time() - t0)

    # ---- NORMAL_TRANSIENT supply ----------------------------------------
    normal: List[Dict] = []
    if not args.skip_normal:
        normal_csv = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Testing.csv"
        for _f, run, raw in iter_runs(normal_csv, faults=None, run_range=(1, 500),
                                      samples_per_run=960):
            component = split_for_run(run) or "train"
            series = det.score_run(raw)
            alarm = (series["fused"] > 1.0).astype(int)
            confirmed = segment_events(alarm, series["starts"], det.persistence,
                                       det.separate, det.max_event_windows)
            if confirmed:
                open_idx, close_idx = confirmed[0]
                is_confirmed = True
            else:
                window_runs = np.convolve(alarm, np.ones(det.persistence, dtype=int), mode="valid")
                if window_runs.size and window_runs.max() >= det.persistence:
                    continue  # reaches 12 somewhere: not an unconfirmed excursion
                loose = segment_events(alarm, series["starts"], 3, det.separate,
                                       det.max_event_windows)
                if not loose:
                    continue
                open_idx, close_idx = loose[0]
                is_confirmed = False
            event_id = f"NORM-{component}-r{run:03d}"
            rec = det.build(event_id, raw, None, window_block=(open_idx, close_idx))
            rec.update({"source_file": normal_csv.name, "fault": None,
                        "label": "NORMAL_TRANSIENT", "simulationRun": run,
                        "split": component, "seed": 42,
                        "confirmed_false_alarm": is_confirmed,
                        "supply_source": "TEP_FaultFree_Testing"})
            normal.append(rec)
            ensure_dir(out / "normal")
            save_json(out / "normal" / f"{component}_r{run:03d}.json", rec)
            logger.info("normal run %3d (%s): confirmed=%s windows=%d",
                        run, component, is_confirmed, rec["n_windows"])

    # ---- summary ---------------------------------------------------------
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "operating_point": op["operating_point_id"],
        "operating_point_thresholds": op["thresholds"],
        "persistence_windows": det.persistence,
        "onset_sample": det.onset,
        "split_runs": SPLIT_RUNS,
        "budgets": {"train_runs_per_fault": TRAIN_RUNS_PER_FAULT,
                    "val_runs_per_fault": VAL_RUNS_PER_FAULT,
                    "test_runs_per_fault": "all 100 (401-500)"},
        "events_per_split": {s: len(per_split.get(s, [])) for s in SPLIT_RUNS},
        "events_total": len(all_faulty),
        "misses": {"total": len(misses),
                   "by_fault": {str(f): sum(1 for m in misses if int(m["fault"]) == f)
                                for f in faults},
                   "by_split": dict(Counter(m["split"] for m in misses)),
                   "by_reason": dict(Counter(m.get("reason") for m in misses)),
                   "unclosed_by_fault": dict(Counter(str(m["fault"]) for m in misses
                                                    if m.get("reason") == "unclosed_at_run_end")),
                   "detector_miss_by_fault": dict(Counter(str(m["fault"]) for m in misses
                                                       if m.get("reason") == "detector_miss"))},
        "oracle_events": len(oracle),
        "oracle_offset": {"sample": train_median_open,
                          "delay_after_onset": train_median_open - det.onset,
                          "source": "median event-open sample over the TRAIN split only"},
        "normal": {"total": len(normal),
                   "by_split": dict(Counter(n["split"] for n in normal)),
                   "confirmed_false_alarm_by_split":
                       dict(Counter(n["split"] for n in normal if n["confirmed_false_alarm"])),
                   "unconfirmed_by_split":
                       dict(Counter(n["split"] for n in normal if not n["confirmed_false_alarm"]))},
        "event_open_delay_by_fault": {str(f): dist([r["event_open_delay"] for r in all_faulty
                                                    if int(r["fault"]) == f]) for f in faults},
        "event_open_delay_by_subsystem": delay_by_subsystem(all_faulty),
        "first_alarm_delay_by_fault": {str(f): dist([r["first_alarm_delay"] for r in all_faulty
                                                     if int(r["fault"]) == f]) for f in faults},
        "event_length_windows_by_fault": {str(f): dist([r["n_windows"] for r in all_faulty
                                                        if int(r["fault"]) == f]) for f in faults},
        "triggered_by": dict(Counter(r["triggered_by"] for r in all_faulty)),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(out / "build_summary.json", summary)
    for name, rows in (("misses.jsonl", misses), ("oracle.jsonl", oracle), ("normal.jsonl", normal)):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    print(f"\nevents: {summary['events_per_split']}  misses: {len(misses)}  "
          f"oracle: {len(oracle)}  normal: {len(normal)}  ({summary['runtime_seconds']}s)")
    print(f"wrote {out / 'build_summary.json'}")


if __name__ == "__main__":
    main()
