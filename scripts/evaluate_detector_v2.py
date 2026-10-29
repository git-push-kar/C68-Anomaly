"""Phase 1: detector evaluation v2 -- streamed, split-aware, per-run alarm table.

Why a second evaluator instead of extending scripts/evaluate_anomaly_detector.py:

* It streams the raw TEP_Faulty_Testing.csv grouped by (faultNumber,
  simulationRun) and uses every run as-is (960 samples, onset 160). It never
  touches data/processed/fault_values/*.npy or the 500-sample blocking in
  prepare_tep.py, both of which pool two different run shapes.
* It takes the split from data/splits_v2.json and passes the split name
  ("faulty_testing") explicitly to utils.get_fault_onset. It never relies on
  utils.resolve_onset_for_path, which defaults to faulty_training (onset 20)
  for an ambiguous filename.
* It reports per-run alarms, so an FDR number can be traced back to a run.

Metric definitions (kept comparable with the legacy numbers):
  window start  = i * stride, windows built by preprocessing.windowing.to_windows
  window end    = start + window_size - 1
  window label  = 1 if the window CONTAINS any post-onset sample
                  (start + window_size > onset), else 0. The legacy evaluator
                  labelled by start >= onset, which counts a window that
                  already contains fault samples as a negative and then reports
                  its alarm as a pre-onset false alarm (with a 60-sample window
                  and stride 5 that is 12 windows per run, and it is why the
                  published 'mean false_before' is 21 for every strong fault).
  FDR           = fraction of labelled-positive windows above threshold
                  (the legacy window-level definition; NOT run-level detection)
  delay         = (end of the first confirmed alarm window) - onset, in
                  samples. A windowed detector cannot report earlier than the
                  sample that completes its window, so the legacy
                  (start - onset) delay is smaller by up to window_size - 1.
  events        = main.py's aggregator semantics: an event opens when the last
                  `confirm` windows are all anomalous, closes after `separate`
                  consecutive normal windows or at `max_event_windows`.

Non-LSTM detectors produce per-sample statistics; their window score is the
MEAN over the window's samples. This is deliberate: the LSTM score is itself a
mean reconstruction error over the window, so a mean keeps every detector at
the same aggregation level, and it leaves their 1%-per-sample thresholds at
roughly their calibrated operating point. (A max over 60 samples turns a 1%
sample threshold into a ~45% window false-alarm rate; that is an artefact of
the aggregation, not a detector property.) Sample-level FAR is reported too.

Usage:
    python scripts/evaluate_detector_v2.py --protocol A --component val
    python scripts/evaluate_detector_v2.py --protocol A --component test --faults 1 2 --max-runs 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.tep_loader import CANONICAL_NAMES, _normalize_rieth_columns  # noqa: E402
from preprocessing.windowing import to_windows  # noqa: E402
from utils import ensure_dir, get_fault_onset, load_config, save_json  # noqa: E402

logger = logging.getLogger("evaluate_detector_v2")

SPLIT = "faulty_testing"  # resolved explicitly, never inferred from a path


# ----------------------------------------------------------------------------
# pure metric helpers (unit-tested on tiny arrays)
# ----------------------------------------------------------------------------
def window_starts(n_windows: int, stride: int) -> np.ndarray:
    return np.arange(int(n_windows), dtype=np.int64) * int(stride)


def window_labels(starts: np.ndarray, onset: int, window_size: int = 1) -> np.ndarray:
    """1 if the window contains at least one post-onset sample."""
    return (np.asarray(starts, dtype=np.int64) + int(window_size) > int(onset)).astype(int)


def rate_above(scores: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    return float((scores > float(threshold)).mean())


def detection_index(alarm: Sequence[int], confirm: int) -> Optional[int]:
    """First index where `confirm` consecutive windows are alarmed."""
    confirm = max(1, int(confirm))
    flags = np.asarray(alarm, dtype=int)
    if flags.size < confirm:
        return None
    run = np.convolve(flags, np.ones(confirm, dtype=int), mode="valid")
    hits = np.flatnonzero(run == confirm)
    return int(hits[0]) if hits.size else None


def _eligible_mask(starts: np.ndarray, onset: int, window_size: int) -> np.ndarray:
    """Windows that contain data from the fault (can legitimately detect it)."""
    return np.asarray(starts, dtype=np.int64) + int(window_size) > int(onset)


def first_alarm_at_or_after(
    starts: np.ndarray, alarm: Sequence[int], onset: int, confirm: int, window_size: int = 1
) -> Optional[int]:
    """Index of the first confirmed alarm in a window that contains the fault.

    Windows that end before the onset cannot have seen the fault, so their
    alarms are false alarms, never detections.
    """
    starts = np.asarray(starts)
    flags = np.asarray(alarm, dtype=int)
    first = int(np.flatnonzero(_eligible_mask(starts, onset, window_size))[0]) if np.any(
        _eligible_mask(starts, onset, window_size)
    ) else len(flags)
    idx = detection_index(flags[first:], confirm)
    return None if idx is None else first + idx


def detection_delay(
    starts: np.ndarray, alarm: Sequence[int], onset: int, confirm: int, window_size: int = 1
) -> Optional[int]:
    """Samples from onset to the completing sample of the first confirmed alarm."""
    idx = first_alarm_at_or_after(starts, alarm, onset, confirm, window_size)
    if idx is None:
        return None
    return int(np.asarray(starts)[idx]) + int(window_size) - 1 - int(onset)


def false_alarm_before_onset(
    starts: np.ndarray, alarm: Sequence[int], onset: int, confirm: int, window_size: int = 1
) -> bool:
    starts = np.asarray(starts)
    flags = np.asarray(alarm, dtype=int)
    pre = ~_eligible_mask(starts, onset, window_size)
    return detection_index(flags[pre], confirm) is not None


def segment_events(
    alarm: Sequence[int],
    starts: np.ndarray,
    confirm: int = 3,
    separate: int = 20,
    max_event_windows: int = 200,
) -> List[Tuple[int, int]]:
    """Group alarmed windows into events; returns [(open_idx, close_idx), ...].

    Mirrors main.py: a candidate opens on `confirm` consecutive alarms and stays
    open while alarmed, closing after `separate` consecutive quiet windows or
    once `max_event_windows` alarmed windows have accumulated.
    """
    flags = np.asarray(alarm, dtype=int)
    events: List[Tuple[int, int]] = []
    open_idx: Optional[int] = None
    quiet = 0
    last_alarm = 0
    for i in range(len(flags)):
        if open_idx is None:
            if i + 1 >= confirm and flags[i - confirm + 1 : i + 1].all():
                open_idx, quiet, last_alarm = i, 0, i
            continue
        if flags[i]:
            quiet, last_alarm = 0, i
            if last_alarm - open_idx + 1 >= max_event_windows:
                events.append((open_idx, last_alarm))
                open_idx = None
        else:
            quiet += 1
            if quiet >= separate:
                events.append((open_idx, last_alarm))
                open_idx = None
    if open_idx is not None:
        events.append((open_idx, last_alarm))
    return events


def auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.min() == labels.max():
        return None
    if not np.all(np.isfinite(scores)):
        keep = np.isfinite(scores)
        labels, scores = labels[keep], scores[keep]
        if labels.size == 0 or labels.min() == labels.max():
            return None
    return float(roc_auc_score(labels, scores))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.min() == labels.max():
        return None
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    if labels.size == 0 or labels.min() == labels.max():
        return None
    return float(average_precision_score(labels, scores))


def align_samples_to_windows(
    sample_idx: np.ndarray, values: np.ndarray, starts: np.ndarray, window_size: int, reduce: str = "mean"
) -> np.ndarray:
    """Aggregate per-sample statistics onto window starts (mean over the window)."""
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    lo = np.searchsorted(sample_idx, starts, side="left")
    hi = np.searchsorted(sample_idx, starts + window_size, side="left")
    out = np.full(starts.size, np.nan, dtype=np.float64)
    for i in range(starts.size):
        if hi[i] > lo[i]:
            out[i] = np.max(values[lo[i] : hi[i]]) if reduce == "max" else np.mean(values[lo[i] : hi[i]])
    return out


def summarise_series(
    scores: np.ndarray,
    starts: np.ndarray,
    threshold: float,
    onset: int,
    confirm: int,
    separate: int,
    max_event_windows: int,
    window_size: int = 1,
) -> Dict[str, object]:
    """Per-run summary for one detector series."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = window_labels(starts, onset, window_size)
    alarm = (scores > float(threshold)).astype(int)
    events = segment_events(alarm, starts, confirm, separate, max_event_windows)
    delay = detection_delay(starts, alarm, onset, confirm, window_size)
    alarm_idx = first_alarm_at_or_after(starts, alarm, onset, confirm, window_size)
    post = labels == 1
    return {
        "fdr": rate_above(scores[post], threshold),
        "delay": delay,
        "detected": delay is not None,
        "false_before_onset": false_alarm_before_onset(starts, alarm, onset, confirm, window_size),
        "n_events": len(events),
        "n_events_post_onset": sum(1 for o, _ in events if int(starts[o]) + int(window_size) > int(onset)),
        "event_windows": [[int(starts[o]), int(starts[c])] for o, c in events],
        "max_score": float(np.nanmax(scores)) if scores.size else 0.0,
        "alarm_sample": (int(starts[alarm_idx]) + int(window_size) - 1) if alarm_idx is not None else None,
    }


# ----------------------------------------------------------------------------
# data access: stream raw CSVs, never the processed arrays
# ----------------------------------------------------------------------------
def _header_sensor_cols(path: Path) -> Tuple[List[str], List[str]]:
    head = pd.read_csv(path, nrows=1)
    head.columns = [str(c).strip() for c in head.columns]
    sensors = [c for c in head.columns if c.lower().startswith(("xmeas", "xmv"))]
    keys = [c for c in ("faultNumber", "simulationRun", "sample") if c in head.columns]
    return sensors, keys


def iter_runs(
    path: Path,
    faults: Optional[Iterable[int]] = None,
    run_range: Optional[Tuple[int, int]] = None,
    samples_per_run: int = 960,
    chunksize: int = 250_000,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield (fault, run, raw[960,52]) streamed from the CSV.

    Grouping is by (faultNumber, simulationRun); a row count of exactly
    `samples_per_run` is asserted per yielded run, because a partial run would
    silently corrupt every downstream number.
    """
    path = Path(path)
    sensors, keys = _header_sensor_cols(path)
    usecols = keys + sensors
    faults = None if faults is None else {int(f) for f in faults}
    acc: Dict[Tuple[int, int], List[np.ndarray]] = {}
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    for chunk in reader:
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _normalize_rieth_columns(chunk)
        if faults is not None and "faultNumber" in chunk.columns:
            chunk = chunk[chunk["faultNumber"].isin(faults)]
        if run_range is not None and "simulationRun" in chunk.columns:
            chunk = chunk[chunk["simulationRun"].between(int(run_range[0]), int(run_range[1]))]
        if chunk.empty:
            continue
        chunk = chunk.sort_values(["faultNumber", "simulationRun", "sample"])
        for (f_num, run), group in chunk.groupby(["faultNumber", "simulationRun"], sort=True):
            acc.setdefault((int(f_num), int(run)), []).append(
                group[CANONICAL_NAMES].to_numpy(dtype=np.float32)
            )
        ready = sorted(k for k, parts in acc.items() if sum(len(p) for p in parts) >= samples_per_run)
        for key in ready:
            arr = np.concatenate(acc.pop(key), axis=0)
            if arr.shape[0] != samples_per_run:
                raise RuntimeError(f"{path.name} key={key}: expected {samples_per_run} rows, got {arr.shape[0]}")
            yield key[0], key[1], arr
    for key in sorted(acc):
        arr = np.concatenate(acc[key], axis=0)
        if arr.shape[0] != samples_per_run:
            raise RuntimeError(f"{path.name} key={key}: expected {samples_per_run} rows, got {arr.shape[0]}")
        yield key[0], key[1], arr


# ----------------------------------------------------------------------------
# detectors
# ----------------------------------------------------------------------------
class DetectorBundle:
    """LSTM AE + CVA + DPCA + PCA/IF baselines, sharing one scaler."""

    def __init__(self, config: Dict, thresholds: Dict, device: str = "cpu", window_size: int = 60, stride: int = 5):
        import torch

        from anomaly_detection.dynamic.cva import CVAModel
        from anomaly_detection.dynamic.dpca import DPCAModel
        from anomaly_detection.inference import AnomalyDetector, build_autoencoder
        from anomaly_detection.threshold import ThresholdConfig
        from preprocessing.scaler import load_scaler

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.thresholds = thresholds
        self.scaler = load_scaler(config["preprocessing"]["scaler_dir"])

        model_dir = Path(config["anomaly_detector"]["model_dir"])
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        lstm_cfg = config["anomaly_detector"].get("lstm", {})
        dec_ih = state["model_state_dict"].get("decoder.lstm.weight_ih_l0")
        model = build_autoencoder(
            {"anomaly_detector": {"lstm": lstm_cfg}},
            int(state["num_features"]),
            int(state["sequence_length"]),
            use_cell_proj="decoder.cell_proj.weight" in state["model_state_dict"],
            decoder_input_dim=int(dec_ih.shape[1]) if dec_ih is not None else None,
        )
        model.load_state_dict(state["model_state_dict"])
        lstm_thr = float(thresholds["lstm_ae"]["threshold"])
        self.lstm = AnomalyDetector(model, self.scaler, ThresholdConfig(method="manual", threshold=lstm_thr), device=device)
        self.lstm_threshold = lstm_thr

        dyn_dir = model_dir / "dynamic"
        self.cva = CVAModel.load(dyn_dir / "cva_model.npz") if (dyn_dir / "cva_model.npz").exists() else None
        self.dpca = DPCAModel.load(dyn_dir / "dpca_model.npz") if (dyn_dir / "dpca_model.npz").exists() else None
        self.dyn_cfg = config.get("dynamic_detector", {})
        self.dyn_thresholds = thresholds["dynamic"]

    # -- scaling ---------------------------------------------------------
    def scale(self, raw: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(raw, dtype=np.float32)).astype(np.float64)

    # -- LSTM ------------------------------------------------------------
    def score_lstm(self, scaled_run: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        windows = to_windows(scaled_run.astype(np.float32), self.window_size, self.stride)
        scores, _ = self.lstm.score_windows(windows, already_scaled=True)
        return window_starts(len(scores), self.stride), np.asarray(scores, dtype=np.float64)

    # -- CVA / DPCA ------------------------------------------------------
    def score_dynamic_samples(self, scaled_run: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Per-sample dynamic statistics: name -> (sample_index, values)."""
        from anomaly_detection.dynamic.lag_builder import build_augmented_matrix, build_past_future

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if self.cva is not None:
            P, _F, t_idx = build_past_future(scaled_run, self.cva.n_past, self.cva.n_future)
            if P.shape[0]:
                keep = set(self.dyn_cfg.get("cva", {}).get("statistics", ["T2", "Q", "Tr"]))
                for name, vals in self.cva.score(P).items():
                    if name in keep and np.ndim(vals) == 1:
                        out[f"cva_{name}"] = (t_idx, np.asarray(vals, dtype=np.float64))
        if self.dpca is not None:
            Xa, t_idx = build_augmented_matrix(scaled_run, self.dpca.n_lags)
            if Xa.shape[0]:
                keep = set(self.dyn_cfg.get("dpca", {}).get("statistics", ["T2", "SPE"]))
                for name, vals in self.dpca.score(Xa).items():
                    if name in keep and np.ndim(vals) == 1:
                        out[f"dpca_{name}"] = (t_idx, np.asarray(vals, dtype=np.float64))
        return out

    def score_dynamic(self, scaled_run: np.ndarray, starts: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            name: align_samples_to_windows(t_idx, vals, starts, self.window_size)
            for name, (t_idx, vals) in self.score_dynamic_samples(scaled_run).items()
        }

    # -- fusion (runner.fuse, mode=or) -----------------------------------
    def fused_series(self, lstm_scores: np.ndarray, dyn_scores: np.ndarray, dyn_threshold: float) -> np.ndarray:
        norm_lstm = lstm_scores / max(self.lstm_threshold, 1e-9)
        norm_dyn = dyn_scores / max(float(dyn_threshold), 1e-9)
        return np.maximum(norm_lstm, norm_dyn)  # threshold 1.0

    def primary_dynamic_key(self) -> str:
        return f"cva_{self.dyn_cfg.get('cva', {}).get('primary_statistic', 'Q')}"

    def dynamic_threshold(self, key: str) -> float:
        model, stat = key.split("_", 1)
        return float(self.dyn_thresholds[model][stat])


class BaselineBundle:
    """Static PCA (T2/SPE) and Isolation Forest fitted on normal TRAINING data only."""

    def __init__(self, config: Dict, seed: int = 42, n_fit: int = 100_000, target_far: float = 0.01):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest

        from preprocessing.scaler import load_scaler

        self.scaler = load_scaler(config["preprocessing"]["scaler_dir"])
        normal_csv = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv"
        sensors, keys = _header_sensor_cols(normal_csv)
        rng = np.random.default_rng(seed)
        parts: List[np.ndarray] = []
        seen = 0
        for chunk in pd.read_csv(normal_csv, usecols=keys + sensors, chunksize=250_000):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            chunk = _normalize_rieth_columns(chunk)
            if "faultNumber" in chunk.columns:
                chunk = chunk[chunk["faultNumber"] == 0]
            arr = chunk[CANONICAL_NAMES].to_numpy(np.float32)
            parts.append(arr)
            seen += len(arr)
            if seen >= n_fit:
                break
        X = np.concatenate(parts, axis=0)
        if len(X) > n_fit:
            idx = np.sort(rng.choice(len(X), size=n_fit, replace=False))
            X = X[idx]
        Z = self.scaler.transform(X).astype(np.float64)

        self.pca = PCA(n_components=0.9, svd_solver="full", random_state=seed).fit(Z)
        s = self.pca.transform(Z)
        self.pca_mean = s.mean(axis=0)
        self.pca_std = s.std(axis=0) + 1e-12
        t2 = (((s - self.pca_mean) / self.pca_std) ** 2).sum(axis=1)
        resid = Z - self.pca.inverse_transform(s)
        spe = (resid ** 2).sum(axis=1)
        self.t2_threshold = float(np.quantile(t2, 1.0 - target_far))
        self.spe_threshold = float(np.quantile(spe, 1.0 - target_far))

        self.iforest = IsolationForest(n_estimators=100, max_samples="auto", random_state=seed, n_jobs=-1)
        self.iforest.fit(Z)
        if_scores = -self.iforest.score_samples(Z)
        self.if_threshold = float(np.quantile(if_scores, 1.0 - target_far))
        self.n_fit_samples = int(len(Z))
        self.n_pca_components = int(self.pca.n_components_)

    def score_samples(self, scaled_run: np.ndarray) -> Dict[str, np.ndarray]:
        s = self.pca.transform(scaled_run)
        t2 = (((s - self.pca_mean) / self.pca_std) ** 2).sum(axis=1)
        resid = scaled_run - self.pca.inverse_transform(s)
        spe = (resid ** 2).sum(axis=1)
        ifs = -self.iforest.score_samples(scaled_run)
        return {"pca_t2": t2, "pca_spe": spe, "iforest": ifs}

    def score(self, scaled_run: np.ndarray, starts: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
        idx = np.arange(scaled_run.shape[0])
        return {
            name: align_samples_to_windows(idx, vals, starts, window_size)
            for name, vals in self.score_samples(scaled_run).items()
        }


# ----------------------------------------------------------------------------
# split + calibration bookkeeping
# ----------------------------------------------------------------------------
def load_split_component(splits_path: Path, protocol: str, component: str) -> Dict[str, object]:
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    protos = data["protocols"]
    key = {"A": "A_seen_faults_run_disjoint"}.get(protocol, protocol)
    if key not in protos:
        raise KeyError(f"protocol {protocol!r} not found; have {sorted(protos)}")
    block = protos[key]
    container = block.get("splits") if "splits" in block else block.get("folds", {})
    if component == "val":
        comp = container.get("val") or container.get("validation")
    else:
        comp = container.get(component)
    if comp is None:
        raise KeyError(f"component {component!r} not found in {protocol}; have {sorted(container)}")
    faults = sorted({int(f["faultNumber"]) for f in comp["faults"]})
    ranges = {int(f["faultNumber"]): tuple(int(x) for x in f["simulationRunRange"]) for f in comp["faults"]}
    return {
        "protocol": protocol,
        "component": component,
        "source_file": comp["source_file"],
        "faults": faults,
        "ranges": ranges,
        "sha256": comp.get("sha256"),
    }


def verify_far_disjoint(
    far_file: str,
    far_runs: Sequence[int],
    calibration: Dict[str, Sequence[int]],
) -> Dict[str, object]:
    """Prove that no calibration (normal TRAINING) run is used for FAR."""
    far_name = Path(far_file).name
    # Compare by basename: calibration keys carry a different path prefix (or
    # a different separator) on another checkout, and matching on the full
    # string would silently report "no overlap" when the files are the same.
    matching = {name: runs for name, runs in calibration.items() if Path(name).name == far_name}
    calibration_runs = {int(r) for runs in matching.values() for r in runs}
    overlap_files = sorted(matching)
    overlap_runs = sorted({int(r) for r in far_runs} & calibration_runs)
    return {
        "far_source_file": far_file,
        "far_runs": [int(min(far_runs)), int(max(far_runs))] if far_runs else None,
        "n_far_runs": len(far_runs),
        "calibration_sources": {name: len(runs) for name, runs in calibration.items()},
        "overlapping_files": overlap_files,
        "overlapping_runs": overlap_runs,
        "ok": not overlap_files and not overlap_runs,
    }


def calibrate_window_thresholds(
    bundle: "DetectorBundle",
    baselines: Optional["BaselineBundle"],
    config: Dict,
    run_range: Tuple[int, int] = (401, 500),
    target_far: float = 0.01,
) -> Dict[str, float]:
    """Window-level thresholds at `target_far` on the calibration holdout runs.

    The stored artifacts were thresholded per sample (p99); with a 60-sample
    window mean that leaves the dynamic detectors at ~0% window FAR while the
    LSTM sits near 1-2%. These thresholds put every detector at the same
    window operating point, using only FaultFree_Training runs (the dynamic
    threshold holdout), never the Testing runs used to measure FAR.
    """
    path = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv"
    series: Dict[str, List[np.ndarray]] = {}
    for _f, _r, raw in iter_runs(path, faults=None, run_range=run_range, samples_per_run=500):
        scaled = bundle.scale(raw)
        starts, lstm_scores = bundle.score_lstm(scaled)
        row: Dict[str, np.ndarray] = {"lstm_ae": lstm_scores, **bundle.score_dynamic(scaled, starts)}
        if baselines is not None:
            row.update(baselines.score(scaled, starts, bundle.window_size))
        prim = bundle.primary_dynamic_key()
        if prim in row:
            row["fused"] = bundle.fused_series(lstm_scores, row[prim], bundle.dynamic_threshold(prim))
        for name, vals in row.items():
            series.setdefault(name, []).append(np.asarray(vals, dtype=np.float64))
    out = {name: float(np.quantile(np.concatenate(chunks), 1.0 - target_far)) for name, chunks in series.items()}
    logger.info("window-calibrated thresholds over runs %s-%s: %s", run_range[0], run_range[1],
                {k: round(v, 4) for k, v in out.items()})
    return out


def load_calibration_sets(config: Dict, n_fault_free_training_runs: int = 500) -> Dict[str, Sequence[int]]:
    """Which normal runs produced the detector artifacts.

    * scaler + LSTM weights + LSTM threshold: TEP_FaultFree_Training
      (prepare_tep builds windows from that file; detector_split records the
      run-level train/validation/test partition, all inside it).
    * CVA/DPCA model + thresholds: TEP_FaultFree_Training runs 1-400 (fit) and
      401-500 (threshold holdout), per dynamic/fit_metadata.json.
    """
    key = str(Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv")
    runs: set = set()
    man = Path(config["paths"]["processed_data_path"]) / "manifests" / "detector_split.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        for name, count_key in (
            ("train_runs", "num_train_runs"),
            ("validation_runs", "num_validation_runs"),
            ("test_runs", "num_test_runs"),
        ):
            listed = m.get(name) or []
            if listed:
                runs.update(int(r) for r in listed)
            n = int(m.get(count_key) or 0)
            if n and listed and len(listed) < n:
                start = int(min(listed))
                runs.update(range(start, start + n))
    fit_meta = Path(config["anomaly_detector"]["model_dir"]) / "dynamic" / "fit_metadata.json"
    if fit_meta.exists():
        m = json.loads(fit_meta.read_text(encoding="utf-8"))
        # fit_metadata stores the first ten of each partition plus the counts;
        # expand from the count so the FAR check covers all 500 runs.
        for listed_key, count_key in (("fit_runs", "fit_n_runs"), ("holdout_runs", "holdout_n_runs")):
            listed = m.get(listed_key) or []
            if not listed:
                continue
            start = int(min(listed))
            n = int(m.get(count_key) or len(listed))
            runs.update(range(start, start + n))
    if not runs:
        runs.update(range(1, n_fault_free_training_runs + 1))
    return {key: sorted(runs)}


def evaluate_normal_far(
    bundle: DetectorBundle,
    baselines: BaselineBundle,
    config: Dict,
    run_range: Tuple[int, int],
    confirm: int,
    separate: int,
    max_event_windows: int,
    max_runs: Optional[int] = None,
    alt_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    path = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Testing.csv"
    stride = int(config["windowing"]["stride"])
    series: Dict[str, List[np.ndarray]] = {}
    sample_series: Dict[str, List[np.ndarray]] = {}
    run_summaries: Dict[str, List[Dict]] = {}
    runs_seen: List[int] = []
    t0 = time.time()
    for _fault, run, raw in iter_runs(path, faults=None, run_range=run_range, samples_per_run=960):
        if max_runs is not None and len(runs_seen) >= max_runs:
            break
        runs_seen.append(run)
        scaled = bundle.scale(raw)
        starts, lstm_scores = bundle.score_lstm(scaled)
        dyn = bundle.score_dynamic(scaled, starts)
        base = baselines.score(scaled, starts, bundle.window_size) if baselines is not None else {}
        row: Dict[str, np.ndarray] = {"lstm_ae": lstm_scores, **dyn, **base}
        prim = bundle.primary_dynamic_key()
        if prim in row:
            row["fused"] = bundle.fused_series(lstm_scores, row[prim], bundle.dynamic_threshold(prim))
        thresholds = {
            "lstm_ae": bundle.lstm_threshold,
            **{k: bundle.dynamic_threshold(k) for k in ("cva_Q", "cva_T2", "cva_Tr", "dpca_SPE", "dpca_T2") if k in row},
            "fused": 1.0,
        }
        if baselines is not None:
            thresholds.update(
                {"pca_t2": baselines.t2_threshold, "pca_spe": baselines.spe_threshold, "iforest": baselines.if_threshold}
            )
        for name, vals in row.items():
            if name not in thresholds:
                continue
            series.setdefault(name, []).append(np.asarray(vals, dtype=np.float64))
            s = summarise_series(vals, starts, thresholds[name], 0, confirm, separate, max_event_windows, bundle.window_size)
            run_summaries.setdefault(name, []).append({"run": run, "alarm": s["alarm_sample"], "events": s["n_events"]})
        # per-sample FAR, the level at which the dynamic/baseline thresholds were calibrated
        for name, (_t, vals) in bundle.score_dynamic_samples(scaled).items():
            sample_series.setdefault(name, []).append(vals)
        if baselines is not None:
            for name, vals in baselines.score_samples(scaled).items():
                sample_series.setdefault(name, []).append(np.asarray(vals, dtype=np.float64))
    logger.info("FAR: scored %d normal runs in %.1fs", len(runs_seen), time.time() - t0)

    thresholds = {
        "lstm_ae": bundle.lstm_threshold,
        **{k: bundle.dynamic_threshold(k) for k in ("cva_Q", "cva_T2", "cva_Tr", "dpca_SPE", "dpca_T2")},
        "fused": 1.0,
    }
    if baselines is not None:
        thresholds.update({"pca_t2": baselines.t2_threshold, "pca_spe": baselines.spe_threshold, "iforest": baselines.if_threshold})
    out: Dict[str, object] = {"n_runs": len(runs_seen), "runs": [int(min(runs_seen)), int(max(runs_seen))] if runs_seen else None}
    for name, chunks in series.items():
        allv = np.concatenate(chunks)
        alarms = [r["alarm"] is not None for r in run_summaries[name]]
        total_events = sum(r["events"] for r in run_summaries[name])
        out[name] = {
            "threshold": thresholds[name],
            "far_window": rate_above(allv, thresholds[name]),
            "far_run": float(np.mean(alarms)) if alarms else 0.0,
            "events_per_run": float(total_events / max(1, len(runs_seen))),
            "n_windows": int(allv.size),
        }
        if alt_thresholds and name in alt_thresholds:
            out[name]["far_window_at_recal"] = rate_above(allv, alt_thresholds[name])
    for name, chunks in sample_series.items():
        if name not in out:
            continue
        allv = np.concatenate(chunks)
        out[name]["far_sample"] = rate_above(allv, thresholds[name])
        out[name]["n_samples"] = int(allv.size)
    return out


# ----------------------------------------------------------------------------
# main evaluation
# ----------------------------------------------------------------------------
def evaluate_component(
    config: Dict,
    protocol: str,
    component: str,
    splits_path: Path,
    thresholds_path: Path,
    out_dir: Path,
    device: str = "cpu",
    max_faults: Optional[int] = None,
    max_runs: Optional[int] = None,
    skip_baselines: bool = False,
) -> Dict[str, object]:
    thresholds = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
    comp = load_split_component(splits_path, protocol, component)
    faults = comp["faults"][: max_faults or len(comp["faults"])]
    onset = int(get_fault_onset(config, SPLIT))  # explicit split, no path inference
    window_size = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])
    ev = config.get("events", {})
    confirm = int(ev.get("consecutive_windows_to_confirm", 3))
    separate = int(ev.get("min_separation_windows", 20))
    max_event_windows = int(ev.get("max_event_windows", 200))

    logger.info("protocol=%s component=%s faults=%d onset=%d ws=%d stride=%d device=%s",
                protocol, component, len(faults), onset, window_size, stride, device)
    bundle = DetectorBundle(config, thresholds, device=device, window_size=window_size, stride=stride)
    if bundle.cva is None or bundle.dpca is None:
        raise RuntimeError("CVA/DPCA artifacts missing; evaluation would be partial")

    t_fit = time.time()
    baselines = None if skip_baselines else BaselineBundle(config)
    fit_seconds = time.time() - t_fit
    logger.info(
        "baselines fitted in %.1fs (fit samples=%s, PCA components=%s)",
        fit_seconds,
        getattr(baselines, "n_fit_samples", None),
        getattr(baselines, "n_pca_components", None),
    )

    dyn_keys = ["cva_Q", "cva_T2", "cva_Tr", "dpca_SPE", "dpca_T2"]
    th: Dict[str, float] = {"lstm_ae": bundle.lstm_threshold, "fused": 1.0}
    for k in dyn_keys:
        th[k] = bundle.dynamic_threshold(k)
    if baselines is not None:
        th.update({"pca_t2": baselines.t2_threshold, "pca_spe": baselines.spe_threshold, "iforest": baselines.if_threshold})

    runs_path = Path(comp["source_file"])
    if not runs_path.is_absolute():
        runs_path = ROOT / runs_path
    fault_scores: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {f: {} for f in faults}
    fault_labels: Dict[int, List[np.ndarray]] = {f: [] for f in faults}
    rows_by_fault: Dict[int, List[Dict[str, object]]] = {f: [] for f in faults}
    per_run_rows: List[Dict[str, object]] = []
    per_fault_runs: Dict[int, int] = {f: 0 for f in faults}
    t0 = time.time()
    n_runs = 0
    # One pass over the CSV for all faults; re-scanning per fault would read
    # 3.6 GB each time. Membership is still decided per (fault, run) from the
    # split file, so a fold with different ranges per fault is handled.
    lo_all = min(r[0] for r in comp["ranges"].values())
    hi_all = max(r[1] for r in comp["ranges"].values())
    for f_num, run, raw in iter_runs(runs_path, faults=faults, run_range=(lo_all, hi_all), samples_per_run=960):
        fault = int(f_num)
        if fault not in comp["ranges"]:
            continue
        lo, hi = comp["ranges"][fault]
        if not (lo <= run <= hi):
            continue
        if max_runs is not None and per_fault_runs[fault] >= max_runs:
            continue
        per_fault_runs[fault] += 1
        n_runs += 1
        scaled = bundle.scale(raw)
        starts, lstm_scores = bundle.score_lstm(scaled)
        labels = window_labels(starts, onset, window_size)
        row: Dict[str, np.ndarray] = {"lstm_ae": lstm_scores}
        row.update(bundle.score_dynamic(scaled, starts))
        if baselines is not None:
            row.update(baselines.score(scaled, starts, window_size))
        prim = bundle.primary_dynamic_key()
        row["fused"] = bundle.fused_series(lstm_scores, row[prim], bundle.dynamic_threshold(prim))

        for name, vals in row.items():
            bucket = fault_scores[fault].setdefault(name, {"scores": [], "starts": []})
            bucket["scores"].append(np.asarray(vals, dtype=np.float64))
            bucket["starts"].append(starts)
        fault_labels[fault].append(labels)

        summaries = {
            name: summarise_series(
                vals, starts, th[name], onset, confirm, separate, max_event_windows, window_size
            )
            for name, vals in row.items()
        }
        fused = summaries["fused"]
        fired = "none"
        if fused["alarm_sample"] is not None:
            idx = first_alarm_at_or_after(starts, (row["fused"] > 1.0).astype(int), onset, confirm, window_size)
            fired_lstm = bool(idx is not None and lstm_scores[idx] > bundle.lstm_threshold)
            fired_dyn = bool(idx is not None and row[prim][idx] > bundle.dynamic_threshold(prim))
            fired = "both" if fired_lstm and fired_dyn else ("lstm_ae" if fired_lstm else "dynamic")
        event_id = f"f{fault:02d}-r{run:03d}-e01" if fused["event_windows"] else None
        row_record = {
            "source_file": runs_path.name,
            "fault": fault,
            "simulationRun": run,
            "onset": onset,
            "alarm_sample": fused["alarm_sample"],
            "alarm_window": ((fused["alarm_sample"] - window_size + 1) // stride)
            if fused["alarm_sample"] is not None
            else None,
            "delay_samples": fused["delay"],
            "detector": fired,
            "n_events": fused["n_events"],
            "event_id": event_id,
            "max_fused": fused["max_score"],
            "fdr_fused": fused["fdr"],
            "max_lstm": summaries["lstm_ae"]["max_score"],
            "alarm_lstm": summaries["lstm_ae"]["alarm_sample"],
            "alarm_cva_Q": summaries.get("cva_Q", {}).get("alarm_sample"),
            "alarm_dpca_SPE": summaries.get("dpca_SPE", {}).get("alarm_sample"),
            "false_before_onset": fused["false_before_onset"],
        }
        per_run_rows.append(row_record)
        rows_by_fault[fault].append(row_record)
        if n_runs % 200 == 0:
            logger.info("  scored %d runs (fault %d, run %d) in %.1fs", n_runs, fault, run, time.time() - t0)
    elapsed = time.time() - t0
    logger.info("scored %d faulty runs in %.1fs (%.2f s/run)", n_runs, elapsed, elapsed / max(1, n_runs))

    per_fault: Dict[str, Dict] = {}
    for fault in faults:
        k = per_fault_runs[fault]
        rows = rows_by_fault[fault]
        labels = np.concatenate(fault_labels[fault]) if k else np.array([])
        delays = [r["delay_samples"] for r in rows if r.get("delay_samples") is not None and r["delay_samples"] >= 0]
        dets: Dict[str, object] = {}
        for name, buckets in fault_scores[fault].items():
            scores = np.concatenate(buckets["scores"])
            if scores.size == 0:
                continue
            post = labels == 1
            dets[name] = {
                "threshold": th[name],
                "fdr": rate_above(scores[post], th[name]),
                "run_detection_rate": len(delays) / max(1, k),
                "delay_mean": float(np.mean(delays)) if delays else None,
                "delay_median": float(np.median(delays)) if delays else None,
                "false_before_onset_rate": sum(1 for r in rows if r.get("false_before_onset")) / max(1, k),
                "auroc": auroc(labels, scores),
                "pr_auc": pr_auc(labels, scores),
                "n_windows": int(scores.size),
                "n_runs": k,
            }
        dets["events"] = {
            "events_per_run": float(np.mean([r["n_events"] for r in rows])) if rows else 0.0,
            "runs_with_event": int(sum(1 for r in rows if r["n_events"] > 0)),
        }
        per_fault[str(fault)] = dets

    # Secondary operating point: all detectors at 1% window FAR, thresholds set
    # on the calibration holdout (Training 401-500), not on the FAR set.
    win_thr = calibrate_window_thresholds(bundle, baselines, config)
    per_fault_recal: Dict[str, Dict] = {}
    for fault in faults:
        rows = rows_by_fault[fault]
        labels = np.concatenate(fault_labels[fault]) if rows else np.array([])
        if labels.size == 0:
            continue
        rec: Dict[str, object] = {}
        for name, buckets in fault_scores[fault].items():
            scores = np.concatenate(buckets["scores"])
            thr = win_thr.get(name)
            if thr is None:
                continue
            rec[name] = {
                "threshold": thr,
                "fdr": rate_above(scores[labels == 1], thr),
                "auroc": auroc(labels, scores),
                "pr_auc": pr_auc(labels, scores),
            }
        per_fault_recal[str(fault)] = rec
        # fused under the recalibrated detector thresholds
        prim = bundle.primary_dynamic_key()
        if prim in fault_scores[fault] and "lstm_ae" in fault_scores[fault] and "fused" in win_thr:
            lstm_all = np.concatenate(fault_scores[fault]["lstm_ae"]["scores"])
            dyn_all = np.concatenate(fault_scores[fault][prim]["scores"])
            fused_norm = np.maximum(
                lstm_all / max(win_thr["lstm_ae"], 1e-9), dyn_all / max(win_thr[prim], 1e-9)
            )
            rec["fused"] = {
                "threshold": 1.0,
                "fdr": rate_above(fused_norm[labels == 1], 1.0),
                "auroc": auroc(labels, fused_norm),
                "pr_auc": pr_auc(labels, fused_norm),
                "note": "OR of lstm_ae and " + prim + " at their 1% window thresholds",
            }

    far_range = (1, 500)
    far = evaluate_normal_far(
        bundle, baselines, config, far_range, confirm, separate, max_event_windows,
        max_runs=max_runs, alt_thresholds=win_thr,
    )
    calibration = load_calibration_sets(config)
    disjoint = verify_far_disjoint(str(Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Testing.csv"), list(range(far_range[0], far_range[1] + 1)), calibration)

    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": protocol,
        "component": component,
        "split": SPLIT,
        "onset": onset,
        "window_size": window_size,
        "stride": stride,
        "confirm": confirm,
        "min_separation": separate,
        "max_event_windows": max_event_windows,
        "source_file": str(comp["source_file"]),
        "component_sha256": comp["sha256"],
        "devices": {"detector": device},
        "thresholds": th,
        "threshold_source": str(thresholds_path),
        "n_runs_scored": n_runs,
        "runtime_seconds": elapsed,
        "per_fault": per_fault,
        "per_fault_window_calibrated": per_fault_recal,
        "window_calibrated_thresholds": win_thr,
        "window_calibration": {
            "source": str(Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv"),
            "run_range": [401, 500],
            "target_far": 0.01,
            "note": "Quantile 0.99 of per-detector window scores on the dynamic threshold holdout runs. Used only for the secondary operating point; the headline numbers use the stored artifact thresholds.",
        },
        "far": far,
        "far_disjoint_from_calibration": disjoint,
        "baselines": None
        if baselines is None
        else {
            "fit_source": str(Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv"),
            "n_fit_samples": baselines.n_fit_samples,
            "pca_components": baselines.n_pca_components,
            "target_far": 0.01,
        },
    }

    ensure_dir(out_dir)
    tag = f"{protocol}_{component}"
    save_json(Path(out_dir) / f"{tag}.json", result)
    pd.DataFrame(per_run_rows).to_csv(Path(out_dir) / f"{tag}_per_run.csv", index=False)
    (Path(out_dir) / f"{tag}.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: Dict) -> str:
    lines = [
        f"# Detector evaluation v2 - protocol {result['protocol']} / {result['component']}",
        "",
        f"Source: `{result['source_file']}` ({result['n_runs_scored']} runs, onset {result['onset']}, "
        f"window {result['window_size']}/stride {result['stride']})",
        f"Split label: `{result['split']}` (explicit, not path-inferred)",
        "",
        "FDR is the legacy definition: fraction of post-onset windows above threshold.",
        "",
        "| Fault | Runs | LSTM FDR | CVA Q FDR | DPCA SPE FDR | Fused FDR | Fused det. rate | Fused delay (mean, samples) | Fused AUROC | Fused PR-AUC | Events/run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fid, dets in result["per_fault"].items():
        def fmt(name, key, pct=True, nd=2):
            v = dets.get(name, {}).get(key)
            if v is None:
                return "-"
            return f"{v*100:.1f}%" if pct else f"{v:.{nd}f}"

        lines.append(
            "| {f} | {n} | {l} | {c} | {d} | {fu} | {dr} | {dl} | {au} | {pr} | {ev} |".format(
                f=fid,
                n=dets.get("fused", {}).get("n_runs", 0),
                l=fmt("lstm_ae", "fdr"),
                c=fmt("cva_Q", "fdr"),
                d=fmt("dpca_SPE", "fdr"),
                fu=fmt("fused", "fdr"),
                dr=fmt("fused", "run_detection_rate"),
                dl=fmt("fused", "delay_mean", pct=False, nd=1),
                au=fmt("fused", "auroc", pct=False, nd=3),
                pr=fmt("fused", "pr_auc", pct=False, nd=3),
                ev=f"{dets.get('events', {}).get('events_per_run', 0):.2f}",
            )
        )
    far = result["far"]
    lines += [
        "",
        "## False alarm rate on fault-free Testing runs (never used for calibration)",
        "",
        "| Detector | Threshold (artifact) | FAR window | FAR sample | FAR at 1% window recal | FAR (run) | Events/run |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, vals in far.items():
        if not isinstance(vals, dict):
            continue
        fs = vals.get("far_sample")
        fr = vals.get("far_window_at_recal")
        lines.append(
            "| {n} | {t:.4g} | {fw:.2f}% | {fs} | {fr} | {frun:.1f}% | {ev:.2f} |".format(
                n=name,
                t=vals["threshold"],
                fw=vals["far_window"] * 100,
                fs=f"{fs*100:.2f}%" if fs is not None else "-",
                fr=f"{fr*100:.2f}%" if fr is not None else "-",
                frun=vals["far_run"] * 100,
                ev=vals["events_per_run"],
            )
        )
    lines += ["", "## Fused FDR at the 1% window operating point", "", "| Fault | Fused FDR (artifact thr) | Fused FDR (window-calibrated thr) |", "|---|---|---|"]
    for fid in result["per_fault"]:
        art = result["per_fault"][fid].get("fused", {}).get("fdr")
        rec = result.get("per_fault_window_calibrated", {}).get(fid, {}).get("fused", {}).get("fdr")
        lines.append(f"| {fid} | {art*100:.1f}% | {rec*100:.1f}% |" if rec is not None else f"| {fid} | {art*100:.1f}% | - |")
    lines += ["", f"Calibration/FAR disjointness: `{json.dumps(result['far_disjoint_from_calibration'])}`", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config_a5000.yaml")
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--component", default="val", choices=["val", "test", "test_seen"])
    ap.add_argument("--splits", default="data/splits_v2.json")
    ap.add_argument("--thresholds", default="configs/detector_thresholds_v2.json")
    ap.add_argument("--out-dir", default="outputs/evaluation/detector_v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-faults", type=int, default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    t0 = time.time()
    result = evaluate_component(
        config,
        args.protocol,
        args.component,
        Path(args.splits),
        Path(args.thresholds),
        Path(args.out_dir),
        device=args.device,
        max_faults=args.max_faults,
        max_runs=args.max_runs,
        skip_baselines=args.skip_baselines,
    )
    print(f"\nwrote {args.out_dir}/{args.protocol}_{args.component}.json in {time.time()-t0:.1f}s")
    for fid, dets in result["per_fault"].items():
        f = dets.get("fused", {})
        print(
            f"fault {int(fid):2d}: fused FDR={f.get('fdr', 0)*100:5.1f}% | det={f.get('run_detection_rate', 0)*100:5.1f}% | "
            f"delay={f.get('delay_mean')} | auroc={f.get('auroc')}"
        )


if __name__ == "__main__":
    main()
