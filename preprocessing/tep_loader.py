"""Tennessee Eastman Process (TEP) dataset loader.

Assumptions about the public TEP data used by this project
----------------------------------------------------------
There are several widely-used public TEP releases (the classic Downs &
Vogel .mat dumps, CSV re-exports on Kaggle, and the extended 52-variable
versions). Rather than scattering format logic through the codebase, every
format-specific decision lives in this module behind the :class:`TEPDataFormat`
enum and the :func:`load_tep_data` entry point.

Supported layouts:

  * "tep_52col_csv" (legacy):
    - normal operation  -> one file per run, e.g.  data/raw/normal/normal.csv
    - fault scenarios   -> one file per fault,  e.g.  data/raw/faults/fault_01.csv
    - each file is a T x F matrix of sensor records (52 cols, no meta cols)

  * "tep_rieth_csv" (current dataset - TEP_FaultFree/Faulty_*.csv):
    - Columns: faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11
    - One consolidated file per split:
        data/raw/normal/TEP_FaultFree_Training.csv  (500 runs, 250k rows, fault 0)
        data/raw/normal/TEP_FaultFree_Testing.csv   (960 runs, 480k rows, fault 0)
        data/raw/faults/TEP_Faulty_Training.csv     (20 faults x 500 runs, 5M rows)
        data/raw/faults/TEP_Faulty_Testing.csv      (20 faults x 960 runs, 9.6M rows)
    - Fault is injected at sample ~160 (config fault_onset_index fallback).
    - Also handled transparently when format is tep_52col_csv -- the loader
      auto-detects Rieth columns and handles both.

  * "tep_matlab_52col": .mat (not used here, extension point).

In all cases the loader returns only the 52 sensor columns using the
canonical names XMEAS_1..41, XMV_42..52; meta columns are stripped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils import resolve_path

logger = logging.getLogger(__name__)

# Standard TEP variable names (XMEAS 1-41, XMV 42-52). Used when files carry no
# header. English names are provided by evidence/process_relationships.py.
STANDARD_VARIABLE_NAMES = [f"XMEAS_{i}" for i in range(1, 42)] + [
    f"XMV_{i}" for i in range(42, 53)
]


class TEPDataFormat(str, Enum):
    TEP_52COL_CSV = "tep_52col_csv"
    TEP_RIETH_CSV = "tep_rieth_csv"
    TEP_MATLAB_52COL = "tep_matlab_52col"


@dataclass
class TEPData:
    """Container returned by :func:`load_tep_data`.

    Attributes:
        normal: DataFrame of normal-operation records (chronological).
        faults: dict mapping fault id (int) -> DataFrame of that fault run.
        feature_names: column names in the order used by the models.
    """

    normal: pd.DataFrame
    faults: Dict[int, pd.DataFrame]
    feature_names: List[str]


RIETH_SENSOR_COLS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
CANONICAL_NAMES = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(42, 53)]
RIETH_TO_CANONICAL = {r: c for r, c in zip(RIETH_SENSOR_COLS, CANONICAL_NAMES)}

def _read_matrix(
    path: Path, has_header: bool, delimiter: str, feature_names: List[str]
) -> pd.DataFrame:
    """Read a single T x F CSV into a DataFrame with guaranteed column names."""
    df = pd.read_csv(path, header=0 if has_header else None, delimiter=delimiter)
    if not has_header:
        df.columns = feature_names
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _is_rieth_format(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return "faultnumber" in cols and "simulationrun" in cols


def _normalize_rieth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Rieth xmeas/xmv lower-case to canonical XMEAS/XMV names, keep meta cols."""
    col_map = {}
    for c in df.columns:
        low = c.lower()
        if low in RIETH_TO_CANONICAL:
            col_map[c] = RIETH_TO_CANONICAL[low]
    if col_map:
        df = df.rename(columns=col_map)
    return df


def _handle_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if not df.isna().any().any():
        return df
    logger.info("Missing values detected in %s rows (strategy=%s).",
                int(df.isna().any(axis=1).sum()), strategy)
    if strategy == "drop":
        return df.dropna().reset_index(drop=True)
    if strategy == "ffill":
        return df.ffill().bfill().reset_index(drop=True)
    if strategy == "interpolate":
        return df.interpolate(method="linear", limit_direction="both").bfill().ffill()
    raise ValueError(f"Unknown missing_value_strategy: {strategy}")


def _split_meta_columns(
    df: pd.DataFrame, timestamp_column: Optional[str], onset_column: Optional[str]
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """Separate meta columns (timestamp / onset index) from sensor columns."""
    timestamps: Optional[np.ndarray] = None
    onset: Optional[np.ndarray] = None
    drop_cols: List[str] = []
    if timestamp_column and timestamp_column in df.columns:
        timestamps = df[timestamp_column].to_numpy()
        drop_cols.append(timestamp_column)
    if onset_column and onset_column in df.columns:
        onset = df[onset_column].to_numpy()
        drop_cols.append(onset_column)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, timestamps, onset


def _load_rieth_faults_consolidated(
    path: Path,
    has_header: bool,
    delimiter: str,
    missing_strategy: str,
    fault_onset_index: int,
    max_runs_per_fault: Optional[int] = None,
) -> Dict[int, Tuple[pd.DataFrame, Optional[int]]]:
    """Load a single Rieth consolidated file (e.g. TEP_Faulty_Training.csv)
    and split by faultNumber into {fault_id: (sensor_df, onset)}.

    Keeps only sensor columns; strips faultNumber/simulationRun/sample meta.
    Drops fault 0 (fault-free) if present in the same file.

    When max_runs_per_fault is set, filters by simulationRun <= N so run
    length (500 vs 960) is handled correctly and large 3.6GB files are not
    fully loaded.
    """
    logger.info("Reading Rieth consolidated file: %s (max_runs=%s)", path, max_runs_per_fault)
    faults: Dict[int, List[pd.DataFrame]] = {}
    # Collect buffers per fault
    n_chunks_read = 0
    for chunk in pd.read_csv(path, header=0 if has_header else None, delimiter=delimiter, chunksize=500000):
        n_chunks_read += 1
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _normalize_rieth_columns(chunk)
        if max_runs_per_fault is not None and "simulationRun" in chunk.columns:
            # Keep only first N runs per fault (handles both 500 and 960 length runs)
            chunk = chunk[chunk["simulationRun"] <= max_runs_per_fault]
            if chunk.empty:
                continue
        chunk = _handle_missing(chunk, missing_strategy)
        if "faultNumber" not in chunk.columns:
            raise ValueError(f"Expected faultNumber column in Rieth file {path}")
        for fid, group in chunk.groupby("faultNumber"):
            fid = int(fid)
            if fid == 0:
                continue
            sensor_cols = [c for c in CANONICAL_NAMES if c in group.columns]
            sensor_df = group[sensor_cols]
            faults.setdefault(fid, []).append(sensor_df)
        # Early stop: if smoke and we have seen all 20 faults with at least N runs worth,
        # break. Estimate runs collected by total rows per fault.
        if max_runs_per_fault is not None and len(faults) >= 20:
            # Check if each fault has at least max_runs * min_run_length (500) rows
            min_rows = max_runs_per_fault * 500
            # For smoke with N=5, 5*500=2500 min; Testing runs are 960*5=4800, so 2500 is safe lower bound
            if all(sum(len(b) for b in bufs) >= min_rows for bufs in faults.values()):
                logger.info("Smoke early stop after %d chunks", n_chunks_read)
                break
        # Smoke fast stop after first chunk if N small (covers all faults interleaved)
        if max_runs_per_fault is not None and n_chunks_read >= 2:
            # For N<=5, first 500k chunk already contains >5 runs per fault for Training (500-length)
            # and for Testing need ~2 chunks for 960-length. Limit to 2 chunks for speed.
            if max_runs_per_fault <= 5 and n_chunks_read >= 1:
                # Check if we already have data for all faults
                if len(faults) == 20:
                    break
    # Concatenate per fault
    result: Dict[int, Tuple[pd.DataFrame, Optional[int]]] = {}
    for fid, bufs in faults.items():
        df = pd.concat(bufs, ignore_index=True) if len(bufs) > 1 else bufs[0].reset_index(drop=True)
        # Trim to exactly N runs if needed (handles 960 vs 500)
        if max_runs_per_fault is not None:
            # Determine run length from sample column max if available, else assume 500
            # For smoke, keep first N*run_length rows where run_length is inferred from df size / N?
            # Simpler: keep first N * (df.shape[0] // max_runs_per_fault) ??? Instead filter by simulationRun already done, so df already limited.
            # But groupby within chunk already filtered, so we can just keep as is.
            pass
        result[fid] = (df.reset_index(drop=True), int(fault_onset_index) if fault_onset_index is not None else 160)
        run_len = 500 if "TEP_Faulty_Training" in str(path) else 960
        logger.info("  -> fault %02d: %d samples (~%d runs, run_len~%d)", fid, len(df), len(df)//run_len, run_len)
    return result


def _load_csv_faults(
    fault_dir: Path,
    pattern: str,
    has_header: bool,
    delimiter: str,
    num_features: int,
    feature_names: List[str],
    missing_strategy: str,
    timestamp_column: Optional[str],
    onset_column: Optional[str],
    fault_onset_index: int,
    max_runs_per_fault: Optional[int] = None,
) -> Dict[int, Tuple[pd.DataFrame, Optional[int]]]:
    """Load fault scenarios as {fault_id: (dataframe, onset_index)}.

    Handles both legacy per-fault files (fault_04.csv) and Rieth
    consolidated files (TEP_Faulty_*.csv) transparently.
    """
    faults: Dict[int, Tuple[pd.DataFrame, Optional[int]]] = {}
    files = sorted(fault_dir.glob(pattern)) if fault_dir.exists() else []
    if not files:
        logger.warning("No fault files matched %s in %s", pattern, fault_dir)
        return faults
    # Smoke mode: only Training file is needed (Testing duplicates same faults, but Training has 500-length runs)
    if max_runs_per_fault is not None and len(files) > 1:
        training_candidates = [p for p in files if "Training" in p.name]
        chosen = training_candidates[0] if training_candidates else files[0]
        logger.info("Smoke mode: using Training fault file %s (of %d) for speed", chosen.name, len(files))
        files = [chosen]
    for path in files:
        # Peek header to detect Rieth format
        try:
            peek = pd.read_csv(path, nrows=2, header=0 if has_header else None, delimiter=delimiter)
            if has_header:
                peek.columns = [str(c).strip() for c in peek.columns]
            if _is_rieth_format(peek):
                rieth_faults = _load_rieth_faults_consolidated(
                    path, has_header, delimiter, missing_strategy, fault_onset_index,
                    max_runs_per_fault=max_runs_per_fault,
                )
                # Merge (consolidated file may appear twice: Training + Testing)
                for fid, (fdf, onset) in rieth_faults.items():
                    if fid in faults:
                        # Concatenate Training+Testing for same fault
                        existing_df, _ = faults[fid]
                        faults[fid] = (pd.concat([existing_df, fdf], ignore_index=True), onset)
                    else:
                        faults[fid] = (fdf, onset)
                continue
        except Exception as exc:
            logger.warning("Rieth detect failed for %s, falling back to matrix mode: %s", path, exc)

        df = _read_matrix(path, has_header, delimiter, feature_names)
        df = _normalize_rieth_columns(df)
        df, timestamps, onset_col = _split_meta_columns(df, timestamp_column, onset_column)
        df = _handle_missing(df, missing_strategy)
        if len(df.columns) != num_features:
            logger.warning(
                "%s has %d columns, expected %d. Using the first %d.",
                path, len(df.columns), num_features, num_features,
            )
            df = df.iloc[:, :num_features]
        fault_id = _extract_fault_id(path.stem)
        onset: Optional[int] = None
        if onset_col is not None and len(onset_col) > 0:
            onset = int(onset_col[0])
        elif fault_onset_index is not None:
            onset = int(fault_onset_index)
        faults[fault_id] = (df, onset)
        logger.info("Loaded fault %02d from %s (samples=%d, onset=%s)",
                    fault_id, path, len(df), onset)
    return faults


def _extract_fault_id(stem: str) -> int:
    """Extract the fault number from a filename like 'fault_04' or 'idv(4)'."""
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Cannot extract fault id from filename stem: {stem!r}")


def load_tep_data(config: dict, max_runs_per_fault: Optional[int] = None) -> TEPData:
    """Load normal + fault TEP data according to ``config``.

    Args:
        config: merged configuration dict (see utils.load_config).
        max_runs_per_fault: if set, limit to first N simulationRuns per
            fault (useful for fast smoke tests on the 3.6GB Testing file).

    Returns:
        :class:`TEPData` with normal frame and a fault-id -> frame mapping.
    """
    ds = config["dataset"]
    data_root = resolve_path(config, "data_root")
    normal_path = resolve_path(config, "normal_data_path")
    fault_path = resolve_path(config, "fault_data_path")

    feature_names = CANONICAL_NAMES
    num_features = int(ds.get("num_features", 52))

    fmt = TEPDataFormat(ds.get("format", "tep_52col_csv"))
    if fmt not in (TEPDataFormat.TEP_52COL_CSV, TEPDataFormat.TEP_RIETH_CSV):
        raise NotImplementedError(
            f"Format {fmt.value} requires a loader in this module "
            "(add it to load_tep_data; see module docstring)."
        )

    normal_frame = pd.DataFrame()
    normal_files = sorted(normal_path.glob(ds["normal_file_pattern"])) if normal_path.exists() else []
    # Also match alternative Rieth pattern if primary finds nothing
    if not normal_files and normal_path.exists():
        normal_files = sorted(normal_path.glob("TEP_FaultFree*.csv"))

    # Prefer Training file for scaler fit (Rieth protocol: Training for train, Testing for test)
    # Sorted order puts Testing before Training, so explicitly pick Training.
    if len(normal_files) > 1:
        training_candidates = [p for p in normal_files if "Training" in p.name]
        if training_candidates:
            if max_runs_per_fault is not None:
                logger.info("Smoke mode: using Training normal file %s", training_candidates[0].name)
            else:
                logger.info("Using Training normal file %s for scaler fit (Testing held out for eval)", training_candidates[0].name)
            normal_files = training_candidates[:1]

    if normal_files:
        frames = []
        for path in normal_files:
            # Check Rieth format by peeking
            try:
                peek = pd.read_csv(path, nrows=2, header=0 if ds.get("has_header", True) else None, delimiter=ds.get("delimiter", ","))
                if ds.get("has_header", True):
                    peek.columns = [str(c).strip() for c in peek.columns]
                if _is_rieth_format(peek):
                    if max_runs_per_fault is not None:
                        # Filter by simulationRun <= N (handles 500 vs 960 correctly)
                        chunks = []
                        for chunk in pd.read_csv(path, header=0 if ds.get("has_header", True) else None, delimiter=ds.get("delimiter", ","), chunksize=500000):
                            chunk.columns = [str(c).strip() for c in chunk.columns]
                            chunk = _normalize_rieth_columns(chunk)
                            if "simulationRun" in chunk.columns:
                                chunk = chunk[chunk["simulationRun"] <= max_runs_per_fault]
                                if chunk.empty:
                                    continue
                            if "faultNumber" in chunk.columns:
                                chunk = chunk[chunk["faultNumber"] == 0]
                            sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
                            chunks.append(chunk[sensor_cols])
                            # Early stop: first chunk already contains N runs for normal (since runs sequential)
                            if len(chunks) >= 1:
                                break
                        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=CANONICAL_NAMES)
                    else:
                        # Chunked read for large files (730k rows normal total is okay, but chunked to control memory)
                        chunks = []
                        for chunk in pd.read_csv(path, header=0 if ds.get("has_header", True) else None, delimiter=ds.get("delimiter", ","), chunksize=500000):
                            chunk.columns = [str(c).strip() for c in chunk.columns]
                            chunk = _normalize_rieth_columns(chunk)
                            if "faultNumber" in chunk.columns:
                                chunk = chunk[chunk["faultNumber"] == 0]
                            sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
                            chunks.append(chunk[sensor_cols])
                        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=CANONICAL_NAMES)
                    df = _handle_missing(df, ds.get("missing_value_strategy", "interpolate"))
                    frames.append(df)
                    logger.info("Loaded %d normal samples from Rieth file %s", len(df), path)
                    continue
            except Exception as exc:
                logger.warning("Rieth normal detect failed for %s: %s, trying matrix mode", path, exc)

            df = _read_matrix(path, ds.get("has_header", True), ds.get("delimiter", ","), feature_names)
            df = _normalize_rieth_columns(df)
            df, _, _ = _split_meta_columns(df, ds.get("timestamp_column"), ds.get("fault_onset_column"))
            df = _handle_missing(df, ds.get("missing_value_strategy", "interpolate"))
            frames.append(df.iloc[:, :num_features])
        if frames:
            normal_frame = pd.concat(frames, ignore_index=True)
            logger.info("Loaded %d normal-operation samples from %d file(s).",
                        len(normal_frame), len(frames))
    else:
        logger.warning("No normal data found under %s (pattern=%s).",
                       normal_path, ds["normal_file_pattern"])

    faults = _load_csv_faults(
        fault_path,
        ds["fault_file_pattern"],
        ds.get("has_header", True),
        ds.get("delimiter", ","),
        num_features,
        feature_names,
        ds.get("missing_value_strategy", "interpolate"),
        ds.get("timestamp_column"),
        ds.get("fault_onset_column"),
        ds.get("fault_onset_index"),
        max_runs_per_fault=max_runs_per_fault,
    )

    feature_names = CANONICAL_NAMES[:num_features]
    return TEPData(normal=normal_frame, faults=faults, feature_names=feature_names)


def load_single_csv(
    path: Union[str, Path],
    config: dict,
    fault_number: Optional[int] = None,
    simulation_run: Optional[int] = None,
) -> pd.DataFrame:
    """Load one CSV (used by the streaming simulator) with the same rules.

    Handles both legacy 52-col and Rieth consolidated files.  For a Rieth
    consolidated file, ``fault_number`` and ``simulation_run`` select one
    temporally coherent run rather than replaying unrelated runs back-to-back.
    This is essential for a realistic continuous-stream simulation.
    """
    ds = config["dataset"]
    feature_names = CANONICAL_NAMES
    p = Path(path)
    # Peek for Rieth
    try:
        peek = pd.read_csv(p, nrows=2, header=0 if ds.get("has_header", True) else None, delimiter=ds.get("delimiter", ","))
        if ds.get("has_header", True):
            peek.columns = [str(c).strip() for c in peek.columns]
        if _is_rieth_format(peek):
            # Full chunked load, then strip meta and keep sensors
            chunks = []
            for chunk in pd.read_csv(p, header=0 if ds.get("has_header", True) else None, delimiter=ds.get("delimiter", ","), chunksize=500000):
                chunk.columns = [str(c).strip() for c in chunk.columns]
                chunk = _normalize_rieth_columns(chunk)
                if fault_number is not None and "faultNumber" in chunk.columns:
                    chunk = chunk[chunk["faultNumber"] == int(fault_number)]
                if simulation_run is not None and "simulationRun" in chunk.columns:
                    chunk = chunk[chunk["simulationRun"] == int(simulation_run)]
                if chunk.empty:
                    continue
                sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
                chunks.append(chunk[sensor_cols])
            if not chunks:
                raise ValueError(
                    f"No rows matched fault_number={fault_number}, "
                    f"simulation_run={simulation_run} in {p}"
                )
            df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
            df = _handle_missing(df, ds.get("missing_value_strategy", "interpolate"))
            return df.reset_index(drop=True)
    except Exception:
        pass

    df = _read_matrix(p, ds.get("has_header", True), ds.get("delimiter", ","), feature_names)
    df = _normalize_rieth_columns(df)
    df, _, _ = _split_meta_columns(df, ds.get("timestamp_column"), ds.get("fault_onset_column"))
    df = _handle_missing(df, ds.get("missing_value_strategy", "interpolate"))
    df = df.iloc[:, : int(ds.get("num_features", 52))]
    return df.reset_index(drop=True)