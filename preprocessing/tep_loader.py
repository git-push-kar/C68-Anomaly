"""Tennessee Eastman Process (TEP) dataset loader.

Assumptions about the public TEP data used by this project
----------------------------------------------------------
There are several widely-used public TEP releases (the classic Downs &
Vogel .mat dumps, CSV re-exports on Kaggle, and the extended 52-variable
versions). Rather than scattering format logic through the codebase, every
format-specific decision lives in this module behind the :class:`TEPDataFormat`
enum and the :func:`load_tep_data` entry point.

Canonical layout expected by this project ("tep_52col_csv"):
  * normal operation  -> one file per run, e.g.  data/raw/normal/normal.csv
  * fault scenarios   -> one file per fault,  e.g.  data/raw/faults/fault_01.csv
  * each file is a T x F matrix of sensor records:
        - one row per sample, in chronological order (sample index 1..T)
        - F columns = process variables (52 where applicable)
  * optional columns (configurable):
        - a timestamp column   (``timestamp_column``)
        - a fault-onset column (``fault_onset_column``) holding the sample
          index at which the disturbance was injected
  * when no header is present, columns are named XMEAS_1..XMEAS_41,
    XMV_42..XMV_52 (the standard TEP variable order).
  * missing values are handled according to ``missing_value_strategy``.

Adding another public TEP format (e.g. .mat) only requires adding a loader
function and a ``TEPDataFormat`` member -- no other module needs to change.
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
) -> Dict[int, Tuple[pd.DataFrame, Optional[int]]]:
    """Load fault scenarios as {fault_id: (dataframe, onset_index)}."""
    faults: Dict[int, Tuple[pd.DataFrame, Optional[int]]] = {}
    files = sorted(fault_dir.glob(pattern)) if fault_dir.exists() else []
    if not files:
        logger.warning("No fault files matched %s in %s", pattern, fault_dir)
        return faults
    for path in files:
        df = _read_matrix(path, has_header, delimiter, feature_names)
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


def load_tep_data(config: dict) -> TEPData:
    """Load normal + fault TEP data according to ``config``.

    Args:
        config: merged configuration dict (see utils.load_config).

    Returns:
        :class:`TEPData` with normal frame and a fault-id -> frame mapping.
    """
    ds = config["dataset"]
    data_root = resolve_path(config, "data_root")
    normal_path = resolve_path(config, "normal_data_path")
    fault_path = resolve_path(config, "fault_data_path")

    feature_names = [
        f"XMEAS_{i}" for i in range(1, 42)
    ] + [f"XMV_{i}" for i in range(42, 53)]
    num_features = int(ds.get("num_features", 52))

    fmt = TEPDataFormat(ds.get("format", "tep_52col_csv"))
    if fmt != TEPDataFormat.TEP_52COL_CSV:
        raise NotImplementedError(
            f"Format {fmt.value} requires a loader in this module "
            "(add it to load_tep_data; see module docstring)."
        )

    normal_frame = pd.DataFrame()
    normal_files = sorted(normal_path.glob(ds["normal_file_pattern"])) if normal_path.exists() else []
    if normal_files:
        frames = []
        for path in normal_files:
            df = _read_matrix(path, ds.get("has_header", True), ds.get("delimiter", ","), feature_names)
            df, _, _ = _split_meta_columns(df, ds.get("timestamp_column"), ds.get("fault_onset_column"))
            df = _handle_missing(df, ds.get("missing_value_strategy", "interpolate"))
            frames.append(df.iloc[:, :num_features])
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
    )

    feature_names = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(42, 53)]
    feature_names = feature_names[:num_features]
    return TEPData(normal=normal_frame, faults=faults, feature_names=feature_names)


def load_single_csv(path: Union[str, Path], config: dict) -> pd.DataFrame:
    """Load one CSV (used by the streaming simulator) with the same rules."""
    ds = config["dataset"]
    feature_names = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(42, 53)]
    df = _read_matrix(Path(path), ds.get("has_header", True), ds.get("delimiter", ","), feature_names)
    df, _, _ = _split_meta_columns(df, ds.get("timestamp_column"), ds.get("fault_onset_column"))
    df = _handle_missing(df, ds.get("missing_value_strategy", "interpolate"))
    df = df.iloc[:, : int(ds.get("num_features", 52))]
    return df.reset_index(drop=True)