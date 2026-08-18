"""Sliding-window generation and leakage-free splitting.

Important: adjacent windows overlap and are strongly correlated, so randomly
splitting all windows leaks information between train and validation. Instead we
split by *segment* -- a contiguous run of records (a fault run, a normal run, or
a temporal block). All windows that touch a segment belong to that segment's
split.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def to_windows(
    data: np.ndarray,
    window_size: int,
    stride: int,
    dtype: str = "float32",
) -> np.ndarray:
    """Convert a [T, F] matrix into overlapping windows [N, W, F].

    Only complete windows are returned (windows shorter than ``window_size`` at
    the tail are dropped).
    """
    data = np.asarray(data, dtype=dtype)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D [T, F] input, got shape {data.shape}")
    n_samples, num_features = data.shape
    if window_size > n_samples:
        raise ValueError(
            f"window_size={window_size} exceeds available samples ({n_samples})"
        )
    starts = np.arange(0, n_samples - window_size + 1, stride)
    indices = starts[:, None] + np.arange(window_size)[None, :]
    windows = data[indices].astype(dtype)
    return windows


def segment_ids_for_runs(run_lengths: Sequence[int], window_size: int, stride: int) -> List[int]:
    """Assign a segment id to every window generated from concatenated runs.

    ``run_lengths`` lists the length of each contiguous run (normal block or
    fault run). Returns one segment id per window in generation order.
    """
    segment_ids: List[int] = []
    for seg_id, length in enumerate(run_lengths):
        n_windows = max(0, (length - window_size) // stride + 1)
        segment_ids.extend([seg_id] * n_windows)
    return segment_ids


def split_windows_by_segment(
    windows: np.ndarray,
    segment_ids: Sequence[int],
    val_ratio: float,
    seed: int,
    test_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Split windows by whole segments to avoid overlap leakage.

    Args:
        windows: [N, W, F] window array.
        segment_ids: one segment id per window (len == N).
        val_ratio: fraction of segments held out for validation.
        seed: RNG seed so the split is reproducible.
        test_ratio: optional extra hold-out for testing.

    Returns:
        (train_idx, val_idx, test_idx) index arrays into ``windows``.
    """
    windows = np.asarray(windows)
    if len(windows) != len(segment_ids):
        raise ValueError(
            f"len(windows)={len(windows)} != len(segment_ids)={len(segment_ids)}"
        )
    unique = np.unique(np.asarray(segment_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)

    n_val = int(round(len(unique) * val_ratio))
    n_test = int(round(len(unique) * test_ratio))
    test_segments = unique[:n_test]
    val_segments = unique[n_test:n_test + n_val]
    train_segments = unique[n_test + n_val:]

    seg_arr = np.asarray(segment_ids)
    train_idx = np.flatnonzero(np.isin(seg_arr, train_segments))
    val_idx = np.flatnonzero(np.isin(seg_arr, val_segments))
    test_idx = np.flatnonzero(np.isin(seg_arr, test_segments)) if n_test > 0 else None

    logger.info(
        "Segment split: %d segments -> train %d windows, val %d windows%s.",
        len(unique), len(train_idx), len(val_idx),
        f", test {len(test_idx)} windows" if test_idx is not None else "",
    )
    return train_idx, val_idx, test_idx


def windows_metadata(
    windows: np.ndarray, segment_ids: Sequence[int], window_size: int, stride: int
) -> dict:
    """Attach metadata (window index -> start sample) for traceability."""
    seg_arr = np.asarray(segment_ids)
    sample_starts = []
    for i, seg in enumerate(seg_arr):
        seg_offset = seg * stride  # valid when segments are back-to-back
        sample_starts.append(int(seg_offset + i * stride))
    return {
        "num_windows": int(windows.shape[0]),
        "window_size": int(window_size),
        "stride": int(stride),
        "window_start_samples": sample_starts,
    }