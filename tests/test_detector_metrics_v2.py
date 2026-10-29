"""Unit tests for the metric helpers in scripts/evaluate_detector_v2.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_detector_v2 import (  # noqa: E402
    align_samples_to_windows,
    auroc,
    detection_delay,
    detection_index,
    false_alarm_before_onset,
    first_alarm_at_or_after,
    pr_auc,
    rate_above,
    segment_events,
    verify_far_disjoint,
    window_labels,
    window_starts,
)


def test_window_starts_and_labels():
    starts = window_starts(5, 5)
    assert list(starts) == [0, 5, 10, 15, 20]
    # window_size 1: label by start
    assert list(window_labels(starts, 10)) == [0, 0, 1, 1, 1]
    # window_size 6: the window starting at 5 already contains sample 10
    assert list(window_labels(starts, 10, window_size=6)) == [0, 1, 1, 1, 1]


def test_rate_above_strict():
    assert rate_above(np.array([0.5, 1.0, 2.0]), 1.0) == pytest.approx(1 / 3)
    assert rate_above(np.array([]), 1.0) == 0.0


def test_detection_index_requires_consecutive():
    assert detection_index([0, 1, 1, 1], 3) == 1
    assert detection_index([0, 1, 0, 1], 3) is None
    assert detection_index([1], 1) == 0
    assert detection_index([], 3) is None


def test_detection_delay_uses_window_end():
    starts = np.arange(5) * 5
    alarm = [0, 1, 1, 1, 1]
    # window_size 5: the window at start 5 covers 5..9, which ends before the
    # onset, so the first eligible window starts at 10 and completes at 14.
    assert detection_delay(starts, alarm, onset=10, confirm=3, window_size=5) == 4
    assert first_alarm_at_or_after(starts, alarm, onset=10, confirm=3, window_size=5) == 2
    # window_size 6: the window at start 5 already covers sample 10
    assert detection_delay(starts, alarm, onset=10, confirm=3, window_size=6) == 0
    # not enough eligible alarms even though the run is alarmed at the end
    assert detection_delay(starts, [1, 1, 0, 1, 0], onset=10, confirm=3, window_size=5) is None


def test_pre_onset_alarm_is_flagged_not_credited_as_detection():
    starts = np.arange(3) * 5
    alarm = [1, 1, 0]
    # window_size 5: only the window at start 0 ends before onset 10
    assert false_alarm_before_onset(starts, [1, 0, 1], onset=10, confirm=2, window_size=5) is False
    starts_long = np.arange(4) * 5
    assert false_alarm_before_onset(starts_long, [1, 1, 0, 1], onset=10, confirm=2, window_size=5) is True
    # only two eligible windows and they are not alarmed
    assert detection_delay(np.arange(4) * 5, [1, 1, 0, 0], onset=10, confirm=2, window_size=5) is None


def test_segment_events_merges_alarms_and_splits_on_separation():
    starts = np.arange(6) * 5
    assert segment_events([1, 1, 1, 0, 0, 0], starts, confirm=3, separate=2) == [(2, 2)]
    assert segment_events([1, 1, 1, 0, 0, 0], starts, confirm=3, separate=3) == [(2, 2)]
    split = segment_events([1, 1, 0, 0, 1, 1], starts, confirm=2, separate=2)
    assert split == [(1, 1), (5, 5)]


def test_segment_events_respects_max_event_windows():
    starts = np.arange(5) * 5
    events = segment_events([1, 1, 1, 1, 1], starts, confirm=2, separate=20, max_event_windows=3)
    assert events == [(1, 3), (4, 4)]


def test_auroc_and_pr_auc():
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(labels, perfect) == pytest.approx(1.0)
    assert pr_auc(labels, perfect) == pytest.approx(1.0)
    assert auroc(np.zeros(4), perfect) is None
    assert pr_auc(np.zeros(4), perfect) is None
    # NaN scores are dropped rather than poisoning the metric
    assert auroc(labels, np.array([0.1, np.nan, 0.8, 0.9])) == pytest.approx(1.0)


def test_align_samples_to_windows_mean_and_max():
    idx = np.arange(10)
    vals = np.arange(10, dtype=float)
    starts = np.array([0, 5])
    assert list(align_samples_to_windows(idx, vals, starts, 5)) == [2.0, 7.0]
    assert list(align_samples_to_windows(idx, vals, starts, 5, reduce="max")) == [4.0, 9.0]


def test_align_handles_lag_offset_index():
    # CVA/DPCA scores start at sample 5, not 0
    idx = np.arange(5, 15)
    vals = np.arange(10, dtype=float)
    out = align_samples_to_windows(idx, vals, np.array([0, 5, 10]), 5)
    assert np.isnan(out[0])  # no scored sample before index 5
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(7.0)


def test_verify_far_disjoint_detects_overlap():
    far_file = "data/raw/normal/TEP_FaultFree_Testing.csv"
    calib = {"data/raw/normal/TEP_FaultFree_Training.csv": [1, 2, 3]}
    ok = verify_far_disjoint(far_file, list(range(1, 501)), calib)
    assert ok["ok"] is True
    assert ok["overlapping_files"] == []

    bad = verify_far_disjoint(far_file, [1, 2, 3], {far_file: [2, 3, 4]})
    assert bad["ok"] is False
    assert bad["overlapping_runs"] == [2, 3]
    assert bad["overlapping_files"] == [far_file]


def test_verify_far_disjoint_matches_basenames_with_other_path_prefix():
    # same file, different path prefix / separator must still be detected
    bad = verify_far_disjoint(
        "data/raw/normal/TEP_FaultFree_Testing.csv",
        [10, 11],
        {"C:\\work\\data\\raw\\normal\\TEP_FaultFree_Testing.csv": [11, 12]},
    )
    assert bad["ok"] is False
    assert bad["overlapping_runs"] == [11]
