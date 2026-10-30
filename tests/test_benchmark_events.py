"""Tests for the event-selection rules shared by the benchmark scripts.

The bug these were written for: run 86 of every fault was reported as
``unclosed_at_run_end``. The aggregator had emitted its event correctly -- open
at window 20, still open at the end of the stream -- but the builder asked
whether the *opening* window ended after onset (sample 159 vs onset 160) and threw
the event away. The rule now asks whether the event's span reaches the onset.
"""
from __future__ import annotations

import numpy as np
import pytest

from llm.benchmark.events import (
    DETECTOR_MISS_REASON,
    UNCLOSED_REASON,
    miss_reason,
    oracle_candidates,
    select_event,
    snapshot_block,
)


def _ends(n_windows: int = 181, window_size: int = 60, stride: int = 5) -> np.ndarray:
    return np.asarray([start + window_size - 1 for start in range(0, n_windows * stride, stride)])


# -- selecting the event that covers the onset --------------------------------
def test_event_opening_just_before_onset_and_running_on_is_selected():
    # The run-86 shape: open window 20 ends at sample 159, onset is 160.
    events = [(20, 180)]
    chosen = select_event(events, _ends(), onset=160)
    assert chosen == (20, 180), "an event spanning the onset must not be discarded"


def test_event_that_closes_before_onset_is_not_selected():
    # Ends at sample 159, i.e. entirely before the fault is injected.
    assert select_event([(4, 20)], _ends(), onset=160) is None


def test_first_qualifying_event_wins():
    events = [(2, 10), (25, 40), (60, 90)]
    assert select_event(events, _ends(), onset=160) == (25, 40)


def test_no_events_means_no_selection():
    assert select_event([], _ends(), onset=160) is None


# -- the fixed horizon --------------------------------------------------------
def test_snapshot_is_the_horizon_from_the_open_window():
    assert snapshot_block(20, 20, 181) == (20, 39)


def test_snapshot_is_capped_at_the_last_window():
    first, last = snapshot_block(175, 20, 181)
    assert (first, last) == (175, 180), "the snapshot cannot read past the stream"


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        snapshot_block(10, 0, 181)


# -- which misses may be patched ---------------------------------------------
def test_only_unclosed_runs_are_oracle_candidates():
    misses = [
        {"fault": 3, "simulationRun": 12, "reason": DETECTOR_MISS_REASON},
        {"fault": 4, "simulationRun": 86, "reason": UNCLOSED_REASON},
        {"fault": 9, "simulationRun": 7, "reason": DETECTOR_MISS_REASON},
    ]
    assert oracle_candidates(misses) == [misses[1]]


def test_detector_misses_are_never_patched():
    misses = [{"fault": 3, "simulationRun": n, "reason": DETECTOR_MISS_REASON} for n in range(50)]
    assert oracle_candidates(misses) == []


def test_a_miss_without_a_reason_is_not_patched():
    assert oracle_candidates([{"fault": 4, "simulationRun": 1}]) == []


# -- why a run produced nothing ----------------------------------------------
def test_confirmed_alarm_with_no_selected_event_is_unclosed():
    alarm = np.zeros(181, dtype=int)
    alarm[20:181] = 1
    assert miss_reason([], alarm, confirm=12) == UNCLOSED_REASON


def test_an_alarm_that_never_confirms_is_a_detector_miss():
    alarm = np.zeros(181, dtype=int)
    alarm[[10, 40, 90]] = 1          # never 12 consecutive
    assert miss_reason([], alarm, confirm=12) == DETECTOR_MISS_REASON


def test_empty_alarm_series_is_a_detector_miss():
    assert miss_reason([], np.zeros(0, dtype=int), confirm=12) == DETECTOR_MISS_REASON
