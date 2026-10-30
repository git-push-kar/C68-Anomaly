"""Event-selection rules shared by the benchmark scripts.

Three small decisions live here instead of inside `build_benchmark_events.py`,
because each one is a place where an off-by-one silently changes the dataset and
each is worth a test on its own:

* **which event belongs to a run** -- the first event whose span reaches the
  onset, not the first event that *starts* after it. A persistent fault keeps the
  alarm on to the end of the stream, so the run's only event often opens a
  sample or two before onset (a pre-onset alarm) and runs to the end. Testing the
  opening window instead of the span discarded that event and reported the run as
  "unclosed", which made one run in twenty look like a stream artifact.
* **how much of it to show** -- a fixed horizon from the event's open window.
  The horizon is what makes closure irrelevant: the snapshot exists whether or
  not the aggregator ever found `min_separation_windows` quiet windows.
* **which misses may be replaced by an oracle window** -- only runs whose alarm
  was confirmed but never closed. Ordinary detector misses are not patched: a
  fault the detector cannot see is a fact about the system, not a gap to fill.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Miss reasons, as recorded by the event builder.
UNCLOSED_REASON = "unclosed_at_run_end"
DETECTOR_MISS_REASON = "detector_miss"

#: Default horizon in *windows* (window_size 60, stride 5 -> 100 samples).
DEFAULT_HORIZON_WINDOWS = 20


def select_event(
    events: Sequence[Tuple[int, int]],
    end_samples: np.ndarray,
    onset: int,
) -> Optional[Tuple[int, int]]:
    """The event that covers the onset, or ``None``.

    An event qualifies when its span reaches the onset -- its closing window ends
    at or after ``onset``. That includes an event that opened before onset and is
    still open at the end of the stream, which is the normal shape of a
    persistent fault rather than an error.
    """
    for open_idx, close_idx in events:
        if int(end_samples[close_idx]) >= int(onset):
            return int(open_idx), int(close_idx)
    return None


def snapshot_block(open_idx: int, horizon: int, n_windows: int) -> Tuple[int, int]:
    """Window block for the evidence snapshot: ``horizon`` windows from open.

    Capped at the last window the run has, so a late event gets a shorter
    snapshot instead of reading past the end of the stream.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    last = min(int(open_idx) + int(horizon) - 1, int(n_windows) - 1)
    return int(open_idx), max(int(open_idx), last)


def oracle_candidates(misses: Iterable[Dict]) -> List[Dict]:
    """Misses that may receive a fixed-offset oracle window.

    Only runs the detector *saw* and whose event had not closed when the stream
    ended. Manufacturing windows for plain detector misses would invent events
    for faults the detector cannot see and inflate training coverage with items
    the system never produces.
    """
    return [m for m in misses if m.get("reason") == UNCLOSED_REASON]


def miss_reason(events: Sequence[Tuple[int, int]], alarm: np.ndarray, confirm: int) -> str:
    """Why a run produced no post-onset event.

    ``unclosed_at_run_end`` means the alarm confirmed but no event reached the
    onset; ``detector_miss`` means it never confirmed at all.
    """
    if not len(alarm):
        return DETECTOR_MISS_REASON
    runs = np.convolve(np.asarray(alarm, dtype=int), np.ones(confirm, dtype=int), mode="valid")
    confirmed = bool(runs.size and runs.max() >= confirm)
    return UNCLOSED_REASON if confirmed else DETECTOR_MISS_REASON


__all__ = [
    "DEFAULT_HORIZON_WINDOWS",
    "DETECTOR_MISS_REASON",
    "UNCLOSED_REASON",
    "miss_reason",
    "oracle_candidates",
    "select_event",
    "snapshot_block",
]
