#!/usr/bin/env python
"""Oracle windows for detector misses.

A missed run produces no closed event, so the response layer never sees that
fault at all. To keep such faults in the response benchmark, one synthetic
("oracle") event is built per missed run at a fixed offset after onset.

Rules this script enforces:

* the offset is the **median event-open sample over the TRAIN split only**,
  computed from events already on disk. It is therefore a training constant, not
  a per-run fit: a test-range miss is patched with a window chosen from training
  timing, so nothing about test timing leaks into it;
* the window block is exactly ``persistence`` windows long, i.e. the minimum an
  event needs to be confirmed, so the oracle says nothing about how the event
  would actually have developed;
* oracle records are marked ``"oracle": true`` so every downstream consumer can
  report them separately from real detector events.

Usage:
    python scripts/build_benchmark_oracles.py [--config ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_benchmark_events import OUT, FrozenDetector, split_for_run  # noqa: E402
from evaluate_detector_v2 import iter_runs  # noqa: E402
from utils import ensure_dir, load_config, save_json  # noqa: E402

logger = logging.getLogger("build_oracles")

#: Faults whose runs are almost all detector misses (fault 3 dribbling feed and
#: fault 9 random variation are unobservable at this operating point). Building
#: oracle windows for them would manufacture an event for a fault the detector
#: cannot see, and they are excluded from the observable set anyway, so their
#: misses stay misses. The one run per fault that closes the stream mid-event is
#: still patched, because that is an aggregation artifact rather than a miss.
UNOBSERVABLE = (3, 9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config_a5000.yaml")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--include-unobservable", action="store_true",
                    help="also patch detector misses of faults 3 and 9")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-8s %(message)s")

    out = Path(args.out)
    config = load_config(args.config)
    detector = FrozenDetector(config, json.loads(
        (Path("configs/detector_v2_operating_point.json")).read_text(encoding="utf-8")))

    train_opens: List[int] = []
    for path in (out / "faulty" / "train").glob("*.json"):
        if path.stem.endswith("_oracle"):
            continue
        train_opens.append(int(json.loads(path.read_text(encoding="utf-8"))["event_open_sample"]))
    if not train_opens:
        logger.error("no train events on disk: run build_benchmark_events.py first")
        return 1
    train_median_open = int(statistics.median(train_opens))
    logger.info("train median event-open sample %d from %d train events",
                train_median_open, len(train_opens))

    misses = [json.loads(line) for line in (out / "misses.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    by_fault: Dict[int, set] = defaultdict(set)
    skipped = 0
    for miss in misses:
        fault = int(miss["fault"])
        if (not args.include_unobservable and fault in UNOBSERVABLE
                and miss.get("reason") == "detector_miss"):
            skipped += 1
            continue
        by_fault[fault].add(int(miss["simulationRun"]))
    if skipped:
        logger.info("skipped %d misses of unobservable faults %s", skipped, list(UNOBSERVABLE))
    if not by_fault:
        logger.info("no misses recorded; nothing to do")
        return 0

    target_start = train_median_open - (detector.window_size - 1)
    start = max(0, min(int(round(target_start / detector.stride)), detector.n_windows() - 1))
    block = (start, min(start + max(detector.persistence, 1) - 1, detector.n_windows() - 1))
    logger.info("oracle block = windows %d..%d (start sample %d)", block[0], block[1],
                start * detector.stride)

    faulty_csv = Path(config["paths"]["fault_data_path"]) / "TEP_Faulty_Testing.csv"
    t0 = time.time()
    written = 0
    records: List[Dict] = []
    for fault, run, raw in iter_runs(faulty_csv, faults=sorted(by_fault),
                                     run_range=(1, 500), samples_per_run=960):
        fault = int(fault)
        if run not in by_fault.get(fault, set()):
            continue
        component = split_for_run(run) or "train"
        event_id = f"EVT-{component}-f{fault:02d}-r{run:03d}-oracle"
        rec = detector.build(event_id, raw, fault, window_block=block)
        rec.update({
            "source_file": faulty_csv.name, "fault": fault, "simulationRun": run,
            "split": component, "seed": 42, "oracle": True,
            "oracle_offset_sample": train_median_open,
            "oracle_offset_source": "median event-open sample over the TRAIN split only",
        })
        records.append(rec)
        ensure_dir(out / "faulty" / component)
        save_json(out / "faulty" / component / f"f{fault:02d}_r{run:03d}_oracle.json", rec)
        written += 1
    with open(out / "oracle.jsonl", "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")
    logger.info("wrote %d oracle events in %.0fs", written, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
