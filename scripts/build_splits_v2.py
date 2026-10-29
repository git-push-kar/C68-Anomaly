"""Build data/splits_v2.json from the protocol definitions below.

Deterministic: re-running produces byte-identical output, including the SHA-256
fingerprint of each split's expanded sorted key list. The validator
(``scripts/check_splits_v2.py``) recomputes those fingerprints, so the file
cannot drift from its declared splits.

Why one source file: the two faulty CSVs are not interchangeable. Measured on
this checkout (2026-10-29):

======================  ==================  ==============  ===================
file                    samples per run     fault onset      rows per fault
======================  ==================  ==============  ===================
TEP_Faulty_Training.csv 500                 ~sample 20      250,000
TEP_Faulty_Testing.csv  960                 ~sample 160     480,000
======================  ==================  ==============  ===================

``windowing.block_samples`` (500) and ``prepare_tep.py`` assume the Training
shape, so the benchmark takes every LLM event from the Testing file, where all
splits share one run length and one onset.

    python scripts/build_splits_v2.py            # write the file
    python scripts/build_splits_v2.py --check    # fail if the file is stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "splits_v2.json"

sys.path.insert(0, str(REPO_ROOT))
from scripts.check_splits_v2 import expand_split, key_digest  # noqa: E402

SOURCE_FILE = "data/raw/faults/TEP_Faulty_Testing.csv"
RUNS = 500
SEEN_SPLIT_RANGES = {"train": (1, 300), "val": (301, 400), "test": (401, 500)}
SYNTHETIC_FAULTS = [21, 22]
OPEN_SET_FAULTS = [16, 17, 18, 19, 20]
UNOBSERVABLE_FAULTS = [3, 9]


COMPONENT_ORDER = ("train", "val", "test", "test_seen")


def _split_block(ranges: Dict[str, tuple], faults: Dict[str, List[int]], note: str) -> Dict:
    """Build a protocol's components (train/val/test and optional test_seen).

    Every component records its own SHA-256, plus an ``all`` fingerprint over
    the union, so a component can be added without invalidating the others.
    """
    present = [name for name in COMPONENT_ORDER if name in ranges]
    splits: Dict[str, Dict] = {}
    for name in present:
        start, end = ranges[name]
        splits[name] = {
            "source_file": SOURCE_FILE,
            "assignment": note,
            "faults": [
                {"faultNumber": fault, "simulationRunRange": [start, end], "runCount": end - start + 1}
                for fault in faults[name]
            ],
        }
    union = [key for block in splits.values() for key in expand_split(block)]
    fingerprint = {name: key_digest(expand_split(splits[name])) for name in present}
    fingerprint["all"] = key_digest(union)
    for name in present:
        splits[name]["sha256"] = fingerprint
    return splits


def build() -> Dict:
    all_faults = list(range(1, 21))

    protocol_a = {
        "description": (
            "Seen faults, run-disjoint. Every real fault 1-20 appears in train, val and test, "
            "but each split uses a disjoint block of simulation runs (1-300 / 301-400 / 401-500), "
            "so no run is shared. This measures in-distribution diagnosis with run-level holdout."
        ),
        "constraints": {"fault_disjoint_across_splits": False},
        "splits": _split_block(
            SEEN_SPLIT_RANGES,
            {name: all_faults for name in ("train", "val", "test")},
            "All faults 1-20, run-disjoint blocks.",
        ),
    }

    def leave_out_fold(held: List[int], fold_note: str) -> Dict:
        seen = [f for f in all_faults if f not in held]
        splits = _split_block(
            {"train": (1, 300), "val": (301, 400), "test": (1, 500), "test_seen": (401, 500)},
            {"train": seen, "val": seen, "test": held, "test_seen": seen},
            "Seen faults: train 1-300, val 301-400, test_seen 401-500. Held-out faults: all 500 runs in test only.",
        )
        return {
            "held_out_faults": held,
            "note": fold_note,
            "constraints": {"fault_disjoint_across_splits": True},
            "components": {
                "test": "UNSEEN component: held-out faults, all 500 runs. Abstention is scored here.",
                "test_seen": "SEEN component: kept faults, runs 401-500. False abstention is scored here.",
            },
            "splits": splits,
        }

    protocol_b = {
        "description": (
            "Leave-faults-out. Held-out faults are removed from train and val entirely and evaluated "
            "in test over all 500 runs (the UNSEEN component). Seen faults' runs 401-500 form "
            "test_seen (the SEEN component), which measures false abstention while test stays "
            "fault-disjoint from train/val."
        ),
        "constraints": {"fault_disjoint_across_splits": True},
        "folds": {
            "fold_1_cooling_loop_holdout": leave_out_fold(
                [4, 11, 15],
                "Holds out the reactor-cooling step fault (4), its random-variation analogue (11) and the "
                "condenser sticking fault (15). Their KB differential siblings stay in train: 14 (sibling of 4), "
                "5 (sibling of 15) and 12 (condenser-loop counterpart of 11). The KB links 11 to 4/14, not to 12; "
                "12 is kept for structural symmetry only.",
            ),
            "fold_2_valve_sticking_family": leave_out_fold(
                [14, 15],
                "Holds out the valve-sticking family (14, 15) so both members of the stiction pair are unseen "
                "while their step-change siblings 4 and 5 remain in train.",
            ),
        },
    }

    return {
        "version": "v2",
        "repo": str(REPO_ROOT).replace("\\", "/"),
        "base_commit": "4d4b3be",
        "created": "2026-10-29",
        "built_by": "scripts/build_splits_v2.py",
        "unique_key": ["source_file", "faultNumber", "simulationRun"],
        "key_note": (
            "Every scored item is identified by this triple. source_file is part of the key because the two "
            "faulty CSVs use different run lengths and onsets and must never be pooled."
        ),
        "source": {
            "file": SOURCE_FILE,
            "samplesPerRun": 960,
            "faultOnsetIndex": 160,
            "runsPerFault": RUNS,
            "simulationRunMin": 1,
            "simulationRunMax": RUNS,
            "faults": all_faults,
            "measured": (
                "Verified on 2026-10-29 by scanning the raw CSV: 500 runs/fault, 960 samples/run, "
                "fault-free baseline from TEP_FaultFree_Testing.csv, and the first sustained >=4-sigma "
                "departure of the across-run mean at sample 161 for the step faults (faults 4/5/6/7), "
                "i.e. the documented 0-based onset 160."
            ),
            "not_used": {
                "data/raw/faults/TEP_Faulty_Training.csv": (
                    "500 samples/run and onset 20 -- a different run shape. Pumping both files into one "
                    "benchmark would make run length and pre-anomaly context depend on the file, and "
                    "windowing.block_samples=500 assumes this shape."
                )
            },
        },
        "fault_scope": {
            "headline": all_faults,
            "open_set": OPEN_SET_FAULTS,
            "unobservable": UNOBSERVABLE_FAULTS,
            "synthetic_only": SYNTHETIC_FAULTS,
        },
        "label_policy": {
            "FAULT_01..FAULT_15": "the adapter must name the fault",
            "UNCLASSIFIED_OPEN_SET": (
                "faults 16-20: their KB names are 'Unknown disturbance in ...', so the accepted label is the "
                "open-set label, not the specific number"
            ),
            "FAULT_03/FAULT_09": "flagged unobservable; scored and reported separately, not dropped",
            "synthetic": "faults 21/22 are synthetic-only, non-headline, and labelled 'synthetic' wherever shown",
        },
        "protocols": {
            "A_seen_faults_run_disjoint": protocol_a,
            "B_leave_faults_out": protocol_b,
        },
        "synthetic": {
            "faults": SYNTHETIC_FAULTS,
            "headline_eligible": False,
            "label_required": "synthetic",
            "note": (
                "These faults exist only in generated data and in no split above. If scored at all, the row, "
                "table cell and caption must say synthetic."
            ),
            "provenance": {
                "data/llm": "v0 distillation: all 22 fault IDs in train/val/test (all_faults_in_train=true).",
                "data/llm_v1": "v1 fault-disjoint holdout: train includes 22, test includes 21, so its test set mixes real fault 15 with synthetic 21.",
            },
        },
    }


def render(doc: Dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    args = parser.parse_args(argv)

    text = render(build())
    if args.check:
        existing = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if existing != text:
            print(f"STALE: {OUTPUT} does not match the builder output")
            return 1
        print(f"UP TO DATE: {OUTPUT}")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
