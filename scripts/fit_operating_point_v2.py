"""A2/A3: freeze a window-level operating point for the fused detector (CPU only).

The deployed artifacts are thresholded per SAMPLE (p99 of 500-sample dynamic
statistics), but the evaluator and the event builder compare a per-WINDOW mean.
That mismatch leaves the dynamic detectors at ~0% window FAR while the LSTM sits
at ~1.7%, so the OR-gate is effectively the LSTM alone (Phase 1 finding).

This script sweeps a window-level operating point on CALIBRATION DATA ONLY and
freezes one combination:

  target window FAR in {0.1%, 0.5%, 1%}
      LSTM        -> normal validation windows (detector_split.json, 75 runs)
      CVA Q       -> TEP_FaultFree_Training runs 401-500 (the artifacts' own
                     threshold holdout, 100 runs)
      DPCA SPE    -> same holdout
  persistence     in {3, 5, 8, 12} consecutive alarmed windows

Selection rule (fixed before looking at any fault-free Testing number):
    among combinations whose RUN-level FAR on fault-free Testing runs 1-250
    (selection half) is <= 5%, take the smallest mean detection delay on
    Protocol A validation over the 18-fault observable set. Run-level FAR on
    runs 251-500 (reporting half) is reported, never used to choose.

Protocol A TEST is never read. Protocol A validation FDR is reported in the
Pareto table but does not enter the selection. Pre-onset windows of a faulty run
are not part of FDR and are not counted as tests of the operating point.

    python scripts/fit_operating_point_v2.py                 # sweep + freeze
    python scripts/fit_operating_point_v2.py --no-freeze     # sweep only
    python scripts/fit_operating_point_v2.py --rebuild-cache # re-score from CSV

Estimated runtime: ~5-6 min from CSV on CPU; seconds from cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_detector_v2 import (  # noqa: E402
    DetectorBundle,
    detection_delay,
    iter_runs,
    load_split_component,
    rate_above,
    summarise_series,
    window_labels,
    window_starts,
)
from utils import ensure_dir, get_fault_onset, load_config, save_json  # noqa: E402

logger = logging.getLogger("fit_operating_point_v2")

TARGET_FARS = (0.001, 0.005, 0.01)
PERSISTENCE = (3, 5, 8, 12)
OBSERVABLE_V1 = [f for f in range(1, 21) if f not in (3, 9)]  # Phase 1 deployed set
SELECT_HALF = (1, 250)
REPORT_HALF = (251, 500)

CACHE = ROOT / "outputs" / "evaluation" / "detector_v2" / "op_point_cache.npz"
SWEEP_JSON = ROOT / "outputs" / "evaluation" / "detector_v2" / "operating_point_sweep.json"
SWEEP_MD = ROOT / "outputs" / "evaluation" / "detector_v2" / "operating_point_sweep.md"
FROZEN = ROOT / "configs" / "detector_v2_operating_point.json"
CFG_COPY = ROOT / "configs" / "config_a5000_no_smoothing.yaml"

DETECTORS = ("lstm_ae", "cva_Q", "dpca_SPE")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# scoring + cache
# ---------------------------------------------------------------------------
def _score_run(bundle: DetectorBundle, raw: np.ndarray) -> Dict[str, np.ndarray]:
    scaled = bundle.scale(raw)
    starts, lstm = bundle.score_lstm(scaled)
    dyn = bundle.score_dynamic(scaled, starts)
    out = {"lstm_ae": np.asarray(lstm, dtype=np.float32)}
    for name in ("cva_Q", "dpca_SPE"):
        if name in dyn:
            out[name] = np.asarray(dyn[name], dtype=np.float32)
    return out


def build_cache(config: Dict, protocol: str, splits: Path) -> Dict[str, np.ndarray]:
    bundle = DetectorBundle(
        config,
        {"lstm_ae": {"threshold": 0.687}, "dynamic": {"cva": {}, "dpca": {}}},
        device="cpu",
        window_size=int(config["windowing"]["window_size"]),
        stride=int(config["windowing"]["stride"]),
    )
    normal_csv = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv"
    testing_csv = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Testing.csv"

    man = Path(config["paths"]["processed_data_path"]) / "manifests" / "detector_split.json"
    val_runs = sorted(int(r) for r in json.loads(man.read_text(encoding="utf-8"))["validation_runs"])

    out: Dict[str, list] = {}

    # --- calibration: LSTM normal validation windows (500-sample runs) ------
    t0 = time.time()
    wanted = set(val_runs)
    acc: Dict[str, List[np.ndarray]] = {k: [] for k in DETECTORS}
    runs: List[int] = []
    for _f, run, raw in iter_runs(normal_csv, faults=None, run_range=(min(val_runs), max(val_runs)), samples_per_run=500):
        if run not in wanted:
            continue
        s = _score_run(bundle, raw)
        for k in DETECTORS:
            acc[k].append(s[k])
        runs.append(run)
    for k in DETECTORS:
        out[f"cal_lstm_{k}"] = np.stack(acc[k])
    out["cal_lstm_runs"] = np.asarray(runs)
    logger.info("calibration (LSTM val windows): %d runs in %.1fs", len(runs), time.time() - t0)

    # --- calibration: dynamic holdout runs 401-500 (500-sample runs) --------
    t0 = time.time()
    acc = {k: [] for k in DETECTORS}
    runs = []
    for _f, run, raw in iter_runs(normal_csv, faults=None, run_range=(401, 500), samples_per_run=500):
        s = _score_run(bundle, raw)
        for k in DETECTORS:
            acc[k].append(s[k])
        runs.append(run)
    for k in DETECTORS:
        out[f"cal_dyn_{k}"] = np.stack(acc[k])
    out["cal_dyn_runs"] = np.asarray(runs)
    logger.info("calibration (dynamic holdout): %d runs in %.1fs", len(runs), time.time() - t0)

    # --- fault-free Testing 1-500 (960-sample runs) -----------------------
    t0 = time.time()
    acc = {k: [] for k in DETECTORS}
    runs = []
    for _f, run, raw in iter_runs(testing_csv, faults=None, run_range=(1, 500), samples_per_run=960):
        s = _score_run(bundle, raw)
        for k in DETECTORS:
            acc[k].append(s[k])
        runs.append(run)
    for k in DETECTORS:
        out[f"far_{k}"] = np.stack(acc[k])
    out["far_runs"] = np.asarray(runs)
    logger.info("fault-free Testing: %d runs in %.1fs", len(runs), time.time() - t0)

    # --- Protocol A validation faulty runs (960-sample each) --------------
    comp = load_split_component(splits, protocol, "val")
    runs_path = Path(comp["source_file"])
    if not runs_path.is_absolute():
        runs_path = ROOT / runs_path
    lo = min(r[0] for r in comp["ranges"].values())
    hi = max(r[1] for r in comp["ranges"].values())
    t0 = time.time()
    acc = {k: [] for k in DETECTORS}
    faults: List[int] = []
    runs = []
    n = 0
    for f_num, run, raw in iter_runs(runs_path, faults=comp["faults"], run_range=(lo, hi), samples_per_run=960):
        fault = int(f_num)
        if fault not in comp["ranges"]:
            continue
        lo_f, hi_f = comp["ranges"][fault]
        if not (lo_f <= run <= hi_f):
            continue
        s = _score_run(bundle, raw)
        for k in DETECTORS:
            acc[k].append(s[k])
        faults.append(fault)
        runs.append(run)
        n += 1
        if n % 400 == 0:
            logger.info("  val runs scored: %d (%.1fs)", n, time.time() - t0)
    for k in DETECTORS:
        out[f"val_{k}"] = np.stack(acc[k])
    out["val_faults"] = np.asarray(faults)
    out["val_runs"] = np.asarray(runs)
    logger.info("Protocol A validation: %d runs in %.1fs", n, time.time() - t0)
    return {k: v for k, v in out.items()}


def load_or_build(config: Dict, protocol: str, splits: Path, rebuild: bool) -> Dict[str, np.ndarray]:
    if CACHE.exists() and not rebuild:
        data = dict(np.load(CACHE))
        logger.info("loaded score cache %s (%d arrays)", CACHE.name, len(data))
        return data
    data = build_cache(config, protocol, splits)
    ensure_dir(CACHE.parent)
    np.savez_compressed(CACHE, **data)
    logger.info("wrote score cache %s", CACHE)
    return data


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def window_thresholds(data: Dict[str, np.ndarray], target_far: float) -> Dict[str, float]:
    return {
        "lstm_ae": float(np.quantile(np.concatenate(data["cal_lstm_lstm_ae"]).ravel(), 1.0 - target_far)),
        "cva_Q": float(np.quantile(np.concatenate(data["cal_dyn_cva_Q"]).ravel(), 1.0 - target_far)),
        "dpca_SPE": float(np.quantile(np.concatenate(data["cal_dyn_dpca_SPE"]).ravel(), 1.0 - target_far)),
    }


def fused_series(w: Dict[str, np.ndarray], thr: Dict[str, float]) -> np.ndarray:
    """Deployed OR fusion: max(norm_lstm, norm_primary_dynamic), primary = CVA Q."""
    return np.maximum(w["lstm_ae"] / max(thr["lstm_ae"], 1e-9), w["cva_Q"] / max(thr["cva_Q"], 1e-9))


def run_far(data: Dict[str, np.ndarray], thr: Dict[str, float], confirm: int,
            separate: int, max_ev: int, window_size: int, stride: int,
            runs_range: Tuple[int, int]) -> Dict[str, float]:
    starts = window_starts(data["far_lstm_ae"].shape[1], stride)
    runs = data["far_runs"]
    mask = (runs >= runs_range[0]) & (runs <= runs_range[1])
    per_run_events = []
    alarmed = 0
    for i in np.flatnonzero(mask):
        w = {k: data[f"far_{k}"][i] for k in DETECTORS}
        fused = fused_series(w, thr)
        s = summarise_series(fused, starts, 1.0, 0, confirm, separate, max_ev, window_size)
        per_run_events.append(s["n_events"])
        alarmed += int(s["alarm_sample"] is not None)
    n = int(mask.sum())
    return {
        "n_runs": n,
        "run_far": alarmed / max(1, n),
        "events_per_run": float(np.mean(per_run_events)) if per_run_events else 0.0,
    }


def val_operating_stats(data: Dict[str, np.ndarray], thr: Dict[str, float], onset: int,
                        confirm: int, separate: int, max_ev: int, window_size: int,
                        stride: int) -> Dict[str, Dict[str, Optional[float]]]:
    starts = window_starts(data["val_lstm_ae"].shape[1], stride)
    labels = window_labels(starts, onset, window_size)
    faults = data["val_faults"]
    per_fault: Dict[int, Dict[str, list]] = {}
    for i in range(data["val_lstm_ae"].shape[0]):
        fault = int(faults[i])
        w = {k: data[f"val_{k}"][i] for k in DETECTORS}
        fused = fused_series(w, thr)
        post = labels == 1
        fdr = rate_above(fused[post], 1.0)
        delay = detection_delay(starts, (fused > 1.0).astype(int), onset, confirm, window_size)
        bucket = per_fault.setdefault(fault, {"fdr": [], "delay": [], "detected": []})
        bucket["fdr"].append(fdr)
        bucket["detected"].append(delay is not None)
        if delay is not None:
            bucket["delay"].append(float(delay))
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for fault, b in per_fault.items():
        out[str(fault)] = {
            "fdr": float(np.mean(b["fdr"])) if b["fdr"] else None,
            "run_detection_rate": float(np.mean(b["detected"])) if b["detected"] else None,
            "delay_mean": float(np.mean(b["delay"])) if b["delay"] else None,
            "n_runs": len(b["fdr"]),
        }
    return out


def sweep(data: Dict[str, np.ndarray], config: Dict, onset: int) -> Dict[str, object]:
    window_size = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])
    separate = int(config["events"]["min_separation_windows"])
    max_ev = int(config["events"]["max_event_windows"])

    rows: List[Dict[str, object]] = []
    thr_table: Dict[str, Dict[str, float]] = {}
    for tf in TARGET_FARS:
        thr = window_thresholds(data, tf)
        thr_table[str(tf)] = thr
        for confirm in PERSISTENCE:
            sel = run_far(data, thr, confirm, separate, max_ev, window_size, stride, SELECT_HALF)
            rep = run_far(data, thr, confirm, separate, max_ev, window_size, stride, REPORT_HALF)
            val = val_operating_stats(data, thr, onset, confirm, separate, max_ev, window_size, stride)
            obs = [val[str(f)] for f in OBSERVABLE_V1 if str(f) in val]
            delays = [v["delay_mean"] for v in obs if v["delay_mean"] is not None]
            row = {
                "target_far": tf,
                "persistence": confirm,
                "thresholds": thr,
                "far_selection_half": sel,
                "far_reporting_half": rep,
                "mean_fdr_observable": float(np.mean([v["fdr"] for v in obs])) if obs else None,
                "mean_fdr_all20": float(np.mean([v["fdr"] for v in val.values()])),
                "mean_detection_rate_observable": float(np.mean([v["run_detection_rate"] for v in obs])),
                "mean_delay_observable": float(np.mean(delays)) if delays else None,
                "fdr_by_fault": {k: v["fdr"] for k, v in val.items()},
            }
            row["passes_run_far"] = bool(sel["run_far"] <= 0.05)
            rows.append(row)
            logger.info(
                "tf=%.3f p=%2d | runFAR sel %.1f%% rep %.1f%% | meanFDR %.1f%% | delay %s",
                tf, confirm, sel["run_far"] * 100, rep["run_far"] * 100,
                (row["mean_fdr_observable"] or 0) * 100,
                f"{row['mean_delay_observable']:.1f}" if row["mean_delay_observable"] is not None else "-",
            )
    return {"rows": rows, "thresholds": thr_table}


def select(rows: List[Dict[str, object]]) -> Dict[str, object]:
    eligible = [r for r in rows if r["passes_run_far"] and r["mean_delay_observable"] is not None]
    if not eligible:
        raise SystemExit("no combination meets the 5% run-level FAR constraint on the selection half")
    # Smallest delay; deterministic tie-break on the selection-half run FAR.
    eligible.sort(key=lambda r: (round(r["mean_delay_observable"], 3), r["far_selection_half"]["run_far"]))
    return eligible[0]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def render_md(result: Dict[str, object], chosen: Dict[str, object]) -> str:
    lines = [
        "# Detector operating point v2 — window-level sweep (calibration data only)",
        "",
        "Selection: run-level FAR <= 5% on fault-free Testing runs 1-250, then smallest mean",
        "delay on Protocol A validation over the 18-fault observable set. Runs 251-500 are",
        "reported, never used. Protocol A test was not read.",
        "",
        "| target window FAR | persistence | thr LSTM | thr CVA Q | thr DPCA SPE | run FAR sel (1-250) | run FAR rep (251-500) | events/run sel | mean FDR (18 obs) | mean det. rate | mean delay (samples) | eligible |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        t = r["thresholds"]
        lines.append(
            "| {tf:.1f}% | {p} | {l:.4f} | {c:.2f} | {d:.2f} | {fs:.1f}% | {fr:.1f}% | {ev:.2f} | "
            "{fdr:.1f}% | {dr:.1f}% | {dl} | {ok} |".format(
                tf=r["target_far"] * 100,
                p=r["persistence"],
                l=t["lstm_ae"],
                c=t["cva_Q"],
                d=t["dpca_SPE"],
                fs=r["far_selection_half"]["run_far"] * 100,
                fr=r["far_reporting_half"]["run_far"] * 100,
                ev=r["far_selection_half"]["events_per_run"],
                fdr=(r["mean_fdr_observable"] or 0) * 100,
                dr=r["mean_detection_rate_observable"] * 100,
                dl=f"{r['mean_delay_observable']:.1f}" if r["mean_delay_observable"] is not None else "-",
                ok="yes" if r["passes_run_far"] else "no",
            )
        )
    lines += [
        "",
        "**Selected:** target window FAR {tf:.1f}%, persistence {p}. "
        "LSTM {l:.4f}, CVA Q {c:.3f}, DPCA SPE {d:.3f}.".format(
            tf=chosen["target_far"] * 100, p=chosen["persistence"],
            l=chosen["thresholds"]["lstm_ae"], c=chosen["thresholds"]["cva_Q"],
            d=chosen["thresholds"]["dpca_SPE"],
        ),
        "",
        f"Run-level FAR: {chosen['far_selection_half']['run_far']*100:.1f}% on the selection half, "
        f"{chosen['far_reporting_half']['run_far']*100:.1f}% on the reporting half.",
        "",
        "## Per-fault FDR at the selected point (Protocol A validation)",
        "",
        "| Fault | FDR | run detection rate | mean delay |",
        "|---|---|---|---|",
    ]
    for f in range(1, 21):
        v = chosen["fdr_by_fault"].get(str(f))
        if v is None:
            lines.append(f"| {f} | - | - | - |")
            continue
        lines.append(f"| {f}{' (unobservable)' if f in (3, 9) else ''} | {v*100:.1f}% | - | - |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config_a5000.yaml")
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--splits", default="data/splits_v2.json")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    onset = int(get_fault_onset(config, "faulty_testing"))
    data = load_or_build(config, args.protocol, Path(args.splits), args.rebuild_cache)

    t0 = time.time()
    result = sweep(data, config, onset)
    chosen = select(result["rows"])
    logger.info("sweep of %d combinations in %.1fs", len(result["rows"]), time.time() - t0)

    ensure_dir(SWEEP_JSON.parent)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection_rule": "run-level FAR <= 5% on fault-free Testing 1-250, then smallest mean delay on Protocol A val (18 observable faults)",
        "target_fars": list(TARGET_FARS),
        "persistence": list(PERSISTENCE),
        "calibration_sources": {
            "lstm_ae": "TEP_FaultFree_Training validation_runs (75 runs), detector_split.json",
            "cva_Q": "TEP_FaultFree_Training runs 401-500 (dynamic threshold holdout)",
            "dpca_SPE": "TEP_FaultFree_Training runs 401-500 (dynamic threshold holdout)",
            "far_set": "TEP_FaultFree_Testing runs 1-500",
        },
        "rows": result["rows"],
        "selected": chosen,
    }
    save_json(SWEEP_JSON, payload)
    SWEEP_MD.write_text(render_md(result, chosen), encoding="utf-8")
    print(f"\nwrote {SWEEP_JSON}")
    print(f"wrote {SWEEP_MD}")

    if args.no_freeze:
        return

    artifacts = {
        "lstm_model": ROOT / "outputs" / "anomaly_detector" / "model.pt",
        "cva_model": ROOT / "outputs" / "anomaly_detector" / "dynamic" / "cva_model.npz",
        "dpca_model": ROOT / "outputs" / "anomaly_detector" / "dynamic" / "dpca_model.npz",
        "dynamic_thresholds_v1": ROOT / "outputs" / "anomaly_detector" / "dynamic" / "dynamic_thresholds.json",
        "detector_split": ROOT / "data" / "processed" / "manifests" / "detector_split.json",
        "splits_v2": ROOT / args.splits,
        "sweep_results": SWEEP_JSON,
    }
    frozen = {
        "_note": (
            "Frozen window-level operating point (Phase 1 patch, A3). Deployed v1 numbers "
            "keep using configs/detector_thresholds_v2.json; this file is the v2 point and is "
            "never mixed into a v1 table."
        ),
        "operating_point_id": "detector_v2_window_op_2026-10-30",
        "smoothing": "none",
        "fusion": "or",
        "fusion_primary_dynamic": "cva_Q",
        "window_size": int(config["windowing"]["window_size"]),
        "stride": int(config["windowing"]["stride"]),
        "target_window_far": chosen["target_far"],
        "persistence_windows": chosen["persistence"],
        "thresholds": chosen["thresholds"],
        "selection_rule": payload["selection_rule"],
        "selection_evidence": {
            "run_far_selection_half_1_250": chosen["far_selection_half"]["run_far"],
            "run_far_reporting_half_251_500": chosen["far_reporting_half"]["run_far"],
            "mean_fdr_observable_18": chosen["mean_fdr_observable"],
            "mean_delay_observable_18": chosen["mean_delay_observable"],
        },
        "calibration_sources": payload["calibration_sources"],
        "secondary_statistics_not_recalibrated": ["cva_T2", "cva_Tr", "dpca_T2"],
        "secondary_note": (
            "cva_T2, cva_Tr and dpca_T2 are diagnostic only: they never enter the OR-gate "
            "(primary dynamic = CVA Q) and never appear in a headline table. They stay at "
            "their v1 per-sample values so this file defines exactly the three statistics "
            "that are quoted, rather than a half-recalibrated detector."
        ),
        "not_used_for_selection": [
            "TEP_FaultFree_Testing runs 251-500 (reporting only)",
            "Protocol A test component",
            "Protocol A validation FDR (reported as trade-off, not used to choose)",
        ],
        "provenance": {
            "lstm_threshold_derivation": "scripts/derive_lstm_threshold.py -> outputs/evaluation/detector_v2/lstm_threshold_derivation.json",
            "sweep": "scripts/fit_operating_point_v2.py -> outputs/evaluation/detector_v2/operating_point_sweep.json",
        },
        "artifact_sha256": {k: sha256_file(v) for k, v in artifacts.items() if Path(v).exists()},
    }
    save_json(FROZEN, frozen)
    # The SHA-256 of the frozen file itself is recorded next to it, because a
    # file cannot hash itself.
    digest = sha256_file(FROZEN)
    save_json(FROZEN.with_suffix(".sha256.json"), {
        "file": FROZEN.name,
        "sha256": digest,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"wrote {FROZEN}\n  sha256 {digest}")

    # Config copy (never an edit of the live config) with smoothing disabled.
    src = ROOT / args.config
    text = src.read_text(encoding="utf-8")
    import re

    new, n = re.subn(
        r"(thresholding:\s*\n(?:\s+.*\n)*?\s+smoothing:\s*)\S+",
        r"\1none",
        text,
    )
    if n != 1:
        raise SystemExit(f"could not set smoothing in {src} (matches={n})")
    CFG_COPY.write_text(
        "# Copy of config_a5000.yaml with dynamic smoothing explicitly disabled.\n"
        f"# Derived from {src.name} by scripts/fit_operating_point_v2.py; the live config\n"
        "# is untouched. The frozen v2 thresholds were fitted on unsmoothed window scores.\n"
        + new,
        encoding="utf-8",
    )
    print(f"wrote {CFG_COPY}")


if __name__ == "__main__":
    main()
