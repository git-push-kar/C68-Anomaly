"""Restore outputs/preprocessing/baseline_stats.json and prove it matches the scaler.

`BaselineScaler.load` returns ``baseline=None`` when that file is absent, and
every consumer dereferences it (`build_event(baseline=self.scaler.baseline)`),
so the app cannot close an event without it. The file was missing from the
checkout along with the detector artifacts.

The statistics are recomputed from the same normal-operation file the scaler was
fitted on (TEP_FaultFree_Training.csv, 52 canonical columns), and then checked
against the fitted StandardScaler's own ``mean_`` and ``var_``: if the numbers
agree, the file describes the same data the scaler saw, and the check is the
reason to trust it rather than the comment.

    python scripts/restore_baseline_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preprocessing.scaler import BASELINE_FILE, load_scaler  # noqa: E402
from preprocessing.tep_loader import CANONICAL_NAMES, _normalize_rieth_columns  # noqa: E402
from utils import ensure_dir, load_config, save_json  # noqa: E402

TOL_REL = 1e-4


def main() -> None:
    config = load_config("configs/config_a5000.yaml")
    scaler_dir = Path(config["preprocessing"]["scaler_dir"])
    normal_csv = Path(config["paths"]["normal_data_path"]) / "TEP_FaultFree_Training.csv"

    scaler = load_scaler(scaler_dir)
    sc = scaler.scaler
    if not hasattr(sc, "mean_") or not hasattr(sc, "var_"):
        raise SystemExit(f"scaler is {type(sc).__name__}; cannot recover baseline stats")

    print(f"fitting file: {normal_csv}")
    chunks = []
    n_rows = 0
    for chunk in pd.read_csv(normal_csv, chunksize=250_000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _normalize_rieth_columns(chunk)
        if "faultNumber" in chunk.columns:
            chunk = chunk[chunk["faultNumber"] == 0]
        chunks.append(chunk[CANONICAL_NAMES].to_numpy(np.float32))
        n_rows += len(chunks[-1])
    values = np.concatenate(chunks, axis=0)

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ref_mean = np.asarray(sc.mean_, dtype=np.float64)
    ref_std = np.sqrt(np.asarray(sc.var_, dtype=np.float64))

    rel_mean = float(np.max(np.abs(mean - ref_mean) / np.maximum(np.abs(ref_mean), 1e-9)))
    rel_std = float(np.max(np.abs(std - ref_std) / np.maximum(np.abs(ref_std), 1e-9)))
    ok = rel_mean <= TOL_REL and rel_std <= TOL_REL

    payload = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "feature_names": list(CANONICAL_NAMES),
        "n_samples": int(n_rows),
    }
    out = ensure_dir(scaler_dir) / BASELINE_FILE
    save_json(out, payload)

    save_json(scaler_dir / "baseline_provenance.json", {
        "written_by": "scripts/restore_baseline_stats.py",
        "source_file": str(normal_csv),
        "n_samples": int(n_rows),
        "feature_order": "preprocessing.tep_loader.CANONICAL_NAMES",
        "verification": {
            "scaler_class": type(sc).__name__,
            "max_relative_error_mean": rel_mean,
            "max_relative_error_std": rel_std,
            "tolerance": TOL_REL,
            "matches_fitted_scaler": ok,
        },
    })

    print(f"rows: {n_rows}")
    print(f"max relative error vs scaler.mean_: {rel_mean:.3e}")
    print(f"max relative error vs scaler.std_:  {rel_std:.3e}")
    print(f"matches fitted scaler (tol {TOL_REL}): {ok}")
    print(f"wrote {out}")
    if not ok:
        raise SystemExit("baseline statistics do not match the fitted scaler")


if __name__ == "__main__":
    main()
