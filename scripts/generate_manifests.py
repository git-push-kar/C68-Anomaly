"""Generate deterministic run-level manifests for detector and RCA.

Creates:
  data/processed/manifests/detector_split.json  -> 350/75/75 normal runs
  data/processed/manifests/rca_known_split.json -> per-fault 350/75/75 runs (known-fault/unseen-run)
  data/processed/manifests/rca_unseen_split.json -> fault-ID holdout (train/val/test fault sets)

All splits are by simulationRun, before windowing, deterministic via seed.
"""
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_dir, load_config, save_json

def deterministic_split(items, n_train, n_val, n_test, seed=42):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train+n_val]),
        "test": sorted(shuffled[n_train+n_val:n_train+n_val+n_test]),
    }

def main():
    config = load_config()
    seed = int(config.get("seed", 42))
    out_dir = ensure_dir(Path(config["paths"]["processed_data_path"]) / "manifests")

    # Detector: normal runs 1..500
    normal_runs = list(range(1, 501))
    det = deterministic_split(normal_runs, 350, 75, 75, seed=seed)
    det_manifest = {
        "seed": seed,
        "method": "simulation_run",
        "window_size": int(config["windowing"]["window_size"]),
        "stride": int(config["windowing"]["stride"]),
        "features": ["XMEAS_1..41", "XMV_1..11"],
        "train_runs": det["train"],
        "validation_runs": det["val"],
        "test_runs": det["test"],
        "num_train_runs": 350,
        "num_validation_runs": 75,
        "num_test_runs": 75,
    }
    save_json(out_dir / "detector_split.json", det_manifest)
    print(f"Detector manifest: {out_dir/'detector_split.json'}")

    # RCA known-fault: per fault 350/75/75 runs
    rca_known = {}
    for fid in range(1, 21):
        split = deterministic_split(list(range(1, 501)), 350, 75, 75, seed=seed+fid)
        rca_known[str(fid)] = split
    save_json(out_dir / "rca_known_split.json", {
        "seed": seed,
        "method": "simulation_run_per_fault",
        "fault_ids": list(range(1, 21)),
        "splits": rca_known,
        "note": "Same fault IDs in all splits, disjoint runs - tests trajectory generalization"
    })
    print(f"RCA known manifest: {out_dir/'rca_known_split.json'}")

    # RCA unseen-fault: hold out fault IDs
    # Use 3 configs for robustness, primary is 15,21 as before plus 2 more
    unseen_configs = {
        "primary": {"train": [1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19,20,22], "val": [4,14], "test": [15,21]},
        "config_A": {"train": [1,2,3,4,5,6,7,8,9,10,11,12,13,14], "val": [16,17], "test": [18,19,20]},
        "config_B": {"train": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], "val": [19,20], "test": [1,2,3]},
    }
    save_json(out_dir / "rca_unseen_split.json", {
        "seed": seed,
        "method": "fault_id_holdout",
        "configs": unseen_configs,
        "note": "No trajectory from test fault IDs appears in train - tests fault generalization, 21/22 synthetic excluded from primary (1-20 only)"
    })
    print(f"RCA unseen manifest: {out_dir/'rca_unseen_split.json'}")

if __name__ == "__main__":
    main()
