"""Pin the untracked detector artifacts and verify them on every build.

The dataset depends on four files that git does not track, because `outputs/` is
ignored:

    outputs/anomaly_detector/model.pt
    outputs/anomaly_detector/dynamic/cva_model.npz
    outputs/anomaly_detector/dynamic/dpca_model.npz
    outputs/preprocessing/scaler.pkl

Two checkouts of the same commit can hold different versions of them, and the
failure mode is silent: a stale `model.pt` can load with mismatched weights, or a
different scaler produces a different evidence vector and a different dataset
under the same name. Both build scripts call `verify_inputs` before doing any
work, so a mismatch stops the build with the file names and both hashes instead
of writing artifacts nobody can identify later.

The pin lives in `configs/benchmark_inputs.json` and is rewritten only by
`--update-inputs`, i.e. deliberately, by whoever re-trained or re-fitted an
artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

#: Artifact name -> path relative to the repo root.
ARTIFACTS: Dict[str, str] = {
    "model.pt": "outputs/anomaly_detector/model.pt",
    "cva_model.npz": "outputs/anomaly_detector/dynamic/cva_model.npz",
    "dpca_model.npz": "outputs/anomaly_detector/dynamic/dpca_model.npz",
    "scaler.pkl": "outputs/preprocessing/scaler.pkl",
}

PIN_PATH = "configs/benchmark_inputs.json"


class InputMismatch(RuntimeError):
    """A pinned input artifact is missing or has changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_artifacts(root: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Current hash and size of every artifact, missing files included."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for name, rel in ARTIFACTS.items():
        path = Path(root) / rel
        if path.exists():
            out[name] = {"path": rel, "sha256": sha256_file(path),
                         "bytes": str(path.stat().st_size)}
        else:
            out[name] = {"path": rel, "sha256": None, "bytes": None}
    return out


def verify_inputs(root: Path, pin_path: Optional[Path] = None,
                  names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Optional[str]]]:
    """Fail loudly unless every pinned artifact is present and unchanged."""
    root = Path(root)
    pin_file = Path(pin_path) if pin_path is not None else root / PIN_PATH
    if not pin_file.exists():
        raise InputMismatch(
            f"input pin {pin_file} is missing; create it with --update-inputs "
            "once the detector artifacts are final"
        )
    pinned = json.loads(pin_file.read_text(encoding="utf-8"))
    expected = pinned.get("artifacts") or {}
    current = hash_artifacts(root)
    wanted = list(names) if names is not None else list(ARTIFACTS)

    problems = []
    for name in wanted:
        if name not in expected:
            problems.append(f"{name}: not pinned")
            continue
        have, want = current[name]["sha256"], expected[name]["sha256"]
        if have is None:
            problems.append(f"{name}: missing at {ARTIFACTS[name]}")
        elif have != want:
            problems.append(f"{name}: sha256 {have[:12]} != pinned {str(want)[:12]}")
    if problems:
        raise InputMismatch(
            "detector artifacts do not match "
            f"{pin_file}:\n  " + "\n  ".join(problems)
            + "\nIf the artifacts were deliberately replaced, re-run with "
              "--update-inputs and rebuild the events."
        )
    return current


def write_pin(root: Path, pin_path: Optional[Path] = None,
              note: str = "") -> Dict:
    root = Path(root)
    pin_file = Path(pin_path) if pin_path is not None else root / PIN_PATH
    payload = {
        "note": note or ("SHA-256 of the untracked detector artifacts the "
                         "benchmark dataset was built from."),
        "artifacts": hash_artifacts(root),
    }
    pin_file.parent.mkdir(parents=True, exist_ok=True)
    pin_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "ARTIFACTS",
    "InputMismatch",
    "PIN_PATH",
    "hash_artifacts",
    "sha256_file",
    "verify_inputs",
    "write_pin",
]
