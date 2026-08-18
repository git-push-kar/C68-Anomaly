"""Shared utilities: configuration, logging, reproducibility, device, IO.

This module is deliberately dependency-light at import time (torch is imported
lazily) so that preprocessing / evidence code can run without the GPU stack.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

# Minimal defaults used only when the config file is absent. The authoritative
# configuration lives in configs/config.yaml.
_DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "log_level": "INFO",
    "paths": {
        "data_root": "data/raw",
        "normal_data_path": "data/raw/normal",
        "fault_data_path": "data/raw/faults",
        "processed_data_path": "data/processed",
        "llm_dataset_path": "data/llm",
        "output_dir": "outputs",
    },
    "dataset": {
        "format": "tep_52col_csv",
        "num_features": 52,
        "has_header": True,
        "delimiter": ",",
        "fault_onset_index": 161,
        "normal_file_pattern": "normal*.csv",
        "fault_file_pattern": "fault_*.csv",
        "missing_value_strategy": "interpolate",
    },
    "preprocessing": {
        "scaler": "standard",
        "scaler_dir": "outputs/preprocessing",
        "baseline_dir": "outputs/preprocessing",
    },
    "windowing": {"window_size": 60, "stride": 5, "dtype": "float32"},
    "anomaly_detector": {
        "model_dir": "outputs/anomaly_detector",
        "lstm": {"hidden_size": 32, "num_layers": 2, "latent_dim": 16, "dropout": 0.1},
        "train": {
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 1.0e-3,
            "val_ratio": 0.15,
            "early_stopping_patience": 8,
        },
        "threshold": {"method": "percentile", "percentile": 99.0, "k_std": 6.0},
    },
    "evidence": {
        "top_k_sensors": 5,
        "min_deviation_percent": 3.0,
        "baseline_windows": 20,
        "trend_smoothing_window": 8,
        "onset_sensitivity": 1.5,
        "max_temporal_events": 6,
    },
    "streaming": {"window_size": 60, "stride": 5, "replay_rate": 0.0},
    "events": {
        "db_path": "outputs/events.db",
        "consecutive_windows_to_confirm": 3,
        "min_separation_windows": 20,
        "max_event_windows": 200,
    },
    "llm": {
        "base_model": "OpenGVLab/InternVL2-2B",
        "adapter_name": "tep_rca",
        "adapter_dir": "outputs/tep_rca_adapter",
        "trust_remote_code": True,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "training": {
            "use_4bit": True,
            "bf16": True,
            "fp16": False,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "learning_rate": 2.0e-4,
            "max_sequence_length": 4096,
            "warmup_ratio": 0.03,
            "lr_scheduler": "cosine",
            "gradient_checkpointing": True,
        },
        "generation": {"max_new_tokens": 768, "do_sample": False},
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (base is not mutated)."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load YAML config from ``configs/config.yaml`` merged over defaults."""
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        logging.getLogger(__name__).warning(
            "Config file %s not found; using built-in defaults.", config_path
        )
        return deepcopy(_DEFAULT_CONFIG)
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(_DEFAULT_CONFIG, loaded)


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    """Resolve a configured path (relative paths are anchored at project root)."""
    value = config.get("paths", {}).get(key)
    if value is None:
        value = _DEFAULT_CONFIG["paths"][key]
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: str = "INFO", log_file: Optional[Union[str, Path]] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    """Seed every reproducibility source we control."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch not installed: still seed numpy/random
        pass


def get_device(prefer_cuda: bool = True) -> Any:
    """Return the best available torch device with GPU detection + CPU fallback."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():  # pragma: no cover - apple silicon
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def gpu_memory_summary() -> str:
    """Human readable GPU summary (torch)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA GPU detected"
        lines = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            lines.append(f"GPU {i}: {props.name} ({total_gb:.1f} GB)")
        return "; ".join(lines)
    except Exception as exc:  # pragma: no cover
        return f"GPU detection failed: {exc}"


def count_parameters(model: Any) -> Tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def save_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types (recursively)."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def print_header(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def bootstrap_sys_path() -> None:
    """Make the project root importable when running scripts/ from anywhere."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def set_torch_threads(max_threads: Optional[int] = None) -> None:
    if max_threads is None:
        max_threads = int(os.environ.get("OMP_NUM_THREADS", 0)) or None
    try:
        import torch

        if max_threads:
            torch.set_num_threads(max_threads)
    except ImportError:
        pass