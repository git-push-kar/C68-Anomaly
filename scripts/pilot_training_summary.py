"""Training summary for the single authorised pilot run (3B-13 step 2). Read only."""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/tep_rca_v2_prose_pilot_seed42"
OUT = ROOT / "outputs/analysis/phase3a/fit/pilot_training_summary.json"
ESTIMATE = 1 * 3600 + 44 * 60


def main() -> int:
    rows = [json.loads(line) for line in (PILOT / "train_telemetry.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [row for row in rows if row.get("loss") is not None]
    log = (PILOT / "train.log").read_text(encoding="utf-8", errors="replace")
    found = re.search(r"'train_runtime': ([0-9.]+)", log)
    runtime = float(found.group(1)) if found else None
    step_seconds = [float(row["step_seconds"]) for row in steps if row.get("step_seconds")]
    adapter = PILOT / "tep_rca_v2_prose_pilot_seed42" / "adapter_model.safetensors"
    preflight = json.loads((PILOT / "preflight_gpu.json").read_text(encoding="utf-8"))
    recorded = json.loads((PILOT / "training_summary.json").read_text(encoding="utf-8"))
    summary = {
        "adapter": "tep_rca_v2_prose_pilot_seed42",
        "steps_completed": max(int(row["step"]) for row in rows),
        "epochs_completed": 3.0,
        "train_examples": 1523,
        "loss_first": steps[0]["loss"], "loss_first_step": steps[0]["step"],
        "loss_last": steps[-1]["loss"], "loss_last_step": steps[-1]["step"],
        "loss_min": min(row["loss"] for row in steps),
        "final_reported_train_loss": 0.4263756589757072,
        "logged_points": len(steps),
        "mean_step_seconds": round(statistics.mean(step_seconds), 2),
        "median_step_seconds": round(statistics.median(step_seconds), 2),
        "total_train_seconds": runtime,
        "total_train_hours": round(runtime / 3600.0, 3) if runtime else None,
        "estimate_seconds": ESTIMATE,
        "estimate_hours": 1.734,
        "ratio_to_estimate": round(runtime / ESTIMATE, 3) if runtime else None,
        "max_gpu_mem_mib": max(float(row["gpu_mem_mib"]) for row in rows
                               if row.get("gpu_mem_mib") is not None),
        "gpu_total_mib": 24564,
        "max_temperature_c": max(float(row["gpu_temp_c"]) for row in rows
                                 if row.get("gpu_temp_c") is not None),
        "mean_gpu_util_pct": round(statistics.mean(float(row["gpu_util_pct"]) for row in rows
                                                   if row.get("gpu_util_pct") is not None), 1),
        "max_memory_allocated_gib": None,
        "max_memory_allocated_note": ("not logged: telemetry wrote nvidia-smi gpu_mem_mib, not "
                                      "torch.max_memory_allocated; the allocated figure recorded "
                                      "is the pre-flight one"),
        "preflight_max_memory_allocated_gib": preflight.get("max_memory_allocated_gib"),
        "adapter_file": str(adapter.relative_to(ROOT)),
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest() if adapter.exists() else None,
        "adapter_sha256_recorded_at_save": recorded["files_sha256"][
            "tep_rca_v2_prose_pilot_seed42/adapter_model.safetensors"],
        "final_checkpoint": "checkpoint-288",
        "checkpoints": sorted(path.name for path in PILOT.glob("checkpoint-*")),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
