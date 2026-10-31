#!/usr/bin/env python
"""Refresh a follow-up manifest's measured cost and file hashes, without calls.

A manifest is written at the end of a build, so a build that bought no prose (a
cache replay, or a structural val/test rebuild) leaves the *train* run's
``cost_estimate`` block saying what was projected before the run rather than what
the model actually took. This recomputes that block from the artefacts on disk --
the teacher cache's own per-response timings and the train turns -- and rehashes
every file, so the manifest describes the directory rather than the last command.

It never calls a model and never rewrites the dataset files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.followup.teacher import TeacherCache  # noqa: E402

SECONDS_PER_CALL_ASSUMPTION = 10.9


def load_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="outputs/benchmark_v2/followup")
    args = ap.parse_args()
    out = Path(args.dir)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_path = out / "train.jsonl"
    teacher_turns = [
        turn
        for row in load_rows(train_path)
        for turn in row["turns"]
        if turn.get("answer_source") == "teacher"
    ]
    fresh = [turn for turn in teacher_turns if not turn.get("prompt_cache_hit")]
    seconds_this_build = round(sum(float(turn.get("teacher_seconds") or 0.0) for turn in teacher_turns), 2)
    per_fresh = (round(sum(float(t.get("teacher_seconds") or 0.0) for t in fresh) / len(fresh), 3)
                 if fresh else None)

    model = manifest.get("teacher_model") or "qwen3:30b"
    cache_path = out / "teacher_cache.jsonl"
    cache = TeacherCache(cache_path, model=model, seed=int(manifest.get("seed", 42)))
    stats = cache.seconds_stats()
    measured_rate = per_fresh or stats.get("mean_seconds")

    calls = len(teacher_turns)
    retry_calls = int(round(calls * float(manifest.get("acceptance", {}).get("rejection_rate", 0.0))))
    manifest["cost_estimate"] = {
        "teacher_turns": calls,
        "distinct_prompts": calls,
        "measured": {
            "fresh_calls_this_build": len(fresh),
            "cache_hits_this_build": len(teacher_turns) - len(fresh),
            "seconds_per_fresh_call": per_fresh,
            "seconds_total_this_build": seconds_this_build,
            "cache": stats,
            "serial_hours_if_all_fresh": (
                round(calls * measured_rate / 3600.0, 2) if measured_rate else None
            ),
        },
        "projection_assumption": {
            "seconds_per_call": SECONDS_PER_CALL_ASSUMPTION,
            "serial_hours": round(calls * SECONDS_PER_CALL_ASSUMPTION / 3600.0, 2),
            "with_one_retry_per_rejection": {
                "extra_calls": retry_calls,
                "serial_hours": round((calls + retry_calls) * SECONDS_PER_CALL_ASSUMPTION / 3600.0, 2),
            },
            "rejection_rate": manifest.get("acceptance", {}).get("rejection_rate"),
        },
        "teacher": manifest.get("teacher"),
    }

    projection = dict(manifest.get("full_run_projection") or {})
    if projection and measured_rate:
        projection["seconds_per_call_basis"] = "measured"
        projection["full_train_serial_hours"] = round(
            int(projection.get("full_train_calls", 0)) * measured_rate / 3600.0, 2)
        projection["full_train_with_retry_hours"] = round(
            int(projection.get("full_train_calls", 0)) * (1 + retry_calls / max(calls, 1))
            * measured_rate / 3600.0, 2)
    manifest["full_run_projection"] = projection

    hashes: Dict[str, str] = {}
    for path in sorted(out.glob("*")):
        if path.suffix.lower() in (".jsonl", ".csv"):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["files_sha256"] = hashes

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"measured seconds/call (cache, n={stats.get('n')}): {stats.get('mean_seconds')}")
    print(f"projection hours at measured rate: {projection.get('full_train_serial_hours')} "
          f"(assumption was {manifest['cost_estimate']['projection_assumption']['serial_hours']})")
    print(f"rehashed {len(hashes)} files -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
