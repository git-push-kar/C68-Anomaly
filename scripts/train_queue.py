"""Run the D7 queue: E seed 43, E seed 44, then F, one at a time.

The step-time guard in llm/train_adapter.py defaults to 45 s and is honest about
being a resource guard; this machine measured 73.1 s/step on 2026-11-05, so the
queue raises it to 180 s through TEP_MAX_STEP_SECONDS and records that in
train_queue.jsonl. A job that fails stops the queue; it never silently skips.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "overnight_2026-11-05"
JOBS = [
    ("E_seed43", "configs/config_prose_e_seed43.yaml"),
    ("E_seed44", "configs/config_prose_e_seed44.yaml"),
    ("F_seed42", "configs/config_followup_f_seed42.yaml"),
]


def preflight() -> None:
    seen = {}
    for tag, cfg in JOBS:
        body = (ROOT / cfg).read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*adapter_dir:\s*(\S+)\s*$", body)
        if not match:
            raise SystemExit("%s: no adapter_dir in %s" % (tag, cfg))
        target = match.group(1)
        if "prose_pilot_seed42" in target:
            raise SystemExit("%s: adapter_dir is the pilot adapter (%s); refusing" % (tag, target))
        if target in seen:
            raise SystemExit("%s: adapter_dir %s already used by %s" % (tag, target, seen[target]))
        seen[target] = tag
        print("%s: adapter_dir %s" % (tag, target), flush=True)


def main() -> int:
    preflight()
    env = dict(os.environ, TEP_MAX_STEP_SECONDS="180")
    for tag, cfg in JOBS:
        started = time.time()
        log = OUT / f"train_{tag}.log"
        with open(log, "w", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {tag} {cfg}\n")
            handle.flush()
            rc = subprocess.call(
                [str(ROOT / "venv_train/Scripts/python.exe"),
                 "scripts/train_tep_adapter.py", "--config", cfg],
                cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env)
        record = {"job": tag, "config": cfg, "returncode": rc,
                  "minutes": round((time.time() - started) / 60.0, 1),
                  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(OUT / "train_queue.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if rc != 0:
            print(f"stopping: {tag} failed rc={rc}", flush=True)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
