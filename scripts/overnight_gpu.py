"""Launch E (retrained adapter) for seeds 43 and 44, sequentially, on the A5000.

D7 approved on 2026-11-05: 3 seeds x 288 steps, teacher corpus fixed before seed 1
(PROTOCOL 30.4 item 19). Seed 42 already exists as the pilot adapter, so only two runs are made
here. Each config is a copy of the pilot config with the seed and adapter identity changed and
nothing else touched.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/config_prose_pilot_seed42.yaml"
PY = ROOT / "venv_train/Scripts/python.exe"
OUT = ROOT / "outputs/overnight_2026-11-05"
LOG = OUT / "gpu_E.log"


def log(message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = "[%s] %s" % (stamp, message)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    text = BASE.read_text(encoding="utf-8")
    status = {}
    for seed in (43, 44):
        config = ROOT / ("configs/config_prose_e_seed%d.yaml" % seed)
        body = (re.sub(r"(?m)^(\s*)seed: 42$",
                     lambda m: m.group(1) + "seed: %d" % seed, text)
                    .replace("adapter_name: tep_rca_v2_prose_pilot_seed42",
                             "adapter_name: tep_rca_v2_prose_e_seed%d" % seed)
                    .replace("ONE authorised run", "D7-authorised run of three (2026-11-05)",
                             1))
        config.write_text(body, encoding="utf-8")
        log("starting E seed %d with %s" % (seed, config.name))
        started = datetime.now()
        result = subprocess.run([str(PY), "scripts/train_tep_adapter.py", "--config",
                                 str(config)], cwd=str(ROOT), capture_output=True, text=True)
        minutes = round((datetime.now() - started).total_seconds() / 60, 1)
        tail = (result.stdout or "")[-1200:] + (result.stderr or "")[-1200:]
        status["seed%d" % seed] = {"returncode": result.returncode, "minutes": minutes,
                                   "tail": tail}
        log("seed %d finished rc=%d in %.1f min" % (seed, result.returncode, minutes))
        (OUT / ("gpu_E_seed%d.json" % seed)).write_text(
            json.dumps(status["seed%d" % seed], indent=2), encoding="utf-8")
        if result.returncode != 0:
            log("stopping: seed %d failed" % seed)
            break
    (OUT / "gpu_E_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
