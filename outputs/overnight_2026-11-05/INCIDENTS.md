# Incidents, overnight run 2026-11-05

## 1. Seed 43 wrote into the seed-42 pilot directory (caught 21:53, stopped)

The generated E configs inherited `adapter_dir: outputs/tep_rca_v2_prose_pilot_seed42`
from the pilot config, and `llm/train_adapter.py` writes through `adapter_dir`. The seed-43
run therefore saved its step-48 checkpoint into the benchmarked seed-42 directory, replacing
`checkpoint-48` there. Stopped at step 54 by killing the queue.

- Surviving and unchanged: the **final pilot adapter**
  `outputs/tep_rca_v2_prose_pilot_seed42/tep_rca_v2_prose_pilot_seed42/adapter_model.safetensors`
  — 73,916,992 bytes, mtime 2026-11-02 16:21, sha256 `1f40b80af495d81e6ecf651ce98a98081936a43b4123233892f06ca09cd612ff`. This is the file the 520 stored
  B generations came from; the B cell and every reading derived from it are unaffected.
- Also unchanged: checkpoints 96, 144, 192, 240, 288 (mtimes 2026-11-02).
- Lost: the pilot's intermediate `checkpoint-48`, overwritten by seed 43's step-48. Moved to
  `quarantine/seed43_into_seed42_checkpoint-48_2026-11-05/` rather than deleted, so the
  substitution stays visible.
- Fix: each job now has its own `adapter_dir`, and `scripts/train_queue.py` refuses to start
  if any job points at the pilot directory or shares a directory with another job.
