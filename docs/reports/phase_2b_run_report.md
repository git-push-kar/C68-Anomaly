# Phase 2B run report — manual execution of the dataset pipeline

Repo: `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
Executed: 2026-10-30. CPU only; no GPU work, no LLM training, no teacher calls.

---

## 1. Summary

The Phase 2B pipeline was executed end to end from a clean tree and every
artifact reproduced **byte for byte**: the three dataset JSONL files and the
audit sheet carry the same SHA-256 as the build that Phase 2B was reported
against, and all 194 tests pass.

One real defect surfaced, and only because the run was done from source rather
than by trusting the artifacts. `build_benchmark_events.py`'s oracle pass took
*every* miss, so a full rebuild produced **462** oracle windows for the 20 the
dataset contains: 442 of them were ordinary detector misses for faults 3 and 9,
i.e. manufactured training items for faults the detector cannot see. The
committed script therefore could not have produced the committed artifacts. It
is fixed in `224b136`, and the rebuild after the fix reproduces the shipped
hashes exactly.

The teacher step was **not** run: it is the one part of Phase 2B that remains
owner-gated (~5.6 h, see §8).

---

## 2. Commands and measured runtimes

| step | command | wall clock | outcome |
|---|---|---|---|
| preflight | `git status`, artifact `ls` | <1 s | clean tree, 7/7 artifacts present |
| smoke | `build_benchmark_events.py --faults 3 --max-runs 5 --skip-normal --out .../_smoke` | 83 s | 15 runs, 12 misses, 12 oracles |
| events (1st) | `build_benchmark_events.py` | 616.1 s | 4,438 events, 462 misses, **462 oracles** ← defect |
| events (2nd, fixed) | `build_benchmark_events.py` | 580.6 s | 4,438 events, 462 misses, **20 oracles** |
| dataset | `build_benchmark_dataset.py` | ~105 s | 2,305 / 959 / 3,668 items |
| tests | `python -m pytest tests -q` | 5.1 s | 194 passed |

Throughput was ~10 runs/s over the 10,000 faulty runs plus 500 fault-free runs.
The first events build ran against the unmodified script; the events directory
was deleted before the second so that no orphan `_oracle.json` record could be
re-read by the dataset loader (it globs `faulty/*/*.json`).

---

## 3. The defect

### 3a. Symptom

`build_summary.json` from the shipped build reported `oracle_events: 0` while
`oracle.jsonl` held 20 rows. That inconsistency is what started the
investigation.

### 3b. What the code actually does

The pass read:

```python
if not args.skip_oracle and misses:
    for m in misses:                      # every miss, both reasons
        by_fault[int(m["fault"])].add(int(m["simulationRun"]))
```

`misses` holds two reasons: `detector_miss` (442, faults 3 and 9 only) and
`unclosed_at_run_end` (20, one per fault). Taking all of them builds a window
for a fault the fused detector never alarmed on, at a fixed offset after onset.
The Phase 2B report argued explicitly against that: it would "manufacture events
for faults 3 and 9" and inflate coverage.

`git log -S"unclosed_at_run_end"` shows the reason string existed from the
first commit of the script, but was only ever *recorded*, never *used as a
filter*. No version of the committed script applied it.

### 3c. Why the shipped artifacts disagreed

The committed script, run in full, produces 462 oracles and a summary saying
462. The shipped artifacts have 20 and a summary saying 0. So the artifact set
was not produced by a single run of the committed script — the evidence is
consistent with a `--skip-oracle` run (which writes `oracle_events: 0` and
explains the stale summary) followed by a separate oracle generation that did
apply the reason filter. That sequence is an inference from the two files'
mtimes and contents; I cannot prove it from git history.

### 3d. Fix

`224b136` filters the oracle pass on the miss reason and records the number of
skipped misses so the count can be read against the miss list:

```python
oracle_misses = [m for m in misses if m.get("reason") == "unclosed_at_run_end"]
...
"oracle_events": len(oracle),
"oracle_skipped_misses": len(misses) - len(oracle_misses),
```

Raw detector misses stay in `misses.jsonl` and are still excluded from every
split component, as before.

---

## 4. Reproduction evidence

Hashes captured from the shipped manifest **before** the rebuild, compared with
the manifest produced **after** it:

| file | result | sha256 (16) |
|---|---|---|
| `train.jsonl` | MATCH | `81dced9e23d739a8` |
| `val.jsonl` | MATCH | `988d1ecb7c2cc231` |
| `test.jsonl` | MATCH | `04a8b40b2d4e2c17` |
| `audit_sample.csv` | MATCH | `55cc5650cfd0124e` |
| `operating_point_sha256` | MATCH | `e7fe73eb0b75b4d5` |
| `kb_sha256` | MATCH | `f01236ffe01b7b4f` |
| `prompt_template_sha256` | MATCH | `fe4b8e0d6971fd9a` |

Item counts are identical to the shipped build:

| split | items | open-set | oracle | NORMAL_TRANSIENT | teacher acc/rej |
|---|---|---|---|---|---|
| train | 2,305 | 195 | 20 | 117 | 0 / 0 |
| val | 959 | 456 | — | 47 | 0 / 0 |
| test | 3,668 | 1,814 | — | 40 | 0 / 0 |

Event-level, from the fixed build: 4,438 faulty events (train 2,168 / val 456 /
test 1,814), 462 misses, 20 oracles at offset sample 229 (delay 69 after onset),
204 NORMAL_TRANSIENT (train 8 confirmed / 109 unconfirmed, val 2 / 45, test
40 / 0).

---

## 5. Split integrity

Emitted by the dataset build itself; an overlap raises instead of being
reported.

| family | components (runs) | pairwise disjoint | held-out faults absent |
|---|---|---|---|
| A_seen_faults_run_disjoint | train 6,000 / val 2,000 / test 2,000 | yes | n/a |
| B fold_1_cooling_loop_holdout | train 5,100 / val 1,700 / test 1,500 / test_seen 1,700 | yes | `[4, 11, 15]` yes |
| B fold_2_valve_sticking_family | train 5,400 / val 1,800 / test 1,000 / test_seen 1,800 | yes | `[14, 15]` yes |

4,458 of the 4,662 built keys (4,438 events + 20 oracles) appear in the
protocol components; the remaining 204 are the fault-free NORMAL_TRANSIENT runs,
which are not part of any faulty-run split by design.

---

## 6. Tests

`python -m pytest tests -q` → **194 passed in 5.10s**, exit 0.

No test was added for the oracle filter. The gate for it is empirical and
stronger than a unit test would be: with the filter, the dataset matches the
shipped hashes; without it, the build emits 462 oracles and cannot. A future
test would have to import the script's `main`, which the events builder does
not expose as a callable unit.

---

## 7. Gate

| gate item | status | evidence |
|---|---|---|
| All runtime artifacts present before the run | **PASS** | 7/7 found, §2 |
| Smoke test exercises the oracle path | **PASS** | 12 misses → 12 oracles, removed after |
| Events build reproduces the reported counts | **PASS** | 4,438 / 462 / 20 / 204 |
| Oracle gating matches documented intent | **PASS** | fixed `224b136`; 20 not 462 |
| Dataset builds with no split overlap | **PASS** | §5, all three families |
| Dataset byte-identical to shipped artifacts | **PASS** | 4 file hashes + 3 input hashes MATCH |
| Full test suite passes | **PASS** | 194 passed |
| Teacher pilot re-run | **NOT RUN** | already measured in Phase 2B (88% accept) |
| Full teacher job | **NOT RUN** | owner-gated, ~5.6 h, §8 |

---

## 8. What is still open

1. **The teacher job.** 2,305 train calls at 7.8 s ≈ 5.6 h serial. It has no
   checkpointing: `build_benchmark_dataset.py` assembles every item in memory
   and writes once at the end, so a crash in hour five loses the whole run. If
   it is to be run at that length, resume support (write accepted items per
   batch, skip on restart) should come first — that is a code change, not a
   rerun.

2. **Regenerating is the only provenance.** `outputs/` is git-ignored, so the
   dataset exists on disk but not in history. The manifest hashes pin it, and
   §4 shows they are reproducible from the committed tree, which is the point of
   having run this.

3. **Artifacts are version-sensitive.** The run depends on four untracked
   files — `model.pt`, the two dynamic models and `scaler.pkl`. A checkout with
   an August-era `model.pt` (173 KB, hidden 32) fails to load against the current
   architecture (hidden 128): this happened in the `C68-Anomaly` clone. Worth a
   recorded hash of those four files in the manifest so a mismatch fails loudly
   instead of producing a silently different dataset.

---

## 9. Deviations

- **D1 — the defect was fixed, not worked around.** The alternative was to keep
  the ad-hoc 20-oracle artifact and leave the script unable to reproduce it. The
  script was corrected so source and artifacts agree.
- **D2 — events directory deleted before the rebuild** rather than patched in
  place, because the dataset loader globs the directory and would have picked up
  442 orphan oracle records.
- **D3 — no new test for the filter** (§6); justification given rather than
  adding a test that only re-asserts the build's own output.
