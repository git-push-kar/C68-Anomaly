# Phase 2B close-out report

Repo: `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
CPU only. Teacher: 200 calls (the budgeted pilot). No LLM training.

---

## 1. Summary

All twelve close-out items are implemented and the dataset was rebuilt from
source. The two findings that matter:

1. **The 20 "unclosed" runs were a selection bug, not a stream artifact.** All of
   them were run 86, one per fault, and the aggregator had emitted a correct event
   that opens at window 20 and stays open to the end of the stream. The builder
   asked whether the event's *opening* window ended after onset (sample 159 vs
   onset 160) and discarded it. With the rule fixed and a fixed 20-window horizon,
   the miss list falls from 462 to **442, all detector misses**, and **no oracle
   window is needed or built**.

2. **Faults 16-20 were demanding an answer the menu forbade.** Their `menu`
   variant targeted `FAULT_17` while the menu excludes 17, and the "removed" twin
   was byte-identical to it. Both variants now target the open set and one item is
   emitted.

Pilot: **200 calls, 172 accepted (86.0%)**, mean 8.89 s. Rejections are 12
numbers, 11 invented sensor names, 5 ungrounded sensors — and inspection shows
they are genuine failures, not filter bugs (see §4.12).

---

## 2. Changes

| commit | what |
|---|---|
| `2f8db65` | `select_event` / `snapshot_block` / `oracle_candidates` in `llm/benchmark/events.py`; fixed-horizon builder; artifact pin + verification |
| `0a543b0` | opaque prompt ids, leak patterns and `prompt_leaks`; tests for twins, 3/9, cap, percent rendering |
| `1fbb233` | 16-20 unanswerable in both variants; 3/9 out of train; open-set cap; teacher resume, concurrency, provenance; licence hash |
| `5a45faa` | confidence leaves the model body; renderer prints "uncalibrated"; fixtures regenerated |

---

## 3. Events rebuilt

`python scripts/build_benchmark_events.py` → 418.7 s (was 580.6 s at the full
event length; the horizon cuts the evidence pass).

| quantity | before | after |
|---|---|---|
| events | 4,438 | **4,458** (train 2,188 / val 456 / test 1,814) |
| misses | 462 (442 detector + 20 unclosed) | **442, all `detector_miss`** |
| oracle windows | 20 | **0** |
| NORMAL_TRANSIENT | 204 | 204 (unchanged) |
| snapshot length | median 149 windows | **20 windows, every event** |
| events covering the onset | 4,438 | **4,458 / 4,458** |

The 20 extra events are the run-86 events recovered by the fix.

Event-open delay (open minus onset 160), median per fault: 59 for faults 1/4/14,
64-74 for 2/5/6/16, 84-94 for 15/20, and 414 (fault 3) / 359 (fault 9) — the
unobservable pair, as before. Runs with fewer than H windows left: **5** of 4,458
(their snapshot is truncated by the end of the stream, not by a choice).
Windows left at open: median 146, p05 137, p95 148.

The full-event build is kept at `outputs/benchmark_v2/events_full/` and no script
reads it.

---

## 4. Task-by-task

### 4.1 Fixed-horizon evidence view — done
H = 20 windows (100 samples), fixed before the rebuild, applied to faulty,
NORMAL_TRANSIENT and oracle events. A benchmark-only `--max-windows` argument
(default 20; `0` restores the full event). The live aggregator and `configs/*.yaml`
are untouched.

### 4.2 Unclosed runs — diagnosed and fixed at the cause
Run 86 for all 20 faults, 172 alarmed windows each, identical `max_consecutive`
= 12. The raw data is normal (max |z| 0.4 over the first 120 samples, no frozen
columns, distinct per fault), so this was never a data or flush problem. See §1.
**Oracles are no longer needed**: with no unclosed runs there is nothing to
patch, and detector misses are still never patched.

### 4.3 Oracle code — one shared, tested filter
`oracle_candidates()` in `llm/benchmark/events.py` filters on
`reason == "unclosed_at_run_end"`; the builder uses it, and the same module owns
`miss_reason()`, so the reason taxonomy exists once. Nine tests cover it
(`tests/test_benchmark_events.py`), including "a detector miss is never patched".

### 4.4 Leak check — clean on every prompt
`render_prompt` now hashes whatever id it is given, so the internal
`EVT-<split>-f<fault>-r<run>` cannot reach the prompt by accident. `prompt_leaks`
checks each prompt for the id shape, split names, run-number tokens, `EVT`/`NORM`,
`NORMAL_TRANSIENT`, and fault references (`FAULT_n`, `fault n`, KB fault names)
outside the menu block.

**4,634 prompts checked, 0 violations.** The build aborts on the first one.

### 4.5 Twins — collapsed for 16-20

| split | before | after | why |
|---|---|---|---|
| val | 959 | **834** | 125 events of 16-20 no longer duplicated |
| test | 3,668 | **3,168** | 500 events of 16-20 no longer duplicated |

val = 47 normal + 456 events + 331 twins (faults 1-15 only);
test = 40 normal + 1,814 events + 1,314 twins.

### 4.6 Faults 3 and 9

| split | items | events from 3/9 | flag |
|---|---|---|---|
| train | 0 | 0 (28 excluded during the build) | — |
| val | 12 | 6 | `unobservable: true` |
| test | 28 | 14 | `unobservable: true` |

They are also absent from teacher input, because the teacher only sees train.

### 4.7 Class balance

| train quantity | value |
|---|---|
| items | 1,939 = 1,367 classified + 455 unclassified + 117 NORMAL_TRANSIENT |
| unclassified before the cap | 793 (193 episodes + 600 from faults 16-20) |
| dropped | **338** (all from 16-20) |
| unclassified after | 455 (193 episodes + 262 from 16-20) |
| share of faulty train | **0.2497** |

The cap is computed on the total *after* dropping (`u ≤ cap·classified/(1−cap)`),
which is why the bound is 455 rather than 540. Test is untouched.

### 4.8 Teacher infrastructure
- **Resume**: one JSONL line per finished call in `teacher_cache.jsonl`, flushed
  immediately, keyed by item id.
- **Kill test**: started with `--teacher-limit 5`, killed at 2 records, restarted
  → `cached: 2, pending: 3, calls: 3`, five unique records, the first two ids
  preserved unchanged.
- **Budget bug found and fixed during that test**: `--teacher-limit` set the
  stratification order but not a cap, so the restart ran 34 calls instead of 5.
- **Concurrency**: `--concurrency N` (default 1) via a thread pool; Ollama serves
  one request at a time, so the default is right and the flag is for other
  servers.
- **Per-record provenance**: teacher tag, temperature, seed, `filter_sha256`
  (four filter files), `license_sha256`, `system_prompt_sha256`. A cached record
  produced under a different filter hash is re-run rather than reused.
- **Licence**: hashed from what the local runtime reports
  (`b87250e4478f…`). No licence is asserted from memory.
- **`qwen3:30b`** is recorded in `docs/PROTOCOL.md` §10.7 and in the manifest.
  `configs/*.yaml` were not edited.

### 4.9 Manifest input hashes — verified at the start of every build
`configs/benchmark_inputs.json` pins `model.pt` (`d0ec3f386f11`), `cva_model.npz`
(`97f27ae9ab10`), `dpca_model.npz` (`a1b29a5b086a`) and `scaler.pkl`
(`6b67a6a18e74`). Both builders call `verify_inputs` first and fail with the file
name and both hashes on a mismatch. This is the failure mode that bit the
`C68-Anomaly` checkout, where a stale 32-unit `model.pt` loaded against a 128-unit
architecture.

### 4.10 Confidence contract
`confidence` is removed from `ReportBody`; a model that emits it is a schema error
(asserted). `Report` carries an optional code-computed value that must name a
fitted calibrator, and the renderer prints **"uncalibrated (no calibrator
fitted)"**. Three renderer tests pin: no value without a calibrator, a calibrated
value shows its source, a model-authored value is rejected. No calibrator is
implemented.

### 4.11 Report fixes
- "2,288 faulty" was arithmetically wrong (2,288 + 117 ≠ 2,305); corrected to
  **2,188 = 2,168 events + 20 oracles**, with a note that the horizon rebuild
  supersedes the totals.
- NORMAL_TRANSIENT is now a per-split table: **8/109, 2/45, 0/40** confirmed vs
  unconfirmed, with the 10 confirmed false alarms called out separately.
- **Pilot example 4 was a wrong rejection.** "70.2% share" for a stored share of
  0.702 is a faithful unit conversion, not a fabricated number; the factor-100
  allowance already in the filter fixes it, and a test now pins both directions
  (70.2 accepted, a truncated 206 for 206.82 still rejected).

### 4.12 Fresh pilot — 200 calls on the rebuilt data
`python scripts/build_benchmark_dataset.py --teacher --teacher-limit 200`,
mean **8.89 s/call**, wall clock ~31 min (the first call carries the model load).

| family | accepted | family | accepted | family | accepted |
|---|---|---|---|---|---|
| fault_01 | 7/10 | fault_08 | 10/10 | fault_15 | 10/10 |
| fault_02 | 8/10 | fault_10 | 10/10 | fault_16 | 8/10 |
| fault_04 | 10/10 | fault_11 | 10/10 | fault_17 | 10/10 |
| fault_05 | **3/10** | fault_12 | 7/10 | fault_18 | 9/10 |
| fault_06 | 10/10 | fault_13 | 7/10 | fault_19 | 8/10 |
| fault_07 | 10/10 | fault_14 | 9/10 | fault_20 | 9/10 |
| normal | 10/10 | open_set | 7/10 | **all** | **172/200 = 86.0%** |

Rejections: 12 `number_not_in_store`, 11 `unknown_tag`, 5 `sensor_not_grounded`.

They are real failures, checked at the cause:
- `221%` was written for a *temperature* printed as `221.56`/`221.63` — a
  unit confusion, correctly rejected;
- `Condenser_Temperature`, `Condenser_Cooling_Water_Inlet_Temperature` and
  `Reactor_Cooling_Water_Valve` are not canonical tags in this repo's vocabulary:
  the model invented plausible sensor names;
- `A_Feed_Flow` resolves to `XMV_44` but was not in the evidence of a fault-20
  event, so it is a genuine ungrounded claim.

**Acceptance is not correctness.** An accepted answer can still be physically
sloppy — one fault-08 item says *"increasing trends in Stripper_Pressure
(-0.15%)"*, an increase described with a negative deviation. The filters check
grounding and schema, not internal consistency; `direction_conflict` only covers
the KB's initiating variable. Stated as a limitation below.

### 4.13 Audit sheet regenerated
`audit_sample.csv`, 100 rows from the rebuilt data (34 train / 12 val / 54 test),
stratified by family, with the human columns left blank.

---

## 5. Dataset as rebuilt

| split | items | unclassified | NORMAL_TRANSIENT | teacher accepted |
|---|---|---|---|---|
| train | 1,939 | 455 | 117 | 172 |
| val | 834 | 456 | 47 | — |
| test | 3,168 | 1,814 | 40 | — |

sha256: train `9281c67d9936…`, val `0ff423b9e268…`, test `4bdd0a935aae…`,
audit `0c907ed5e668…`.

Split integrity (checked during the build, aborts on violation): all three families
pairwise disjoint; fold 1 holds out `[4, 11, 15]`, fold 2 `[14, 15]`, both absent
from every seen component.

---

## 6. Commands

| step | command | wall clock |
|---|---|---|
| events | `python scripts/build_benchmark_events.py` | 418.7 s |
| dataset | `python scripts/build_benchmark_dataset.py` | 96 s |
| resume test | `--teacher --pilot 1 --teacher-limit 5` (killed, restarted) | 2 runs, ~90 s |
| pilot | `--teacher --teacher-limit 200` | ~31 min |
| tests | `python -m pytest tests -q` | 5.1 s |

---

## 7. Tests

**234 passed in 5.13 s** (194 before this phase; 40 added):

- `tests/test_benchmark_events.py` — 13 tests: onset-span selection (the run-86
  shape), horizon capping, oracle gating, miss reasons.
- `tests/test_benchmark_dataset.py` — leak detection (each token, plus a KB fault
  name in the evidence and the menu exemption), opaque ids, twins for 16-20, 3/9
  flag, open-set cap (including the post-drop bound), percent vs truncated numbers.
- `tests/test_response_renderer.py` — 4 confidence-contract tests.

---

## 8. Gate

| gate item | status | evidence |
|---|---|---|
| Leak check clean on every prompt | **PASS** | 4,634 checked, 0 violations; build aborts on one |
| Split overlap still zero | **PASS** | §5, all three families |
| Rebuilt counts reported | **PASS** | §3, §5 |
| Twins handled for 16-20 | **PASS** | val 959→834, test 3,668→3,168 |
| Faults 3/9 handled | **PASS** | out of train/teacher, flagged in val/test |
| Class balance ≤ 25% | **PASS** | 0.2497, 338 dropped, test untouched |
| Resume works (kill and restart a 5-call test) | **PASS** | cached 2 → calls 3 → 5 unique, ids preserved |
| Manifest verifies input hashes | **PASS** | `verify_inputs` at both build starts |
| Fresh pilot acceptance reported | **PASS** | 172/200 = 86.0%, by family and reason |
| Full test suite passes | **PASS** | 234 passed |
| Event-length and open-delay distributions re-reported | **PASS** | §3 |
| Full teacher job | **NOT RUN** | budgeted pilot only, by instruction |

---

## 9. Deviations

- **D1 — the oracle pass stays, as a no-op.** Removing it entirely would leave
  the reason taxonomy with one caller and no test surface. It is driven by the
  shared filter and produces 0 records on this data; the report says so rather
  than implying coverage.
- **D2 — a 20-window horizon truncates 5 runs.** Their snapshot is shorter than
  H because the event opens near the end of the stream. Recorded, not patched.
- **D3 — the cap drops 338 items** (35% of the 16-20 supply). The brief asks for
  ≤25% by subsampling 16-20, which is what this is, but it is a large share of that
  family and worth a second look before training.
- **D4 — `--teacher-limit` was broken** (ordering only, not a cap) and the resume
  test found it. Fixed in `1fbb233`; the 34 extra calls it produced went only to a
  scratch dataset that was deleted.
- **D5 — the audit sheet is unstratified by fault family in the strict sense**:
  it round-robins over families, so a 20-fault population gets ~5 rows each, and
  normal/open-set get one row per pass. It is a sample to *audit*, not a
  balanced estimator.

---

## 10. Open questions

1. **Physical consistency is unchecked.** Fault 08's accepted text pairs
   "increasing" with "-0.15%". A rule comparing trend words to the sign of the
   deviation in the same clause would catch it; it is not implemented because it
   is easy to fire on negations ("no longer increasing").
2. **Full teacher run.** 1,939 train items × 8.89 s ≈ **4.8 h**, resumable, so a
   crash costs at most one call. Approve when ready.
3. **Open-set cap share.** 262 of 600 faults-16-20 items survive. If that family
   is supposed to carry its own signal, the cap should be stated as a per-family
   budget instead of a global share.
4. **Fault 5's 30% acceptance** is the worst family and the most common failure is
   an invented sensor name. Either the prompt should list the sensor vocabulary
   (which risks hinting) or the teacher needs a sentence naming only shown sensors
   (it has one). Worth one experiment before the full run.
