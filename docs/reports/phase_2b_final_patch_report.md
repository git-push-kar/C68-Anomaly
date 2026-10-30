# Phase 2B final patch — fault-13 sign change, the last 52 re-calls, and the accounting

**Repo:** `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade` (head `0aaf1dd`).

**Teacher budget:** 60 of the 90 allowed API attempts (52 items, 8 retried, 4 rejected
twice). Estimated ~15 min before starting; measured **9.3 min of calls + ~4 min build**.
No training was run.

**Headline: the fault-13 KB change was priced and stopped, not applied.** Its menu line
is rendered into 5,731 of 5,870 prompts, so it is a dataset-wide re-render rather than a
one-line fix — the stop condition you set in 1(a). Everything else in the patch ran.

---

## 1. Fault 13: impact first, then stop

### (a) What would move

`scripts/replay_fault13_change.py impact`:

| quantity | value |
|---|---|
| old line | `FAULT_13: Reaction kinetics slow drift — Reactor_Pressure decreases first` |
| new line | `FAULT_13: Reaction kinetics slow drift — Reactor_Pressure increases first` |
| prompts containing the line | **5,731 / 5,870** (train 1,854/1,868, val 809/834, test 3,068/3,168) |
| prompts without it | 139 — every one a fault-13 open-set twin (14/25/100), the case where the entry is removed on purpose |
| same line in the teacher prompt | yes — `teacher_menu` rewrites only valve positions, so a measurement's line is carried through unchanged |
| teacher prompt changes | yes, for the same 5,731 |

**Stopped.** The student-visible menu text changes for 5,676 items that are not about
fault 13, which is exactly the condition you said to stop on.

A second, sharper reason came out of the same check: `FILTER_CODE_FILES` lists
`prompt_v2.py` but **not** `menu.py` or `fault_knowledge.py`. A KB-only edit therefore
moves the prompt text while leaving `filter_sha256` untouched, and both `run_teacher`'s
pending rule and `replay_cache` decide staleness from that hash alone. The 1,868 cached
answers written from the old line would be re-attached silently, with no record that they
came from a different menu. That has to be fixed before any KB edit is safe.

### (b) Why the sign is wrong, and where the record was wrong

| field | measurement over fault-13 train events |
|---|---|
| `trend` of `XMEAS_7` (ranked in 102 of 120) | `increasing` in **102/102** |
| `deviation_percent` sign | negative 41, positive 61 |

`PROTOCOL.md` said the KB "contradicts in **0/102**". That figure matches neither field
and is withdrawn in this commit. The KB says `decrease`; relative to the trend that
contradicts 102/102, relative to the deviation sign it contradicts 61/102. The trend is
the field the rejection rule already reads for every other fault, so the KB is the odd
one out — but "the deviation is negative in 41" is why the earlier note could look
defensible, and it belongs in the record rather than being replaced by a cleaner claim.

### (c) Replay of the cache (`scripts/replay_fault13_change.py replay`)

119 fault-13 train items, all 119 accepted today.

| scenario | newly rejected | note |
|---|---|---|
| gerunds added, KB unchanged | **21** | answers that correctly call `XMEAS_7` increasing — the damage of the vocabulary edit alone |
| KB flipped, vocabulary unchanged | **0** | nothing changes: `decrease` is not a substring of `decreasing`, so the rule sees no direction word at all |
| KB flipped **and** gerunds added | **5** | the five that call it decreasing; 114 stay valid |

The 5, with the exact text the rule reads (60 chars after the tag):

| item | window | narrative |
|---|---|---|
| `EVT-train-f13-r030` | `') with a decreasing trend (z=-8.87, deviation -2.47%) starti'` | "…slow drift in pressure and temperature trends…" |
| `EVT-train-f13-r038` | `'), and separator (XMEAS_13) systems, alongside a decreasing '` | "…slow drift in pressure trends with decreasing stripper steam flow…" |
| `EVT-train-f13-r070` | `') with a decreasing trend starting at +0.0 min, followed by '` | "The reactor pressure is slowly decreasing while stripper pressure…" |
| `EVT-train-f13-r098` | `') with a decreasing trend starting at +0.0 min, followed by '` | "The reactor pressure is gradually decreasing…" |
| `EVT-train-f13-r100` | `') at onset, with a trend decreasing first, consistent with F'` | "…a gradual decrease in Reactor_Pressure." |

`direction_inconsistent` flags **none** of them: it needs a sentence naming exactly one
ranked sensor, and these sentences name several. The rejection rule and the flag disagree
by construction, which is worth knowing before the flag is quoted as a proxy for the rule.

The pairing is the interesting part: the 5 rejections only exist because the gerund was
added, and adding the gerund is only safe because the sign was flipped. Neither edit is
correct alone.

---

## 2. Re-call of the 52 unreached items

`--teacher-items outputs/benchmark_v2/dataset/recall_unreached.txt --max-calls 88`.

| pass | items requested | calls | API attempts | retried | rejected | dropped | seconds | s/call | s/attempt |
|---|---|---|---|---|---|---|---|---|---|
| 1 (scaffolding) | 71 | 60 | 81 | 21 | 9 | 9 | 801.5 | 13.36 | 9.90 |
| 2 (scaffolding) | 118 | 57 | 69 | 12 | 1 | 1 | 615.2 | 10.79 | 8.92 |
| **3 (this patch)** | **52** | **52** | **60** | **8** | **4** | **4** | **559.3** | **10.76** | **9.32** |
| total | — | 169 | 210 | 41 | 14 | 14 | 1,976.0 | 11.69 | 9.41 |

Pass 3 recovered **48 of 52** items; 4 were rejected twice and dropped. Composition of the
52: f14 18, f15 15, NORMAL 17, f16 1, f20 1.

### Per-fault retention (kept ÷ intended)

| family | kept | dropped | intended | retention |
|---|---|---|---|---|
| FAULT 01, 02, 06, 07, 08, 11, 14 | 120 each | 0 | 120 | 1.000 |
| FAULT 04, 05, 12, 13 | 119 each | 1 | 120 | 0.992 |
| FAULT 10, 15 | 118 each | 2 | 120 | 0.983 |
| FAULT 16, 17, 18 | 52 each | 0 | 52 | 1.000 |
| FAULT 19, 20 | 53 each | 0 | 53 | 1.000 |
| NORMAL_TRANSIENT | 111 | 6 | 117 | 0.949 |
| **total** | **1,925** | **14** | **1,939** | **0.993** |

**No family is below the 0.85 floor**; the worst is NORMAL_TRANSIENT at 0.949. The two
families that were the previous report's weak point are now at 1.000 (fault 14) and 0.983
(fault 15), which is what the 33 recovered f14/f15 items bought.

---

## 3. Accounting

### (i) Seconds per call — both denominators were right, and mixed

The previous report gave "13.4 s/call" for pass 1 and "1,416 s / 150 → 11.9 s". Both came
from dividing different things:

* `801.5 / 81 = 9.90 s per API attempt`; `801.5 / 60 = 13.36 s per item` (an item may need
  a retry, so the two cannot be compared).
* `1,416.7 / 117 = 12.11 s per item`; the reported 11.9 divided by **119**, which is
  neither the item count (117) nor the attempt count (150).

Corrected table is the one in §2. The pass-over-pass improvement (13.36 → 10.76 s per
item) is real: `think: false` and shorter prompts, plus pass 3's items carrying less
retry pressure.

### (ii) 90 dropped, 118 planned, 67 "with no prose" — the 23-item gap

Every number is from an artifact, and one of them was not:

| number | where it comes from | what it means |
|---|---|---|
| **90** | `phase_2b_full_teacher_report.md`: `1,849 + 90 = 1,939` | items rejected twice in the full run |
| **20** | `recall_plan.json → still_dropped_ids` | of those 90, the ones still invalid after the closeout-3 rule changes |
| **98** | `recall_plan.json → newly_invalid_ids` | newly invalid because of the scaffolding rule |
| **118** | `118 = 98 + 20`, and `1,939 − 1,821 = 118` | items with no valid prose at plan time; the plan's real size |
| **109** | pass-2 log: `restricted to 118 ids`, `pending 109` | 9 of the 118 already had current-hash verdicts |
| **52** | this patch | 109 − 57 called in pass 2 |
| **14** | final build: `excluded 14 items without valid prose` | 4 from pass 3 + 10 residual |
| **67** | previous report, `165 = 98 + 67` | **withdrawn — not a measured quantity** |

The gap you asked about: **67 was a forecast, not a count.** It equals `1,939 − 1,872`,
i.e. it assumed pass 1's 51 acceptances added 51 newly-valid items. The pass-1 build log
shows `cache replay: 1821 valid` *after* those calls — unchanged. The 51 items were
already replay-valid and were re-called only because their cache entry carried a stale
filter hash (`run_teacher`'s pending rule keys on hash, not on current validity), so the
remainder stayed at 118 and pass 2 correctly requested 118, not 67. The forecast `165`
was therefore wrong by 47, and `90 − 67 = 23` compares a drop count with a forecast
remainder — there is no 23-item set.

The real chain, measured: **90 original drops → 70 recovered by the closeout-3 rules,
20 still invalid → + 98 scaffolding = 118 planned → 9 already current → 109 pending →
57 called → 52 unreached → all 52 called now → 48 recovered, 4 dropped → 14 final drops.**

---

## 4. Rebuild and checks

`python scripts/build_benchmark_dataset.py --events ... --out ... --teacher --teacher-items ... --max-calls 88 --exclude-rejected`

| check | result |
|---|---|
| train items | **1,925** (1,939 − 14), all with prose |
| class counts | 1,361 classified / 453 `UNCLASSIFIED_OPEN_SET` / 111 `NORMAL_TRANSIENT` |
| unclassified share of faulty | **24.97%** (cap 25%, `capped: false`) |
| leak check | **4,634 prompts, 0 violations** |
| split integrity | `check_splits_v2.py` → **PASS**: unique keys, no train/val/test overlap, fault-disjoint folds, recorded SHA-256s match |
| open-set scan | 487 abstention records, 482 accepted, **0 naming the removed fault** |
| SFT export | `python -m llm.benchmark.to_sft --all` → **1,925 rows**, guards on, sha256 `5f43d1a0dc638ba5…`, 12.5 MB |
| audit sheet v3 | 100 rows, 5 per family, regenerated from the new accepted set |
| manifest | train `1072416c…`, val `38c1c948…`, test `699331ae…`, kb `f01236ff…`, operating point `e7fe73eb…`, template `fe4b8e0d…` |

### Drop bias — it moved, and the composition is now the problem

| group | n | median fused ratio | median top-\|z\| | median open-delay | normal share |
|---|---|---|---|---|---|
| accepted | 1,925 | **26.16** | **3.88** | 64 | 5.8% |
| dropped | 14 | **1.10** | **0.27** | 94 | **42.9%** |

Previous values (against the 90-item drop): 27.86 vs 11.74 ratio, 3.97 vs 0.39 top-|z|.
The accepted side barely moved; the dropped side moved a long way down, because the 70
recovered items were the moderate-signal ones and what is left is dominated by fault-free
excursions — 6 of the 14 final drops are `NORMAL_TRANSIENT`, against 5.8% of the kept set.
The bias is now *worse in kind* even though the drop is 6× smaller: what remains missing
from training is mostly normal behaviour, not hard faults.

---

## 5. Tests

| environment | command | result |
|---|---|---|
| app | `python -m pytest tests -q` | **358 passed** in 5.75 s |
| venv_train | `venv_train/Scripts/python -m pytest tests -q` | **358 passed** in 18.34 s |

`venv_train` had no `pytest`; it was installed there (that environment only, per the
standing authorization) — recorded here as the one environment deviation.

---

## 6. Gate table

| gate item | status | evidence |
|---|---|---|
| fault-13 impact reported **before** applying; applied or stopped | **PASS (stopped)** | 5,731/5,870 prompts carry the line; §1(a) |
| the reason for stopping is not just the stop condition | **PASS** | KB/menu not in `FILTER_CODE_FILES` → silent prose reuse; §1(a) |
| re-call results by fault | **PASS** | 52 called, 48 recovered, 4 dropped; §2 |
| per-fault retention table, nothing below 0.85 | **PASS** | min 0.949 (NORMAL); §2 |
| accounting reconciled | **PASS with one number withdrawn** | 90/20/98/118/109/52/14 all traced; the 67 was a forecast; §3 |
| final class counts | **PASS** | 24.97% unclassified, cap not needed |
| checks clean | **PASS** | 0 leaks / 4,634 prompts; splits PASS; 0/482 open-set leaks |
| tests pass, both environments | **PASS** | 358 / 358 |
| audit sheet v3 regenerated | **PASS** | 100 rows, 5 per family |

---

## 7. Deviations

1. **The fault-13 KB change was not applied** — stopped on the condition you set, and
   additionally blocked by the cache-hash gap in §1(a).
2. **`PROTOCOL.md`'s `0/102` claim is withdrawn** and replaced with the trend/deviation
   split. This is a correction to a published number, not a new measurement.
3. **The previous report's 67 and 165 are withdrawn.** The plan's real size is 118.
4. `pytest` installed into `venv_train` only.
5. Nothing else was changed after seeing results; the four pass-3 rejections were not
   re-tuned.

## 8. Open questions

1. **Fault 13** — approve the dataset-wide re-render (5,731 prompts, all
   `prompt_sha256` values, and 1,868 cached answers that would need re-asking or an
   explicit re-render check) knowing the prose gain is 5 corrected answers? My reading is
   that the cited reason has to be right, but it is 5 answers against a full rebuild, so
   it is your call. Either way, `menu.py` and `fault_knowledge.py` should join
   `FILTER_CODE_FILES` first.
2. **The remaining 14 drops are 43% normal excursions.** Accept, or lower the NORMAL
   floor further? The 70 recovered items show the filter review worked; this residue is a
   different failure mode.
3. Still unaddressed from earlier: **generation is ~4.0 tokens/s** on the app's
   forward-only decode path, which is what makes Phase 4 evaluation unaffordable as
   currently written.
