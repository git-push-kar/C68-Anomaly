# Phase 2B close-out 3 report

Repo: `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
CPU + Ollama. **Teacher calls this phase: 26 of 30** (15 targeted + 5 resume + 6 from
the kill/restart test). No full job.

---

## 1. Summary

The open-set leak was real and bigger than a phrasing problem. The teacher prompt
printed the removed fault's cause under `DECIDED DIAGNOSIS`, and the filter
accepted that same phrase because it trusted `truth_fault_number`. On top of that,
`nearest_candidates` could pad its list with the fault itself, so some items named
the answer twice. Before: **63 of 63** abstention prompts carried the true cause,
and **2 of 60** accepted answers named it. After: **0 of 73** answers, prompts
checked by the builder, which now refuses to emit an abstention item whose prompt
names the fault.

The valve/flow question turned out to be the same kind of ambiguity one level
down: eleven XMV tags are displayed with names that read like process flows
(`A_Feed_Flow`, `Stripper_Steam_Valve`), and the prompt did not say which was
which. Every tag is now labelled with its kind, the teacher is told a valve
position is not a flow, and the direction flag skips valve sentences. Flags over
the accepted cache fell from 8 to **2**, both measurements.

Counts reconcile exactly: 186 items with prose was 143 pilot survivors + 43
re-run, not 172 + 43; the missing 29 were cap-dropped, not lost.

---

## 2. Task 1 — open-set leak

### What the prompt said

`build_teacher_prompt` appended `- the disturbance: <KB name>` whenever the target
had a `truth_fault_number` — which every abstention item has, because the
simulator's number is kept for audit. The old code at `44a9cae` has no branch for
the abstention label, so this applied to **all 63** abstention items in the cache:
open-set episodes of faults 1-15 *and* faults 16-20, whose entry never existed.

### What the filter allowed

`check_answer` set `kb_number = truth_fault_number` unconditionally and then
`is_cause_mention(token, kb_number)`, so the removed fault's own wording was an
accepted "non-sensor cause mention". The number was also added to `permitted`.

### Two failures in one

| layer | before | after |
|---|---|---|
| teacher prompt names the removed fault | 63 / 63 | 0 / 73 (builder raises) |
| filter accepts the removed fault's cause phrase | 63 / 63 | 0 (scoped to menu) |
| accepted prose actually naming it | 2 / 60 | **0 / 73** |

The two prose violations were `EVT-train-f05-r027::openset` (fault 5's KB name) and
`EVT-train-f19-r021::menu` (fault 19's).

### The fix

- **Prompt**: on an abstention target, no disturbance line. Instead:
  `- no listed candidate explains this signature: say so, and discuss only the
  nearest listed candidates you rule out (FAULT_04 (...); FAULT_07 (...))`.
  `teacher_prompt_leaks` runs on every abstention prompt and raises on the removed
  fault's number or *distinctive* phrasing; the dataset builder calls it for every
  abstention item, so a build cannot ship one.
- **Filter**: cause mentions are allowed only for faults the item's menu offered
  (`menu_ids`), and the truth's number is removed from that set on abstention
  items. `permitted` likewise.
- **New rule** `names_removed_fault`: rejects prose containing the removed fault's
  KB name, a distinctive signature phrase, or `fault N` / `FAULT_N`. Sensors the KB
  names stay grounded, because they are measurements the evidence already shows.

### A trap worth recording: faults 5 and 12 share a cause

Fault 5 is "Condenser cooling water inlet temperature **step**", fault 12 is
"… **random variation**". Fault 12 must stay on a fault-5 open-set menu, so its
line necessarily contains fault 5's prefix. Treating that as a leak made the
builder abort on the first item:

```
ValueError: teacher prompt leaks the removed fault: ['fault_name:2', 'fault_number:2']
```

…which exposed the second bug: `nearest_candidates(2)` returned `FAULT_02` as its
own nearest candidate, through the fill loop that pads the list to *k*. Fixed, with
a test over all 15 classifiable faults.

The signature rule is therefore **distinctive**: a phrase counts only if no other
menu entry's phrase contains it. Fault 5's `… temperature step` is distinctive;
`… inlet temperature` is also fault 12's wording and is allowed, so ruling 12 out
stays possible. Both directions are tested.

### Re-run: 15 affected items

Estimated before starting: ~2.5 min at 9 s/call, ~4 min at the 20-call cap. Actual:
**15 calls, 15 accepted, no retries**, cache 245 → 260. The two violations are now
clean abstentions. Items spanned 14 faults.

---

## 3. Task 2 — valve vs flow

### Collisions found (Jaccard on tokens minus flow/valve/stream/rate/temperature/pressure/level)

| XMV | display name | near-duplicate XMEAS | shared tokens |
|---|---|---|---|
| XMV_42 | D_Feed_Flow | XMEAS_2 D_Feed_Stream2 | D, Feed |
| XMV_43 | E_Feed_Flow | XMEAS_3 E_Feed_Stream3 | E, Feed |
| XMV_44 | A_Feed_Flow | XMEAS_1 A_Feed_Stream1 | A, Feed |
| XMV_45 | A_C_Feed_Flow | XMEAS_4 A_C_Feed_Stream4 | A, C, Feed |
| XMV_47 | Purge_Valve | XMEAS_10 Purge_Rate | Purge (1.00) |
| XMV_48 | Separator_Pot_Liquid_Flow | XMEAS_14 Separator_Underflow_Stream10 | Separator |
| XMV_49 | Stripper_Liquid_Product_Flow | XMEAS_17 Stripper_Underflow_Stream11 | Stripper |
| XMV_50 | Stripper_Steam_Valve | XMEAS_19 Stripper_Steam_Flow | Steam, Stripper (**1.00**) |
| XMV_51 | Reactor_Cooling_Water_Flow | XMEAS_21 …_Outlet_Temperature | Cooling, Reactor, Water |
| XMV_52 | Condenser_Cooling_Water_Flow | XMEAS_22 …_Outlet_Temperature | Condenser, Cooling, Water |

`XMV_50 Stripper_Steam_Valve` / `XMEAS_19 Stripper_Steam_Flow` and
`XMV_47 Purge_Valve` / `XMEAS_10 Purge_Rate` are the worst: after dropping the
suffix they are the *same phrase*, one a valve position and one a flow.

### How it looked in the prompt (before)

```
- A_Feed_Flow: z=17.97, deviation 221.58%, trend increasing
- A_Feed_Stream1: z=17.97, deviation 221.52%, trend increasing
```
Two lines that read like the same variable.

### After — every tag carries its kind

```
- A_Feed_Flow [XMV_44, valve position, manipulated variable]: z=17.97, deviation 221.58%, trend increasing
- A_Feed_Stream1 [XMEAS_1, measurement]: z=17.97, deviation 221.52%, trend increasing
...
- XMEAS_23 [measurement]: residual share 0.595
- XMV_44 [valve position, manipulated variable]: residual share 0.175
...
1. A_Feed_Stream1 [XMEAS_1, measurement] at +0.0 min
3. A_Feed_Flow [XMV_44, valve position, manipulated variable] at +0.0 min
```

The teacher system prompt gains rule 2b: an XMV is a valve position, an XMEAS is a
measurement, a valve position is not a flow even when its name says "flow", and a
flow's direction must not be inferred from a valve's.

### Direction flags, re-run over 205 accepted answers

Valve sentences are now **skipped**: 207 of them across 11 XMV tags, listed per
answer as `direction_valves_skipped`. Two flags remain, both measurements, both
by inspection false positives of the rule:

| item | sensor | kind | stored | basis | words | sentence |
|---|---|---|---|---|---|---|
| `EVT-train-f01-r054::openset` | XMEAS_16 | measurement | increase | trend | decrease | "The onset order shows A_Feed_Stream1, A_C_Feed_Stream4, Stripper_Pressure, and A_Feed_Flow …" (the word belongs to another clause) |
| `EVT-train-f05-r009::menu` | XMEAS_22 | measurement | increase | trend | reduced | "The top deviating sensors include Condenser_Cooling_Water_Outlet_Temperature (z=0.87, +0.2…" (quantifier, not a movement) |

Still a flag, never a rejection.

---

## 4. Task 3 — report corrections

### (a) Fault 5 with one denominator (all train items)

Fault 5 has **120** train items (106 menu, 14 open-set episodes); 31 have been
attempted.

| stage | accepted | rate of attempted |
|---|---|---|
| first attempt | 30 | 0.968 |
| after one retry | +1 | **1.000** |
| dropped | 0 | — |
| valid prose now | 31 | 1.000 |

The earlier "3/10 → 27/27" compared two different populations (10 pilot items,
then 27 items after the cap and the fix). Against the 120-item denominator nothing
is hidden: 31 attempted, 31 valid, 89 never called.

### (b) The 45-item re-run, consistent denominators

| stage | count | rate of 45 |
|---|---|---|
| first-pass accepted | 36 | 0.800 |
| recovered by the single retry | 7 | — |
| dropped after a second rejection | 2 | — |
| **accepted** | **43** | **0.956** |

### (c) 186 versus 172 + 43

The cache held **245 lines / 220 unique items** (25 ids were re-run). Accepted:
**215**. Of those, **186** were live train items: **143** pilot + **43** re-run.
The other **29** accepted pilot answers were items the open-set cap had dropped
(faults 16-20) or ids that no longer exist; they were never in `train.jsonl`.

So `172 + 43 = 215` counts every accepted answer, and `143 + 43 = 186` counts the
ones the dataset actually had. The previous report used 186 for the dataset and 172
for the direction scan without saying they had different denominators. After this
phase the numbers are: **207 attempted, 198 first-pass, 7 by retry, 2 dropped, 205
valid prose** (0.957 first-pass, 0.990 after retry of attempted).

---

## 5. Task 4 — full-run readiness

### Command

```bash
python scripts/build_benchmark_dataset.py --teacher --model qwen3:30b --concurrency 1
```

(`--teacher-limit N` / `--max-calls N` cap a partial run; `--teacher-items FILE`
restricts to ids; the cache is appended per call and resumed automatically.)

### Pinned hashes

| item | value |
|---|---|
| teacher tag | `qwen3:30b` (digest `ad815644…`) |
| filter code | `c3888e7d3509e9546d7ffd0029ae81c3e90791e4fa54b9820e549a0398711dbb` |
| teacher system prompt | `1912531322b2399f5bd106ad351db590ebb4a0a3e9fa64e06d83abf2c85ba20c` |
| student template (prompt_v2 system prompt) | `fe4b8e0d6971fd9ac3cb2f5110fad2769cd2ebbd622af6ca3369acb7ba5b3a45` |
| licence (as the local runtime reports it) | `b87250e4478f8f0673bc61cd397e402ac33ca3a6f23a7ad9512618ab33f18fbe` |
| operating point | `e7fe73eb0b75b4d5…` |
| KB | `f01236ffe01b7b4f…` |
| inputs (`model.pt`, `cva_model.npz`, `dpca_model.npz`, `scaler.pkl`) | verified at build start |

### Remaining calls

**1,734** of 1,939 train items have no valid prose (205 do). At the measured
**10.9 s/call** and a 4.3% first-pass rejection rate (≈1,810 API calls with
retries): **≈5.2-5.5 h**, resumable per call.

### Resume, under the final hashes

Started 6 items, killed with `SIGKILL` after 2 calls (cache 264 → 266), restarted
with the same command:

```
INFO teacher: calls=4 cached=235 rejections=0
```

The two finished ids were not called again, the four remaining ran, cache
266 → 270. Cached ids were preserved through a hard kill.

### Audit sheet v2

Regenerated from valid accepted outputs: **100 rows, teacher reasoning and
narrative on 100/100**, 5 open-set and 5 normal_transient rows, 18 faults
represented, human columns blank. (`audit_v2_prefix.csv` keeps the previous
sheet for comparison; its rows were built before the tag labels.)

### Re-verification

| check | result |
|---|---|
| prompt leak check | 4,634 prompts, **0 violations** (builder raises) |
| abstention prompt check | every abstention prompt; **0** after fix (63/63 before) |
| split overlap | pairwise disjoint in all three components; held-out faults absent |
| input artifact hashes | `verify_inputs` passed at build start |
| cache replay | 199 valid, 2 no longer valid under the final filters |
| full test suite | **263 passed** in 7.03 s (252 before; 11 added) |

---

## 6. Gate

| gate item | status | evidence |
|---|---|---|
| Open-set leak count before/after with fix | **PASS** | prompts 63/63 → 0/73; prose 2/60 → 0/73; §2 |
| Open-set tests both ways | **PASS** | reject removed name/number; accept a listed candidate; 16-20 item; nearest-candidate test |
| Valve/flow decision shown with a prompt example | **PASS** | 10 collisions tabled, labelled prompt excerpt, §3 |
| Direction flags re-listed with tag kind | **PASS** | 2 measurement flags, 207 valve sentences skipped, §3 |
| Reconciled counts | **PASS** | fault 5 31/31; re-run 36→43/45; 186 = 143 + 43, §4 |
| Resume verified under final hashes | **PASS** | kill after 2, restart `calls=4 cached=235`, §5 |
| Audit sheet regenerated | **PASS** | 100 rows, 100 with teacher text, §5 |
| All checks pass | **PASS** | 263 tests, 0 prompt leaks, splits disjoint, hashes verified |
| Teacher budget respected | **PASS** | 26 of 30 calls |
| Full teacher job | **NOT RUN** | by instruction |

---

## 7. Deviations

- **D1 — the valve-label change invalidated the whole cache, so I added a replay
  path instead of discarding 186 accepted answers.** The builder re-judges each
  cached raw answer with the current filters (`--replay-cache`, no model call) and
  attaches only what passes. This is stricter than a hash match: 4 of the first
  188 and 2 of 201 were rejected by today's rules. It also means the "cache valid
  under the final hash" count is 0 by construction, but 205 items are valid under
  the final *rules*.
- **D2 — only distinctive signature phrases are rejected.** Required by the
  fault-5/fault-12 sibling pair: the shared prefix is also a live menu entry's
  wording. Exact-name and number rejection are unaffected.
- **D3 — the re-run covered 15 items, not all 63 that had a leaking prompt.** The
  leak was in the prompt, not the answers; 13 random abstention items across 14
  faults plus the 2 violations is enough to measure the new behaviour, and the
  budget was 30 calls.
- **D4 — `names_removed_fault` also fires on faults 16-20**, which never had a menu
  entry: naming one is wrong by construction, so prose naming it is rejected the
  same way. Number matching relies on `fault N` / `FAULT_N`; a bare "17" is caught
  by the number rule instead.

## 8. Open questions

1. **Faults 5 and 12 are the same cause with different dynamics.** An open-set
   item for one is only distinguishable from the other by the disturbance tail and
   the evidence's timing. If open-set recall for 5/12 turns out low, the honest
   fix is to treat them as one family in the menu, not to widen the filter.
2. **The direction flag is now nearly quiet (2 of 205), both false positives.**
   Either it earns its place as a diagnostic or it should be dropped; that needs
   the human audit sheet v2 to answer.
3. **Full teacher run ≈5.2 h** at 10.9 s/call; the estimate assumes the retry rate
   holds at 4.3% over 1,734 items, which is a small-sample figure.
