# Phase 2B close-out 2 report

Repo: `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
CPU + Ollama. **Teacher calls this phase: 54** (budget 60). No full job.

---

## 1. Summary

The fault-5 hypothesis was **partly right, and the part that was right mattered**:
every fault's KB *name* describes an unmeasured cause (20/20), while every
initiating sensor is a measured tag (20/20). The filter was rejecting prose that
named the cause correctly.

After teaching the prompt the KB wording and letting the filter accept exactly
that phrase: **fault 5 went from 3/10 (30%) to 27/27 (100%)**, and a mixed 45-item
re-run accepted **43 (95.6%)** in 54 API calls, with 9 retries recovering 7.

The direction check went the other way and found a bug in my own rule first. As
written it compared direction words against `deviation_percent`, which is the
*level* offset, not the movement. `trend` and deviation disagree in **442 of 2,151**
ranked sensors (a level below baseline but climbing), so the rule flagged 29
correct sentences. Comparing against `trend` — the movement — leaves **8 flags in
7 answers**, of which roughly half are genuine.

---

## 2. Task 1 — fault-5 diagnosis

### The 7 rejected fault-5 outputs

| event | reason |
|---|---|
| `EVT-train-f05-r002` | `unknown_tag:Condenser_Temperature` |
| `EVT-train-f05-r003` | `unknown_tag:Condenser_Cooling_Water_Valve` |
| `EVT-train-f05-r006` | `unknown_tag:Condenser_Pressure` |
| `EVT-train-f05-r007` | `unknown_tag:Condenser_Temperature` |
| `EVT-train-f05-r008` | `unknown_tag:Condenser_Temperature` |
| `EVT-train-f05-r009` | `unknown_tag:Condenser_Cooling_Water_Inlet_Temperature` |
| `EVT-train-f05-r010` | `unknown_tag:Condenser_Cooling_Water_Inlet_Temperature` |

### Hypothesis test — partially confirmed

Fault 5's KB name is **"Condenser cooling water inlet temperature step"**. The
cooling-water inlet temperature is not instrumented: `XMEAS_22` is the condenser
cooling-water *outlet* temperature and `XMV_52` is the flow. So the cause is
unmeasured by construction.

Of the 7 tokens, **2 are that exact phrase** and 5 are looser paraphrases the KB
does not contain (`Condenser_Temperature`, `Condenser_Pressure`,
`Condenser_Cooling_Water_Valve`). The hypothesis holds for the cause-naming
behaviour and not for every token.

### Per-fault: how many KB name/initiating phrases name something unmeasured

| phrase kind | measured | unmeasured | total |
|---|---|---|---|
| KB `name` (the disturbance) | 0 | **20** | 20 |
| KB `initiating.sensor` | **20** | 0 | 20 |

Every fault has exactly **one** unmeasured phrase — its name — and it is the
phrase a teacher needs in order to say what happened.

### What was implemented

- `llm/benchmark/kb_causes.py`: `disturbance_phrase()`, `unmeasured_cause_phrases()`,
  `is_cause_mention()`. Phrases are read out of the KB; there is no list of
  sensor-shaped words and no regex over plant names.
- The teacher prompt now carries the KB wording under `DECIDED DIAGNOSIS`:
  `- the disturbance: Condenser cooling water inlet temperature step`, and the
  system prompt says naming it is allowed even though it has no tag.
- `check_answer` accepts a token only when it is *this* fault's KB phrase, or that
  phrase with its disturbance tail dropped, three words minimum. Everything else
  is still a fabricated tag. This is the student-prompt filter; the KB wording is
  added to the **teacher** prompt only, so the student prompt is unchanged.

Tests, both directions: the exact phrase and its tail-dropped form are accepted;
`Condenser_Pressure`, `Condenser_Cooling_Water_Valve`, a generic `cooling water`,
and any measured-but-unshown tag still reject.

### Re-run: 28 rejected + 20 fresh fault-5

Estimated before running: ~7 min for 48 first attempts at 8.9 s, worst case ~9 min
at the 60-call cap. Actual: **54 API calls, ~9 min**.

| scope | before | after |
|---|---|---|
| fault 5 (all items) | 3/10 = 30% | **27/27 = 100%** |
| this 45-item re-run | 3/45 (the previously accepted) | **43/45 = 95.6%** |
| retries | — | 9 retried, **7 recovered**, 2 dropped |

3 of the 48 requested ids no longer exist: the per-fault cap (§5) dropped their
items before the run, so they could not be called. That is why 45, not 48.

The 2 remaining failures are genuine and unchanged by the fix:

- `fault_02`: `unknown_tag:C_Feed_Flow` (the KB says `A_C_Feed_Flow`) plus
  ungrounded `A_Feed_Flow`;
- `fault_13`: `number_not_in_store:-21.0`.

---

## 3. Task 2 — direction check

Implemented as `direction_inconsistent()` in `filters.py`: a **flag**, never a
rejection. It compares direction words attached to a single named sensor against
that sensor's stored direction, skips negated sentences, and records which basis it
used (`trend` or `deviation`).

### The rule was wrong first

My first version used `deviation_percent`'s sign. It flagged **29 sentences in 27
of 172 answers** — and checking each against the stored `trend` showed **0** were
supported by it. The evidence carries two different quantities:

| trend | deviation sign | count |
|---|---|---|
| increasing | positive | 1,116 |
| decreasing | negative | 347 |
| **increasing** | **negative** | **334** |
| **decreasing** | **positive** | **108** |
| stable | either | 95 |

442 of 2,151 ranked sensors are below baseline while climbing. "Increasing" for
those is correct, and the flag was contradicting the evidence's own trend column.
Now `trend` is preferred (`increasing`/`decreasing`), `stable` makes no claim, and
the deviation sign is only a fallback when `trend` is absent. Tests pin the
climbing-below-baseline case, the basis reported, and the stable case.

### The 8 remaining flags (7 answers, of 172 accepted)

| family | event | sensor | stored | basis | words | sentence |
|---|---|---|---|---|---|---|
| fault_04 | f04-r002 | XMV_51 | increase | trend | decreasing | "… ruling out FAULT_14 which would show decreasing flow." |
| fault_07 | f07-r008 | XMV_50 | increase | trend | reduced | "… for possible blockage or reduced availability first." |
| fault_07 | f07-r010 | XMV_45 | increase | trend | decrease, decreasing | "FAULT_07 … matches the observed A_C_Feed_Flow decrease trend …" |
| fault_07 | f07-r010 | XMV_45 | increase | trend | decreasing | "… verify if A_C_Feed_Flow is decreasing as observed." |
| fault_07 | f07-r011 | XMV_50 | increase | trend | reduced | "… control for possible blockage or reduced availability." |
| open_set | f01-r054 | XMEAS_16 | increase | trend | decrease | "… which contradicts the expected sequence …" |
| fault_01 | f01-r012 | XMV_44 | increase | trend | decreasing | "The A/C feed ratio is changing, with A feed flow decreasing." |
| fault_07 | f07-r014 | XMV_45 | increase | trend | decrease | "The C header pressure loss is causing a decrease in A_C_Feed_Flow." |

For you to judge, but by inspection: **4 look genuine** (`f07-r010` ×2, `f01-r012`,
`f07-r014` — they assert the sensor decreases while the evidence trends it up) and
**4 look like false positives** of the flag: a conditional ("would show
decreasing"), and "reduced availability" / "blockage" where the word attaches to a
noun rather than to the sensor's movement.

---

## 4. Task 3 — rejection policy

- On rejection: **one** retry, with the rejection codes and the instruction to fix
  exactly those points, at `seed + 1`.
- The retry is judged against the **original** prompt. The feedback names the
  offending token, so counting the retry prompt as "shown" would let the model
  quote the token back and pass. Tested with `87.3`, a number that exists only in
  the rejection reason: it is still rejected on the second attempt.
- A twice-rejected item is **dropped** (`dropped: true`) and kept in the cache for
  audit, not in the dataset as accepted-looking prose.
- Tested with a fake teacher: reject-then-accept (`attempts: 2`, accepted),
  reject-twice (`dropped: true`, `retry:` prefixed codes), and the prompt-grounding
  case above.

Per-fault coverage is now in the manifest with a floor:

```
teacher_coverage: {floor: 0.70, faults: {...}, below_floor: [...]}
```

**18 faults are below the floor** at this point, which is expected: only 186 of
1,939 train items have prose (a 200-call pilot plus this 45-item re-run). The floor
is doing its job — it will flag the same faults after the full run if they stay
under 70%.

---

## 5. Task 4 — per-fault cap

The pooled shuffle is replaced by an equal per-fault budget over faults 16-20,
allocated by quota and then round-robin for any remainder, seeded.

| fault | items | kept |
|---|---|---|
| FAULT_16 | 120 | 52 |
| FAULT_17 | 120 | 52 |
| FAULT_18 | 120 | 52 |
| FAULT_19 | 120 | 53 |
| FAULT_20 | 120 | 53 |
| **total** | 600 | **262** |

`budget_per_fault` = 67 requested, 52-53 kept after the round-robin. Unclassified
455 of 1,822 faulty train = **0.2497**; the 193 open-set episodes are untouched;
test is untouched. A test asserts the kept counts differ by at most 1.

---

## 6. Task 5 — audit sheets

**Sheet v1 contains no teacher text at all: 0 of 100 rows.** Its columns are
`item_id, event_id, split, variant, truth_fault_number` plus the five blank human
columns — there is no reasoning or narrative field, so an auditor could not see
what was being judged.

**Sheet v2** (`audit_v2.csv`, 100 rows) is built only from accepted teacher output
by `scripts/make_audit_sheet.py`:

| family | rows | family | rows |
|---|---|---|---|
| FAULT_01-08, 10-15 | 5-6 each | FAULT_16 | 1 |
| FAULT_17 | 2 | FAULT_18-20 | 4-5 each |
| **open_set** | **5** | **normal_transient** | **5** |

Columns: opaque item id, family, evidence summary (detector scores/ratios, top-3
sensors with z and deviation, lag shares), menu variant, target label, teacher
reasoning, teacher narrative, then blank `correct / grounded /
direction_consistent / useful / notes`. Nothing is scored.

---

## 7. Task 6 — re-verification

| check | result |
|---|---|
| prompt leak check | 4,634 prompts, **0 violations** (build aborts on one) |
| split overlap | all three families pairwise disjoint; held-out `[4,11,15]`, `[14,15]` absent from seen components |
| input artifact hashes | `verify_inputs` passed at build start (4 artifacts) |
| full test suite | **252 passed** in 5.11 s (234 before; 18 added) |

---

## 8. Gate

| gate item | status | evidence |
|---|---|---|
| Fault-5 acceptance reported before/after | **PASS** | 3/10 (30%) → 27/27 (100%); re-run 43/45 |
| Direction flags listed | **PASS** | 8 flags in 7 answers, §3, with stored sign and basis |
| Retry policy tests pass | **PASS** | 3 fake-teacher tests; 9 real retries, 7 recovered |
| Per-fault cap counts shown | **PASS** | 52/52/52/53/53 of 120 each, §5 |
| Audit sheet v2 exists with teacher text on every row | **PASS** | 100 rows, 186 accepted available |
| All checks still pass | **PASS** | §7 |
| Teacher budget respected | **PASS** | 54 of 60 calls |
| Full teacher job | **NOT RUN** | by instruction |

---

## 9. Deviations

- **D1 — 45 items, not 48.** The per-fault cap had already dropped 3 of the 28
  previously rejected items, so their ids matched nothing. The alternative was to
  bypass the cap for the re-run, which would have measured acceptance on items the
  dataset no longer contains.
- **D2 — the direction rule was rewritten after seeing its output.** Not a filter
  change made to improve a rate: the first version measured the wrong quantity and
  flagged 29 correct sentences. Before/after is 29 → 8, and the reason is in §3.
- **D3 — the two flags that remain ambiguous are left in.** Tightening them (e.g.
  ignoring "would show", or requiring the direction word to be adjacent to the
  sensor) is a judgement call about prose, and the flag is advisory.
- **D4 — `is_cause_mention` accepts a whole-word prefix.** "Condenser cooling water
  inlet temperature" is accepted for "… temperature step", three words minimum.
  Exact-only would have fixed 0 of the 7 fault-5 tokens instead of 2.

---

## 10. Open questions

1. **5 of the 7 fault-5 tokens are still rejected by rule** — they were
   paraphrases, not the KB phrase. Fault 5 now passes because the prompt supplies
   the wording, not because the filter got looser. If other families paraphrase
   instead of quoting, the same fix applies (the prompt), not a wider allowlist.
2. **The floor of 70% is currently tripped by 18 faults**, correctly, because the
   teacher has only run on ~10% of train. It becomes a real signal after the full
   run.
3. **Full teacher run**: 1,939 items, now with retries. At 8.9 s and a 9/45 retry
   rate, expect ~1,939 × 9.7 s ≈ **5.2 h**, resumable per call.
4. **`direction_inconsistent` needs a precision estimate** against human judgement
   before it is reported anywhere; on this sample it is roughly 50% precise.
