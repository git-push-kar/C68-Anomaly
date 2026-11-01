# RUN 1 PARITY AND ERROR ANATOMY — the field was scored as a string the model cannot write

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Environment:** `venv_train` (GPU), app env (CPU)
**Date:** 2026-11-01 · **Scope:** parity between the decoder and the label scorer, then what the step-500 errors are made of. No training, no test component opened, run-1 adapter and checkpoints untouched (read only).

---

## 0. Gate table

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1a | Parity: decoded string vs scorer sum, mean abs diff, count > 0.01 | **PASS** | §1.1 — 22.80 nats, 300/300 |
| 1b | Token ids alone vs in context for all 17 candidates, boundary/terminator mismatch stated | **PASS** | §1.2 — 1,700 pairs, no boundary mismatch; terminator mismatch found |
| 1c | Fix, tests, rerun (a)+(b) at step 500 on subsample and full validation, joint-vs-decode | **PASS** | §1.3–1.5 — fix committed, 3 tests, tables below |
| 1d | Probability vector, top-3, ECE, Brier at step 500 on full validation | **PASS** | §1.5 — ECE 0.0236, Brier 0.0402, top-3 0.9951 |
| 2 | Prefix diagnosis recomputed with correct alignment, reconciled with the decoded answer | **PASS** | §2 — the earlier reading was an artefact |
| 3 | Error anatomy at step 500, menu-present, n=497 | **PASS** | §3 — confusions, examples, strength, position, GBDT |
| 4 | Label-loss share, per-field teacher-forced loss (200 train + 200 val rows) | **PASS** | §4 |
| 5 | Evidence swap and menu shuffle (100 items) | **PASS** | §5 — 90 items (quota per gold) |
| 6 | Follow-up rescoring from the router/store, plus train decline fractions | **PASS** | §6 |
| 7 | PROTOCOL notes, full tests in both environments, commit | **PASS** | PROTOCOL §15; 518 + 518 passed |

**Headline.** The two instruments were not measuring the same string. The label scorer ended
a candidate with the bare quote `"` (id 1) while the exported answer continues `",` (id 497)
as one token, so it scored a continuation the model cannot produce: `log P('"')` after a
label is **−22.8 nats** on average (P ≈ 1e−10), against **−0.0002** for `",`. The summed
score was unaffected in ranking (the term is near-constant across candidates) — but the
per-token mean divides it by the candidate's own length, so the *longest* label won almost
every item. Every number in §14 that came from the mean instrument, and the whole
"prefix-sharing / first-token is lost" diagnosis in §1 of the previous report, was that
artefact. After the fix the four readings collapse onto each other (mean and joint agree on
**300/300** items, joint and decode on **298/300**) and the model turns out to be decisive:
on the ten items that used to look like failures it now names the right fault on nine, with
`P(number | FAULT)` of 0.98–1.00.

---

## 1. Scorer parity

### 1.1 The measurement, before the fix

300-item stratified validation subsample, step 500, batch 16 for scoring and 8 for decoding.
For each item: decode the field, then ask the scorer for the summed log-probability of the
**same string**.

| Quantity | Value |
|---|---|
| mean \\|decode chain − scorer sum\\| | **22.80 nats** |
| max \\|difference\\| | 25.69 nats |
| items differing by > 0.01 nats | **300 / 300** |
| mean \\|difference\\| over the label tokens only (terminator removed from the scorer) | **0.0126 nats**, 67/300 > 0.01 |
| log P(bare `"`) after a label | mean −22.80, range −25.69 … −13.75 |
| log P(`",`) after a label | mean −0.0002, range −0.0034 … 0.0000 |

The difference is exactly one token per item, always the same one: the terminator. Its
per-label spread is 12 nats, which is why the family masses in the old diagnosis moved
around: `exp(joint)` differs between candidates by `exp(12) ≈ 1.6e5` on that term alone.

### 1.2 Tokenisation audit

`scripts/scorer_parity.py tokens --prompts 100`: for every candidate, the tokens after the
real prompt (`prompt_for(prompt) + '{"fault_id": "'`) compared with the tokens after the
prefix alone, and with the tokens alone.

| Check | Result |
|---|---|
| prompt's own tokens end with the prefix's tokens | yes, 100/100 prompts |
| candidate suffix in context == suffix after the prefix alone | yes, **1,700 / 1,700 pairs** |
| candidate tokenised alone | `[5291, 62, 15, 16]` = `FAULT _ 0 1` (4 tokens, no terminator) |
| suffix with the bare quote | `[5291, 62, 15, 16, 1]` = `… 1 "` |
| suffix with the real terminator | `[5291, 62, 15, 16, 497]` = `… 1 ",` |
| suffix lengths after the fix | 5 (`FAULT_nn`), 4 (`NORMAL_TRANSIENT`), 6 (`UNCLASSIFIED_OPEN_SET`) |

So there is no BPE boundary mismatch, and there never was: the defect is that the string
being scored was not the string being written. `tokenizer('"') == [1]` and
`tokenizer('",') == [497]` are different tokens, and only the second one occurs in the
answer (the JSON continues with the next key).

### 1.3 The fix

One commit, three places:

1. `llm/benchmark/label_scorer.py`: `FIELD_TERMINATOR = '",'` replaces the bare quote, and
   `suffix_ids(tokenizer, label, context=None)` now takes the **real** context and raises if
   the continuation does not tokenise the same way after the full prompt as after the prefix
   alone. `score_batch` runs that check once per call, on a real prompt.
2. `scripts/instrument_check.py::decode_first_field` scores the token that carries the
   closing quote instead of dropping it, so both instruments cover the same string. The
   decode *prediction* is unchanged (the terminator carries no label text).
3. `tests/test_label_scorer.py`: three tests — the terminator is `",` (id 497) and never the
   bare quote, every candidate round-trips as a continuation of a real validation prompt, and
   the decoder is structurally required to import the scorer's terminator and to add its
   log-probability.

### 1.4 Parity after the fix

Same strings, same model, three batchings (`scripts/scorer_parity.py residual`):

| Comparison | mean \\|Δ\\| | max \\|Δ\\| | over 1e-3 |
|---|---|---|
| scorer batch 1 vs decoder batch 1, 20 items | **0.000000** | 0.000000 | 0 |
| scorer batch 16 vs decoder batch 8, 20 items | 0.000824 | 0.002810 | 6 |
| scorer batch 1 vs scorer batch 16, 20 items | 0.001621 | 0.009306 | 7 |
| decoder batch 1 vs decoder batch 8, 20 items | 0.001582 | 0.009306 | 7 |
| the 6 largest-residual items, batch 1 vs batch 1 | **0.000000** | 0.000000 | 0 |
| the same 6 items, scorer b1 vs b16 | 0.148010 | 0.397522 | 6 |
| the same 6 items, decoder b1 vs b8 | 0.076000 | 0.157666 | 6 |

At matched batching the two instruments are **bit-identical**. The residual under production
batching is bf16 with left padding: it is largest on the six longest prompts (>3,300
characters), reaches 0.5 nats on one, and is a property of the batch, not of the definition.
That is worth knowing independently: the reported confidence of the frozen instrument moves
by up to half a nat depending on which rows share its batch.

### 1.5 Instruments rerun at step 500

300-item subsample, before and after (accuracy / macro-F1):

| Instrument | before | after |
|---|---|---|
| (a) mean | 0.9033 / 0.8540 | **0.9367 / 0.9328** |
| (b) joint | 0.4100 / 0.5345 | **0.9367 / 0.9328** |
| (d) hierarchical | 0.4067 / 0.5318 | 0.9333 / 0.9311 |
| (c) decode (frozen) | 0.9367 / 0.9353 | 0.9367 / 0.9353 |

Full Protocol A validation, 822 observable rows, step 500, after the fix:

| Instrument | acc | macro-F1 | top-3 | abstain P | abstain R | ECE | Brier |
|---|---|---|---|---|---|---|---|
| (a) mean | 0.9453 | 0.9280 | 0.9939 | 0.9885 | 0.9511 | 0.5830 | 0.3847 |
| (b) joint | **0.9453** | **0.9280** | **0.9951** | 0.9885 | 0.9511 | **0.0236** | **0.0402** |
| (d) hierarchical | 0.9380 | 0.9215 | 0.9951 | 0.9929 | 0.9356 | 0.0235 | 0.0391 |
| (c) decode (frozen) | 0.9416 | 0.9263 | n/a | 0.9907 | 0.9422 | n/a | n/a |

Menu-present / removed split (822 = 497 + 325): joint 0.9095 / 0.9223 on the 497 and
1.0000 / 1.0000 on the 325; decode 0.9034 / 0.9199 and 1.0000 / 1.0000. The 12 unobservable
rows score 0.5000 for every instrument, as before — faults 3 and 9 are not candidates, so
half is the ceiling.

**Does the joint argmax now match decode?** On the subsample, 298/300; on full validation,
**828/834 (99.3 %)**. Before the fix they could not have: joint's own accuracy was 0.4100
against decode's 0.9367, so they disagreed on at least 158 of 300 items. The six remaining
disagreements are all near-ties between `UNCLASSIFIED_OPEN_SET` and `FAULT_10`/`FAULT_12`/
`FAULT_15`, i.e. the batching noise of §1.4 flipping items that are within 0.1 nats.

**Mean vs joint:** 300/300 on the subsample and 834/834 on full validation. With the constant
gone, dividing by the token count no longer changes the ranking. **The mean instrument and
the joint instrument are the same instrument**, which is the cleanest statement of what the
artefact was doing.

**Consequence for the freeze.** The rule in §14.3 (highest full-validation macro-F1 with
abstention precision ≥ 0.95) would now select the joint scorer: 0.9280 against decode's
0.9263 — **two items in 822**, against a measured batching sensitivity that flips items of
their own. The freeze stands, and PROTOCOL §15.1 records both numbers and the reason. What
changed is the *reading* of the numbers, not the run.

**Calibration.** The joint vector is a softmax over the 17 candidate strings, i.e. the
distribution *conditional on the answer being one of them*, with no calibrator fitted. ECE
0.0236 and Brier 0.0402 are raw. The mean instrument's vector is not a probability
(ECE 0.583): a softmax over per-token averages has no interpretation, and it should not be
quoted as confidence anywhere.

---

## 2. Prefix diagnosis, recomputed with correct alignment

The ten items are the union of the decode-vs-scorer disagreements at steps 300 and 400 from
the previous report. Recomputed at the frozen checkpoint (step 500) with the corrected
terminator; `dec` is the decoded answer, `mean`/`joint` the two scorer argmaxes.

| Item | Gold | dec | mean | joint | P(FAULT / NORMAL / UNCL) | P(number \\| FAULT), top 3 |
|---|---|---|---|---|---|---|
| `EVT-val-f11-r317::menu` | FAULT_11 | FAULT_11 | FAULT_11 | FAULT_11 | 0.98 / 0.00 / 0.02 | 11: 0.99 |
| `EVT-val-f12-r311::menu` | FAULT_12 | FAULT_12 | FAULT_12 | FAULT_12 | 0.94 / 0.00 / 0.06 | 12: 1.00 |
| `EVT-val-f14-r310::menu` | FAULT_14 | FAULT_14 | FAULT_14 | FAULT_14 | 0.99 / 0.00 / 0.01 | 14: 0.99 |
| `EVT-val-f14-r324::menu` | FAULT_14 | FAULT_14 | FAULT_14 | FAULT_14 | 0.99 / 0.00 / 0.01 | 14: 0.98 |
| `EVT-val-f15-r308::menu` | FAULT_15 | FAULT_15 | FAULT_15 | FAULT_15 | 0.89 / 0.04 / 0.06 | 15: 0.98 |
| `EVT-val-f15-r323::openset` | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | 0.00 / 0.00 / 1.00 | — |
| `EVT-val-f16-r317::menu` | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | 0.49 / 0.02 / 0.49 | 15: 0.62 · 10: 0.37 |
| `EVT-val-f18-r307::menu` | UNCLASSIFIED | FAULT_12 | FAULT_05 | FAULT_05 | 0.92 / 0.00 / 0.08 | 05: 0.50 · 12: 0.46 |
| `EVT-val-f19-r302::menu` | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | 0.11 / 0.00 / 0.89 | 10: 0.54 · 15: 0.33 |
| `EVT-val-f19-r315::menu` | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | 0.22 / 0.01 / 0.77 | 15: 0.33 · 10: 0.25 |

Reconciliation, item by item:

* **nine of ten are now correct** and all three readings agree on them. The previous report
  read these as "the model's first token votes NORMAL with 0.9–0.99 mass even when the truth
  is a fault". With the impossible terminator removed, the same items show `P(NORMAL) ≈ 0.00`
  and `P(FAULT) = 0.89–0.99`. The model was not undecided; the score was.
* **one row is a genuine tie**: `EVT-val-f18-r307::menu`, where the fault family gets 0.92 but
  splits 0.50 / 0.46 between faults 5 and 12, decode picks 12 and the scorer picks 5. Gold is
  UNCLASSIFIED, so all three are wrong, but not for the reason the old table gave.
* `EVT-val-f16-r317::menu` is a real ambiguity (0.49 fault family, 0.49 abstain) resolved
  correctly by all three readings; the fault mass splitting 0.62 / 0.37 between 15 and 10 is
  visible in the number column and is exactly what the abstain should be buying.

So the previous §1 diagnosis — "no, the argmax does not follow a unique-first-token
candidate; the per-token average loses the information" — is **withdrawn**. It followed a
−22.8 nat term with a 12-nat spread, not a property of the model. The prefix structure
(15 faults sharing `FAULT`) is real, but it costs nothing once the scoring string is correct.

---

## 3. Error anatomy at step 500 (menu-present, n = 497)

48 errors, accuracy 0.9034. They are not scattered:

| Gold | Errors | Predicted as |
|---|---|---|
| UNCLASSIFIED_OPEN_SET | 26 | FAULT_10 ×20, FAULT_15 ×4, other ×2 |
| FAULT_15 | 8 | NORMAL_TRANSIENT ×4, FAULT_10 ×2, FAULT_07 ×1, FAULT_11 ×1 |
| NORMAL_TRANSIENT | 6 | FAULT_15 ×5, FAULT_01 ×1 |
| FAULT_08 | 2 | FAULT_01, FAULT_13 |
| FAULT_10 | 2 | FAULT_15 ×2 |
| FAULT_13 | 2 | UNCLASSIFIED ×2 |
| FAULT_06, FAULT_12 | 1 each | FAULT_08, UNCLASSIFIED |

Per gold class, error rate: FAULT_15 **0.32**, UNCLASSIFIED **0.208**, NORMAL_TRANSIENT
0.128, every other fault 0–0.08. Faults 1, 2, 4, 5, 7, 11 and 14 have **zero** errors.
`FAULT_10` and `FAULT_15` act as sinks: 22 of the 48 errors are predicted `FAULT_10`, 11 are
predicted `FAULT_15`. The dominant single confusion is **missed abstention**: 20 of the 26
open-set errors are answered `FAULT_10` and 4 `FAULT_15`.

Evidence strength, errors against correct items (permutation test, 2,000 shuffles):

| Quantity | errors (median) | correct (median) | p |
|---|---|---|---|
| top \\|z\\| in the prompt | **0.405** | **4.100** | **0.0005** |
| LSTM ratio | 1.03 | 8.80 | 0.0395 |
| fused ratio | 3.45 | 30.50 | 0.2324 |
| overall anomaly score | 3.02 | 46.03 | 0.1839 |
| onset delay (first onset) | 0.0 | 0.0 | 1.0 |
| number of onsets | 6 | 6 | 0.0005 |

The errors are the events where the **prompt carries almost no evidence**: top |z| 0.4 means
no sensor in the evidence block moved. Ten examples each for the two sinks:

`FAULT_10` (2 errors, both predicted `FAULT_15`)

| Item | score | fused | top \\|z\\| | gold menu line | GBDT P(gold) |
|---|---|---|---|---|
| `EVT-val-f10-r312::menu` | 13.99 | 3.5 | 3.35 | C feed temperature random variation — Reactor_Temperature increases first | 1.000 |
| `EVT-val-f10-r320::menu` | 4.05 | 4.3 | 0.42 | (same) | 0.994 |

`FAULT_15` (8 errors; top |z| never above 0.39)

| Item | pred | score | fused | top \\|z\\| | GBDT P(gold) |
|---|---|---|---|---|
| `EVT-val-f15-r303::menu` | FAULT_07 | 0.86 | 1.17 | 0.22 | 1.000 |
| `EVT-val-f15-r305::menu` | NORMAL | 0.79 | 1.00 | 0.17 | 0.989 |
| `EVT-val-f15-r306::menu` | NORMAL | 0.81 | 1.09 | 0.28 | 1.000 |
| `EVT-val-f15-r309::menu` | NORMAL | 0.78 | 1.00 | 0.23 | 0.993 |
| `EVT-val-f15-r310::menu` | FAULT_11 | 0.90 | 1.14 | 0.32 | 0.998 |
| `EVT-val-f15-r311::menu` | FAULT_10 | 0.85 | 1.10 | 0.39 | 1.000 |
| `EVT-val-f15-r317::menu` | FAULT_10 | 0.81 | 1.11 | 0.14 | 0.990 |
| `EVT-val-f15-r318::menu` | NORMAL | 0.72 | 1.00 | 0.08 | **0.008** |

Both sinks behave the same way: a barely-alarmed event whose top-three sensors are noise
(z < 0.4, every trend reading "increasing"), a menu line that names the mechanism, and a
GBDT that is certain anyway.

**What the GBDT thinks of the right answer on exactly these rows:** of the 48 errors, the
400-tree classifier predicts the gold class on **46** (median gold probability **0.999**,
p10 0.863); only 2 rows have gold probability below 0.5 (`EVT-val-f15-r318` at 0.008 and one
more). So the information is in the event — it is in the 55-sensor feature vector the
classifier reads and **not** in the three "top deviating sensors" the prompt prints. The
prompt's evidence block is a lossy projection that, for exactly the events that matter,
projects to nothing.

**Menu position.** Errors by the gold entry's position in the menu: 0/25 at positions 1, 2,
4, 5, 7, 11, 14; 1 at 6 and 12; 2 at 8, 10, 13; **8 at 15**; **26 at 16** (abstain row).
Because the split has one item per (event, fault) pair and the menu is in KB order, position
and fault identity are the same variable here, and **the two cannot be separated from this
table** — §5 is what separates them, and it separates them decisively.

---

## 4. What share of the loss is the label

**Supervised-token share** (1,919 report rows of `train_mixed.jsonl`, real tokenizer, the
trainer's own answer+EOS masking):

| Field | tokens | share of supervised tokens |
|---|---|---|
| `fault_id` (key, value, terminator) | 19,531 | **3.40 %** |
| `fault_id` value only | 11,855 | 2.06 % |
| `verdict` | 7,224 | 1.26 % |
| `ranked_hypotheses` | 23,739 | 4.13 % |
| `physical_reasoning` | 393,287 | **68.40 %** |
| `narrative` | 83,213 | 14.47 % |
| all supervised | 574,980 | 100 % (mean 299.6 tokens/row, median 297) |

**Teacher-forced per-field loss, step 500** (200 train rows with prose, 200 validation rows):

| Field | train tokens | train loss | ppl | val tokens | val loss | ppl |
|---|---|---|---|---|---|---|
| `fault_id` | 1,031 | 0.0292 | 1.03 | 1,083 | 0.0253 | 1.03 |
| `ranked_hypotheses` | 2,191 | 0.0024 | 1.00 | 2,237 | 0.0091 | 1.01 |
| `verdict` | 553 | 0.0001 | 1.00 | 701 | 0.0001 | 1.00 |
| `physical_reasoning` | 41,314 | **0.2535** | 1.29 | — | — | — |
| `narrative` | 8,733 | **0.3967** | 1.49 | — | — | — |
| overall | 60,222 | 0.2320 | 1.26 | 10,421 | 0.6126 | 1.85 |

Two readings, both worth stating:

* the label field is **memorised** — 0.025–0.029 nats per token on train *and* validation
  (ppl 1.03), no gap at all. The adapter knows the menu ids; the closed-set errors are not a
  failure to recall the format.
* the prose is where the loss lives, and `narrative` (ppl 1.49) is twice as costly per token
  as `physical_reasoning` (1.29) even though it is the shorter field.
* the validation "overall" is **not comparable** to the train one: the structural split has no
  teacher prose, so its 200 rows contain only 10,421 supervised tokens (the deterministic
  fields), and 0.6126 is the loss of a JSON *skeleton* rather than of an answer. This is the
  concrete reason eval-loss was disabled for this run, and it is why the per-field table is
  the honest comparison.

The consequence for any claim about the closed-set metric: the field it scores received
**one token in thirty** of the gradient. A mix or filter change cannot be credited with
moving it unless the objective weights it.

---

## 5. Evidence or menu? (swap and shuffle, 90 items)

100 items were requested; the stratified quota (6 per gold class over 15 classes present)
yields **90**. Three decodes per item, 0.236 s/decode, 63.6 s total.

| Intervention | Result |
|---|---|
| original prompt | accuracy **0.900** |
| menu lines shuffled (content identical, order changed) | accuracy **0.267**, prediction changes on **72.2 %** of items |
| evidence replaced by another fault's event, own menu kept | follows the **donor's** fault **83.3 %** |
| | follows the **original** fault 1.1 % |
| | follows neither 15.6 % |

So both statements are true at once, and they are not in tension:

* **the evidence is being read** — give the item a different event and the answer moves to
  that event's fault four times out of five, and it does *not* stay on the original label
  (1.1 %);
* **the answer is also a position lookup** — reorder the same lines and accuracy collapses
  from 0.90 to 0.27 with three quarters of the predictions changing.

The reconciliation is that the model identifies the fault *number* from the evidence and then
emits the entry at that number's position in the list, rather than copying the matched line.
The deployed menu is always in KB order, so this never shows up in validation accuracy — which
is exactly why the number is 0.90 and not 0.27. Per class, `follows_evidence` is 1.00 for
faults 4, 5, 14 and 15, and 0.67 for faults 1, 2, 6, 11 and 13.

This is the most actionable finding in the run: **train and evaluate with a shuffled menu.**
The current 0.90 menu-present accuracy is a number about a fixed presentation, and a
deployment that renders the menu in a different order (alphabetical, filtered, translated)
would drop toward 0.27 without any change in the model.

---

## 6. Follow-up rescoring: is the refusal right?

179 validation turns, `expected_decline` computed from the router (no case labels), with the
event fetched from the store by `source|fault|run`. Two readings, as recorded in PROTOCOL
§15.3: **store** = no store key resolves; **router** = the router's own verdict is `refuse`.

| Case | n | store coverage | exp. decline (router) | exp. decline (store) | adapter refusal | over-refusal (router) | under-refusal (router) | train decline fraction (n turns) |
|---|---|---|---|---|---|---|---|---|
| out_of_scope | 13 | 0.00 | 1.00 | 1.00 | **1.00** | — | **0.00** | 0.763 (114) |
| false_premise | 15 | 0.00 | 1.00 | 1.00 | **1.00** | — | **0.00** | 0.750 (120) |
| in_scope_control | 40 | 0.50 | 0.25 | 0.50 | 0.05 | **0.067** | 1.00 | — |
| causal_why | 15 | 0.00 | 0.40 | 1.00 | 0.73 | 0.778 | 0.333 | 0.450 (100) |
| mechanism | 15 | 0.00 | 0.60 | 1.00 | 0.47 | 0.667 | 0.667 | 0.605 (81) |
| valve_actuator | 15 | 0.00 | 0.00 | 1.00 | 0.40 | 0.400 | — | 0.543 (140) |
| repeated_question | 17 | 0.00 | 0.00 | 1.00 | 0.53 | 0.529 | — | 0.350 (80) |
| hypothesis_comparison | 15 | 0.00 | 0.00 | 1.00 | 0.00 | 0.000 | — | 0.292 (120) |
| ambiguous_question | 12 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | — | 0.297 (101) |
| clarification_reply | 15 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | — | 0.099 (81) |
| template_sample | 7 | 1.00 | 0.00 | 0.00 | 0.14 | 0.143 | — | 0.042 (24) |
| **all** | 179 | — | 53 | 125 | **64** | **29** | **18** | — |

Agreement with the adapter's refusal: **0.737** under the router reading, 0.626 under the
store reading.

* The refusals that are *right* are right for the right reason: 100 % on out-of-scope and
  false-premise, with **zero** under-refusal on either.
* **Over-refusal on the controls is low** — 2 of 30 answerable controls (0.067) — so the model
  is not simply refusing everything.
* The prose cases are where the number looks bad and the cause is in the data: `causal_why`
  refuses 73 % of the time and the **training corpus declines 45 %** of its accepted
  `causal_why` turns; `mechanism` 47 % against 60 % in the corpus; `valve_actuator` 40 %
  against 54 %; `repeated_question` 53 % against 35 %; `hypothesis_comparison` 0 % against
  29 %. The adapter is reproducing, not inventing, the corpus's refusal habit — with one
  exception: on `repeated_question` it refuses *more* than it was taught (53 % vs 35 %).
* The store reading (which marks every prose question unanswerable because no lookup key
  exists for "why") turns 64 answered turns into "under-refusals". That is the reading's
  artefact, not the model's: refusing a mechanism question the KB covers is not correct
  behaviour, and PROTOCOL §15.3 records why the router reading is the one to quote.

---

## 7. What changed on disk

* `llm/benchmark/label_scorer.py` — `FIELD_TERMINATOR`, context-verified `suffix_ids`, the
  normalisation docstring corrected to say what the artefact was.
* `scripts/instrument_check.py` — decode scores the terminator; `prefix` reports the joint
  argmax and the decoded answer; `fullval` writes all four instruments; `sweep` takes
  `--step`.
* `scripts/scorer_parity.py` (new) — `tokens`, `parity`, `residual`.
* `scripts/run1_error_anatomy.py` (new), `scripts/label_loss_share.py` (new),
  `scripts/evidence_swap_probe.py` (new), `scripts/followup_rescore.py` (new).
* `tests/test_label_scorer.py` — three tests on the terminator and the continuation.
* `docs/PROTOCOL.md` §15 — the correction, the validation terminology, the two refusal
  readings, and the label's 3 % share.
* Artifacts under `outputs/analysis/phase3a/` (`scorer_parity/`, `instrument_check/`,
  `label_loss/`, `evidence_swap/`, `run1_error_anatomy.json`, `followup_rescore.json`).

Tests: **518 passed** in the app environment, **518 passed** in `venv_train`.

*Note added 2026-11-01, after the hybrid transition:* the suite has since grown by the 26 hybrid
tests, so the same command now reports **544 passed** in the app environment and **541 passed,
3 skipped** in `venv_train` (the three skips are the LightGBM tests, which that environment has
no LightGBM for). 518 is the count at the time of this run, not a current figure.

---

## 8. Limitations

* The ten prefix items were disagreement items at steps 300/400; the table is recomputed at
  step 500 because that is the frozen checkpoint, so it shows the items where the model has
  already converged. The prior report's numbers for those items at 300/400 are not
  reproducible with the fixed scorer and are superseded rather than compared.
* Parity at production batching is not exact (0.5 nats worst case on the longest prompts,
  §1.4). The six joint-vs-decode disagreements on full validation are inside that noise, so
  "99.3 % agreement" is a floor on the definitional agreement, not a statement about ties.
* The swap probe is 90 items, not 100 (quota per gold class), and the evidence swap keeps the
  *menu* fixed but changes the incident hash and the onset block too — the "follows the
  donor" rate therefore measures "reads the evidence block", not "reads the sensors only".
* Menu position and gold fault are the same variable in the error anatomy (§3); the shuffle
  experiment in §5 is what separates them, and it says position matters a great deal.
* Validation has no teacher prose, so the per-field loss comparison covers three fields on
  validation and five on train (§4).
* Test components remain unopened; nothing here touches them.
