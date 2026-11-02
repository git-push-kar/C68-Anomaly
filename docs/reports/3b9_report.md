# 3B-9 — THE A/B RECORD, THE ADAPTER-TEXT CHECKS, FALLBACK V3, THE FRESH SHEET

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Date:** 2026-11-01
**Environment:** app env (CPU) for every check and for v3; `venv_train` for the tokenizer counts and
its test suite. **No adapter trained, no test component opened, no teacher call, no run-1 or KB file
modified.** GPU not used — see §6 for the reading and for a correction to what 3B-8 claimed.

---

## 0. Pre-registration first

§21 was appended to `docs/PROTOCOL.md` and committed as **`f097b4c`** before any number in this
round was computed. It records the scored A/B result (§21.1), the **post-hoc, disclosed** per-stratum
amendment (§21.2) that applies **only** to the fresh sheet, the pending human spot-check (§21.3), and
the round's own constraints (§21.4). The scored sheet the user supplied in chat was materialised as
`outputs/analysis/phase3a/fit/audit_ab_100_v2_scored_model.csv` (100 rows) so the record can be
verified as data rather than trusted as prose.

**Transcription check.** Every aggregate in §21.1 reproduces from those rows except one cell:

| | reported | recomputed | |
|---|---|---|---|
| adapter strict correct | 48 % | **49 %** | one cell lands `y` where the rater had `p`; the lenient figure is unchanged |
| adapter lenient correct / grounded | 95 / 98 % | **95 / 98 %** | exact |
| adapter grounded / consistent / useful (strict) | 79 / 79 / 65 % | **79 / 79 / 65 %** | exact |
| fallback v2 correct / grounded / consistent / useful | 31 / 95 / 96 / 34 % | **31 / 95 / 96 / 34 %** | exact, once `=` is read as (n, y, y, n) |
| preference A / F / T | 61 / 16 / 23 | **61 / 16 / 23** | exact |
| F stratum adapter | 50 / 72 / 67 / 65 | **50 / 72 / 67 / 65** | exact |
| N+O+R stratum adapter correct | 51 % | **53 %** | the same single cell |
| U stratum | 0 / 100 / 0 / 0 | **0 / 100 / 0 / 0** | exact |

The `=` cells are the whole explanation of the odd denominators: the fallback side was not separately
scored on the 54 N/O/R rows, and mapping `=` to (n, y, y, n) reproduces 31 / 95 / 96 / 34 exactly.

---

## 1. The stored adapter texts, checked (step 2)

`scripts/adapter_checks_3b9.py`; artefact `outputs/analysis/phase3a/fit/adapter_checks_3b9.json`.
Four checks, three populations, split by verdict stratum. **Every check here is a heuristic with an
unmeasured false-flag rate** — that is why nothing in this section enters the live path (§21.4).

| population | stratum | n | a: ranked-top ≠ diagnosis | b: number flags / rows | c: binding flags / rows | d: trend flags / rows |
|---|---|---|---|---|---|---|
| main 242 | all | 242 | **88** (any) / **0** (classified) | 112 / 52 | 461 / 179 | 11 / 8 |
| main 242 | classified | 110 | **0** | 15 / 11 | 270 / 101 | 5 / 4 |
| main 242 | abstention | 132 | **88** | 97 / 41 | 191 / 78 | 6 / 4 |
| counterfactual 58 | all | 58 | 0 | 3 / 3 | 153 / 55 | 4 / 3 |
| A/B 100 | all | 100 | 0 | 198 / 99 | 120 / 72 | 12 / 11 |
| A/B 100 | classified | 56 | 0 | 112 / 56 | 83 / 47 | 10 / 9 |

### (a) The "0 of 242" claim, reconciled

The 88 flags are **all** on rows whose verdict is an abstention, where the ranked list is *supposed* to
carry a candidate. Under the original §16.4 definition — rank check applied to every verdict — the
main batch gives **88 flags**, exactly the number §5.2 of the transition report reported as
`ranked_top_mismatch` before it was retired. **Verdict-scoping those flags to 0 is the rule that
dropped the check**, and with it applied the main batch and the counterfactual batch both give **0**,
which is the "0 of 242" that was published. Both readings are now printed side by side.

**The 13 contradicting A/B rows do not reproduce.** On the 56 classified A/B rows, `ranked[0]` equals
the supplied diagnosis **56/56**; on all 224 classified rows across the three populations it is
**224/224**. A crude independent test — a fault other than the diagnosis named anywhere outside a
±90-character rejection window — flags 56 rows, but reading them shows the *lookalike differential*
(`Ranked: FAULT_02, FAULT_01`) doing its job. **Reported as: 0 by both closed-list readings, 56 as an
upper bound from a test that cannot tell a ranked alternative from a rejected one, 13 not
reproduced.** §5 of the spec update is written against this measurement.

### (b) Numbers against the detectors' own quantities

Three kinds: `number_not_the_detectors` (78), `direction_below_but_value_above_threshold` (22),
`direction_above_but_value_below_threshold` (12) on the main batch. **The first kind is noisy and its
examples say why:** in `EVT-val-f10-r317` the sentence "The CVA Q-statistic (2966.1) and LSTM
reconstruction error (0.8919) confirm…" is attributed to *both* detectors, so each detector's allow-list
rejects the other's number. The 34 direction flags are the useful part and are not noise of that kind:
they mark rows where the text says "above" about a detector whose stored value is below its own
threshold. **Precision is unmeasured; no threshold is set on these.**

### (c) Tag-name binding

After the refinement — a claim is only judged when both sides carry an identity token — the main batch
has **457 unverifiable** (`Check the A feed valve position (XMV_44)` names a role, not a measurement)
and **4 mismatches**; the counterfactual batch 148 / 5; the A/B 100 has 117 unverifiable and **3
unknown tags**. The three originally-known bad bindings (XMEAS_17 as condenser outlet temperature,
XMV_47 as condenser valve, XMEAS_21 as flow) are in the fixture set and are flagged; the check's
problem is the opposite of the ranked-top one — it is too permissive by construction, because most
adapter names are roles the canonical table does not contain.

### (d) Trend and sign

**11 flags on the main batch (8 rows), 12 on the A/B (11 rows).** Reading them, the majority are
sentences that name a sensor and then compare it to *another* sensor's direction
(`Stripper_Temperature … and Separator_Temperature …`), which a sentence-level direction word cannot
attribute. Two are genuine: `EVT-val-f01-r313` says `A_C_Feed_Stream4 [XMEAS_4] decreases first` while
the store prints it rising, and that is exactly the failure mode the check exists for.

---

## 2. Fallback v3 (step 3)

`scripts/fallback_v3.py` + `configs/kb_overlay_v1.json`; artefact
`outputs/analysis/phase3a/fit/fallback_v3.json`; per-row output `fallback_v3_rows.jsonl`.

**The guard first, then the fixes.** v2 was re-measured under the stricter guard (signed numbers,
number+unit pairs, ranges, tags):

| guard class | **v2** | **v3** |
|---|---|---|
| numbers not in the prompt | 666 | **0** |
| **number+unit pairs not in the prompt** | **37** | **0** |
| ranges not in the prompt | 0 | **0** |
| tags not in the prompt | 0 | **0** |
| **rows with any violation, prompt + supplied classifier values** | **429** | **0** |

**The five known rows are flagged by the new guard and clean in v3**, and the pair class is what
catches them — the KB's own `+25%` and `+5 °C` printed as though they were this event's numbers:

| item | v2 prompt-only | v2's pairs | v3 strict | v3 prompt-only |
|---|---|---|---|---|
| `EVT-val-f04-r306::menu` | 3 | `+25%`, `+5 °C` | **0** | 1 |
| `EVT-val-f04-r307::menu` | 2 | `+5 °C` | **0** | 1 |
| `EVT-val-f04-r310::menu` | 2 | `+5 °C` | **0** | 1 |
| `EVT-val-f04-r317::menu` | 3 | `+25%`, `+5 °C` | **0** | 1 |
| `EVT-val-f11-r307::menu` | 4 | `+25%`, `+5 °C` | **0** | 1 |

The residual 1 is a supplied calibrated probability (`1.00`), which is the class §20.6 counts
separately and which the spec requires be printed in the prompt if v3 ships prose.

**The seven fixes, and what each changed on 834 rows:**

| fix | effect |
|---|---|
| (i) strict guard, re-run on the finished string | 429 → **0** rows; every remaining number is the store's or the classifier's |
| (ii) direction-conflict rule | **50 rows** had a KB clause dropped for pointing the wrong way |
| (iii) differential only for a menu second candidate at p ≥ 0.05 | **405 rows** now carry no differential line at all (abstentions), and no "FAULT_01 at 0.00" remains |
| (iv) action wording gated by `compute_deterministic_severity` | **22 rows** had `CRITICAL`/`EMERGENCY`/`SAFETY INTERLOCK` stripped; severities returned: 403 critical, 4 high, 5 medium, 17 low |
| (v) top sensor = max \|z\| | every narrative now names the largest deviation rather than the first printed |
| (vi) abstention / `NORMAL_TRANSIENT` / unclassified explainer | detector ratios, leading sensors, the listed faults whose **initiating tags are absent** from them, one subsystem check action; no "fault family" wording |
| (vii) KB overlay with its own hash | `FAULT_14`'s mangled degree sign and its emergency action repaired at load time; **KB untouched** (`git status` for the KB, the dataset and the menu config: **clean**), overlay `sha256 10c82130…f7eb6` |

**v3 against v2 on all 834 validation rows:** schema **834/834**; mean words 82.6 / 54.0 (v2 102.5 /
34.5); default-mechanism rows **520** (unchanged); byte-identical reasoning inside a diagnosis group
**4.8 %** (v2 5.04 %). v3 is *shorter* than v2 in `physical_reasoning` because the differential line is
now absent on half the rows and the alarm wording is gone.

Ten examples, including the three abstention items and two `NORMAL_TRANSIENT` items 3B-7 used, are in
the artefact. One, verbatim:

> **`EVT-val-f03-r302::menu`** — `UNCLASSIFIED_OPEN_SET`
> "No listed fault is asserted for this event (UNCLASSIFIED_OPEN_SET). Evidence: detector ratios are LSTM
> autoencoder reconstruction error 1.02x and CVA Q-statistic 0.98x; the leading deviations are
> Reactor_Feed_Rate (XMEAS_6, z=-0.71, stable) and Reactor_Pressure (XMEAS_7, z=0.44, stable); of the
> listed faults, FAULT_02 (initiating XMV_47) and FAULT_04 (initiating XMV_50) would start at a tag that
> is not among those deviations, so neither is supported by this page. Recommended: check the feed header
> flow transmitter and its ratio controller, starting with Reactor_Feed_Rate (XMEAS_6)."

**Nothing was iterated on v3 after the fresh sheet was built.**

---

## 3. The fresh sheet (step 4)

`outputs/analysis/phase3a/fit/audit_ab_100_v3.csv` — **40 rows, 40 distinct items, none of the 100
already seen** — with `audit_ab_100_v3_key.csv` and `audit_ab_100_v3_meta.json`, seed 42.

| property | value |
|---|---|
| classified faults | **20**, round-robin across fault labels |
| abstention | **10** (5 `open_set` on a menu, 5 menu-removed) |
| `NORMAL_TRANSIENT` | **10** |
| adapter side A / B | **19 / 21** |
| comparator | **fallback v3** |
| scoring columns filled | **0 of 9** |
| check flags shown | **none** |
| classifier-error rows available to include | **0** — the 242 main batch's classified rows all agree with their gold label, so none could be drawn |
| unused pool left | fault_menu 44, removed 24, normal 17, open_set 10, unobservable 7 |

Every row's adapter text is the stored answer from the 242 main batch; the other side is v3 rendered
in the same four-section shape the v2 sheet used, so the two sheets are scored the same way.

---

## 4. Spec update (step 5)

`docs/TRAINING_SPEC_PROSE_V1.md` **Part II**, counts only, nothing built:

* the `ranked_hypotheses`-supplied-and-masked change, stated **against the measurement** — the claim
  was 13 of 46 classified rows, the measurement is **0 of 224 closed-list mismatches** — with the
  honest consequence that adopting it makes the answer fully determined by the prompt and removes the
  disagreement question rather than measuring it;
* the **abstention-only** variant: **563** report rows + **961** dialogue = **1,524 rows**, **233,360**
  supervised tokens, **96** steps/epoch, **288** steps, **1 h 44 m** per run at run-1's 21.71 s/step
  (against 3 h 20 m for the full mix) — a 30 % cheaper run needing **no new data**;
* which rebuild steps can spend teacher calls: **only** dialogue regeneration, and neither variant
  changes a dialogue prompt, so both are **teacher-call-free end to end**.

---

## 5. GPU (step 6): **skipped, and the condition is the reason**

`nvidia-smi` read **1,024 MiB, 34 % utilisation, 33 °C** on the second sample (1,013 MiB / 9 % on the
first). No compute process was present — the occupants are the desktop compositor and `EdgeWebView` —
but **34 % is not idle**, and §21.4 makes idleness a precondition rather than a preference. The job was
therefore not started, and no forced-ranking numbers exist for this round.

**A correction to 3B-8, because it matters for the next attempt.** 3B-8 reported that "the repo has no
validated harness" for 512-token generation. That was wrong: `llm/hybrid/generate.py::decode_text(model,
tokenizer, prompt, max_new=512)` exists and `scripts/hybrid_faithfulness.py` builds the merged model and
calls it — they are what 3B-4 and 3B-5 used to produce the 300 answers. 3B-8 looked only at
`instrument_check.py`, `decode_bench.py` and `menu_order_decode.py` and concluded from those three.
The forced-ranking test is **runnable next round without writing a generator**: `build_model(CONFIG,
ADAPTER, merged=True)` + `continuation_prompt(prompt, diagnosis, verdict)` + `decode_text` +
`parse_slim`, with `ranked_hypotheses[0]` added to the forced prefix.

---

## 6. Gate

| # | gate item | status | evidence |
|---|---|---|---|
| 1 | §21 dated and committed **first** | **pass** | `f097b4c`, before any number this round |
| 2 | check table with the reconciled 0-of-242 claim | **pass** | 88 flags any-verdict, all abstention; **0** classified; both readings printed; the 13 does not reproduce and is reported as 0 / 56-upper-bound |
| 3 | v3 table v2 vs v3, five known rows flagged by the new guard | **pass** | v2 429 rows / 37 pairs → v3 **0**; all five named rows flagged then cleared |
| 4 | `audit_ab_100_v3.csv` blank + separate key | **pass** | 40 rows / 40 items, 20/10/10, 0 of 9 columns filled, no flag in the sheet |
| 5 | spec update with counts | **pass** | Part II: 563+961 rows, 233,360 tokens, 288 steps, 1 h 44 m |
| 6 | GPU table **or explicit skip with reason** | **skip, stated** | 34 % utilisation — not idle; plus the 3B-8 harness claim corrected |
| 7 | tests pass | **pass** | app env and `venv_train`, both below |
| 8 | no test data, no teacher calls, no training, run-1 and KB untouched | **pass** | KB `git status` clean inside the v3 run output; run-1 adapter read-only |

**Tests:** app env **588 passed**; `venv_train` **585 passed, 3 skipped**.

---

## 7. What this changes, and what it does not

1. **The A/B record is now verifiable data**, and its odd denominators are explained (`=` → n/y/y/n).
   The rule as written still does not decide the case; §21.2 decides it for the *next* sheet only.
2. **v3 is grounded under a much stricter guard** — unit pairs, not just bare numbers — and that guard
   is what exposes the five rows v2 passed.
3. **The ranked-list premise of the training-spec change is not supported by the stored texts.** That is
   the round's most consequential negative result: it removes an argument for masking
   `ranked_hypotheses`, and leaves determinism as the only reason to do it.
4. **The fresh 40-row sheet is ready to score**, on items nobody has seen, in the two strata §21.2
   decides separately.

**Does not change:** the frozen hybrid, the closed instruments, the fallback's role as comparator, and
the fact that §19.2 remains undecided for the 100 rows already scored — those are recorded, not
reinterpreted.
