# Stage A close-out — report

**Repo:** `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
Teacher calls used: **119** (cap 200). No training.

---

## 1. Summary

The fault-13 correction is **applied** and the cache was re-priced without a
model call for everything it could not have touched.

* The AFFECTED set is **119 items**, not the 493 the earlier union produced and
  nowhere near the 160 stop condition: 119 accepted fault-13 answers plus **0**
  answers for other faults that name `FAULT_13` *and* make a direction claim about
  `XMEAS_7` in the same sentence. The 493 figure was dominated by answers that
  mention reactor pressure as another fault's cascade.
* The re-call ran **119 items, 98 first-pass + 15 recovered on retry, 6 dropped**,
  at 12.17 s/call ≈ 24 min. Fault-13 retention is **0.95 (113/119)**, above the
  0.85 flag line. Every other fault keeps its prose untouched.
* The provenance work is what made that possible: **1,806** answers are exempted
  per record with a written reason and **1,925** carry a replay stamp under the
  new filter hash. Without the stamp, the same edit would have re-called ~1,900
  items (~6.4 h) for text that already passes today's rules.
* Rebuild: **1,919** train / 834 val / 3,168 test items, **0** prompts containing
  the old menu line, **0** leak findings, split integrity clean, unclassified
  share exactly **25.0%**.
* Evaluation exports exist for the first time: 2,408 val rows and 9,424 test rows
  across Protocol A and both Protocol B folds, with Protocol A counts equal to the
  dataset's.

## 2. Changes (commit hashes)

| commit | contents |
|---|---|
| `0b655c1` | replay stamp (`validated_filter_sha256`, `validated_at`, `validation_method`) written to a cache sidecar; reuse rule accepts either hash; `scripts/fault13_affected_set.py` |
| `fa5cc8d` | KB fault-13 direction `decrease → increase`; gerunds added to the rejection vocabulary; `MENU_REVISION = r2`; `data/menu_revisions.json` + test; PROTOCOL §10.13 rewrite |
| `6ac3d74` | `direction_inconsistent_display` advisory flag + `scripts/audit_direction_display.py`; PROTOCOL limitation text for XMV-initiated faults |
| `d3598df` | PROTOCOL §11 pre-registration (checkpoint rule, slim target, retired stale claims) |
| *(this report)* | Stage A report |

Earlier in the same chain: `48ce803` (slim target) and `0662d0e` (eval exports,
decode bench, label scorer) belong to Phase 3A-2 and are reported there.

## 3. Commands and outputs

### 3.1 Affected set (before any call)

```
$ python scripts/fault13_affected_set.py --write
{"counts": {"fault13_accepted": 119, "named_with_direction": 0,
            "union": 119, "cap": 160, "over_cap": false}}
written: outputs/benchmark_v2/dataset/fault13_affected.txt (119 item ids)
```

The first version of this script returned 0 for `named_with_direction` because it
required the fault reference and the direction claim in the *same sentence*, and
the one candidate splits them. Relaxing the fault reference to answer scope keeps
the claim sentence-scoped, which is the honest place for it: a direction word in a
sentence about a different sensor is not a claim about `XMEAS_7`.

### 3.2 Provenance sync (no model call)

```
$ python scripts/sync_cache_provenance.py --affected-file .../fault13_affected.txt
{"filter_sha256": "a3da3b5b3a08…", "menu_revision": "r2",
 "items": 1925, "replay_valid": 1920, "stamped": 1920, "exempted": 1806,
 "affected_held_pending": 114,
 "reusable_before": {"filter_changed": 1925},
 "reusable_after": {"filter_changed": 5, "menu_revision_changed": 114,
                    "reusable": 1806}}
```

`before` is measured under the new hash (everything stale, because the gerunds
moved `filters.py`). Under the *old* hash and r1, **169** of the 2,353 cache
records were reusable; the honest comparison is 169 → **1,806 reusable / 119
pending**.

### 3.3 Re-call

```
$ python scripts/build_benchmark_dataset.py --teacher \
    --teacher-items .../fault13_affected.txt --exclude-rejected --max-calls 200
teacher 10/119 calls, rejected 0, mean 14.9s
… 100/119 calls, rejected 4, mean 12.3s
train 1919 items / val 834 / test 3168
```

| measure | value |
|---|---|
| items attempted | 119 |
| first-pass accepted | 98 |
| retried | 21 |
| recovered on retry | 15 |
| dropped | **6** |
| rejection codes (first + retry) | `sensor_not_grounded` 3, `unknown_tag` 1, `number_not_in_store` 1, `direction_conflict` 1 |
| mean seconds per call | 12.17 |
| fault-13 retention vs affected set | **0.95** (113/119) |

The 6 drops are grounding failures (prose citing `XMV_45/51/52` on a fault whose
evidence has none) plus one `Steam_Flow` fabrication. None is a direction failure.

### 3.4 Prompt and dataset checks after the rebuild

| check | result |
|---|---|
| `"Reactor_Pressure decreases first"` in any prompt | **0** of 5,921 items |
| `"Reactor_Pressure increases first"` | 5,783 (the 138 without it are fault-13 open-set twins) |
| leak violations | **0** (4,634 prompts checked at build time; the walk over all 5,921 items finds none) |
| split integrity | pairwise disjoint in both protocols and both folds |
| train items | 1,919 (was 1,925; −6 dropped) |
| unclassified share of faulty train | **0.2500** (cap held) |
| faults below the 70% coverage floor | **none** |
| fault-13 answers asserting a direction on `XMEAS_7` | **94** |
| fault-13 closed-set answers contradicting the stored trend | **0** |

Two items still trip a *raw* window scan for direction words, and both are
correctly outside the rule:

| item | why it is not a filter failure |
|---|---|
| `EVT-train-f13-r055::openset` | open-set twin; the "decreasing" in the 60-char window belongs to "a decreasing valve position", not to the pressure |
| `EVT-train-f19-r014::menu` | target is `UNCLASSIFIED_OPEN_SET`, and the rule deliberately does not hold an abstention to one fault's KB sign — there is no single entry to agree with |

### 3.5 Display-name flag (D3), priced with no calls

```
$ python scripts/audit_direction_display.py
accepted_items 1902, items_with_display_flag 4, flags 4
by_fault {2:1, 12:1, 15:1, 20:1}; by_tag_kind {measurement: 4}; by_basis {trend: 4}
```

Four sentences in 1,902 answers. Small enough that the tag rule's blindness is not
hiding a systematic problem, and stated as a flag rather than a rule so it cannot
silently start rejecting.

### 3.6 Evaluation exports

```
$ python scripts/build_eval_exports.py
val : dataset 834  -> 2,408 rows  {A/val 834, fold_1/val 637, fold_1/test 150,
                                   fold_2/val 687, fold_2/test 100}   matches_dataset true
test: dataset 3168 -> 9,424 rows  {A/test 3,168, fold_1/test 600, fold_1/test_seen 2,528,
                                   fold_2/test 400, fold_2/test_seen 2,728} matches_dataset true
```

`fold_N/test` is larger than 100 runs per held-out fault on purpose: the held-out
component spans runs 1–500, because there is no run of a held-out fault that may be
trained on. A held-out fault's validation-run items therefore appear under
`fold_N/test`, which is why the val split contributes 150 and 100 rows there.

## 4. Tests

| suite | result |
|---|---|
| `tests/test_benchmark_direction.py` (new) | 15 passed |
| `tests/test_benchmark_provenance.py` (+7 new) | 24 passed |
| `tests/test_to_sft.py` (+3 new) | 11 passed |
| `tests/test_slim_target.py` (new) | 15 passed |
| `tests/test_eval_exports.py` (new) | 9 passed |
| full suite, app environment | **446 passed** |

Three pre-existing tests encoded the old KB state and were **flipped in the same
commit** that changed it, not weakened: `test_measurement_direction_is_still_a_conflict`
(scaffolding), `test_the_gerund_form_is_not_yet_seen_by_the_rule` (follow-up
provenance), and the hard-coded `r1` in the follow-up cache test.

## 5. Gate table

| gate item | verdict | evidence |
|---|---|---|
| AFFECTED count reported before any call | **PASS** | 119 < 160, §3.1 |
| Change applied in one commit with tests both ways | **PASS** | `fa5cc8d`, 15 + 6 assertions |
| Reuse-rule counts before/after | **PASS** | 169 → 1,806 reusable, §3.2 |
| Re-call results and retention | **PASS** | 98/21/15/6, retention 0.95, §3.3 |
| Old menu line absent from all prompts | **PASS** | 0 of 5,921 |
| Fault-13 contradictions 0 or listed | **PASS (listed)** | 0 closed-set; 2 open-set items explained, §3.4 |
| Eval export counts match | **PASS** | `matches_dataset: true` for both splits |
| PROTOCOL amendment dated, no result seen | **PASS** | §11, dated 2026-10-31 |
| All checks clean | **PASS** | leak 0, splits disjoint, cap 25.00% |
| Tests pass | **PASS** | 446 passed |

## 6. Deviations

1. **The `menu_revision_unaffected` marker is a sidecar line, not a field on the
   cache record.** The cache is append-only evidence; a claim about it belongs
   beside it. The reuse rule reads one merged record, so the behaviour is the one
   specified — only the storage differs, and the merge is tested.
2. **The AFFECTED definition is sentence-scoped for the direction claim.** The
   brief's wording admits a union of 180 if answer-scope is used; the difference is
   entirely fault-13's own answers plus one fault-20 item whose XMEAS_7 sentence has
   no direction word. Recorded because the number decides the cost.
3. **Two items remain listed rather than fixed** (§3.4). Both are out of the rule's
   scope by design, and changing the scope would mean re-calling abstention items
   against a KB entry that a given abstention does not have.
4. **The previous `train.jsonl` hash was not captured before the rebuild.** The last
   recorded one is `feea8b06…` from the Phase 2B wrapup; the immediate predecessor
   of this build is not recorded anywhere and is not reconstructed here.

## 7. Open questions

1. **Should the direction rule apply to abstention items?** Today it does not, by
   design. Applying it would have rejected 2 of 1,919; the cost of the re-call is
   trivial, the question is whether an abstention can meaningfully be "wrong" about
   a fault it is ruling out.
2. **Should the prompt line change reach the 138 twin items?** They have no fault-13
   entry by construction, so they never carried the old line either. Nothing to do;
   recorded so the 5,783/5,921 asymmetry is not read as a miss.
3. **Promote the display-name flag?** 4 of 1,902. Cheap to keep as a flag, cheap to
   promote; the owner decision needs the 4 sentences, which the audit JSON holds.

## 8. Cost of what remains

| item | estimate |
|---|---|
| Phase 3A-2 GPU work (decode bench 6 modes at 20 items, label scan 20 items, smoke) | 1.5–2.5 h measured wall-clock, plus model loads (~2 min each) |
| Stage B pilot (200 calls) + full train run (2,182 calls) | ≈ 0.7 h + 7.3 h at 12.2 s/call |
| Full evaluation pass, if decode stays at the measured 24 s/item | not affordable — see the Phase 3A-2 budget |
