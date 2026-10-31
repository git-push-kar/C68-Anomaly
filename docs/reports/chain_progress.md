# Chained run progress (Stage A -> Phase 3A-2 -> Stage B)

Repo: C:\Users\Admin\Desktop\anomaly, branch `benchmark-upgrade`.

| step | state | note |
|---|---|---|
| A1 provenance (replay stamp + sidecar) | done | reuse rule accepts `validated_filter_sha256`; sidecar keeps the cache append-only (`0b655c1`) |
| A2 affected-set count | done | 119 fault-13 answers + **0** named-with-direction = 119, under the 160 cap (`scripts/fault13_affected_set.py`) |
| A3 apply fault-13 + gerunds + r2 | done | `fa5cc8d`; filter hash `a3da3b5b3a08…` |
| A4 display-name flag | done | `6ac3d74`; 4 of 1,902 accepted answers flagged, advisory only |
| A5 D4 XMV scoping | done | already scoped in code; PROTOCOL §10.13 relabelled APPLIED with the 120/120, 120/120, 26/26 limitation |
| A6 re-call 119 | running | 70/119 calls, 4 dropped (grounding), ~12.4 s/call |
| A7 rebuild | pending | |
| A8 eval exports | code done | `scripts/build_eval_exports.py`; val 834 -> 2,408 rows, test 3,168 -> 9,424, Protocol A counts match |
| A9 PROTOCOL amendment | done | §11 pre-registration, dated 2026-10-31, no model result seen |

Phase 3A-2 code (CPU parts, written while the teacher ran):

| item | state |
|---|---|
| slim target + hydrate + guards | done (`48ce803`), answer tokens p50 402 -> 296 |
| fold exports | done, fold_1 1,568 rows, fold_2 1,687 (held out 4/11/15 and 14/15) |
| decode bench + label scorer | code done (`0662d0e`), GPU run pending |
2026-10-31T04:04:53 followup train: 200 teacher calls, 89 dialogues, 194 accepted, 10 rejected, 65s elapsed
2026-10-31T04:15:26 followup train: 402 teacher calls, 178 dialogues, 390 accepted, 24 rejected, 698s elapsed
2026-10-31T04:25:16 followup train: 600 teacher calls, 268 dialogues, 587 accepted, 35 rejected, 1288s elapsed
2026-10-31T04:35:46 followup train: 801 teacher calls, 361 dialogues, 778 accepted, 51 rejected, 1918s elapsed
2026-10-31T04:45:28 followup train: 1000 teacher calls, 449 dialogues, 972 accepted, 62 rejected, 2500s elapsed
2026-10-31T04:55:56 followup train: 1201 teacher calls, 539 dialogues, 1170 accepted, 79 rejected, 3128s elapsed
2026-10-31T05:05:31 followup train: 1400 teacher calls, 626 dialogues, 1356 accepted, 102 rejected, 3703s elapsed
2026-10-31T05:15:54 followup train: 1600 teacher calls, 713 dialogues, 1549 accepted, 119 rejected, 4326s elapsed
2026-10-31T05:25:56 followup train: 1801 teacher calls, 803 dialogues, 1738 accepted, 131 rejected, 4928s elapsed
2026-10-31T05:36:36 followup train: 2000 teacher calls, 892 dialogues, 1921 accepted, 147 rejected, 5568s elapsed
2026-10-31T05:47:30 followup train: 2200 teacher calls, 980 dialogues, 2109 accepted, 159 rejected, 6222s elapsed
2026-10-31T05:58:48 followup train: 2401 teacher calls, 1069 dialogues, 2292 accepted, 177 rejected, 6900s elapsed
2026-10-31T06:19:21 followup train: 402 teacher calls, 178 dialogues, 391 accepted, 23 rejected, 137s elapsed
2026-10-31T06:22:46 followup train: 801 teacher calls, 361 dialogues, 782 accepted, 47 rejected, 341s elapsed
2026-10-31T06:25:50 followup train: 1201 teacher calls, 539 dialogues, 1176 accepted, 73 rejected, 525s elapsed
2026-10-31T06:28:56 followup train: 1600 teacher calls, 713 dialogues, 1558 accepted, 110 rejected, 711s elapsed
2026-10-31T06:32:01 followup train: 2000 teacher calls, 892 dialogues, 1935 accepted, 133 rejected, 896s elapsed
2026-10-31T06:36:23 followup train: 2401 teacher calls, 1069 dialogues, 2306 accepted, 163 rejected, 1158s elapsed
2026-10-31T06:44:23 followup train: 600 teacher calls, 268 dialogues, 588 accepted, 34 rejected, 184s elapsed
2026-10-31T06:48:21 followup train: 1201 teacher calls, 539 dialogues, 1176 accepted, 73 rejected, 422s elapsed
2026-10-31T06:52:02 followup train: 1801 teacher calls, 803 dialogues, 1750 accepted, 119 rejected, 644s elapsed
2026-10-31T06:57:13 followup train: 2401 teacher calls, 1069 dialogues, 2305 accepted, 164 rejected, 955s elapsed
2026-10-31T06:56 followup: full train run done — 2,562 calls, 2,377 accepted (92.8%), 7,361s; open-set echoes 3 -> 0 after the paraphrase-picker fix
2026-10-31T07:02 followup: SFT mix 1,919 report + 959 dialogue rows (ratio 2.001), audit sheet 100 rows, leak/split/open-set clean, 480 tests pass in both envs
| Stage B | done: corpus, SFT mix, audit sheet, checks | 92.8% acceptance, ratio 2.001, 0 leaks |
## $ (date -u +%FT%TZ) close-out
2026-10-31T14:05 pre-training close-out: mixed-file smoke 14.31 s/step (17.02 GiB, 80 C, 1,658 MHz), label scoring invariant at batch 16 (max delta p 0.008), LM-direct decode 4.42x with identical probabilities, fold test n 957/638, 483 tests pass in both envs; 50 refusal_missing turns still unrecovered pending D1
2026-10-31 pre-run step: follow-up corpus replayed under the fixed refusal rule (2,377 -> 2,418 accepted, 0 calls); mix filled by case priority (2,878 rows, 959 dialogue rows); PROTOCOL 12.1/12.2 dated; checks and 491 tests green in both envs.
2026-10-31 mix floor: per-case dialogue targets replace the tier fill (961 rows, deviation +2); PROTOCOL 12.1b dated; 494 tests green in both envs; no training.
