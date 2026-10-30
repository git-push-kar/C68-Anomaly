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
