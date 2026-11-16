"""Correct the confirmatory-pass section of the final report after the 2026-11-16 batch-size measurements.

The earlier text assumed batch 1 was the reference and therefore quoted 33.3 h. The stored fit set was
generated at batch 16 / 2.47 s per item, so the admissible configuration is that one and the pass is 2.2 h.
Each replacement is asserted so a partial apply cannot pass silently.
"""
import pathlib

PATH = pathlib.Path("docs/reports/BENCHMARK_FINAL_REPORT_2026-11-16.md")

EXEC_OLD = "is 33.3 hours. The one shot is unspent."
EXEC_NEW = ("runs at 2.2 h at the batch size the scored cells were generated with (16), so it is admissible\n"
            "rather than blocked. The one shot is unspent.")

WHY_OLD = """**Why it has not run.** Two reasons, both measured today.

1. **The batched path is not output-equivalent.** Batch 1 and batch 4 were run on the same four test items
   with the same adapter and cap: **0 of 4 texts were byte-identical** (lengths 1117 vs 1234, 1223 vs 1222,
   1285 vs 1369, 1260 vs 1208). Batching is therefore not usable for a one-shot measurement — it would
   change the text being scored. Measured speed at batch 4 is 45.5 s for 4 items against 146.0 s at batch 1
   (3.2×), which is exactly the saving that cannot be taken.
2. **At the admissible speed the job is 33.3 h.** 3,168 items × 37.83 s/item (measured, cap 512) =
   119,845 s.

**What this costs and what it needs.** Either a decode path proven byte-identical to batch 1 (a rewrite of
the manual cache loop to a masked `generate` call, then a fresh equivalence check), or a single 33.3-hour
GPU run. The one shot remains unspent; spending it on a non-equivalent path would be worse than not
spending it."""

WHY_NEW = """**What the batch-size measurements settle.** Three comparisons, all on stored prompts:

| comparison | identical |
|---|---|
| batch 1 vs batch 4, same 4 test items | **0 / 4** |
| batch 1 vs **padding-free** length-grouped batch 8, same 8 test items | **3 / 8** |
| stored probes of the same 12 validation prompts at batch 16 vs batch 8 | **1 / 12** |

Padding is therefore not the cause: grouping prompts by exact token length and batching only inside a group
removes every pad token and still changes 5 of 8 texts. Generation is **not batch-invariant**, and no
masking or grouping trick recovers the batch-1 text — I implemented the padding-free version, measured it,
and reverted it.

That result does not block the pass, because **batch 1 was never the reference.** Every scored cell comes
from texts generated at **batch 16, cap 512, 2.47 s/item** (`prose_pilot/raw_pilot_run.json`). A batch-1 pass
would be a *different generator*, not a canonical one; the admissible configuration is the one the fit set
used. At that speed the pass costs 3,168 × 2.47 s = 7,825 s ≈ **2.2 h**, not the 33.3 h quoted earlier (that
figure applied a single-item decode time of 37.83 s to a batched pipeline).

**Caveat on reproduction of stored texts.** Re-running eight stored validation prompts at batch 16 did not
reproduce the stored texts, and the fresh output used a different JSON schema (`fault_id` first), so the
failure is at least partly a prompt-build difference and cannot be read as non-determinism of the same call.
An exact reproduction check would need the original prompt builder; until then this pass is comparable to the
fit set by configuration, not by replayed text.

**State.** The one shot remains unspent. Spending it at batch 16 is admissible and takes 2.2 h; launching it
was not part of what was authorised today."""

E_OLD = """scored into the cell table**: scoring needs 2 × 520 generations, i.e. 10.9 h at the admissible single-item
speed (the same batching limitation as §9 applies)."""

E_NEW = """scored into the cell table**: scoring needs 2 × 520 generations, i.e. about 0.7 h at the batch-16 speed the
fit set was generated with (2.47 s/item)."""

LIM_OLD = """6. **Batching is not output-equivalent**, which is why the confirmatory pass is 33.3 h rather than ~10 h,
   and why E is unscored."""

LIM_NEW = """6. **Generation is not batch-invariant** (1 of 12 identical between batch 16 and batch 8, 3 of 8 with
   padding removed). The pass must therefore run at the fit set's batch size, cost 2.2 h, and no speedup is
   available by batching; E's scoring is 0.7 h on the same basis."""

REPRO_OLD = "rows_test.jsonl --orders kb --batch 1"
REPRO_NEW = "rows_test.jsonl --orders kb --batch 16"

SUBS = [(EXEC_OLD, EXEC_NEW), (WHY_OLD, WHY_NEW), (E_OLD, E_NEW), (LIM_OLD, LIM_NEW), (REPRO_OLD, REPRO_NEW)]


def main() -> int:
    text = PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    for index, (old, new) in enumerate(SUBS, 1):
        if old not in text:
            print("PATTERN MISS on edit %d: %s" % (index, old.splitlines()[0][:70]))
            return 2
        text = text.replace(old, new, 1)
    PATH.write_text(text, encoding="utf-8")
    print("applied %d edits; %d lines" % (len(SUBS), len(text.splitlines())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
