"""A5: event-level statistics at the frozen v2 operating point.

The per-run alarm table records how many events a run produced but not where
each event started, so "did this event begin before the fault was injected?"
cannot be answered from the CSV alone. This script re-segments events from the
cached window scores with exactly the frozen thresholds and persistence, and
splits every event by whether its opening window contains a post-onset sample.

Definitions:
  event          main.py aggregator: opens after `persistence` consecutive
                 alarmed windows, closes after `min_separation` quiet windows or
                 at `max_event_windows`.
  pre-onset      opening window ends at or before the onset, so the windowed
                 detector had not seen any fault sample yet.
  post-onset     opening window contains at least one post-onset sample.

    python scripts/event_level_stats_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_detector_v2 import segment_events, window_starts  # noqa: E402
from utils import ensure_dir, get_fault_onset, load_config, save_json  # noqa: E402

CACHE = ROOT / "outputs" / "evaluation" / "detector_v2" / "op_point_cache.npz"
FROZEN = ROOT / "configs" / "detector_v2_operating_point.json"
OUT = ROOT / "outputs" / "evaluation" / "detector_v2" / "event_level_stats_v2.json"
MD = ROOT / "outputs" / "evaluation" / "detector_v2" / "event_level_stats_v2.md"


def fused(w, thr):
    return np.maximum(w["lstm_ae"] / max(thr["lstm_ae"], 1e-9), w["cva_Q"] / max(thr["cva_Q"], 1e-9))


def main() -> None:
    config = load_config("configs/config_a5000.yaml")
    op = json.loads(FROZEN.read_text(encoding="utf-8"))
    thr = op["thresholds"]
    confirm = int(op["persistence_windows"])
    separate = int(config["events"]["min_separation_windows"])
    max_ev = int(config["events"]["max_event_windows"])
    ws = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])
    onset = int(get_fault_onset(config, "faulty_testing"))

    data = dict(np.load(CACHE))

    # --- fault-free runs -------------------------------------------------
    starts = window_starts(data["far_lstm_ae"].shape[1], stride)
    ff_events = []
    ff_runs_with_event = 0
    for i in range(data["far_lstm_ae"].shape[0]):
        w = {k: data[f"far_{k}"][i] for k in ("lstm_ae", "cva_Q", "dpca_SPE")}
        s = fused(w, thr)
        ev = segment_events((s > 1.0).astype(int), starts, confirm, separate, max_ev)
        ff_events.append(len(ev))
        ff_runs_with_event += int(bool(ev))
    ff = {
        "n_runs": int(data["far_lstm_ae"].shape[0]),
        "events_per_run": float(np.mean(ff_events)),
        "runs_with_event": int(ff_runs_with_event),
        "run_far": float(ff_runs_with_event / len(ff_events)),
        "event_count_histogram": {str(k): int(v) for k, v in sorted(
            {n: ff_events.count(n) for n in set(ff_events)}.items())},
    }

    # --- faulty runs -----------------------------------------------------
    faults = data["val_faults"]
    per_fault = {}
    all_pre = []
    all_post = []
    for i in range(data["val_lstm_ae"].shape[0]):
        fault = int(faults[i])
        w = {k: data[f"val_{k}"][i] for k in ("lstm_ae", "cva_Q", "dpca_SPE")}
        s = fused(w, thr)
        ev = segment_events((s > 1.0).astype(int), starts, confirm, separate, max_ev)
        pre = [o for o, _ in ev if int(starts[o]) + ws <= onset]
        post = [o for o, _ in ev if int(starts[o]) + ws > onset]
        all_pre.extend(pre)
        all_post.extend(post)
        b = per_fault.setdefault(fault, {"runs": 0, "events": 0, "pre": 0, "post": 0,
                                         "runs_with_post": 0, "runs_with_pre": 0})
        b["runs"] += 1
        b["events"] += len(ev)
        b["pre"] += len(pre)
        b["post"] += len(post)
        b["runs_with_post"] += int(bool(post))
        b["runs_with_pre"] += int(bool(pre))

    total_events = len(all_pre) + len(all_post)
    faulty = {
        "n_runs": int(data["val_lstm_ae"].shape[0]),
        "events_per_run": total_events / max(1, data["val_lstm_ae"].shape[0]),
        "n_events": total_events,
        "n_pre_onset": len(all_pre),
        "n_post_onset": len(all_post),
        "fraction_pre_onset": len(all_pre) / max(1, total_events),
        "runs_with_event": sum(1 for f in per_fault.values() for _ in range(0)),
        "per_fault": {
            str(k): {
                **v,
                "events_per_run": v["events"] / max(1, v["runs"]),
                "post_onset_run_rate": v["runs_with_post"] / max(1, v["runs"]),
                "pre_onset_event_fraction": v["pre"] / max(1, v["events"]),
            }
            for k, v in sorted(per_fault.items())
        },
    }
    faulty["runs_with_event"] = sum(
        1 for i in range(data["val_lstm_ae"].shape[0])
        if per_fault[int(faults[i])]["runs"] > 0
    )

    f15 = faulty["per_fault"]["15"]
    record = {
        "operating_point": op["operating_point_id"],
        "thresholds": thr,
        "persistence_windows": confirm,
        "min_separation": separate,
        "max_event_windows": max_ev,
        "onset_sample": onset,
        "window_size": ws,
        "fault_free": ff,
        "faulty": faulty,
        "fault_15": {
            "runs": f15["runs"],
            "runs_with_post_onset_event": f15["runs_with_post"],
            "post_onset_run_rate": f15["post_onset_run_rate"],
            "events_per_run": f15["events_per_run"],
        },
    }
    ensure_dir(OUT.parent)
    save_json(OUT, record)

    lines = [
        "# Event-level statistics at the frozen v2 operating point",
        "",
        f"`{op['operating_point_id']}` — persistence {confirm} consecutive alarmed windows, "
        f"min separation {separate}, max event {max_ev}, onset sample {onset}.",
        "",
        "## Fault-free runs (TEP_FaultFree_Testing 1-500)",
        "",
        f"- events per run: **{ff['events_per_run']:.3f}**",
        f"- runs with at least one event: **{ff['runs_with_event']} / {ff['n_runs']} "
        f"({ff['run_far']*100:.1f}%)**",
        f"- event-count histogram: `{json.dumps(ff['event_count_histogram'])}`",
        "",
        "## Faulty runs (Protocol A validation, 20 faults x 100 runs)",
        "",
        f"- events per run: **{faulty['events_per_run']:.3f}**",
        f"- events: {total_events} total — **{len(all_pre)} pre-onset ({len(all_pre)/max(1,total_events)*100:.1f}%)**, "
        f"{len(all_post)} post-onset",
        "",
        "| Fault | events/run | runs with post-onset event | post-onset rate | pre-onset share of events |",
        "|---|---|---|---|---|",
    ]
    for f, v in faulty["per_fault"].items():
        lines.append(
            f"| {f} | {v['events_per_run']:.2f} | {v['runs_with_post']}/{v['runs']} | "
            f"{v['post_onset_run_rate']*100:.0f}% | {v['pre_onset_event_fraction']*100:.0f}% |"
        )
    lines += [
        "",
        f"**Fault 15:** {f15['runs_with_post']}/{f15['runs']} runs have a post-onset event "
        f"({f15['post_onset_run_rate']*100:.0f}%), {f15['events_per_run']:.2f} events/run.",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: record[k] for k in ("fault_free", "fault_15")}, indent=1))
    print(f"\nfaulty: {total_events} events, {faulty['events_per_run']:.2f}/run, "
          f"{len(all_pre)/max(1,total_events)*100:.1f}% pre-onset")
    print(f"wrote {OUT}\nwrote {MD}")


if __name__ == "__main__":
    main()
