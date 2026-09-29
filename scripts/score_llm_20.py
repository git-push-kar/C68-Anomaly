"""Full 20-fault LLM rescore + metrics (Stage 2 proof). Reuses single LLM app."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, time
from collections import Counter, defaultdict
from utils import load_config
from main import TEPApp
from scripts.validate_detection import _load_runs, _run_app
from scripts.fault_knowledge import FAULT_KNOWLEDGE

def norm(s):
    return (s or "").lower().strip()

def root_match(pred, kb_name, fid):
    p, k = norm(pred), norm(kb_name)
    if fid in (16,17,18,19,20):
        # unknown faults: accept "unknown" in pred
        return "unknown" in p
    if p == k: return True
    # fuzzy: all significant words of kb in pred or vice versa
    kw = [w for w in k.replace("/"," ").split() if len(w) > 3 and w not in ("with","from","step","change")]
    hit = sum(1 for w in kw if w in p)
    return hit >= max(2, len(kw)-1)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--faults", type=int, nargs="+", default=list(range(1,21)))
    ap.add_argument("--out", type=str, default="outputs/llm_dataset_v2/llm_20fault_metrics.json")
    args = ap.parse_args()
    config = load_config("configs/config_a5000.yaml")
    app = TEPApp(config=config, enable_llm=True)
    print(f"LLM loaded: {app.rca is not None}, thr {app.detector.threshold.threshold}", flush=True)
    runs_per_fault = args.runs
    fault_list = args.faults
    out_path = args.out
    results = []
    for fid in fault_list:
        kb = FAULT_KNOWLEDGE[fid]
        runs = _load_runs("data/raw/faults/TEP_Faulty_Testing.csv", config, list(range(1, runs_per_fault+1)), fault_number=fid)
        for run_id in sorted(runs):
            # fresh state, reuse app object but reset streaming buffers
            app._buffer.clear(); app._recent_flags.clear(); app._open = None
            app._window_id = 0; app._records_since_emit = 0
            events = _run_app(app, runs[run_id])
            if not events:
                results.append({"fault_id": fid, "run": run_id, "detected": False,
                    "sub_ok": False, "exact_ok": False, "sev_ok": False,
                    "pred_sub": None, "pred_cause": None, "pred_sev": None,
                    "kb_sub": kb["subsystem"], "kb_name": kb["name"], "kb_sev": kb["severity"]})
                print(f"Fault {fid:02d} run {run_id}: NOT DETECTED", flush=True)
                continue
            ev = events[0]
            rep = ev.get("report", {}) or {}
            pred_sub, pred_cause, pred_sev = rep.get("affected_subsystem",""), rep.get("root_cause",""), rep.get("severity","")
            # subsystem: allow "X (alternative: Y)" — check primary contains kb
            sub_ok = kb["subsystem"] in norm(pred_sub)
            exact_ok = root_match(pred_cause, kb["name"], fid)
            sev_ok = norm(pred_sev) == norm(kb["severity"])
            results.append({"fault_id": fid, "run": run_id, "detected": True,
                "sub_ok": sub_ok, "exact_ok": exact_ok, "sev_ok": sev_ok,
                "pred_sub": pred_sub, "pred_cause": pred_cause, "pred_sev": pred_sev,
                "kb_sub": kb["subsystem"], "kb_name": kb["name"], "kb_sev": kb["severity"],
                "triggered_by": (ev.get("evidence",{}) or {}).get("detector_evidence",{}).get("triggered_by"),
                "change_type": (ev.get("evidence",{}) or {}).get("detector_evidence",{}).get("change_type")})
            print(f"Fault {fid:02d} run {run_id}: sub={'OK' if sub_ok else 'MISS'} exact={'OK' if exact_ok else 'MISS'} sev={'OK' if sev_ok else 'MISS'} | pred={pred_cause[:50]} -> {pred_sub[:40]} [{pred_sev}]", flush=True)
    n = len(results)
    det = sum(1 for r in results if r["detected"])
    sub = sum(1 for r in results if r["sub_ok"])
    ex = sum(1 for r in results if r["exact_ok"])
    sev = sum(1 for r in results if r["sev_ok"])
    # exclude 3,9 from headline (accepted limitation)
    core = [r for r in results if r["fault_id"] not in (3,9)]
    metrics = {
        "runs_per_fault": runs_per_fault, "total": n, "detected": det,
        "subsystem_acc_all": round(sub/max(n,1),3), "exact_acc_all": round(ex/max(n,1),3), "sev_acc_all": round(sev/max(n,1),3),
        "core_total": len(core), "core_detected": sum(1 for r in core if r["detected"]),
        "core_sub": round(sum(1 for r in core if r["sub_ok"])/max(len(core),1),3),
        "core_exact": round(sum(1 for r in core if r["exact_ok"])/max(len(core),1),3),
        "core_sev": round(sum(1 for r in core if r["sev_ok"])/max(len(core),1),3),
    }
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    # per-fault table
    print("\nPer-fault (detected/sub/exact/sev):")
    for fid in range(1,21):
        sub_r = [r for r in results if r["fault_id"]==fid]
        print(f" {fid:02d} det {sum(r['detected'] for r in sub_r)}/{len(sub_r)} sub {sum(r['sub_ok'] for r in sub_r)} exact {sum(r['exact_ok'] for r in sub_r)} sev {sum(r['sev_ok'] for r in sub_r)} | kb={FAULT_KNOWLEDGE[fid]['name'][:40]}")
    out = Path(out_path)
    out.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2))
    print(f"\nSaved {out}")
    # also save run-level for 3,9 note
    f39 = [r for r in results if r["fault_id"] in (3,9)]
    print(f"3,9 detected {sum(r['detected'] for r in f39)}/{len(f39)} (expected 0 at thr 1.85, accepted limitation)")

if __name__ == "__main__":
    main()
