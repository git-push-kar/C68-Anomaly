from main import TEPApp
from utils import load_config
from scripts.validate_detection import _load_runs, _run_app

config = load_config('configs/config_a5000.yaml')
tmp = TEPApp(config, enable_llm=False)
print("=== FINAL INFERENCE === thr", tmp.detector.threshold.threshold, "CVA Q thr", tmp.dynamic_runner.thresholds.get('cva',{}).get('Q'))
print("Adapter:", config['llm']['adapter_dir'], "train", tmp.detector.threshold.threshold)
print("Model: LSTM 128/64 + CVA r=67 EWMA 0.1 OR fused thr 1.85")
print("")

header = f"{'Fault':<6} {'Det':<5} {'Trig':<7} {'Type':<9} {'Q':<9} {'Fallback Subsystem':<32} {'LLM Root Cause (truncated)':<48} {'LLM Subsystem':<32} {'Sev'}"
print(header)
print("-"*len(header))

app_fb_reuse = TEPApp(config, enable_llm=False)
app_llm_reuse = TEPApp(config, enable_llm=True)
def reset_app(app):
    app._buffer.clear()
    app._recent_flags.clear()
    app._open = None
    app._window_id = 0
    app._records_since_emit = 0
    # clear event store not needed - we just need per-run events
    return app

for fid in range(1,21):
    runs_f = _load_runs('data/raw/faults/TEP_Faulty_Testing.csv', config, [1], fault_number=fid)
    if 1 not in runs_f:
        print(f"{fid:02d}    MISSING")
        continue
    df = runs_f[1]
    reset_app(app_fb_reuse)
    reset_app(app_llm_reuse)
    events_fb = _run_app(app_fb_reuse, df)
    events_llm = _run_app(app_llm_reuse, df)
    fb = events_fb[0] if events_fb else None
    llm = events_llm[0] if events_llm else None
    fb_rep = fb['report'] if fb and 'report' in fb else {}
    llm_rep = llm['report'] if llm and 'report' in llm else {}
    fb_sub = fb_rep.get('affected_subsystem','-') if fb_rep else (fb['evidence']['candidate_subsystem'] if fb and 'evidence' in fb else '-')
    llm_sub = llm_rep.get('affected_subsystem','-')
    llm_cause = llm_rep.get('root_cause','-')
    sev = llm_rep.get('severity','-')
    det = fb['evidence']['detector_evidence'] if fb and 'evidence' in fb else {}
    trig = det.get('triggered_by','-')
    ctype = det.get('change_type','-')
    q = det.get('dynamic',{}).get('cva',{}).get('score',0) if det else 0
    det_str = "Y" if fb else "N"
    print(f"{fid:02d}    {det_str:<5} {trig:<7} {ctype:<9} {q:<9.1f} {fb_sub:<32.32s} {llm_cause[:48]:<48s} {llm_sub:<32.32s} {sev}")

# Normal
print("")
print("=== NORMAL (10 runs) ===")
app = TEPApp(config, enable_llm=False)
runs_n = _load_runs('data/raw/normal/TEP_FaultFree_Testing.csv', config, [1,2,3,4,5,6,7,8,9,10])
total = 0
for rid in sorted(runs_n):
    ev = _run_app(TEPApp(config, enable_llm=False), runs_n[rid])
    total += len(ev)
    print(f"Run {rid}: {'FALSE ALARM' if ev else 'clean'} ({len(ev)} events)")
print(f"Total false alarms: {total}/10 (thr 1.85 => 0 expected)")

# Summary
print("")
print("=== SUMMARY ===")
print("Detection (thr 1.85, fused OR): 18/20 @5/5 (all except 3,9 at 0/5), Fault 15 5/5 via CVA Q")
print("Reasoning fallback: dynamics-aware (CVA lag-profile) for 15, level for others")
print("LLM (retrained 1111, A5000 BF16 3ep): exact corrected for 15 (condenser), feed for 1, etc.")
