import json, subprocess, re

thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.2, 1.5, 2.0, 2.2]
faults = [1,4,14,15,21]
normal_runs = list(range(1,6))

def set_threshold(v):
    j=json.load(open('outputs/anomaly_detector/threshold.json'))
    j['threshold']=v
    json.dump(j, open('outputs/anomaly_detector/threshold.json','w'), indent=2)

def run_cmd(cmd):
    r=subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out=r.stdout+r.stderr
    m=re.search(r'Closed anomaly events:\s*(\d+)', out)
    events=int(m.group(1)) if m else -1
    scores=re.findall(r'score\s+([0-9\.]+)', out)
    return events, scores, out

for th in thresholds:
    set_threshold(th)
    print(f"\n=== THRESHOLD {th} ===")
    normal_events=[]
    for nr in normal_runs:
        ev,_,_ = run_cmd(f'python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 999999 --normal-run {nr} 2>&1')
        normal_events.append(ev)
    print(f"Normal 1..5 events: {normal_events}  false_alarms={sum(1 for x in normal_events if x!=0)}/5")
    for f in faults:
        ev,scores,_= run_cmd(f'python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 160 --normal-run 1 --fault-number {f} --fault-run 1 2>&1')
        try:
            max_score = max([float(s) for s in scores]) if scores else 0
        except:
            max_score=0
        import subprocess as sp
        out2=sp.run(f'python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 160 --normal-run 1 --fault-number {f} --fault-run 1 2>&1', shell=True, capture_output=True, text=True).stdout
        m2=re.search(r'Affected subsystem\s+:\s+(\S+)', out2)
        subsys=m2.group(1) if m2 else '-'
        print(f"Fault {f:2d}: events={ev} max_score={max_score:.2f} subsystem={subsys} {'DETECTED' if ev>0 else 'MISSED'}")

set_threshold(0.75)
print("\nRestored threshold 0.75")
