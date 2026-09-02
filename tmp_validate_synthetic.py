import json
examples = [json.loads(l) for l in open('outputs/llm_dataset_v2/synthetic_rca.jsonl')]
print(f"Total: {len(examples)}")
# Check leakage
leak = sum(1 for ex in examples if ex['target']['fault_name'] in json.dumps(ex['evidence']))
print(f"Leakage fault_name in evidence: {leak} (should be 0- but candidate may match target subsystem, not name)")
# Check candidate vs ground truth
matches = sum(1 for ex in examples if ex['evidence']['candidate_subsystem'] == ex['target']['subsystem'])
print(f"Candidate matches ground truth: {matches}/{len(examples)} ({matches/len(examples):.1%}) - should be ~60-90%, not 100%")
for ex in examples[:1]:
    print("Sample top sensor:", ex['evidence']['top_anomalous_sensors'][0])
    print("Candidate:", ex['evidence']['candidate_subsystem'], "Target:", ex['target']['subsystem'])
max_pct = max(abs(s['deviation_percent']) for ex in examples for s in ex['evidence']['top_anomalous_sensors'])
print(f"Max deviation %: {max_pct} (bounded to 100, good)")
max_z = max(abs(s['z_score']) for ex in examples for s in ex['evidence']['top_anomalous_sensors'])
print(f"Max z: {max_z} (should be <5-10 for most)")
# Check per fault
from collections import Counter
print(Counter(ex['fault_id'] for ex in examples))
