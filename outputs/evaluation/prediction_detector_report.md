# Prediction vs Reconstruction - Faults 3,9,15

Recon threshold: 0.6874 (p99 val), Pred p99: 0.9557

| condition | recon_mean_max | recon_p95 | recon_event | pred_mean_max | pred_p95 | pred_event |
|-----------|--------------|-----------|-------------|-------------|----------|------------|
| normal | 0.683 | 0.680 | 0.093 | 1.003 | 0.994 | 0.000 |
| fault3 | 0.678 | 0.677 | 0.118 | 0.979 | 0.960 | 0.000 |
| fault9 | 0.679 | 0.678 | 0.126 | 0.982 | 0.964 | 0.000 |
| fault15 | 0.684 | 0.683 | 0.196 | 0.983 | 0.966 | 0.000 |

Questions:
- Fault 3: recon not, pred not
- Fault 9: recon not, pred not
- Fault 15: recon not, pred not
