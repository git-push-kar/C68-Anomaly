## Rendered report

## ANOM-0488 — Condenser cooling water valve sticking
**Severity:** Low (authoritative, computed in code)  
**Subsystem:** condenser_cooling_system  
**Confidence:** 0.71

A sustained deviation was detected in the cooling-water loop. The pattern is consistent with a sticking valve actuator rather than a sensor failure.

### Evidence
- CVA Q-statistic stayed above its threshold for the whole event window
- The condenser cooling-water flow sensor is the largest single contributor

### Reasoning
Reduced cooling-water flow raises separator pressure and temperature together, which is a flow-side disturbance rather than a measurement fault.

| Detector | Score | Threshold | Margin | Status |
|---|---|---|---|---|
| LSTM reconstruction error | 0.6352 | 0.69 | 0.92x | below threshold |
| CVA Q-statistic | 248.2991 | 239.75 | 1.04x | ALARMED |
| Overall event score | 0.7577 | — | — | ALARMED |

### Timeline
| # | Sensor | Event | Minutes |
|---|---|---|---|
| 1 | A_Feed_Stream1 | became_anomalous | 0.0 min |
| 2 | D_Feed_Stream2 | became_anomalous | 0.0 min |
| 3 | E_Feed_Stream3 | became_anomalous | 0.0 min |
| 4 | A_C_Feed_Stream4 | became_anomalous | 0.0 min |

### Recommended action
Inspect the condenser cooling-water valve actuator and verify the positioner response.

### Uncertainty
A fouled condenser would produce a similar signature; on-site verification is required.

_Severity basis:_ Low severity from the authoritative production rule; detector margin 1.04x, 2 window(s) at 97% of peak, max |z| 0.3; KB base for FAULT_15 is critical.

_Citations:_ `detector.dynamic.score`, `detector.dynamic.threshold`, `subsystem.candidate`

## Rendered follow-up

### Severity
Severity is LOW, set by the production severity rule in code; detector margin 1.04x, the event spanned 2 window(s), and the knowledge-base base for FAULT_15 is critical.

| Detector | Score | Threshold | Margin | Status |
|---|---|---|---|---|
| LSTM reconstruction error | 0.6352 | 0.69 | 0.92x | below threshold |
| CVA Q-statistic | 248.2991 | 239.75 | 1.04x | ALARMED |
| Overall event score | 0.7577 | — | — | ALARMED |

---

### Severity Assessment for Incident ANOM-0488
- **Assessed Severity Level:** **LOW**
- **Safety Context:** Minor drift within operational safety margins.
- **Suspected Cause:** FAULT_15 affecting `condenser_cooling_system`.

**You might also ask:**
- What are the detector scores?
- What is the recommended action?

_Citations:_ `detector.lstm.score`, `detector.lstm.threshold`, `detector.dynamic.score`, `detector.dynamic.threshold`

## Validation

- report validator: `{"ok": true, "codes": [], "issues": []}`
- severity authority: `evidence.process_limits.compute_deterministic_severity`
- cross-check level (not displayed): `high` — agreement `False`
