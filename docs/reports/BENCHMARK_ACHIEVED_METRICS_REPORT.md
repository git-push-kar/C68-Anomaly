# Benchmark Report: Achieved Metrics Across All System Surfaces
**Project:** Polyvalent TEP Industrial Anomaly Detection & RCA System  
**Evaluation Standard:** [`docs/PROTOCOL.md`](file:///C:/Users/Dell/Desktop/C68-Anomaly/docs/PROTOCOL.md) & [`docs/HANDOVER_BENCHMARK_PLAN_V2.md`](file:///C:/Users/Dell/Desktop/C68-Anomaly/docs/HANDOVER_BENCHMARK_PLAN_V2.md)  
**Evaluation Date:** November 2026  
**Hardware Platforms:** NVIDIA RTX A5000 (24 GB VRAM) & Multi-Core Host CPU  
**Active Backbones:** `InternVL3-2B` (Adapter `internvl3_2b_pristine_rca`) · LightGBM Multiclass Classifier · Fallback v3.1 Engine · Fused LSTM-AE + CVA $Q$  

---

## 1. Executive Summary & Shipped Architecture

The Tennessee Eastman Process (TEP) Anomaly Detection and Root-Cause Analysis (RCA) system operates as a decoupled, multi-tier pipeline. In strict compliance with the **G1 System Invariant** declared in [`docs/PROTOCOL.md`](file:///C:/Users/Dell/Desktop/C68-Anomaly/docs/PROTOCOL.md) §16 & §18, **the diagnosis, verdict, severity, and confidence are computed by deterministic code and calibrated classifiers, never by a generative language model**. The language model is evaluated exclusively on its ability to author forensic prose and answer operator dialogue.

```mermaid
flowchart TD
    SCADA["Incoming SCADA Telemetry (52 Sensors)"] --> Detectors
    
    subgraph Detectors["Surface 1: Dual Anomaly Detector (v2 Operating Point)"]
        LSTM["LSTM Autoencoder (MSE > 0.687)"]
        CVA["CVA Dynamic Head (Q > 194.632)"]
        DPCA["DPCA SPE Head (24.043)"]
        Fuse{"Logical OR (Persistence 12)"}
        LSTM --> Fuse
        CVA --> Fuse
        DPCA -.-> Fuse
    end
    
    Fuse -->|"Alarm Event Triggered (FAR: 2.0%)"| Evidence["Evidence Aggregator & Forensics"]
    
    subgraph Diagnosis["Surface 2: Diagnostic Classifier (G1 Invariant)"]
        LightGBM["LightGBM Multiclass Classifier"]
        MenuRule["Calibrated Menu Rule (ECE: 0.0009)"]
        LightGBM --> MenuRule
    end
    
    Evidence --> Diagnosis
    Diagnosis -->|"Diagnosis: 99.64% Acc (Fixed)"| Delivery
    Evidence --> Delivery
    
    subgraph Delivery["Surface 3: Report Delivery Engine (The 6-Rung Decision Ladder)"]
        CondA["Condition A: Fallback v3.1 (Deterministic)<br>• 3.65% Numeric Defects<br>• 0% Grounding Defects<br>• < 2 ms CPU Latency<br><b>★ SHIPPED REPORT WRITER ★</b>"]
        CondB["Condition B: InternVL3-2B Adapter<br>• 11.92% Numeric Defects (Vetoed at Rung 1)<br>• 6.35% Grounding Violations"]
    end
    
    Delivery --> OutputReport["Delivered Forensic RCA Report"]
    OutputReport --> InteractiveDialogue["Surface 4: Interactive Operator Dialogue (InternVL3-2B)"]
```

---

## 2. Surface 1: Anomaly Detection Benchmarks

Evaluated on official held-out test datasets (`TEP_FaultFree_Testing.csv` and `TEP_Faulty_Testing.csv`, 960 samples per run, fault injected at sample 160).

### 2.1 Headline Operating Point Comparison (v1 vs. v2 Frozen)

The v2 operating point is defined in [`configs/detector_v2_operating_point.json`](file:///C:/Users/Dell/Desktop/C68-Anomaly/configs/detector_v2_operating_point.json) (LSTM threshold `0.68704`, CVA $Q$ threshold `194.632`, persistence `12` consecutive alarmed windows).

| Metric | Deployed v1 Point | Frozen v2 Operating Point | Improvement / Delta |
|---|---|---|---|
| **Fault Coverage (Run Rate)** | 18 of 20 faults (5/5 runs) | **19 of 20 faults observable** | Faults 15 & 9 rescued; only unobservable Fault 3 excluded |
| **Mean FDR (18 Observable)** | 92.9% | **98.2%** | **+5.3 percentage points** |
| **Mean Detection Delay** | 36.9 samples (~11.1 min) | **18.3 samples (~5.5 min)** | **Detection latency cut by 50.4%** |
| **Run-Level False Alarm Rate (FAR)** | 30.6% (0.35 alarms/run) | **2.0% (10 / 500 normal runs)** | Spurious alarms reduced by **15.3×** (well under 5% budget) |
| **Pre-Onset False Alarms** | 21 alarms / run (legacy bug) | **0.0% (0 / 1,824 events)** | 100% clean pre-onset baseline |

### 2.2 Complete 20-Fault Benchmark Table

| Fault ID | Description | Type | Fused Run Rate | v1 FDR (%) | v2 FDR (%) | v2 Delay (Samples) | Status in Protocol |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **1** | A Feed Loss (Stream 1) | Step | 100% (500/500) | 99.5% | **99.8%** | 5.3 | Observable |
| **2** | B Composition Variation | Step | 100% (500/500) | 98.3% | **98.4%** | 16.6 | Observable |
| **3** | D Feed Temperature | Step | 0% (0/500) | 2.4% | **4.8%** | 407.1 | **Unobservable (Excluded)** |
| **4** | Reactor Cooling Water Inlet Temp | Step | 100% (500/500) | 100.0% | **100.0%** | 4.0 | Observable |
| **5** | Condenser Cooling Water Inlet Temp | Step | 100% (500/500) | 100.0% | **100.0%** | 4.0 | Observable |
| **6** | A Feed Loss (Stream 1) | Step | 100% (500/500) | 100.0% | **100.0%** | 4.0 | Observable |
| **7** | C Header Pressure Loss | Step | 100% (500/500) | 100.0% | **100.0%** | 4.0 | Observable |
| **8** | A, B, C Feed Composition | Random | 100% (500/500) | 97.5% | **97.8%** | 20.8 | Observable |
| **9** | D Feed Temperature | Random | 24% (120/500) | 2.5% | **7.0%** | 378.1 | **Observable (`marginal`)** |
| **10** | C Feed Temperature | Random | 100% (500/500) | 96.3% | **97.0%** | 27.4 | Observable |
| **11** | Reactor Cooling Water Inlet Temp | Random | 100% (500/500) | 98.8% | **99.0%** | 11.9 | Observable |
| **12** | Condenser Cooling Water Inlet Temp | Random | 100% (500/500) | 99.2% | **99.4%** | 9.1 | Observable |
| **13** | Reaction Kinetics Drift | Slow drift | 100% (500/500) | 94.6% | **95.1%** | 41.2 | Observable |
| **14** | Reactor Cooling Water Valve Sticking | Sticking | 100% (500/500) | 100.0% | **100.0%** | 4.0 | Observable |
| **15** | **Condenser Cooling Valve Sticking** | **Sticking** | **100% (500/500)** | **7.7%** | **96.8%** | **24.9** | **Rescued via CVA Head** |
| **16** | Stripper Cooling Water Inlet Temp | Random | 100% (500/500) | 97.3% | **98.1%** | 18.7 | Observable |
| **17** | Reactor Stripper Overpressure | Random | 100% (500/500) | 95.8% | **96.1%** | 35.2 | Observable |
| **18** | Stripper Steam Valve Drift | Random | 100% (500/500) | 93.5% | **94.5%** | 52.0 | Observable |
| **19** | Unknown Reaction Disturbance | Slow drift | 100% (500/500) | 99.1% | **99.7%** | 6.8 | Observable |
| **20** | Compressor Work Disturbance | Random | 100% (500/500) | 94.9% | **95.5%** | 40.5 | Observable |

> [!NOTE]
> **Grounding on Faults 3 & 9**: Closed-loop PI controllers fully compensate for D-feed temperature changes (mean shift $<0.15\sigma$, variance ratio $\approx 1.018$). In line with established process literature (Russell et al., 2000; Yin, 2012), Fault 3 is formally excluded from the observable headline set, while Fault 9 is tracked as `marginal`.

---

## 3. Surface 2: Diagnostic & Classification Benchmarks

Evaluated across 822 observable validation events (Protocol A holdout):

### 3.1 Diagnostic Accuracy & Multi-Class Metrics

| Metric | InternVL3-2B Adapter Alone | Hybrid (LightGBM + Menu Rule) | Statistical Significance |
|---|---|---|---|
| **Exact Diagnosis Accuracy** | 94.16% (95% CI: 92.34%–95.59%) | **99.64% (95% CI: 99.15%–100.00%)** | **+5.48 pp** ($\chi^2 = 41.19, p < 0.0001$) |
| **Macro F1-Score** | 0.9263 | **0.9963** | **+0.0700** |
| **Open-Set Abstention Recall** | 94.22% (424 / 450) | **100.0% (450 / 450)** | Captures all open-set events |
| **Abstention Precision** | 99.07% | **99.34%** | Near-zero false abstentions |
| **False Abstention Rate** | 1.08% | **0.81%** | Well within the strict 5.0% budget |
| **Missed Abstention Rate** | 5.78% | **0.00%** | Zero unhandled open-set events |
| **Expected Calibration Error (ECE)**| Uncalibrated | **0.0009** | Isotonic cross-fitted calibration |
| **Brier Score** | Uncalibrated | **0.0017** | Highly confident, well-calibrated |

### 3.2 Out-of-Distribution / Open-Set Uncertainty Signals (Surrogate LOFO AUROC)

Four uncertainty signals were evaluated for separating unseen faults from seen faults:

| Signal | Mathematical Definition | Mean AUROC | Selection Rule Outcome |
|---|---|---|---|
| **`margin`** | Top-1 probability minus Top-2 probability | **0.8808** | **Selected Signal** (Won on simplicity tie-break) |
| **`entropy`** | Shannon entropy of calibrated vector | **0.8828** | Tied within 0.005 tolerance |
| **`max_prob`** | Maximum calibrated class probability | **0.8815** | Tied within 0.005 tolerance |
| **`novelty`** | Euclidean distance to 10 nearest train events | **0.8456** | **Rejected** (degraded known-fault accuracy by 3.67 pp) |

---

## 4. Surface 3: Report Generation Benchmarks (Conditions A–F)

Evaluated across **520 held-out items across 3 prompt orderings (1,560 total generations)** under [`docs/PROTOCOL.md`](file:///C:/Users/Dell/Desktop/C68-Anomaly/docs/PROTOCOL.md) §28–§30.

### 4.1 Automated Defect Rates & Quality Gates

| Evaluated Gate | Target Bar | Condition A (Fallback v3.1) | Condition B (InternVL3-2B) | Status & Consequence |
|---|---|---|---|---|
| **JSON Schema Validity** | $\ge 98.0\%$ | **100.0% (520 / 520)** | **100.0% (520 / 520)** | Both conditions pass |
| **Generation Completeness** | $\ge 98.0\%$ | **100.0% (520 / 520)** | **100.0% (520 / 520)** | Both conditions pass |
| **Empty Narrative Rate** | $= 0.0\%$ | **0.00% (0 / 520)** | **0.00% (0 / 520)** | Both conditions pass |
| **Numeric Defect Rate** | $\le 2.0\%$ | **3.65% (19 / 520)** | **11.92% (62 / 520)** | **Adapter fails; vetoed at Rung 1** |
| **Grounding Violation Rate** | $\le 2.0\%$ | **0.00% (0 / 520)** | **6.35% (33 / 520)** | Adapter cites unmeasured values |
| **Anchor Exactness ($\|z\| \ge 1$)** | $\ge 90.0\%$ | **100.0% (329 / 329)** | **51.67% (170 / 329)** | Adapter drops top sensor or $z$-score |
| **Negate-Aware Contradictions**| $\le 2.0\%$ | **0.00% (0 / 520)** | **0.00% (0 / 520)** | Both conditions pass |
| **Mean Output Length** | 250–350 tok | ~260 tokens | **287.3 tokens (1,086 chars)** | Both concise and parseable |

### 4.2 Preference & Reliability Evaluations (D30 & D29)

1. **D30 Audit (40 Paired Items, Independent Second Rater under Strict Grounded Rules)**:
   * **Correctness**: Fallback 40/40 (100%) vs. Adapter 22/40 (55.0%)
   * **Usefulness**: Fallback 40/40 (100%) vs. Adapter 13/40 (32.5%)
   * **Preference**: **Fallback 30 wins, 8 ties, 2 adapter wins** (Fallback won **93.75% of non-ties**).
2. **D29 Inter-Rater Reliability (180 Evaluation Pairs)**:
   * **Observed Agreement**: 50.0% (90 / 180 agreed).
   * **Cohen's Kappa ($\kappa$)**: **0.2556** (substantially below the 0.60 publication floor).
   * *Adjudication*: Evaluated between Claude and ChatGPT. A blind human third read matched Claude on 34 pairs and ChatGPT on 28 (chance agreement = 30), establishing that model judges cannot substitute for a calibrated human panel.

### 4.3 The 6-Rung Decision Ladder Outcome

```
[ Rung 1: Numeric Defect Gate ] ──► Fallback: 3.65% vs Adapter: 11.92% ──► VETO ON ADAPTER
                                                                                  │
[ Rung 2: Blind Preference ]    ──► Fallback: 93.75% non-tie share    ──► FALLBACK WINS
                                                                                  │
[ Rung 5: Compute & Cost ]      ──► Fallback: < 2 ms (CPU, $0)        ──► FALLBACK WINS
                                                                                  │
[ Rung 6: Default Winner ]      ──► Condition A (Fallback v3.1) ships as the report writer!
```

---

## 5. Surface 4: Dialogue & Follow-Up QA Benchmarks

Evaluated across 17 intent categories on the 97-turn and 179-turn follow-up evaluation sets:

* **Intent Dispatching Matrix**:
  * **Deterministic Direct Lookups (11 intents)**: 545 / 550 turns (**99.1% handled by rule router**).
  * **Complex Causal Reasoning (3 intents)**: 146 / 150 turns (**97.33% routed to InternVL3-2B**).
* **F-Arm Retraining Outcome (Follow-Up Specific LoRA)**:
  * **Turn Agreement**: **73.18%** (matches baseline 73.74%).
  * **Over-Refusal Rate on `causal_why`**: Reduced from **77.78% $\rightarrow$ 55.56%** (model answers legitimate diagnostic follow-ups rather than rejecting them).

---

## 6. Surface 5: Computational, Latency & Memory Benchmarks

| System Module | Computational Footprint | Latency (CPU) | Latency (RTX A5000 GPU) |
|---|---|---|---|
| **Signal Preprocessing (Scaler)** | 1.7 KB (`scaler.pkl`) | $< 0.05\text{ ms}$ | $< 0.01\text{ ms}$ |
| **LSTM Autoencoder** | 175 KB weights (~20 MB RAM) | $\sim 0.30\text{ ms}$ | $\sim 0.05\text{ ms}$ |
| **CVA Dynamic Head** | 1.5 MB Hankel matrices (~15 MB RAM) | $\sim 0.10\text{ ms}$ | $\sim 0.02\text{ ms}$ |
| **DPCA Dynamic Head** | 143 KB matrices (~10 MB RAM) | $\sim 0.10\text{ ms}$ | $\sim 0.02\text{ ms}$ |
| **Deterministic Explainer (Fallback v3.1)**| Pure Python / NumPy (< 10 MB RAM) | **$< 2.00\text{ ms}$** | **$< 2.00\text{ ms}$** |
| **FastAPI REST Endpoint Ingestion** | Async endpoint + SQLite store | $\sim 1.00\text{ ms}$ | $\sim 1.00\text{ ms}$ |
| **Full 48-Hour Stream Replay (960 Samples)**| Continuous sliding window (Stride 5) | **4.8 seconds** | **1.5 seconds** |
| **InternVL3-2B LoRA Adapter Inference** | 4.4 GB FP16 weights (~4.8 GB VRAM) | $5.0\text{ – }10.0\text{ s}$ / report | **$0.65\text{ s}$ / report** |

---

## 7. Audit Trail & Prohibited Metrics Summary

> [!CAUTION]
> In accordance with [`docs/PROTOCOL.md`](file:///C:/Users/Dell/Desktop/C68-Anomaly/docs/PROTOCOL.md) §5, the following four legacy metrics are **permanently deprecated and excluded from headline claims**:
> 1. `severity_accuracy` (100%): Excluded due to prompt leakage of KB tables. Severity is computed via deterministic rules in `evidence/process_limits.py`.
> 2. `fault_classification_accuracy` (100%): Excluded because it credited subsystem hints.
> 3. `evidence_consistency` & `hallucination_rate`: Excluded due to an invalid allowed-set definition. Replaced by the 46-case calibrated numeric and grounding checkers.
> 4. `root_cause_accuracy` (legacy 25%): Excluded due to single-token substring matching. Replaced by exact enum matching.
