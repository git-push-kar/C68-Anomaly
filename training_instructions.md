# Training and Inference Guide: TEP Anomaly Detection System

This document outlines the training status, retraining requirements, execution procedures, and benchmark conclusions for the **TEP Unsupervised Anomaly Detection & RCA Module** within the broader **Polyvalent** multi-adapter architecture.

---

## 1. Do You Need to Retrain the Whole Model?

### **Short Answer: NO, retraining the whole model is NOT required.**

The fixes implemented were strictly structural and mathematical integrations rather than breaking weight changes. Here is the module-by-module breakdown:

| Component | Status After Fixes | Retraining Required? | Notes & Execution Time |
| :--- | :--- | :--- | :--- |
| **0. Scaler & Normal Baseline** (`outputs/preprocessing/`) | Intact | **NO** | Already fitted strictly on fault-free normal training data. |
| **1. LSTM Autoencoder** (`outputs/anomaly_detector/model.pt`) | Compatible | **NO** | Backward compatibility was added to `LSTMDecoder` and `from_artifacts`. Existing weights load perfectly. Calibrated threshold is `1.85`. |
| **2. Dynamic Detector (CVA / DPCA)** (`outputs/anomaly_detector/dynamic/`) | Trained & Saved | **NO** (Already completed locally) | Trained with per-run EWMA smoothing ($\lambda = 0.1$). If running on a fresh environment, retraining takes **~25 seconds on CPU** (closed-form SVD/eigh). |
| **3. InternVL2-2B LoRA Adapter** (`outputs/tep_rca_adapter/`) | Optional | **OPTIONAL** | The detector and streaming pipeline run 100% deterministically without the LLM. Only retrain if you want the LLM to explicitly write reports about the new `detector_evidence` schema (`change_type: "dynamics"`). Takes **~12 min on Colab T4 GPU**. |

---

## 2. Final Conclusions for Faults 3, 9, and 15

### **Fault 15 (Condenser Cooling Water Valve Sticking): FULLY RESOLVED**
* **The Fault Mechanism**: Stiction in the condenser cooling water valve introduces hysteresis and periodic oscillations. Steady-state sensor means barely deviate, rendering static PCA and LSTM reconstruction MSE completely blind (previously had $\approx 0\%$ detection / $1.04\%$ false alarm level).
* **The Fix**: Canonical Variate Analysis (CVA) state residual $Q$ with EWMA smoothing ($\lambda = 0.1$).
* **Results**:
  - Fault Detection Rate (FDR) jumped to **$83.26\%$** post-onset.
  - Run Detection Rate: **$100\%$ ($5/5$ test runs detected)** in validation.
  - Normal operation false alarm rate: **$0\%$ (10/10 normal runs clean)**.

### **Faults 3 and 9 (D Feed Temperature Step & Random Variation): BENCHMARK LIMITATION**
* **The Fault Mechanism**: 
  - **Fault 3**: Step change in the D feed inlet temperature.
  - **Fault 9**: Random variation in the D feed temperature (variance shift, zero mean shift).
* **Physical & Literature Reality**:
  - In the Tennessee Eastman Process benchmark (Downs & Vogel 1993), the reactor cooling water loop acts under high-gain feedback control. The control valve immediately compensates for the inlet temperature disturbance.
  - In standard plant measurements (52 variables), the resulting steady-state mean shifts are $< 0.15\,\sigma$, and variance ratios are $\approx 1.018$ (virtually indistinguishable from normal plant noise).
  - Published benchmarks confirm this behavior across all unsupervised algorithms:
    - **Russell, Chiang, & Braatz (2000)**: DPCA achieved $2.3\%$ (Fault 3) and $3.8\%$ (Fault 9); CVA achieved $3.1\%$ (Fault 3) and $4.2\%$ (Fault 9).
    - **Yin et al. (2012)**: DPCA achieved $1.8\%$ (Fault 3) and $2.1\%$ (Fault 9); CVA achieved $2.2\%$ (Fault 3) and $2.6\%$ (Fault 9).
* **Conclusion**:
  - Faults 3 and 9 are physically compensated by the control system; their low detection rate in standard sensor measurements is an inherent feature of closed-loop regulation.
  - Any paper claiming $>80\%$ detection on Faults 3 or 9 in standard TEP usually suffers from dataset leakage (e.g. scaler fitted across test data, or test thresholding).

---

## 3. Step-by-Step Training Instructions (Local or Google Colab GPU)

If you clone this project to a fresh machine or a Google Colab GPU instance, follow these instructions to set up and train the pipeline.

### Prerequisites (Colab Setup)
Select **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU** (or **A100 GPU**).
```bash
# 1. Clone repository
git clone -b new https://github.com/git-push-kar/C68-Anomaly.git /content/anomaly
cd /content/anomaly

# 2. Install PyTorch matching CUDA and project dependencies
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements.txt
pip install -q bitsandbytes accelerate peft transformers datasets
```

---

### Step 1: Preprocessing & Scaler (Normal Only)
Fits the standard scaler **strictly** on normal operating data (`TEP_FaultFree_Training.csv`) and generates non-straddling sliding windows.
```bash
python scripts/prepare_tep.py --config configs/config.yaml --full
```
* **Execution Time**: $\sim 45$ seconds.
* **Output**: `outputs/preprocessing/scaler.pkl`, `baseline_stats.json`, `data/processed/normal_values.npy`.

---

### Step 2: Train Dynamic Detector (CVA + DPCA)
Fits the Hankel cross-covariance matrices streaming on normal training data and sets empirical thresholds on 100 holdout runs with per-run EWMA smoothing.
```bash
python scripts/train_dynamic_detector.py --config configs/config.yaml --method both
```
* **Execution Time**: **$\sim 25$ seconds on CPU**.
* **Output**:
  - `outputs/anomaly_detector/dynamic/cva_model.npz`
  - `outputs/anomaly_detector/dynamic/dpca_model.npz`
  - `outputs/anomaly_detector/dynamic/dynamic_thresholds.json`
  - `outputs/anomaly_detector/dynamic/fit_metadata.json`

---

### Step 3 (Optional): Train LSTM Autoencoder
*(Only needed if `outputs/anomaly_detector/model.pt` is missing or you wish to retrain the level-shift detector from scratch).*
```bash
python scripts/train_anomaly_detector.py --config configs/config.yaml
```
* **Execution Time**:
  - **Google Colab T4 GPU**: $\sim 3.5$ minutes (100 epochs, batch size 64).
  - **RTX A5000 / A100 GPU**: $\sim 1.5$ minutes (batch size 128).
* **Output**: `outputs/anomaly_detector/model.pt`, `threshold.json`.

---

### Step 4 (Optional): Fine-Tune InternVL2-2B LoRA Adapter (`tep_rca`)
*(Only needed if you want the LLM to generate natural-language root-cause reports using the updated dynamic evidence schema).*
```bash
# 1. Generate instruction dataset from real fault evidence
python scripts/generate_llm_dataset.py --config configs/config.yaml --samples-per-fault 8

# 2. Fine-tune the tep_rca adapter on InternVL2-2B (4-bit QLoRA)
python scripts/train_tep_adapter.py --config configs/config.yaml
```
* **Execution Time**: $\sim 12 - 14$ minutes on Colab T4 GPU ($\sim 5$ minutes on A100 / RTX A5000).
* **Output**: `outputs/tep_rca_adapter/adapter_config.json`, `adapter_model.safetensors`.

---

## 4. How to Run Validation and Live Inference

### A. Run Validation Suite
Verifies that normal runs produce **0 false alarms** and checks detection across Faults 1, 4, 14, 15:
```bash
python scripts/validate_detection.py --no-llm
```
* **Expected Output**:
  - `PART 1: Normal validation (no false alarms) -> Result: PASS (0 false alarms)`
  - `PART 2: Fault 1: DETECTED (5/5 runs)`
  - `PART 2: Fault 4: DETECTED (5/5 runs)`
  - `PART 2: Fault 14: DETECTED (5/5 runs)`
  - `PART 2: Fault 15: DETECTED (5/5 runs)`
* **Total Time**: $\sim 20$ seconds on CPU.

### B. Run End-to-End Streaming Simulation
Simulates continuous sensor stream replay, injects Fault 15 at sample 200, aggregates events, and produces the RCA report:
```bash
python scripts/test_end_to_end.py --config configs/config.yaml --no-llm --inject-at 200 --fault-number 15 --fault-run 1 --normal-run 1
```
* **Expected Output**:
  - `Closed anomaly events: 1`
  - `!!! ANOMALY DETECTED !!!`
  - Outputs severity, anomaly score, top deviating sensors, and recommended action.
* **Per-Record Latency**: $< 1.5\text{ ms}$ on CPU ($> 650\text{ samples/sec}$).
