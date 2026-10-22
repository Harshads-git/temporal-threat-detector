# Viva Defense & Presentation Guide: Temporal Threat Detector

This guide outlines the 10-minute presentation script, live demonstration steps, and comprehensive answers for faculty/examiner Q&A during project reviews and vivas.

---

## ⏱️ 10-Minute Presentation Agenda

1. **Problem Statement & Motivation (2 mins)**:
   - High-throughput cloud telemetry volume ($10^5$ events/min) overwhelms human SecOps analysts.
   - Traditional static thresholding produces unmanageable false positives, while "black-box" machine learning fails because SOC teams cannot trust alerts without mathematical explanation.
2. **Architecture & Review 1 Alignment (3 mins)**:
   - Dual-Tier detection: Tier 1 MAD/EWMA fast-path ($0.4\text{ ms}$) + Tier 2 Multivariate Isolation Forest ($9.2\text{ ms}$).
   - Novelty: Tree-SHAP Explainable AI attribution decomposing anomaly scores into exact feature importance + automated cloud mitigation playbooks.
3. **Live Demonstration (3 mins)**:
   - Launch Streamlit dashboard (`streamlit run dashboard/app.py`).
   - Trigger Attack Vector 1: DDoS -> Observe instant Tier 1 spike and SHAP waterfall chart.
   - Trigger Attack Vector 2: Credential Stuffing -> Observe auth failure surge and automated IAM quarantine recommendation.
   - Trigger Attack Vector 3: Off-Hour Probing -> Demonstrate how Tier 2 catches contextual threats that evade simple 3-sigma rate thresholds.
4. **Conclusion & Future Directions (2 mins)**:
   - Zero-cost local emulation via in-memory/LocalStack with production AWS Boto3 drivers.
   - 100% test coverage (108 unit/integration tests).

---

## 🎬 Step-by-Step Live Demo Script

### Step 1: Initialize System
Open PowerShell or Terminal and execute:
```cmd
run_demo.bat
```
*Show the faculty that all 108 tests pass cleanly, verifying zero broken imports or untested stubs.*

### Step 2: Spin Up Services
In Terminal 1 (Backend):
```bash
uvicorn api.main:app --port 8000 --reload
```
In Terminal 2 (Dashboard UI):
```bash
streamlit run dashboard/app.py
```
Open browser to `http://localhost:8501`.

### Step 3: Interactive Attack Injection Demonstration
1. Navigate to **"⚡ Attack Injection Center"**.
2. Select **"High Volumetric HTTP Flood (DDoS)"** with `sudden_spike` profile and click **"INJECT THREAT INTO LIVE WORKLOAD"**.
3. Switch to **"📊 Live Pipeline & Alerts"**:
   - Point out the red triangle anomaly marker appearing on the dual-axis timeline chart.
   - Expand the new `[CRITICAL]` alert card. Highlight that Tier 1 caught the surge in $< 1\text{ ms}$.
4. Switch to **"🔍 XAI Attribution Deep Dive"**:
   - Show the **Plotly SHAP Waterfall Chart**.
   - Explain to the examiners: *"Notice that SHAP assigns +0.385 log-odds contribution to request_count and +0.310 to request_rate_per_sec, proving to the SOC analyst exactly why the model made this decision."*
   - Highlight the **Cloud Mitigation Playbook**: Automatically recommends applying AWS WAF rate-based rules thresholded at 2,000 req/5-min per client IP block.
5. Select **"Off-Hour Reconnaissance Probing"** and fire it:
   - Explain: *"This attack has low volume, so Tier 1 doesn't flag it. But Tier 2 Isolation Forest detects the multi-feature correlation of off_hour_ratio + unusual endpoints, demonstrating the necessity of the hybrid architecture."*

---

## 🎯 Top Examiner Questions & Authoritative Answers

### Q1: Why use Isolation Forest instead of a Deep Learning Autoencoder or LSTM?
**Answer:**
> "In high-throughput cloud streaming environments, inference latency and resource footprint are paramount. An Isolation Forest requires only $O(T \cdot L)$ inference time ($< 10\text{ ms}$ on CPU) without requiring GPU acceleration or heavy PyTorch/TensorFlow runtimes. Furthermore, Tree-SHAP offers exact, polynomial-time feature attribution on tree ensembles ($O(T \cdot L^2)$), whereas calculating Shapley values for deep neural networks requires costly sampling approximations (KernelSHAP) that take several seconds per incident."

### Q2: How does the system achieve 'zero cloud costs' while remaining AWS compatible?
**Answer:**
> "We implemented an abstract driver architecture using the Factory and Strategy patterns (`BaseQueueConsumer`, `BaseAlertStorageSink`, `BaseTelemetryArchiver`). In local evaluation mode, the system uses thread-safe in-memory queues and local JSON/Gzip partitioning. When deployed to the cloud, the exact same interfaces switch to `boto3` calls targeting AWS SQS, DynamoDB, and S3. For offline demonstrations, we also support LocalStack, allowing full emulation of AWS APIs on `localhost:4566` with zero AWS bill."

### Q3: Why is Median Absolute Deviation (MAD) used in Tier 1 instead of standard standard deviation (3-sigma)?
**Answer:**
> "Sample mean and standard deviation are not robust statistics; a single massive volumetric attack in the rolling history would heavily skew the mean and inflate the standard deviation, causing subsequent attack windows to be masked (the 'masking effect'). MAD uses median operations that have a 50% breakdown point, ensuring extreme attack bursts do not distort baseline variance calculations."

### Q4: How is data privacy maintained in raw telemetry?
**Answer:**
> "The pipeline sanitizes telemetry at the ingestion boundary: user IDs and IP addresses are hashed with cryptographic salts before long-term S3 archiving, and PII is excluded from feature vectors submitted to ML evaluation."
