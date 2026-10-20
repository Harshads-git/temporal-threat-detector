# Docker Orchestration: Temporal Anomaly & Threat Detector

This directory contains the multi-stage `Dockerfile` and `docker-compose.yml` for orchestrating the complete cloud telemetry threat detection pipeline locally.

---

## Architecture Overview

```
                         ┌──────────────────────────────────────────────┐
                         │              Docker Network                  │
                         │       (temporal_threat_detector_net)         │
                         └───────┬──────────────────────────────┬───────┘
                                 │                              │
                                 ▼                              ▼
                 ┌───────────────────────────────┐  ┌───────────────────────────────┐
                 │       FastAPI Backend         │  │     Streamlit Dashboard       │
                 │         (Port 8000)           │◄─┤         (Port 8501)           │
                 │  - Dual-Tier Threat Ensemble  │  │  - Real-Time Timeline Chart   │
                 │  - SHAP TreeExplainer (XAI)   │  │  - Interactive Attack Buttons │
                 │  - Telemetry Ingestion Worker │  │  - SHAP Waterfall Viewer      │
                 └───────────────┬───────────────┘  └───────────────────────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │          LocalStack           │
                 │         (Port 4566)           │
                 │  - SQS + DLQ Telemetry Queue  │
                 │  - DynamoDB ThreatAlerts      │
                 │  - S3 Telemetry Archive       │
                 └───────────────────────────────┘
```

---

## 1-Line Execution Commands

### 1. Build and Spin Up Complete Stack
```bash
docker compose -f docker/docker-compose.yml up --build -d
```

### 2. View Aggregated Real-Time Logs
```bash
docker compose -f docker/docker-compose.yml logs -f
```

### 3. Service Access Endpoints
- **Streamlit Interactive UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Health Probe**: [http://localhost:8000/health](http://localhost:8000/health)
- **LocalStack Health**: [http://localhost:4566/_localstack/health](http://localhost:4566/_localstack/health)

### 4. Teardown Stack
```bash
docker compose -f docker/docker-compose.yml down -v
```
