# ==============================================================================
# Multi-Stage Production Dockerfile: Temporal Anomaly & Threat Detector
# Supports dual runtime modes:
# 1. FastAPI REST Ingestion & Detection Service (Port 8000)
# 2. Streamlit Live Explainable AI Dashboard (Port 8501)
# ==============================================================================

# Stage 1: Build Dependencies & Wheels
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Hardened Minimal Production Runtime
FROM python:3.11-slim AS runtime

# Python optimization flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    PORT=8000 \
    AWS_DEFAULT_REGION=us-east-1

WORKDIR /app

# Install runtime utilities (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group for security hardening
RUN groupadd -r -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

# Copy installed Python packages from builder stage
COPY --from=builder /install /install

# Copy project application source modules
COPY api/ /app/api/
COPY cloud/ /app/cloud/
COPY data/ /app/data/
COPY engine/ /app/engine/
COPY dashboard/ /app/dashboard/
COPY tests/ /app/tests/

# Prepare runtime directories and grant ownership to non-root appuser
RUN mkdir -p /app/data/raw /app/data/benchmarks /app/models && \
    chown -R appuser:appgroup /app

USER appuser

# Expose FastAPI backend (8000) and Streamlit dashboard (8501)
EXPOSE 8000 8501

# Container Healthcheck (defaults to probing FastAPI /health endpoint)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default entrypoint runs the FastAPI detection service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
