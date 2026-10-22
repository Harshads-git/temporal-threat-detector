# ==============================================================================
# Makefile: Temporal Anomaly & Threat Detector for Cloud Workloads
# ==============================================================================

PYTHON ?= python

.PHONY: help install test api dashboard docker-up docker-down benchmark clean

help:
	@echo "Available targets for Temporal Threat Detector:"
	@echo "  make install             - Install all production & test dependencies"
	@echo "  make test                - Run complete pytest test suite (112 tests)"
	@echo "  make api                 - Launch FastAPI detection backend on :8000"
	@echo "  make dashboard           - Launch Streamlit live visualizer on :8501"
	@echo "  make docker-up           - Build and start full stack in Docker Compose"
	@echo "  make docker-down         - Stop and remove Docker Compose containers"
	@echo "  make benchmark-latency   - Profile pipeline component latencies"
	@echo "  make benchmark-accuracy  - Evaluate multi-vector threat detection accuracy"
	@echo "  make clean               - Clean temporary build, cache, and log files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down -v

benchmark-latency:
	$(PYTHON) -m benchmarks.benchmark_pipeline --batch-size 500 --iterations 10

benchmark-accuracy:
	$(PYTHON) -m benchmarks.benchmark_accuracy --events 400

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
