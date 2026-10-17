"""Comprehensive test suite for FastAPI service endpoints and background worker."""

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.schemas import (
    AttackType,
    CloudTelemetryLog,
    DetectorTier,
    FeatureAttribution,
    SeverityLevel,
    ThreatAlert,
    WindowAggregate,
)
from data.synthetic_generator import SyntheticTelemetryGenerator


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient fixture with full lifespan startup and shutdown."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verify root landing endpoint status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "/health" in data["health_url"]


def test_health_check_endpoint(client: TestClient):
    """Verify health check probe returns detector and queue status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["detector_ready"] is True
    assert "queue_depth" in data
    assert "alert_count" in data


def test_metrics_endpoint(client: TestClient):
    """Verify metrics endpoint returns pipeline counters."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_events_ingested" in data
    assert "total_windows_processed" in data
    assert "total_alerts_emitted" in data
    assert "mean_pipeline_latency_ms" in data


def test_ingest_single_event(client: TestClient):
    """Verify single telemetry event ingestion into queue."""
    generator = SyntheticTelemetryGenerator(seed=42)
    sample_log = generator.generate_batch(count=1)[0]

    response = client.post("/ingest", json=sample_log.model_dump(mode="json"))
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert data["event_id"] == sample_log.event_id
    assert "queue_message_id" in data


def test_ingest_batch_events(client: TestClient):
    """Verify batch telemetry event ingestion."""
    generator = SyntheticTelemetryGenerator(seed=42)
    logs = generator.generate_batch(count=15)
    payload = [log.model_dump(mode="json") for log in logs]

    response = client.post("/ingest/batch", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert data["accepted_count"] == 15
    assert len(data["queue_message_ids"]) == 15


def test_ingest_batch_empty_validation(client: TestClient):
    """Verify ingesting an empty batch returns 400 Bad Request."""
    response = client.post("/ingest/batch", json=[])
    assert response.status_code == 400


def test_on_demand_detect_normal_window(client: TestClient):
    """Verify detect endpoint reports nominal status on clean baseline window."""
    now = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)
    clean_window = WindowAggregate(
        window_start=now,
        window_end=now + timedelta(seconds=60),
        duration_seconds=60.0,
        request_count=20,
        request_rate_per_sec=0.33,
        error_count_4xx=1,
        error_count_5xx=0,
        error_rate_4xx=0.05,
        error_rate_5xx=0.0,
        mean_duration_ms=40.0,
        p95_duration_ms=65.0,
        unique_source_ips=20,
        unique_users=5,
        total_request_bytes=10000,
        total_response_bytes=42000,
        mean_response_bytes=2100.0,
        auth_failure_count=0,
        off_hour_ratio=0.0,
        bot_agent_ratio=0.0,
    )

    response = client.post("/detect", json={"window": clean_window.model_dump(mode="json")})
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is False
    assert data["alert"] is None
    assert "nominal" in data["message"].lower()


def test_on_demand_detect_attack_window(client: TestClient):
    """Verify detect endpoint flags anomaly and generates XAI attribution on attack window."""
    now = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)
    ddos_window = WindowAggregate(
        window_start=now,
        window_end=now + timedelta(seconds=60),
        duration_seconds=60.0,
        request_count=3500,  # Massive spike
        request_rate_per_sec=58.33,
        error_count_4xx=50,
        error_count_5xx=150,
        error_rate_4xx=0.014,
        error_rate_5xx=0.043,
        mean_duration_ms=45.0,
        p95_duration_ms=65.0,
        unique_source_ips=800,
        unique_users=450,
        total_request_bytes=1750000,
        total_response_bytes=700000,
        mean_response_bytes=200.0,
        auth_failure_count=5,
        off_hour_ratio=0.0,
        bot_agent_ratio=0.25,
    )

    response = client.post("/detect", json={"window": ddos_window.model_dump(mode="json")})
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is True
    assert data["alert"] is not None
    assert "DDoS" in data["alert"]["threat_hypothesis"] or "Flood" in data["alert"]["threat_hypothesis"]
    assert len(data["alert"]["top_contributions"]) > 0


def test_on_demand_detect_from_events(client: TestClient):
    """Verify detect endpoint aggregates raw events and evaluates anomaly status."""
    generator = SyntheticTelemetryGenerator(seed=42)
    events = generator.generate_batch(count=30)
    payload = {"events": [e.model_dump(mode="json") for e in events]}

    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "window_id" in data


def test_on_demand_detect_empty_request_validation(client: TestClient):
    """Verify detect endpoint requires either window or events."""
    response = client.post("/detect", json={})
    assert response.status_code == 400


def test_attack_injection_endpoint(client: TestClient):
    """Verify attack injection endpoint creates events and publishes to ingestion queue."""
    injection_config = {
        "attack_type": "DDOS_HTTP_FLOOD",
        "record_count": 40,
        "intensity_multiplier": 3.0,
        "burst_mode": "SUDDEN_SPIKE",
    }
    response = client.post("/inject", json=injection_config)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "injected"
    assert data["attack_type"] == "DDOS_HTTP_FLOOD"
    assert data["injected_count"] == 40
    assert len(data["queue_message_ids"]) == 40


def test_alerts_list_and_detail(client: TestClient):
    """Verify querying alerts and fetching single alert by ID."""
    # List alerts
    response = client.get("/alerts?limit=10")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)

    if alerts:
        first_id = alerts[0]["alert_id"]
        detail_resp = client.get(f"/alerts/{first_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["alert_id"] == first_id

        # Waterfall payload
        wf_resp = client.get(f"/xai/waterfall/{first_id}")
        assert wf_resp.status_code == 200
        wf_data = wf_resp.json()
        assert "base_value" in wf_data
        assert "features" in wf_data
        assert "shap_values" in wf_data


def test_alert_not_found_404(client: TestClient):
    """Verify non-existent alert returns 404."""
    response = client.get("/alerts/non-existent-alert-uuid")
    assert response.status_code == 404
