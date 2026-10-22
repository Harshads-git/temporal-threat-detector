"""Unit and integration tests for Streamlit dashboard components and API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests
from dashboard.api_client import DashboardApiClient
from dashboard.components.alert_cards import (
    get_sample_xai_alert,
    get_severity_badge,
)
from dashboard.components.timeline_chart import generate_fallback_timeline_data


def test_dashboard_api_client_initialization() -> None:
    """Verify DashboardApiClient initialization and URL normalization."""
    client = DashboardApiClient(base_url="http://localhost:8000/", timeout_sec=2.5)
    assert client.base_url == "http://localhost:8000"
    assert client.timeout_sec == 2.5


def test_dashboard_api_client_health_check_online() -> None:
    """Verify health check handling on successful response."""
    client = DashboardApiClient(base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "service": "temporal-threat-detector"}

    with patch("requests.get", return_value=mock_resp):
        assert client.is_available() is True
        health = client.get_health()
        assert health["status"] == "healthy"


def test_dashboard_api_client_health_check_offline() -> None:
    """Verify health check returns offline dictionary when connection fails."""
    client = DashboardApiClient(base_url="http://127.0.0.1:8000")
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        assert client.is_available() is False
        health = client.get_health()
        assert health["status"] == "offline"


def test_dashboard_api_client_metrics_retrieval() -> None:
    """Verify metrics endpoint client wrapper."""
    client = DashboardApiClient(base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "total_events_processed": 1050,
        "queue_depth": 0,
        "total_alerts_emitted": 4,
        "tier1_zscore_alerts": 2,
        "tier2_iforest_alerts": 1,
        "hybrid_alerts": 1,
        "mean_pipeline_latency_ms": 11.2,
    }

    with patch("requests.get", return_value=mock_resp):
        metrics = client.get_metrics()
        assert metrics["total_events_processed"] == 1050
        assert metrics["total_alerts_emitted"] == 4


def test_dashboard_api_client_alerts_query() -> None:
    """Verify alerts querying and parameter forwarding."""
    client = DashboardApiClient(base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "alerts": [
            {"alert_id": "alert-1", "severity": "CRITICAL"},
            {"alert_id": "alert-2", "severity": "HIGH"},
        ],
        "count": 2,
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        alerts = client.get_alerts(limit=10, severity="CRITICAL")
        assert len(alerts) == 2
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8000/alerts",
            params={"limit": 10, "severity": "CRITICAL"},
            timeout=3.0,
        )


def test_dashboard_api_client_inject_attack() -> None:
    """Verify attack injection call serialization."""
    client = DashboardApiClient(base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 202
    mock_resp.json.return_value = {
        "status": "injected",
        "attack_type": "ddos",
        "burst_profile": "sudden_spike",
        "injected_count": 300,
        "batch_id": "batch-1234",
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = client.inject_attack(attack_type="ddos", burst_profile="sudden_spike", count=300)
        assert res["status"] == "injected"
        assert res["injected_count"] == 300
        mock_post.assert_called_once()


def test_fallback_timeline_data_generation() -> None:
    """Verify fallback timeline data structure and columns."""
    df = generate_fallback_timeline_data(count=25)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25
    expected_cols = {"timestamp", "request_count", "request_rate", "error_rate", "p95_duration_ms"}
    assert expected_cols.issubset(set(df.columns))
    assert (df["request_count"] > 0).all()


def test_sample_xai_alert_structure() -> None:
    """Verify fallback sample XAI alert format contains all required keys."""
    sample = get_sample_xai_alert()
    assert "alert_id" in sample
    assert "detector_tier" in sample
    assert "severity" in sample
    assert "threat_hypothesis" in sample
    assert "top_contributions" in sample
    assert len(sample["top_contributions"]) == 4
    first_attr = sample["top_contributions"][0]
    assert "feature_name" in first_attr
    assert "observed_value" in first_attr
    assert "baseline_expected" in first_attr
    assert "shap_value" in first_attr


def test_severity_badge_formatting() -> None:
    """Verify markdown badge formatting for each severity category."""
    assert "🔴" in get_severity_badge("CRITICAL")
    assert "🟠" in get_severity_badge("HIGH")
    assert "🟡" in get_severity_badge("MEDIUM")
    assert "🔵" in get_severity_badge("LOW")
    assert "⚪" in get_severity_badge("INFO")


def test_dashboard_api_client_local_evaluation() -> None:
    """Verify DashboardApiClient in-process offline fallback detection."""
    from datetime import datetime, timezone
    from api.schemas import WindowAggregate
    from dashboard.components.inspector import BASELINE_EXPECTED

    client = DashboardApiClient(base_url="http://127.0.0.1:9999")  # Non-existent port to force fallback

    # 1. Test nominal window
    now = datetime.now(timezone.utc)
    nominal_window = WindowAggregate(
        window_start=now,
        window_end=now,
        duration_seconds=60.0,
        request_count=20,
        request_rate_per_sec=0.33,
        error_count_4xx=0,
        error_count_5xx=0,
        error_rate_4xx=0.0,
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
    res_nom = client.detect_window(nominal_window.model_dump(mode="json"))
    assert res_nom["is_anomaly"] is False
    assert res_nom["alert"] is None

    # 2. Test massive volumetric DDoS window
    ddos_window = WindowAggregate(
        window_start=now,
        window_end=now,
        duration_seconds=60.0,
        request_count=3500,
        request_rate_per_sec=58.3,
        error_count_4xx=10,
        error_count_5xx=45,
        error_rate_4xx=0.002,
        error_rate_5xx=0.012,
        mean_duration_ms=180.0,
        p95_duration_ms=450.0,
        unique_source_ips=3000,
        unique_users=90,
        total_request_bytes=1750000,
        total_response_bytes=6500000,
        mean_response_bytes=1857.0,
        auth_failure_count=5,
        off_hour_ratio=0.0,
        bot_agent_ratio=0.75,
    )
    res_attack = client.detect_window(ddos_window.model_dump(mode="json"))
    assert res_attack["is_anomaly"] is True
    assert res_attack["alert"] is not None
    assert "DDoS" in res_attack["alert"]["threat_hypothesis"] or "Volumetric" in res_attack["alert"]["threat_hypothesis"]
    assert len(res_attack["alert"]["top_contributions"]) > 0

    # 3. Test local offline alert retrieval
    alerts = client.get_alerts(limit=5)
    assert len(alerts) >= 1

    # 4. Test local offline metrics
    metrics = client.get_metrics()
    assert metrics["total_events_processed"] > 0


def test_inspector_baseline_and_data_contracts() -> None:
    """Verify inspector baseline expectations and metric definitions."""
    from dashboard.components.inspector import BASELINE_EXPECTED

    required_keys = {
        "request_count",
        "request_rate_per_sec",
        "error_rate_4xx",
        "error_rate_5xx",
        "mean_duration_ms",
        "p95_duration_ms",
        "unique_source_ips",
        "total_response_bytes",
        "auth_failure_count",
        "off_hour_ratio",
    }
    assert required_keys.issubset(set(BASELINE_EXPECTED.keys()))
    assert BASELINE_EXPECTED["request_count"] == 20.0

