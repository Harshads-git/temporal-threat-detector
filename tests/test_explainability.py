"""Unit and integration tests for Explainable AI (SHAP TreeExplainer & Semantic Reasoning)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np

from api.schemas import (
    FeatureAttribution,
    WindowAggregate,
)
from engine.explainability import (
    IncidentResponseRecommender,
    MitigationPlaybook,
    SemanticReasoningEngine,
    ShapExplainerEngine,
)
from engine.isolation_forest import (
    IsolationForestDetector,
)
from engine.windowing import CANONICAL_FEATURE_NAMES


def create_mock_window(
    offset_min: int,
    req_count: int = 100,
    off_hour: float = 0.0,
    resp_bytes: int = 150000,
    err_4xx: int = 2,
    err_5xx: int = 0,
    auth_fail: int = 1,
    bot_ratio: float = 0.02,
    duration_ms: float = 45.0,
) -> WindowAggregate:
    """Helper to generate parameterized window aggregates."""
    now = datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_min)
    return WindowAggregate(
        window_start=now,
        window_end=now + timedelta(seconds=60),
        duration_seconds=60.0,
        request_count=req_count,
        request_rate_per_sec=req_count / 60.0,
        error_count_4xx=err_4xx,
        error_count_5xx=err_5xx,
        error_rate_4xx=err_4xx / max(1, req_count),
        error_rate_5xx=err_5xx / max(1, req_count),
        mean_duration_ms=duration_ms,
        p95_duration_ms=duration_ms * 1.5,
        unique_source_ips=max(1, req_count // 3),
        unique_users=max(1, req_count // 5),
        total_request_bytes=req_count * 500,
        total_response_bytes=resp_bytes,
        mean_response_bytes=resp_bytes / max(1, req_count),
        auth_failure_count=auth_fail,
        off_hour_ratio=off_hour,
        bot_agent_ratio=bot_ratio,
    )


@pytest.fixture(scope="module")
def trained_explainer_engine():
    """Module-level fixture providing a trained IsolationForest and ShapExplainerEngine."""
    # Build baseline with natural slight variance across 40 windows
    baseline_windows = [
        create_mock_window(
            i,
            req_count=100 + (i % 15),
            resp_bytes=150000 + (i * 2000),
            err_4xx=1 + (i % 3),
            auth_fail=1 + (i % 5),
            bot_ratio=0.01 + (0.01 * (i % 3)),
            duration_ms=40.0 + (i % 10),
            off_hour=(i % 4) * 0.02,
        )
        for i in range(40)
    ]

    detector = IsolationForestDetector(n_estimators=40, contamination=0.05, random_state=42)
    detector.fit(baseline_windows)

    # Extract background matrix
    rows = []
    for w in baseline_windows:
        fmap = w.to_feature_vector()
        rows.append([fmap.get(c, 0.0) for c in CANONICAL_FEATURE_NAMES])
    background_matrix = np.array(rows, dtype=np.float64)

    explainer_engine = ShapExplainerEngine(
        model=detector.model,
        scaler=detector.scaler,
        background_data=background_matrix,
        feature_names=CANONICAL_FEATURE_NAMES,
    )
    return explainer_engine


def test_shap_explainer_initialization_and_attribution(trained_explainer_engine):
    """Verify SHAP TreeExplainer computes valid Shapley values across 13 canonical features."""
    test_window = create_mock_window(50, req_count=105)
    shap_vals, raw_vals = trained_explainer_engine.compute_shap_values(test_window)

    assert len(shap_vals) == len(CANONICAL_FEATURE_NAMES)
    assert len(raw_vals) == len(CANONICAL_FEATURE_NAMES)
    assert isinstance(trained_explainer_engine.expected_value, float)
    assert not np.isnan(shap_vals).any()


def test_shap_top_k_rank_ordering(trained_explainer_engine):
    """Verify explain_window returns rank-ordered attributions sorted by absolute magnitude."""
    # Massive anomaly window
    test_window = create_mock_window(51, req_count=1500, resp_bytes=50000000)
    attributions = trained_explainer_engine.explain_window(test_window, top_k=4)

    assert len(attributions) == 4
    assert [a.importance_rank for a in attributions] == [1, 2, 3, 4]

    # Verify monotonic descent in absolute impact
    magnitudes = [abs(a.shap_value) for a in attributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_ddos_shap_attribution_dominance(trained_explainer_engine):
    """Verify volumetric flood identifies request_count as dominant SHAP contributor."""
    ddos_window = create_mock_window(52, req_count=2500, err_5xx=50)
    attributions = trained_explainer_engine.explain_window(ddos_window, top_k=3)

    top_feature_names = [a.feature_name for a in attributions]
    assert "request_count" in top_feature_names or "request_rate_per_sec" in top_feature_names

    hypothesis, narrative = SemanticReasoningEngine.infer_threat_semantics(attributions, ddos_window)
    assert hypothesis == "High Volumetric HTTP Flood (DDoS)"
    assert "SHAP attribution" in narrative


def test_exfiltration_shap_attribution_dominance(trained_explainer_engine):
    """Verify large egress payload identifies total_response_bytes as dominant SHAP contributor."""
    exfil_window = create_mock_window(53, req_count=100, resp_bytes=60 * 1024 * 1024)  # 60 MB
    attributions = trained_explainer_engine.explain_window(exfil_window, top_k=3)

    top_feature_names = [a.feature_name for a in attributions]
    assert any("response_bytes" in f for f in top_feature_names)

    hypothesis, narrative = SemanticReasoningEngine.infer_threat_semantics(attributions, exfil_window)
    assert hypothesis == "Privileged Data Exfiltration Surge"
    assert "MB" in narrative


def test_credential_stuffing_attribution_dominance(trained_explainer_engine):
    """Verify auth failure surge identifies auth_failure_count or bot_agent_ratio in top SHAP contributors."""
    auth_window = create_mock_window(54, req_count=100, auth_fail=80, bot_ratio=0.9, err_4xx=85)
    attributions = trained_explainer_engine.explain_window(auth_window, top_k=3)

    top_feature_names = [a.feature_name for a in attributions]
    assert any(f in top_feature_names for f in ("auth_failure_count", "error_rate_4xx", "bot_agent_ratio"))

    hypothesis, narrative = SemanticReasoningEngine.infer_threat_semantics(attributions, auth_window)
    assert hypothesis == "Distributed Credential Stuffing Campaign"


def test_off_hour_probing_attribution_dominance(trained_explainer_engine):
    """Verify off-hour reconnaissance identifies off_hour_ratio as dominant contributor."""
    probe_window = create_mock_window(55, req_count=100, off_hour=1.0, err_4xx=30)
    attributions = trained_explainer_engine.explain_window(probe_window, top_k=3)

    top_feature_names = [a.feature_name for a in attributions]
    assert "off_hour_ratio" in top_feature_names

    hypothesis, narrative = SemanticReasoningEngine.infer_threat_semantics(attributions, probe_window)
    assert hypothesis == "Off-Hours Reconnaissance Probing"
    assert "off-peak hours" in narrative.lower()


def test_waterfall_payload_structure(trained_explainer_engine):
    """Verify get_waterfall_payload produces chart-ready dictionary structure."""
    test_window = create_mock_window(56, req_count=500)
    payload = trained_explainer_engine.get_waterfall_payload(test_window, top_k=5)

    assert "base_value" in payload
    assert "features" in payload
    assert "shap_values" in payload
    assert "observed_values" in payload
    assert "baseline_expected" in payload
    assert len(payload["features"]) == 5
    assert len(payload["shap_values"]) == 5


def test_incident_response_playbook_mapping():
    """Verify IncidentResponseRecommender maps hypotheses to operational AWS playbooks."""
    ddos_playbook = IncidentResponseRecommender.get_playbook("High Volumetric HTTP Flood (DDoS)")
    assert ddos_playbook.playbook_id == "IR-AWS-DDOS-01"
    assert "AWS WAF" in ddos_playbook.aws_services
    assert len(ddos_playbook.remediation_steps) >= 3

    auth_playbook = IncidentResponseRecommender.get_playbook("Distributed Credential Stuffing Campaign")
    assert auth_playbook.playbook_id == "IR-AWS-AUTH-02"
    assert "Amazon Cognito" in auth_playbook.aws_services

    recon_playbook = IncidentResponseRecommender.get_playbook("Off-Hours Reconnaissance Probing")
    assert recon_playbook.playbook_id == "IR-AWS-RECON-03"

    exfil_playbook = IncidentResponseRecommender.get_playbook("Privileged Data Exfiltration Surge")
    assert exfil_playbook.playbook_id == "IR-AWS-EXFIL-04"

    # Fallback playbook for unknown attack
    unknown_playbook = IncidentResponseRecommender.get_playbook("Novel Zero-Day Zero-Trust Bypass")
    assert unknown_playbook.playbook_id == "IR-AWS-GENERIC-99"
