"""Interactive Live Input -> Output Threat Detection Simulator & Playground.

Allows users to adjust telemetry input parameters or select presets, then
immediately witness the full dual-tier pipeline evaluation, visual threat gauge,
SHAP feature attribution breakdown, and automated cloud mitigation playbooks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.schemas import WindowAggregate
from dashboard.engine_bridge import OfflineDetectionEngine


def get_preset_window(preset_name: str) -> WindowAggregate:
    """Generate canonical WindowAggregate inputs for standard scenario presets."""
    now = datetime.now(timezone.utc)
    base_window = dict(
        window_start=now - timedelta(seconds=60),
        window_end=now,
        duration_seconds=60.0,
    )

    if preset_name == "nominal":
        return WindowAggregate(
            **base_window,
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
    elif preset_name == "ddos":
        return WindowAggregate(
            **base_window,
            request_count=3500,
            request_rate_per_sec=58.33,
            error_count_4xx=15,
            error_count_5xx=120,
            error_rate_4xx=0.004,
            error_rate_5xx=0.034,
            mean_duration_ms=180.0,
            p95_duration_ms=450.0,
            unique_source_ips=3100,
            unique_users=25,
            total_request_bytes=1750000,
            total_response_bytes=7000000,
            mean_response_bytes=2000.0,
            auth_failure_count=5,
            off_hour_ratio=0.0,
            bot_agent_ratio=0.85,
        )
    elif preset_name == "credential_stuffing":
        return WindowAggregate(
            **base_window,
            request_count=450,
            request_rate_per_sec=7.5,
            error_count_4xx=390,
            error_count_5xx=5,
            error_rate_4xx=0.866,
            error_rate_5xx=0.011,
            mean_duration_ms=55.0,
            p95_duration_ms=85.0,
            unique_source_ips=30,
            unique_users=420,
            total_request_bytes=225000,
            total_response_bytes=90000,
            mean_response_bytes=200.0,
            auth_failure_count=385,
            off_hour_ratio=0.1,
            bot_agent_ratio=0.92,
        )
    elif preset_name == "off_hour":
        return WindowAggregate(
            **base_window,
            request_count=35,
            request_rate_per_sec=0.58,
            error_count_4xx=12,
            error_count_5xx=0,
            error_rate_4xx=0.342,
            error_rate_5xx=0.0,
            mean_duration_ms=95.0,
            p95_duration_ms=210.0,
            unique_source_ips=4,
            unique_users=2,
            total_request_bytes=17500,
            total_response_bytes=35000,
            mean_response_bytes=1000.0,
            auth_failure_count=8,
            off_hour_ratio=1.0,
            bot_agent_ratio=0.65,
        )
    else:  # "exfiltration"
        return WindowAggregate(
            **base_window,
            request_count=60,
            request_rate_per_sec=1.0,
            error_count_4xx=0,
            error_count_5xx=0,
            error_rate_4xx=0.0,
            error_rate_5xx=0.0,
            mean_duration_ms=320.0,
            p95_duration_ms=750.0,
            unique_source_ips=1,
            unique_users=1,
            total_request_bytes=30000,
            total_response_bytes=85000000,
            mean_response_bytes=1416666.0,
            auth_failure_count=0,
            off_hour_ratio=0.0,
            bot_agent_ratio=0.0,
        )


def render_risk_gauge(risk_score: float, is_anomaly: bool) -> None:
    """Render a clean, modern Plotly gauge chart for the Threat Risk Score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Threat Risk Index (0 - 100)", "font": {"size": 16, "color": "#94A3B8"}},
            number={"suffix": "/100", "font": {"size": 32, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#EF4444" if is_anomaly else "#10B981"},
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [35, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#F43F5E", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_attribution_bars(attributions: list) -> None:
    """Render horizontal bar chart of top SHAP feature contributions."""
    if not attributions:
        return

    features = [a.get("feature_name", "") for a in attributions][::-1]
    shap_vals = [a.get("shap_value", 0.0) for a in attributions][::-1]
    observed_vals = [a.get("observed_value", 0.0) for a in attributions][::-1]
    baseline_vals = [a.get("baseline_expected", 0.0) for a in attributions][::-1]

    colors = ["#EF4444" if v > 0.05 else "#06B6D4" for v in shap_vals]

    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=features,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#334155", width=1)),
            text=[f"SHAP: +{v:.3f} (Obs: {o} | Exp: {b})" for v, o, b in zip(shap_vals, observed_vals, baseline_vals)],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Novelty: SHAP Feature Importance Attribution (TreeExplainer)",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        xaxis_title="SHAP Importance Delta (Impact toward Anomaly Flag)",
        yaxis_title="Telemetry Feature",
        height=260,
        margin=dict(l=20, r=20, t=40, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_live_threat_simulator(engine: OfflineDetectionEngine) -> None:
    """Render the full interactive User Input -> Output Live Threat Simulator."""
    st.markdown("### 🎮 Live Input → Output Threat Simulator")
    st.markdown(
        "Interactively customize cloud telemetry window metrics or pick standard attack presets, "
        "then execute the live dual-tier pipeline to observe real-time anomaly classification and SHAP XAI attribution."
    )

    # 1. Preset Selector Buttons
    st.markdown("#### 1. Choose Scenario Preset or Customize Parameters")
    preset_col1, preset_col2, preset_col3, preset_col4, preset_col5 = st.columns(5)

    if "current_preset" not in st.session_state:
        st.session_state.current_preset = "nominal"

    with preset_col1:
        if st.button("🟢 Nominal Baseline", use_container_width=True):
            st.session_state.current_preset = "nominal"
    with preset_col2:
        if st.button("🌊 DDoS HTTP Flood", use_container_width=True):
            st.session_state.current_preset = "ddos"
    with preset_col3:
        if st.button("🔑 Credential Stuffing", use_container_width=True):
            st.session_state.current_preset = "credential_stuffing"
    with preset_col4:
        if st.button("🕵️ Off-Hour Probing", use_container_width=True):
            st.session_state.current_preset = "off_hour"
    with preset_col5:
        if st.button("💾 Data Exfiltration", use_container_width=True):
            st.session_state.current_preset = "exfiltration"

    active_preset = get_preset_window(st.session_state.current_preset)

    # 2. Interactive Input Controls (Sliders)
    with st.expander(f"⚙️ Fine-Tune Telemetric Input Parameters (Active: {st.session_state.current_preset.upper()})", expanded=True):
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            req_count = st.number_input("Request Count (in 60s window)", min_value=1, max_value=10000, value=int(active_preset.request_count))
            unique_ips = st.number_input("Unique Client Source IPs", min_value=1, max_value=10000, value=int(active_preset.unique_source_ips))
            unique_users = st.number_input("Unique Authenticated Users", min_value=1, max_value=5000, value=int(active_preset.unique_users))

        with col_in2:
            p95_duration = st.slider("P95 Request Latency (ms)", min_value=10.0, max_value=2000.0, value=float(active_preset.p95_duration_ms), step=5.0)
            err_4xx = st.number_input("4xx Client Errors (401/403/404)", min_value=0, max_value=10000, value=int(active_preset.error_count_4xx))
            err_5xx = st.number_input("5xx Server Errors (500/502/503)", min_value=0, max_value=1000, value=int(active_preset.error_count_5xx))

        with col_in3:
            resp_bytes = st.number_input("Total Outbound Response (Bytes)", min_value=1000, max_value=500000000, value=int(active_preset.total_response_bytes), step=50000)
            auth_failures = st.number_input("Auth Failure Count (401 Unauthorized)", min_value=0, max_value=5000, value=int(active_preset.auth_failure_count))
            off_hour_ratio = st.slider("Off-Hour Window Ratio (Night Activity)", min_value=0.0, max_value=1.0, value=float(active_preset.off_hour_ratio), step=0.05)

    # Reconstruct Window from user input
    now = datetime.now(timezone.utc)
    user_input_window = WindowAggregate(
        window_start=now - timedelta(seconds=60),
        window_end=now,
        duration_seconds=60.0,
        request_count=req_count,
        request_rate_per_sec=round(req_count / 60.0, 2),
        error_count_4xx=err_4xx,
        error_count_5xx=err_5xx,
        error_rate_4xx=round(err_4xx / max(1, req_count), 4),
        error_rate_5xx=round(err_5xx / max(1, req_count), 4),
        mean_duration_ms=round(p95_duration * 0.65, 2),
        p95_duration_ms=p95_duration,
        unique_source_ips=unique_ips,
        unique_users=unique_users,
        total_request_bytes=req_count * 500,
        total_response_bytes=resp_bytes,
        mean_response_bytes=round(resp_bytes / max(1, req_count), 2),
        auth_failure_count=auth_failures,
        off_hour_ratio=off_hour_ratio,
        bot_agent_ratio=0.85 if req_count > 1000 else 0.0,
    )

    # Show User Input Vector
    with st.expander("🔍 View Raw User Input Telemetry Vector (JSON)", expanded=False):
        st.json(user_input_window.model_dump(mode="json"))

    # 3. Execution Trigger
    st.markdown("---")
    execute_button = st.button("⚡ RUN DUAL-TIER THREAT DETECTION PIPELINE", type="primary", use_container_width=True)

    # Run detection
    result = engine.evaluate_window(user_input_window)
    is_anomaly = result["is_anomaly"]
    risk = result["risk_score"]

    st.markdown("#### 2. Pipeline Execution Output")

    # A. Verdict Banner
    if not is_anomaly:
        st.success(f"### 🟢 CLEAN NOMINAL WORKLOAD — Within Expected Diurnal Baseline (Risk: {risk}/100)")
    else:
        hyp = result.get("threat_hypothesis", "Unknown Threat")
        alert = result.get("alert") or {}
        sev = alert.get("severity", "CRITICAL")
        conf = alert.get("confidence_score", 0.95)
        st.error(f"### 🚨 THREAT DETECTED: {hyp}\n**Severity: `{sev}`** | **Confidence: `{conf:.1%}`** | **Risk Index: `{risk}/100`**")

    # B. Side-by-Side: Gauge Meter & Pipeline Stepper
    out_col1, out_col2 = st.columns([1, 2])

    with out_col1:
        render_risk_gauge(risk, is_anomaly)

    with out_col2:
        st.markdown("**Dual-Tier Arbitration Stepper:**")
        t1 = result["tier1"]
        t2 = result["tier2"]

        # Step cards
        s1, s2, s3 = st.columns(3)
        with s1:
            t1_status = "🔴 ANOMALOUS" if t1["flagged"] else "🟢 NORMAL"
            st.metric("Tier 1 (Statistical MAD)", t1_status, help="Volumetric fast-path threshold check")
        with s2:
            t2_status = "🔴 ANOMALOUS" if t2["flagged"] else "🟢 NORMAL"
            st.metric("Tier 2 (Isolation Forest)", t2_status, help="Multivariate unsupervised contextual model")
        with s3:
            ensemble_verdict = "CONFIRMED THREAT" if is_anomaly else "NORMAL"
            st.metric("Hybrid Ensemble", ensemble_verdict, delta="+5% conf" if (t1['flagged'] and t2['flagged']) else None)

    # C. SHAP Feature Attribution Visualization
    st.markdown("---")
    render_shap_attribution_bars(result.get("attributions", []))

    # D. Plain-English Semantic Explanation & Cloud Mitigation Playbook
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown("#### 🧠 Plain-English AI Forensic Narrative")
        st.info(f"**Identified Threat Vector:** {result.get('threat_hypothesis')}\n\n{result.get('explanation_narrative')}")

    with e_col2:
        st.markdown("#### 🛡️ Automated Cloud Mitigation Playbook")
        st.success(f"**Action Plan:**\n\n{result.get('recommended_mitigation')}")
