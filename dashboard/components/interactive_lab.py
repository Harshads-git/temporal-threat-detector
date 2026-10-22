"""Interactive Threat Lab: Live User Input vs AI Detection Output Showcase.

Allows users to input custom telemetry parameters or select attack scenarios,
and displays instant dual-tier detection verdicts with multi-dimensional
visual representations (Gauge, Radar, SHAP Waterfall, and Mitigation Playbooks).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.schemas import WindowAggregate
from dashboard.api_client import DashboardApiClient
from dashboard.embedded_engine import OfflineThreatEngine


def render_threat_gauge(score: float, threshold: float, severity: str) -> go.Figure:
    """Render a modern Plotly speedometer gauge chart indicating anomaly score vs threshold."""
    max_val = max(threshold * 3.0, score * 1.3, 10.0)

    color_map = {
        "NORMAL": "#10B981",
        "LOW": "#3B82F6",
        "MEDIUM": "#F59E0B",
        "HIGH": "#F97316",
        "CRITICAL": "#EF4444",
    }
    bar_color = color_map.get(severity.upper(), "#EF4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            delta={"reference": threshold, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
            title={"text": f"Anomaly Score (Threshold: {threshold:.2f})", "font": {"size": 15, "color": "#E2E8F0"}},
            number={"font": {"size": 28, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": bar_color, "thickness": 0.4},
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, threshold * 0.7], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [threshold * 0.7, threshold], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [threshold, max_val], "color": "rgba(239, 68, 68, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#F87171", "width": 4},
                    "thickness": 0.75,
                    "value": threshold,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        height=260,
        margin=dict(l=25, r=25, t=40, b=25),
    )
    return fig


def render_feature_radar_chart(
    input_features: Dict[str, float],
    baseline_expected: Dict[str, float],
) -> go.Figure:
    """Render a Plotly radar/spider chart comparing User Input features against Diurnal Baseline."""
    # 8 most interpretable features for polar comparison
    compare_features = [
        "request_count",
        "request_rate_per_sec",
        "error_rate_4xx",
        "error_rate_5xx",
        "p95_duration_ms",
        "unique_source_ips",
        "auth_failure_count",
        "off_hour_ratio",
    ]

    labels = [
        "Req Volume",
        "Req Rate",
        "4xx Errors",
        "5xx Errors",
        "P95 Latency",
        "IP Dispersion",
        "Auth Failures",
        "Off-Hour Ratio",
    ]

    obs_norm = []
    exp_norm = []

    for f in compare_features:
        obs = float(input_features.get(f, 0.0))
        exp = float(baseline_expected.get(f, 0.001))
        # Normalize relative to expected baseline (scale around 1.0 = normal)
        ratio = obs / max(1e-4, exp)
        obs_norm.append(min(10.0, ratio))
        exp_norm.append(1.0)

    # Close the radar loop
    obs_norm.append(obs_norm[0])
    exp_norm.append(exp_norm[0])
    labels.append(labels[0])

    fig = go.Figure()

    # Baseline Profile Trace
    fig.add_trace(
        go.Scatterpolar(
            r=exp_norm,
            theta=labels,
            fill="toself",
            name="Diurnal Baseline Expected",
            line=dict(color="#10B981", width=2),
            fillcolor="rgba(16, 185, 129, 0.15)",
        )
    )

    # User Input Telemetry Trace
    fig.add_trace(
        go.Scatterpolar(
            r=obs_norm,
            theta=labels,
            fill="toself",
            name="Current Input Telemetry",
            line=dict(color="#00E5FF", width=2.5),
            fillcolor="rgba(0, 229, 255, 0.25)",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(5.0, max(obs_norm))], tickcolor="#64748B"),
            bgcolor="#1E293B",
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        height=320,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


def render_shap_waterfall_chart(attributions: List[Dict[str, Any]], hypothesis: str = "Threat Attribution") -> go.Figure:
    """Render Plotly waterfall decomposition of SHAP feature attributions."""
    features = [a.get("feature_name", f"feat_{i}") for i, a in enumerate(attributions[:6])]
    shap_vals = [float(a.get("shap_value", 0.0)) for a in attributions[:6]]

    base_score = 0.05
    measures = ["relative"] * len(shap_vals) + ["total"]
    x_labels = features + ["Final Decision"]
    y_values = shap_vals + [round(base_score + sum(shap_vals), 4)]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Attribution",
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in shap_vals] + [f"{y_values[-1]:.3f}"],
            y=y_values,
            connector={"line": {"color": "#64748B"}},
            decreasing={"marker": {"color": "#10B981"}},
            increasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#F59E0B"}},
        )
    )

    fig.update_layout(
        title=f"SHAP TreeExplainer Log-Odds Contribution ({hypothesis})",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        height=320,
        margin=dict(l=35, r=35, t=40, b=30),
        yaxis_title="Feature Impact (Δ Log-Odds)",
    )
    return fig


def render_interactive_threat_lab(
    client: DashboardApiClient,
    engine: OfflineThreatEngine,
) -> None:
    """Main interactive Threat Lab UI mapping User Input to Real-Time AI Output."""
    st.markdown("### 🧪 Interactive Telemetry Input & Threat Output Lab")
    st.caption(
        "Dial in telemetry parameters, choose attack vector presets, or simulate custom cloud workloads. "
        "Observe in real-time how the Dual-Tier ensemble and SHAP TreeExplainer evaluate the input."
    )

    # 1. Preset Selector Buttons
    st.markdown("#### 1. Choose Input Workload Scenario")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)

    preset_name = st.session_state.get("active_preset", "normal")

    with col_p1:
        if st.button("🟢 Nominal Baseline", use_container_width=True):
            st.session_state.active_preset = "normal"
            preset_name = "normal"
    with col_p2:
        if st.button("🌊 DDoS HTTP Flood", use_container_width=True):
            st.session_state.active_preset = "ddos"
            preset_name = "ddos"
    with col_p3:
        if st.button("🔑 Credential Stuffing", use_container_width=True):
            st.session_state.active_preset = "credential_stuffing"
            preset_name = "credential_stuffing"
    with col_p4:
        if st.button("🕵️ Off-Hour Recon", use_container_width=True):
            st.session_state.active_preset = "off_hour_probing"
            preset_name = "off_hour_probing"
    with col_p5:
        if st.button("💾 Data Exfiltration", use_container_width=True):
            st.session_state.active_preset = "data_exfiltration"
            preset_name = "data_exfiltration"

    # Default preset values
    preset_configs = {
        "normal": {
            "req_count": 20, "req_rate": 0.33, "err_4xx": 1, "err_5xx": 0,
            "latency": 40.0, "unique_ips": 20, "auth_fail": 0, "off_hour": 0.0,
            "bytes_mb": 0.04, "bot_ratio": 0.0,
        },
        "ddos": {
            "req_count": 3500, "req_rate": 58.3, "err_4xx": 50, "err_5xx": 120,
            "latency": 280.0, "unique_ips": 3200, "auth_fail": 5, "off_hour": 0.0,
            "bytes_mb": 5.2, "bot_ratio": 0.85,
        },
        "credential_stuffing": {
            "req_count": 450, "req_rate": 7.5, "err_4xx": 380, "err_5xx": 2,
            "latency": 65.0, "unique_ips": 8, "auth_fail": 375, "off_hour": 0.0,
            "bytes_mb": 0.6, "bot_ratio": 0.95,
        },
        "off_hour_probing": {
            "req_count": 45, "req_rate": 0.75, "err_4xx": 12, "err_5xx": 1,
            "latency": 95.0, "unique_ips": 2, "auth_fail": 15, "off_hour": 1.0,
            "bytes_mb": 0.12, "bot_ratio": 0.4,
        },
        "data_exfiltration": {
            "req_count": 80, "req_rate": 1.33, "err_4xx": 0, "err_5xx": 0,
            "latency": 420.0, "unique_ips": 1, "auth_fail": 0, "off_hour": 0.0,
            "bytes_mb": 250.0, "bot_ratio": 0.0,
        },
    }

    curr = preset_configs.get(preset_name, preset_configs["normal"])

    # 2. Interactive Input Controls Expander / Sliders
    with st.expander("🛠️ Customize Input Telemetry Parameters (Fine-Tuning Controls)", expanded=True):
        scol1, scol2, scol3, scol4 = st.columns(4)
        with scol1:
            in_req_count = st.slider("Request Count / Window", 5, 5000, curr["req_count"], 25)
            in_unique_ips = st.slider("Unique Client IPs", 1, 5000, min(in_req_count, curr["unique_ips"]), 10)
        with scol2:
            in_err_4xx = st.slider("4xx Client Errors", 0, 500, min(in_req_count, curr["err_4xx"]), 5)
            in_err_5xx = st.slider("5xx Server Errors", 0, 200, min(in_req_count, curr["err_5xx"]), 2)
        with scol3:
            in_latency = st.slider("P95 Latency (ms)", 10.0, 1000.0, float(curr["latency"]), 10.0)
            in_auth_fail = st.slider("Auth Failures (401/403)", 0, 500, curr["auth_fail"], 5)
        with scol4:
            in_bytes_mb = st.slider("Outbound Data (MB)", 0.01, 300.0, float(curr["bytes_mb"]), 1.0)
            in_off_hour = st.slider("Off-Hour Activity Ratio", 0.0, 1.0, float(curr["off_hour"]), 0.1)

    # 3. Action Execution Button
    btn_eval = st.button("🚀 EVALUATE TELEMETRY & RUN XAI ENGINE", type="primary", use_container_width=True)

    # Construct WindowAggregate from user input
    now = datetime.now(timezone.utc)
    duration = 60.0
    req_rate = in_req_count / duration
    tot_bytes = int(in_bytes_mb * 1024 * 1024)
    mean_bytes = tot_bytes / max(1, in_req_count)

    user_window = WindowAggregate(
        window_start=now - timedelta(seconds=duration),
        window_end=now,
        duration_seconds=duration,
        request_count=in_req_count,
        request_rate_per_sec=round(req_rate, 4),
        error_count_4xx=in_err_4xx,
        error_count_5xx=in_err_5xx,
        error_rate_4xx=round(in_err_4xx / max(1, in_req_count), 4),
        error_rate_5xx=round(in_err_5xx / max(1, in_req_count), 4),
        mean_duration_ms=round(in_latency * 0.7, 2),
        p95_duration_ms=round(in_latency, 2),
        unique_source_ips=in_unique_ips,
        unique_users=max(1, in_unique_ips // 4),
        total_request_bytes=in_req_count * 500,
        total_response_bytes=tot_bytes,
        mean_response_bytes=round(mean_bytes, 2),
        auth_failure_count=in_auth_fail,
        off_hour_ratio=round(in_off_hour, 2),
        bot_agent_ratio=round(curr["bot_ratio"], 2),
    )

    # Evaluate using either FastAPI (if available) or Embedded Engine
    result: Optional[Dict[str, Any]] = None
    if client.is_available():
        try:
            api_res = client.detect_window(user_window.model_dump(mode="json"))
            if api_res.get("status") != "error":
                # Reshape api response into result
                alert_dict = api_res.get("alert") or {}
                result = {
                    "is_anomaly": api_res.get("is_anomaly", False),
                    "verdict": "THREAT DETECTED" if api_res.get("is_anomaly") else "NOMINAL WORKLOAD",
                    "severity": alert_dict.get("severity", "NORMAL"),
                    "confidence_score": alert_dict.get("confidence_score", 0.0),
                    "anomaly_score": alert_dict.get("anomaly_score", 0.0),
                    "decision_threshold": alert_dict.get("decision_threshold", 3.0),
                    "detector_tier": alert_dict.get("detector_tier", "TIER1_STATISTICAL"),
                    "threat_hypothesis": alert_dict.get("threat_hypothesis", "Normal Operational Baseline"),
                    "explanation_narrative": alert_dict.get("explanation_narrative", api_res.get("message", "")),
                    "recommended_mitigation": alert_dict.get("recommended_mitigation", "No action needed."),
                    "top_contributions": alert_dict.get("top_contributions", []),
                    "input_features": user_window.to_feature_vector(),
                    "baseline_expected": engine.baseline_feature_means,
                }
        except Exception:
            result = None

    if result is None:
        # Evaluate via embedded offline engine
        result = engine.evaluate_window(user_window)

    st.markdown("---")
    st.markdown("### 📋 AI Evaluation Results & Visual Forensics")

    # A. Output Verdict Banner
    is_anomaly = result.get("is_anomaly", False)
    sev = result.get("severity", "NORMAL")
    hyp = result.get("threat_hypothesis", "Nominal")

    if is_anomaly:
        st.error(
            f"### 🚨 {result.get('verdict')} : {hyp}\n"
            f"**Severity:** `{sev}` | **Ensemble Tier:** `{result.get('detector_tier')}` | "
            f"**Confidence:** `{result.get('confidence_score', 0.0):.1%}`"
        )
    else:
        st.success(
            f"### 🟢 {result.get('verdict')}\n"
            f"**Status:** Workload telemetry is aligned with diurnal baseline distributions. No anomalies detected."
        )

    # B. Side-by-Side Visualizations (Gauge + Radar)
    vcol1, vcol2 = st.columns([1, 1.2])

    with vcol1:
        st.markdown("#### 1. Threat Severity & Anomaly Score")
        gauge_fig = render_threat_gauge(
            score=result.get("anomaly_score", 0.0),
            threshold=result.get("decision_threshold", 3.0),
            severity=sev,
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with vcol2:
        st.markdown("#### 2. Multi-Feature Radar: Input vs Diurnal Baseline")
        radar_fig = render_feature_radar_chart(
            input_features=result.get("input_features", {}),
            baseline_expected=result.get("baseline_expected", {}),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # C. SHAP Waterfall & Feature Deviation Table
    wcol1, wcol2 = st.columns([1.2, 1])

    with wcol1:
        st.markdown("#### 3. Novelty: SHAP TreeExplainer Waterfall")
        waterfall_fig = render_shap_waterfall_chart(
            attributions=result.get("top_contributions", []),
            hypothesis=hyp,
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

    with wcol2:
        st.markdown("#### 4. Feature Deviation Matrix")
        attrs = result.get("top_contributions", [])
        if attrs:
            df_dev = pd.DataFrame(attrs)
            # Format display
            st.dataframe(
                df_dev[["importance_rank", "feature_name", "observed_value", "baseline_expected", "shap_value"]],
                use_container_width=True,
                column_config={
                    "importance_rank": st.column_config.NumberColumn("Rank", width="small"),
                    "feature_name": st.column_config.TextColumn("Telemetric Feature"),
                    "observed_value": st.column_config.NumberColumn("User Input"),
                    "baseline_expected": st.column_config.NumberColumn("Baseline"),
                    "shap_value": st.column_config.NumberColumn("SHAP Importance", format="%.4f"),
                },
                hide_index=True,
            )

    # D. Cloud Incident Response Playbook
    st.markdown("#### 5. Automated Cloud Incident Response & Mitigation Playbook")
    mitigation_text = result.get("recommended_mitigation", "No action needed.")
    if is_anomaly:
        st.warning(f"🛡️ **SecOps Recommended Action:**\n\n{mitigation_text}")
    else:
        st.info(f"ℹ️ **Operational Guidance:** {mitigation_text}")
