"""Interactive User Input & AI Model Output Demonstration Lab.

Clearly showcases:
1. USER INPUT: Parameterized attack presets, custom telemetry sliders, or raw vectors.
2. PIPELINE PROCESSING: Dual-Tier evaluation (MAD/EWMA Z-Score + Multivariate Isolation Forest).
3. AI MODEL OUTPUT: Threat verdict, confidence, SHAP XAI feature importance, and mitigation playbooks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.schemas import WindowAggregate


def create_window_from_params(
    request_count: int,
    error_count_4xx: int,
    error_count_5xx: int,
    p95_duration_ms: float,
    unique_source_ips: int,
    auth_failure_count: int,
    off_hour_ratio: float,
    mean_response_bytes: float,
) -> WindowAggregate:
    """Construct a validated WindowAggregate object from user input parameters."""
    now = datetime.now(timezone.utc)
    duration_sec = 60.0
    req_rate = round(request_count / duration_sec, 2)
    err_rate_4xx = round(error_count_4xx / max(1, request_count), 4)
    err_rate_5xx = round(error_count_5xx / max(1, request_count), 4)
    total_bytes = int(request_count * mean_response_bytes)

    return WindowAggregate(
        window_start=now - timedelta(seconds=duration_sec),
        window_end=now,
        duration_seconds=duration_sec,
        request_count=request_count,
        request_rate_per_sec=req_rate,
        error_count_4xx=error_count_4xx,
        error_count_5xx=error_count_5xx,
        error_rate_4xx=err_rate_4xx,
        error_rate_5xx=err_rate_5xx,
        mean_duration_ms=round(p95_duration_ms * 0.6, 2),
        p95_duration_ms=p95_duration_ms,
        unique_source_ips=unique_source_ips,
        unique_users=max(1, int(unique_source_ips * 0.25)),
        total_request_bytes=request_count * 512,
        total_response_bytes=total_bytes,
        mean_response_bytes=mean_response_bytes,
        auth_failure_count=auth_failure_count,
        off_hour_ratio=off_hour_ratio,
        bot_agent_ratio=0.85 if auth_failure_count > 50 else 0.0,
    )


def render_shap_importance_barchart(attributions: list, baseline_expected: dict) -> None:
    """Render horizontal Plotly bar chart visualizing SHAP feature importance attributions."""
    if not attributions:
        st.info("No feature attributions available.")
        return

    features = [a.get("feature_name", f"feat_{i}") for i, a in enumerate(attributions)][::-1]
    shap_vals = [a.get("shap_value", 0.0) for a in attributions][::-1]
    observed = [a.get("observed_value", 0.0) for a in attributions][::-1]
    expected = [a.get("baseline_expected", baseline_expected.get(f, 0.0)) for f, a in zip(features, attributions)][::-1]

    hover_texts = [
        f"<b>Feature:</b> {f}<br><b>Observed:</b> {o}<br><b>Baseline:</b> {e}<br><b>SHAP Value:</b> +{s:.4f}"
        for f, o, e, s in zip(features, observed, expected, shap_vals)
    ]

    colors = ["#EF4444" if v > 0.05 else "#F59E0B" for v in shap_vals]

    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=features,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[f"+{v:.3f} (Obs: {o})" for v, o in zip(shap_vals, observed)],
            textposition="auto",
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="SHAP Feature Importance Attribution (Why was this flagged?)",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="Additive SHAP Value (Log-Odds Impact on Decision)"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_interactive_lab(engine: Any) -> None:
    """Render side-by-side User Input ➔ Model Output interactive demonstration lab."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%); padding: 16px; border-radius: 8px; border-left: 5px solid #00E5FF; margin-bottom: 20px;">
            <h3 style="margin: 0; color: #00E5FF;">🎯 Interactive Threat Detection Lab</h3>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Explicitly inspect <b>User Input</b> telemetry, observe <b>Dual-Tier Pipeline Processing</b>, and examine the <b>AI Model & Explainable AI (XAI) Output</b> in real-time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Maintain selected preset in session state
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "ddos"

    col_input, col_output = st.columns([1, 1], gap="medium")

    # =========================================================================
    # LEFT COLUMN: USER INPUT
    # =========================================================================
    with col_input:
        st.markdown("### 📥 1. User Input Telemetry")

        input_mode = st.radio(
            "Input Mode:",
            options=["preset", "custom"],
            format_func=lambda x: "⚡ Quick Threat Presets" if x == "preset" else "🎛️ Custom Parameter Sliders",
            horizontal=True,
        )

        preset_params = {
            "ddos": {
                "name": "🌊 Volumetric HTTP Flood (DDoS)",
                "request_count": 3500,
                "error_4xx": 12,
                "error_5xx": 85,
                "p95_ms": 180.0,
                "ips": 3200,
                "auth_fail": 0,
                "off_hour": 0.0,
                "bytes": 2100.0,
                "desc": "Simulates 3,500 req/min (+1450% surge) across 3,200 dynamic IPs with 5xx server saturation.",
            },
            "cred_stuff": {
                "name": "🔑 Credential Stuffing & Auth Flooding",
                "request_count": 480,
                "error_4xx": 410,
                "error_5xx": 2,
                "p95_ms": 75.0,
                "ips": 45,
                "auth_fail": 395,
                "off_hour": 0.0,
                "bytes": 1200.0,
                "desc": "Simulates brute-force authentication attacks with 395 failed logins (401 Unauthorized) from bot agents.",
            },
            "off_hour": {
                "name": "🕵️ Off-Hour Reconnaissance Probing",
                "request_count": 45,
                "error_4xx": 18,
                "error_5xx": 0,
                "p95_ms": 120.0,
                "ips": 6,
                "auth_fail": 15,
                "off_hour": 1.0,
                "bytes": 1800.0,
                "desc": "Simulates stealth reconnaissance during midnight off-hours (02:00 AM) probing administrative endpoints.",
            },
            "exfiltration": {
                "name": "💾 Massive Data Exfiltration",
                "request_count": 140,
                "error_4xx": 0,
                "error_5xx": 0,
                "p95_ms": 320.0,
                "ips": 3,
                "auth_fail": 0,
                "off_hour": 0.0,
                "bytes": 620000.0,  # 620 KB per response = ~85 MB total
                "desc": "Simulates bulk data exfiltration from /api/v1/reports/export transferring 85 MB in small request bursts.",
            },
            "nominal": {
                "name": "🟢 Nominal Diurnal Baseline Traffic",
                "request_count": 20,
                "error_4xx": 1,
                "error_5xx": 0,
                "p95_ms": 65.0,
                "ips": 20,
                "auth_fail": 0,
                "off_hour": 0.0,
                "bytes": 2100.0,
                "desc": "Nominal, healthy cloud workload traffic matching trained diurnal baseline parameters.",
            },
        }

        if input_mode == "preset":
            selected_key = st.selectbox(
                "Select Attack Preset to Inject as User Input:",
                options=list(preset_params.keys()),
                format_func=lambda k: preset_params[k]["name"],
            )
            p = preset_params[selected_key]
            st.info(f"**Scenario Profile:** {p['desc']}")

            target_window = create_window_from_params(
                request_count=p["request_count"],
                error_count_4xx=p["error_4xx"],
                error_count_5xx=p["error_5xx"],
                p95_duration_ms=p["p95_ms"],
                unique_source_ips=p["ips"],
                auth_failure_count=p["auth_fail"],
                off_hour_ratio=p["off_hour"],
                mean_response_bytes=p["bytes"],
            )
        else:
            st.caption("Fine-tune exact temporal metrics using interactive controls:")
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                req_cnt = st.slider("Request Count (in 60s):", 5, 5000, 450, step=25)
                err_4xx = st.slider("4xx Errors (Client / Auth):", 0, 1000, 20)
                err_5xx = st.slider("5xx Errors (Server Faults):", 0, 500, 0)
                p95_ms = st.slider("P95 Latency (ms):", 10.0, 1500.0, 75.0, step=10.0)
            with c_r2:
                u_ips = st.slider("Unique Source IPs:", 1, 3500, min(req_cnt, 250))
                auth_f = st.slider("Auth Failures (401/403):", 0, 1000, 0)
                off_h = st.slider("Off-Hour Ratio (0.0=Day, 1.0=Night):", 0.0, 1.0, 0.0, step=0.1)
                m_bytes = st.slider("Mean Response Bytes:", 500, 1000000, 2100, step=1000)

            target_window = create_window_from_params(
                request_count=req_cnt,
                error_count_4xx=err_4xx,
                error_count_5xx=err_5xx,
                p95_duration_ms=p95_ms,
                unique_source_ips=u_ips,
                auth_failure_count=auth_f,
                off_hour_ratio=off_h,
                mean_response_bytes=float(m_bytes),
            )

        # Visual User Input Summary Card
        st.markdown("#### Formatted Telemetry Vector (Sent to Model)")
        vec_display = {
            "Request Volume": f"{target_window.request_count:,} req/window ({target_window.request_rate_per_sec:.1f} req/s)",
            "Error Rates": f"4xx: {target_window.error_rate_4xx:.1%} | 5xx: {target_window.error_rate_5xx:.1%}",
            "Latency P95": f"{target_window.p95_duration_ms:.1f} ms",
            "IP Cardinality": f"{target_window.unique_source_ips:,} unique IPs",
            "Auth Failures": f"{target_window.auth_failure_count} failures",
            "Off-Hour Ratio": f"{target_window.off_hour_ratio:.0%}",
            "Outbound Data": f"{target_window.total_response_bytes / 1024:.1f} KB (Mean: {target_window.mean_response_bytes:.0f} B)",
        }
        st.json(vec_display)

    # =========================================================================
    # RIGHT COLUMN: AI MODEL OUTPUT & PROCESSING
    # =========================================================================
    with col_output:
        st.markdown("### 📤 2. AI Model Output & XAI Attribution")

        # Evaluate target window using embedded or API engine
        eval_result = engine.evaluate_window(target_window)
        is_anomaly = eval_result.get("is_anomaly", False)
        severity = eval_result.get("severity", "INFO")
        confidence = eval_result.get("confidence_score", 0.0)
        tier = eval_result.get("detector_tier", "TIER1_STATISTICAL")
        hypothesis = eval_result.get("threat_hypothesis", "Nominal Traffic")
        narrative = eval_result.get("explanation_narrative", "")
        mitigation = eval_result.get("recommended_mitigation", "")
        t1_info = eval_result.get("tier1_result", {})
        t2_info = eval_result.get("tier2_result", {})

        # 1. Detection Verdict Header
        if is_anomaly:
            st.markdown(
                f"""
                <div style="background-color: rgba(239, 68, 68, 0.15); border: 2px solid #EF4444; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25rem; font-weight: 700; color: #EF4444;">🚨 THREAT DETECTED</span>
                        <span style="background-color: #EF4444; color: #FFFFFF; padding: 3px 10px; border-radius: 12px; font-weight: 700; font-size: 0.8rem;">
                            SEVERITY: {severity}
                        </span>
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #F8FAFC; margin-top: 6px;">
                        {hypothesis}
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                        Confidence Score: <b>{confidence:.1%}</b> | Detection Tier: <b>{tier}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background-color: rgba(16, 185, 129, 0.15); border: 2px solid #10B981; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25rem; font-weight: 700; color: #10B981;">✅ NOMINAL STATUS</span>
                        <span style="background-color: #10B981; color: #000000; padding: 3px 10px; border-radius: 12px; font-weight: 700; font-size: 0.8rem;">
                            SEVERITY: NORMAL
                        </span>
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #F8FAFC; margin-top: 6px;">
                        Healthy Workload Telemetry
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                        All multivariate metrics align within diurnal baseline limits.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Dual-Tier Multi-Engine Arbitration Breakdown
        st.markdown("#### Dual-Tier Processing Breakdown")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            t1_flagged = t1_info.get("flagged", False)
            t1_badge = "🔴 FLAGGED" if t1_flagged else "🟢 PASS"
            st.markdown(
                f"""
                <div style="background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Tier 1: Statistical Filter</div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin: 4px 0;">{t1_badge}</div>
                    <div style="font-size: 0.75rem; color: #CBD5E1;">Z-Score: <code>{t1_info.get('z_score', 0.0):.2f}</code> / Thresh: <code>{t1_info.get('threshold', 3.0):.1f}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_t2:
            t2_flagged = t2_info.get("flagged", False)
            t2_badge = "🔴 OUTLIER" if t2_flagged else "🟢 INLIER"
            st.markdown(
                f"""
                <div style="background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Tier 2: Isolation Forest</div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin: 4px 0;">{t2_badge}</div>
                    <div style="font-size: 0.75rem; color: #CBD5E1;">Score: <code>{t2_info.get('raw_decision_score', 0.0):.4f}</code> / Thresh: <code>{t2_info.get('threshold', -0.014):.3f}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Novelty Component: SHAP TreeExplainer Attribution Output
        st.markdown("#### Novelty Component: Explainable AI (XAI) Attribution")
        render_shap_importance_barchart(
            attributions=eval_result.get("top_contributions", []),
            baseline_expected=eval_result.get("baseline_expected", {}),
        )

        # 4. AI Explanation Narrative & Cloud Mitigation Playbook
        st.markdown(f"**🤖 AI Reason Narrative:** {narrative}")

        if is_anomaly and mitigation:
            st.success(f"🛡️ **Automated Incident Response Playbook:**\n\n{mitigation}")

        # 5. Raw Output JSON Expander
        with st.expander("🔍 View Raw Machine-Readable Model Output"):
            st.json(eval_result)
