"""Temporal Anomaly & Threat Detector - Live Interactive Streamlit Dashboard.

Provides 100% offline-capable, real-time situational awareness for cloud workloads:
- Interactive Input ➔ AI Output Lab (User parameter controls & Dual-Tier output)
- Dual-Tier Anomaly Detection (Statistical Filter + Isolation Forest)
- Novelty XAI Explainability (SHAP TreeExplainer Waterfall & Attribution)
- Live Timeline Visualizations with Anomaly Markers
- Interactive Attack Injection Control Panel
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dashboard.api_client import DashboardApiClient
from dashboard.components.alert_cards import render_alert_feed, render_xai_deep_dive
from dashboard.components.attack_injector import render_attack_injector_panel
from dashboard.components.interactive_lab import render_interactive_lab
from dashboard.components.metrics_panel import render_kpi_ribbon
from dashboard.components.timeline_chart import render_temporal_timeline
from dashboard.embedded_engine import EmbeddedOfflinePipeline, get_embedded_pipeline

# Streamlit Page Configuration
st.set_page_config(
    page_title="Temporal Threat Detector | Cloud Security AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber Security UI Theme Styling
CUSTOM_CSS = """
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00E5FF;
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 22px;
    }
    .status-badge-online {
        background-color: #00E676;
        color: #000000;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-badge-standalone {
        background: linear-gradient(90deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        border: 1px solid #38BDF8;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .kpi-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-top: 4px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_api_client() -> DashboardApiClient:
    """Retrieve or initialize DashboardApiClient in Streamlit session state."""
    if "api_client" not in st.session_state:
        api_url = os.getenv("DETECTOR_API_URL", "http://127.0.0.1:8000")
        st.session_state.api_client = DashboardApiClient(base_url=api_url)
    return st.session_state.api_client


def render_sidebar(is_online: bool) -> Dict[str, Any]:
    """Render sidebar navigation, backend connectivity status, and settings."""
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 2rem;">🛡️</span>
            <div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC;">Threat Detector</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Multi-Tier Telemetry AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")

    if is_online:
        st.sidebar.markdown(
            '<span class="status-badge-online">● BACKEND CONNECTED (PORT 8000)</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption("FastAPI REST backend is actively servicing live requests.")
    else:
        st.sidebar.markdown(
            '<span class="status-badge-standalone">● STANDALONE OFFLINE ENGINE</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            "Running 100% locally in-memory. Zero internet connectivity or external servers required."
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Architecture")
    st.sidebar.markdown(
        """
        - **Tier 1:** Sliding MAD & EWMA Z-Score Filter *(O(1) volumetric rate check)*
        - **Tier 2:** Unsupervised Isolation Forest *(Multivariate outlier detector)*
        - **Ensemble:** Disjunctive Arbitration with Dynamic Thresholding
        - **Explainability:** TreeExplainer SHAP *(Additive feature attribution)*
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Controls")
    refresh_rate = st.sidebar.slider("Auto-Refresh Rate (seconds)", 2, 30, 5)
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

    return {
        "is_online": is_online,
        "refresh_rate": refresh_rate,
        "auto_refresh": auto_refresh,
    }


def main() -> None:
    """Main application entry point."""
    client = get_api_client()
    is_online = client.is_available()

    # Retrieve cached embedded engine (100% offline capable)
    embedded_engine = get_embedded_pipeline()

    # Configure active engine based on availability
    active_engine = client if is_online else embedded_engine

    settings = render_sidebar(is_online)

    # Header Ribbon
    st.markdown(
        '<div class="main-header">🛡️ Temporal Anomaly & Threat Detector</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Multi-Tier Explainable AI Telemetry Protection for Cloud Workloads &bull; 100% Offline Capable</div>',
        unsafe_allow_html=True,
    )

    # Main Navigation Tabs - Tab 1 spotlighting the User Input ➔ Model Output Lab
    tab_lab, tab_overview, tab_xai, tab_inject, tab_arch = st.tabs([
        "🎯 Input ➔ Output Interactive Lab",
        "📊 Real-Time Timeline & Metrics",
        "🔍 SHAP Waterfall Deep Dive",
        "⚡ Attack Injection Center",
        "🏛️ Architecture & Math",
    ])

    # =========================================================================
    # TAB 1: INTERACTIVE USER INPUT ➔ AI OUTPUT LAB
    # =========================================================================
    with tab_lab:
        render_interactive_lab(engine=embedded_engine)

    # =========================================================================
    # TAB 2: LIVE PIPELINE OVERVIEW & TIMELINE
    # =========================================================================
    with tab_overview:
        st.subheader("Operational Pipeline KPIs")

        if is_online:
            health = client.get_health()
            metrics = client.get_metrics()
            alerts = client.get_alerts(limit=50)
            timeline_data = None
        else:
            health = embedded_engine.get_health()
            metrics = embedded_engine.get_metrics()
            alerts = embedded_engine.get_alerts(limit=50)
            timeline_data = embedded_engine.recent_windows

        render_kpi_ribbon(metrics=metrics, health=health)

        st.markdown("---")
        render_temporal_timeline(windows_data=timeline_data, alerts=alerts)

        st.markdown("---")
        render_alert_feed(alerts=alerts, client=client if is_online else None)

    # =========================================================================
    # TAB 3: SHAP WATERFALL & EXPLAINABILITY DEEP DIVE
    # =========================================================================
    with tab_xai:
        render_xai_deep_dive(alerts=alerts, client=client if is_online else None)

    # =========================================================================
    # TAB 4: ATTACK INJECTION CENTER
    # =========================================================================
    with tab_inject:
        render_attack_injector_panel(client=active_engine)

    # =========================================================================
    # TAB 5: SYSTEM ARCHITECTURE & MATHEMATICAL FORMULATIONS
    # =========================================================================
    with tab_arch:
        st.subheader("Multi-Tier Architectural Topology")
        st.markdown(
            """
            ```
            ┌────────────────────────────────────────────────────────┐
            │       Cloud Telemetry Generator / Workload Logs        │
            │   (API requests, 4xx/5xx errors, latency, bytes, auth) │
            └───────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
            ┌────────────────────────────────────────────────────────┐
            │        Telemetry Ingestion Worker (Local / SQS)        │
            └───────────────────────────┬────────────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
            ┌─────────────────────────┐   ┌──────────────────────────┐
            │ Local / S3 Raw Archiver │   │ Sliding Window Aggregator│
            │  (Forensic audit trail) │   │  (60s window, 10s hop)   │
            └─────────────────────────┘   └────────────┬─────────────┘
                                                       │
                                  ┌────────────────────┴────────────────────┐
                                  ▼                                         ▼
            ┌──────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
            │       Tier 1: Statistical Filter         │ │     Tier 2: Machine Learning Filter      │
            │  - Median Absolute Deviation (MAD)       │ │  - Unsupervised Isolation Forest        │
            │  - Exponentially Weighted Moving Avg     │ │  - Empirical Quantile Thresholding       │
            │  - O(1) Streaming Evaluation             │ │  - Multivariate Correlation Analysis     │
            └─────────────────────┬────────────────────┘ └────────────────────┬─────────────────────┘
                                  │                                         │
                                  └────────────────────┬────────────────────┘
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │         Hybrid Threat Ensemble          │
                                  │   (Disjunctive Arbitration Engine)      │
                                  └────────────────────┬────────────────────┘
                                                       │ (Triggered Anomaly)
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │       SHAP TreeExplainer Engine         │
                                  │  (Additive Shapley Feature Attribution) │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │      Semantic Reasoning & Playbook      │
                                  │ (Translates SHAP deltas to remediation) │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │   Alert Sink (DynamoDB / In-Memory)     │
                                  │      + Interactive Streamlit UI         │
                                  └─────────────────────────────────────────┘
            ```
            """
        )

        st.markdown("### Detection Mathematics & Formulations")
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown(
                """
                #### Tier 1: Robust Modified Z-Score
                $$M_i = \\frac{0.6745 \\cdot (x_i - \\tilde{x})}{\\text{MAD}}$$
                Where $\\tilde{x} = \\text{median}(X)$, and:
                $$\\text{MAD} = \\text{median}(|x_i - \\tilde{x}|)$$
                Dynamic Volatility EWMA Variance:
                $$\\sigma_t^2 = \\alpha (x_t - \\mu_t)^2 + (1 - \\alpha) \\sigma_{t-1}^2$$
                """
            )
        with c_m2:
            st.markdown(
                """
                #### Tier 2 & XAI: Isolation Forest & SHAP
                Anomaly Score from Average Path Length $E(h(x))$:
                $$s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}}$$
                Local SHAP Feature Attribution (Shapley Value):
                $$\\phi_i(x) = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \\cup \\{i\\}) - f(S)]$$
                """
            )

    # Auto-refresh loop
    if settings.get("auto_refresh", False):
        time.sleep(settings.get("refresh_rate", 5))
        st.rerun()


if __name__ == "__main__":
    main()
