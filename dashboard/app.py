"""Temporal Anomaly & Threat Detector - Live Interactive Streamlit Dashboard.

Provides 100% offline, zero-cloud-cost real-time situational awareness:
- Interactive Live Input -> Output Threat Detection Simulator
- Dual-Tier Anomaly Detection (Statistical Filter + Isolation Forest)
- Novelty XAI Explainability (SHAP TreeExplainer Waterfall & Attribution)
- Live Timeline Visualizations with Anomaly Markers
- Parameterized Attack Injection Center
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import streamlit as st
from dashboard.api_client import DashboardApiClient
from dashboard.components.alert_cards import render_alert_feed, render_xai_deep_dive
from dashboard.components.attack_injector import render_attack_injector_panel
from dashboard.components.live_simulator import render_live_threat_simulator
from dashboard.components.metrics_panel import render_kpi_ribbon
from dashboard.components.timeline_chart import render_temporal_timeline
from dashboard.engine_bridge import OfflineDetectionEngine

# Streamlit Page Configuration
st.set_page_config(
    page_title="Temporal Threat Detector | Cloud Security AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Minimalist Dark Cyber Theme Styling
CUSTOM_CSS = """
<style>
    /* Dark glassmorphic background */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #38BDF8;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 18px;
    }
    .status-badge-online {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
    }
    .status-badge-offline {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 600;
        background-color: #111827;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
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


def render_sidebar(client: DashboardApiClient) -> Dict[str, Any]:
    """Render sidebar navigation, backend connectivity status, and settings."""
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 2.0rem;">🛡️</span>
            <div>
                <strong style="font-size: 1.1rem; color: #F8FAFC;">Threat Detector</strong><br>
                <small style="color: #64748B;">v1.0.0 (Offline Ready)</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Connection Check
    is_online = client.is_available()
    if is_online:
        st.sidebar.markdown(
            '<span class="status-badge-online">● FASTAPI BACKEND (ONLINE)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="status-badge-offline">● LOCAL ENGINE (100% OFFLINE)</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Operating in-process with zero cloud or network dependencies.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Architecture")
    st.sidebar.markdown(
        """
        - **Tier 1:** Robust MAD & EWMA ($< 0.5\\text{ ms}$)
        - **Tier 2:** Multivariate Isolation Forest ($9.2\\text{ ms}$)
        - **Arbitration:** Hybrid Ensemble ($100\\%\\text{ Recall}$)
        - **Novelty:** Tree-SHAP XAI Attribution ($3.3\\text{ ms}$)
        """
    )

    st.sidebar.markdown("---")
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
    settings = render_sidebar(client)
    offline_engine = OfflineDetectionEngine.get_instance()

    st.markdown(
        '<div class="main-header">🛡️ Temporal Anomaly & Threat Detector</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Explainable AI Telemetry Protection for Cloud Workloads &bull; Dual-Tier Ensemble Architecture</div>',
        unsafe_allow_html=True,
    )

    # Main Navigation Tabs (Playground First!)
    tab_simulator, tab_overview, tab_xai, tab_inject, tab_arch = st.tabs([
        "🎮 Live Input → Output Simulator",
        "📊 Pipeline & Telemetry Timeline",
        "🔍 XAI Attribution Deep Dive",
        "⚡ Attack Injection Center",
        "🏛️ System Architecture",
    ])

    with tab_simulator:
        render_live_threat_simulator(engine=offline_engine)

    with tab_overview:
        st.subheader("Operational Pipeline KPIs")
        if settings["is_online"]:
            health = client.get_health()
            metrics = client.get_metrics()
            alerts = client.get_alerts(limit=50)
        else:
            health = {
                "status": "healthy",
                "subsystems": {
                    "queue_driver": "in-memory (offline)",
                    "alert_sink_driver": "in-memory (offline)",
                    "archiver_driver": "local-gzip (offline)",
                    "worker_thread_running": True,
                },
            }
            metrics = offline_engine.get_metrics()
            alerts = offline_engine.get_alerts(limit=50)

        render_kpi_ribbon(metrics=metrics, health=health)
        st.markdown("---")
        render_temporal_timeline(alerts=alerts)
        st.markdown("---")
        render_alert_feed(alerts=alerts, client=client)

    with tab_xai:
        alerts = client.get_alerts(limit=50) if settings["is_online"] else offline_engine.get_alerts(limit=50)
        render_xai_deep_dive(alerts=alerts, client=client)

    with tab_inject:
        render_attack_injector_panel(client=client, engine=offline_engine)

    with tab_arch:
        st.subheader("Architectural Topology & Dataflow")
        st.markdown("""
        ```
        [ Cloud Telemetry Generator / Workload Logs ]
                          │
                          ▼
           [ Telemetry Ingestion Worker (Queue) ]
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
        [ S3 / Local Archiver ] [ Sliding Window Aggregator ]
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
            [ Tier 1: Z-Score/EWMA ]   [ Tier 2: Isolation Forest ]
                        │                         │
                        └────────────┬────────────┘
                                     ▼
                        [ Hybrid Threat Ensemble ]
                                     │ (Triggered Anomaly)
                                     ▼
                        [ SHAP TreeExplainer Engine ]
                                     │
                        [ Semantic Reasoning & Playbooks ]
                                     │
                        [ DynamoDB / Alert Sink ]
        ```
        """)

    # Auto-refresh loop
    if settings.get("auto_refresh", False):
        time.sleep(settings.get("refresh_rate", 5))
        st.rerun()


if __name__ == "__main__":
    main()
