"""Temporal Anomaly & Threat Detector - Live Interactive Streamlit Dashboard.

Provides real-time situational awareness for cloud workloads:
- Dual-Tier Anomaly Detection (Statistical Filter + Isolation Forest)
- Novelty XAI Explainability (SHAP TreeExplainer Waterfall & Attribution)
- Live Timeline Visualizations with Anomaly Markers
- Interactive Attack Injection Control Panel
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
from dashboard.components.metrics_panel import render_kpi_ribbon
from dashboard.components.timeline_chart import render_temporal_timeline

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
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00E5FF;
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #90A4AE;
        margin-bottom: 20px;
    }
    .status-badge-online {
        background-color: #00E676;
        color: #000000;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .status-badge-offline {
        background-color: #FF1744;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
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
    .alert-critical {
        border-left: 5px solid #EF4444;
        background-color: rgba(239, 68, 68, 0.1);
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .alert-high {
        border-left: 5px solid #F97316;
        background-color: rgba(249, 115, 22, 0.1);
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 10px;
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
    st.sidebar.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/AWS-Dark.svg",
        width=50,
    )
    st.sidebar.title("Configuration")

    # API Connection Check
    is_online = client.is_available()
    if is_online:
        st.sidebar.markdown(
            '<span class="status-badge-online">● BACKEND CONNECTED</span>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="status-badge-offline">● BACKEND OFFLINE</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.warning(
            "FastAPI backend is offline or starting up.\n\n"
            "Run in terminal:\n"
            "`uvicorn api.main:app --port 8000`"
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Telemetry Engine")
    st.sidebar.info(
        "**Dual-Tier Engine Active**\n\n"
        "- Tier 1: Sliding MAD / EWMA Filter\n"
        "- Tier 2: Unsupervised Isolation Forest\n"
        "- Explainability: SHAP TreeExplainer"
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

    st.markdown(
        '<div class="main-header">🛡️ Temporal Anomaly & Threat Detector</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Multi-Tier Explainable AI Telemetry Protection for Cloud Workloads</div>',
        unsafe_allow_html=True,
    )

    # Main Navigation Tabs
    tab_overview, tab_xai, tab_inject, tab_arch = st.tabs([
        "📊 Live Pipeline & Alerts",
        "🔍 XAI Attribution Deep Dive",
        "⚡ Attack Injection Center",
        "🏛️ System Architecture",
    ])

    with tab_overview:
        st.subheader("Operational Pipeline KPIs")
        alerts = []
        if settings["is_online"]:
            health = client.get_health()
            metrics = client.get_metrics()
            alerts = client.get_alerts(limit=50)
            render_kpi_ribbon(metrics=metrics, health=health)
        else:
            st.error("Cannot load operational overview while backend is disconnected.")
            render_kpi_ribbon(
                metrics={"total_events_processed": 0, "queue_depth": 0, "total_alerts_emitted": 0, "tier1_zscore_alerts": 0, "tier2_iforest_alerts": 0, "hybrid_alerts": 0, "mean_pipeline_latency_ms": 0.0},
                health={"subsystems": {"queue_driver": "offline", "alert_sink_driver": "offline", "archiver_driver": "offline", "worker_thread_running": False}},
            )

        st.markdown("---")
        render_temporal_timeline(alerts=alerts)

        st.markdown("---")
        render_alert_feed(alerts=alerts, client=client)

    with tab_xai:
        render_xai_deep_dive(alerts=alerts, client=client)

    with tab_inject:
        render_attack_injector_panel(client=client)

    with tab_arch:
        st.subheader("Architectural Topology")
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
