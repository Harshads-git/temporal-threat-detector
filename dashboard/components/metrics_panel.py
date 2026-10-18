"""Operational KPI metrics ribbon and subsystem status panel for Streamlit dashboard."""

from __future__ import annotations

from typing import Any, Dict
import streamlit as st


def render_kpi_ribbon(metrics: Dict[str, Any], health: Dict[str, Any]) -> None:
    """Render high-level operational KPIs and processing statistics."""
    total_events = metrics.get("total_events_processed", 0)
    queue_depth = metrics.get("queue_depth", 0)
    total_alerts = metrics.get("total_alerts_emitted", 0)
    t1_alerts = metrics.get("tier1_zscore_alerts", 0)
    t2_alerts = metrics.get("tier2_iforest_alerts", 0)
    hybrid_alerts = metrics.get("hybrid_alerts", 0)
    latency_ms = metrics.get("mean_pipeline_latency_ms", 0.0)

    # Top KPI metric cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Telemetry Ingested",
            value=f"{total_events:,}",
            help="Total telemetry log events processed through temporal windows",
        )
    with col2:
        st.metric(
            label="Queue Depth",
            value=f"{queue_depth}",
            delta="-0" if queue_depth == 0 else f"{queue_depth} pending",
            delta_color="normal" if queue_depth < 100 else "inverse",
            help="Active messages pending in the event ingestion queue",
        )
    with col3:
        st.metric(
            label="Total Alerts",
            value=f"{total_alerts}",
            delta=f"{hybrid_alerts} hybrid",
            delta_color="inverse" if total_alerts > 0 else "off",
            help="Total temporal anomalies flagged by the detection ensemble",
        )
    with col4:
        st.metric(
            label="Tier 1 (Z-Score)",
            value=f"{t1_alerts}",
            help="Volumetric rate and surge anomalies caught by statistical filters",
        )
    with col5:
        st.metric(
            label="Tier 2 (AI iForest)",
            value=f"{t2_alerts}",
            help="Multivariate contextual anomalies caught by Isolation Forest",
        )

    # Subsystem status badges and latency strip
    subsystems = health.get("subsystems", {})
    queue_driver = subsystems.get("queue_driver", "in-memory")
    alert_driver = subsystems.get("alert_sink_driver", "in-memory")
    archiver_driver = subsystems.get("archiver_driver", "in-memory")
    worker_running = subsystems.get("worker_thread_running", True)

    st.markdown("---")
    scol1, scol2, scol3, scol4, scol5 = st.columns(5)
    with scol1:
        st.caption(f"**Queue Driver:** `{queue_driver}`")
    with scol2:
        st.caption(f"**Alert Sink:** `{alert_driver}`")
    with scol3:
        st.caption(f"**Archiver Sink:** `{archiver_driver}`")
    with scol4:
        worker_badge = "🟢 ACTIVE" if worker_running else "🔴 STOPPED"
        st.caption(f"**Worker Thread:** {worker_badge}")
    with scol5:
        st.caption(f"**Mean Latency:** `{latency_ms:.1f} ms`")
