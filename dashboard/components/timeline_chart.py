"""Interactive multi-series temporal timeline visualizer with anomaly markers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def generate_fallback_timeline_data(count: int = 30) -> pd.DataFrame:
    """Generate synthetic historical window timeline data for cold-start visualization."""
    now = datetime.now(timezone.utc)
    records = []
    for i in range(count):
        ts = now - timedelta(seconds=(count - i) * 10)
        # Baseline traffic with minor variance
        req_count = 20 + int(5 * (i % 4 - 2))
        err_4xx = 1 if i % 7 == 0 else 0
        err_5xx = 1 if i % 15 == 0 else 0
        p95_ms = 60.0 + (i % 5) * 4.0
        records.append({
            "timestamp": ts,
            "request_count": max(5, req_count),
            "request_rate": round(req_count / 10.0, 2),
            "error_rate": round((err_4xx + err_5xx) / max(1, req_count) * 100.0, 2),
            "p95_duration_ms": p95_ms,
            "is_anomaly": False,
            "severity": "NORMAL",
            "threat_label": "Nominal",
        })
    return pd.DataFrame(records)


def render_temporal_timeline(
    windows_data: Optional[List[Dict[str, Any]]] = None,
    alerts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render dual-axis Plotly timeline with request rates, latency, and anomaly markers."""
    st.subheader("Temporal Traffic & Anomaly Timeline")

    # Prepare DataFrame
    if windows_data and len(windows_data) > 0:
        df = pd.DataFrame(windows_data)
        if "window_end" in df.columns:
            df["timestamp"] = pd.to_datetime(df["window_end"])
        elif "timestamp" not in df.columns:
            df["timestamp"] = [datetime.now(timezone.utc) - timedelta(seconds=(len(df)-i)*10) for i in range(len(df))]
    else:
        df = generate_fallback_timeline_data()

    # Match alerts with timeline timestamps
    alert_lookup = {}
    if alerts:
        for a in alerts:
            ts_str = a.get("created_at") or a.get("timestamp")
            if ts_str:
                alert_lookup[ts_str[:19]] = a

    # Create subplots: Top = Request Volume & Latency, Bottom = Error Rate
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Workload Request Volume (Req/s) & P95 Latency (ms)", "Error Rate (%)"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # 1. Request Rate / Count Trace
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df.get("request_rate", df.get("request_count", 0)),
            name="Request Rate (req/s)",
            line=dict(color="#00E5FF", width=2.5),
            mode="lines+markers",
            marker=dict(size=4),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 2. P95 Duration Trace (Secondary Y-Axis)
    if "p95_duration_ms" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["p95_duration_ms"],
                name="P95 Latency (ms)",
                line=dict(color="#FFB300", width=1.5, dash="dot"),
                mode="lines",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    # 3. Error Rate Trace (Bottom Subplot)
    err_col = "error_rate" if "error_rate" in df.columns else "error_count_4xx"
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df.get(err_col, 0),
            name="Error Rate (%)",
            marker_color="#F43F5E",
            opacity=0.75,
        ),
        row=2,
        col=1,
    )

    # 4. Anomaly Markers Overlay
    if alerts and len(alerts) > 0:
        anom_x = []
        anom_y = []
        hover_texts = []
        for a in alerts:
            ts = a.get("created_at") or a.get("timestamp")
            if ts:
                anom_x.append(ts)
                anom_y.append(a.get("anomaly_score", 1.0))
                hover_texts.append(
                    f"<b>THREAT DETECTED</b><br>"
                    f"Tier: {a.get('detector_tier')}<br>"
                    f"Severity: {a.get('severity')}<br>"
                    f"Hypothesis: {a.get('threat_hypothesis')}<br>"
                    f"Confidence: {a.get('confidence_score')}"
                )

        if anom_x:
            fig.add_trace(
                go.Scatter(
                    x=anom_x,
                    y=[df["request_rate"].max() if "request_rate" in df.columns else 25.0] * len(anom_x),
                    name="Anomaly Flagged",
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="#EF4444",
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
    )

    fig.update_yaxes(title_text="Req/s", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Errors (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
