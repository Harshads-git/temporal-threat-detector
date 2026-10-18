"""Live Threat Alert cards and SHAP Explainable AI (XAI) deep-dive component."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.api_client import DashboardApiClient


def get_severity_badge(severity: str) -> str:
    """Format severity level into colored markdown badge."""
    sev = severity.upper()
    if sev == "CRITICAL":
        return "🔴 **CRITICAL**"
    elif sev == "HIGH":
        return "🟠 **HIGH**"
    elif sev == "MEDIUM":
        return "🟡 **MEDIUM**"
    elif sev == "LOW":
        return "🔵 **LOW**"
    return "⚪ **INFO**"


def get_sample_xai_alert() -> Dict[str, Any]:
    """Provide a realistic sample alert with SHAP attribution for cold-start demo inspection."""
    return {
        "alert_id": "demo-ddos-alert-001",
        "created_at": "2026-10-18T12:00:00Z",
        "detector_tier": "HYBRID_ENSEMBLE",
        "severity": "CRITICAL",
        "confidence_score": 0.985,
        "anomaly_score": 14.82,
        "decision_threshold": 3.0,
        "threat_hypothesis": "High Volumetric HTTP Flood (DDoS)",
        "explanation_narrative": (
            "Request rate surged to 58.3 req/s (+1450% above baseline) coupled with "
            "unprecedented source IP dispersion across non-standard endpoints."
        ),
        "recommended_mitigation": (
            "Engage AWS Shield / CloudFront rate limiting rules. Apply WAF rate-based rule "
            "thresholding at 2,000 requests per 5-minute rolling window per client IP block."
        ),
        "top_contributions": [
            {"feature_name": "request_count", "observed_value": 3500.0, "baseline_expected": 20.0, "shap_value": 0.385, "importance_rank": 1},
            {"feature_name": "request_rate_per_sec", "observed_value": 58.33, "baseline_expected": 0.33, "shap_value": 0.310, "importance_rank": 2},
            {"feature_name": "unique_source_ips", "observed_value": 3200.0, "baseline_expected": 19.6, "shap_value": 0.185, "importance_rank": 3},
            {"feature_name": "error_rate_5xx", "observed_value": 0.42, "baseline_expected": 0.008, "shap_value": 0.120, "importance_rank": 4},
        ],
    }


def render_alert_feed(alerts: List[Dict[str, Any]], client: Optional[DashboardApiClient] = None) -> None:
    """Render collapsible alert cards for all flagged temporal threats."""
    st.subheader(f"Active Threat Incident Feed ({len(alerts)} alerts)")

    display_alerts = alerts if alerts else [get_sample_xai_alert()]
    if not alerts:
        st.info("No live alerts currently in storage sink. Showing sample reference threat alert for inspection.")

    for alert in display_alerts:
        sev = alert.get("severity", "MEDIUM")
        hyp = alert.get("threat_hypothesis", "Unknown Anomaly")
        aid = alert.get("alert_id", "N/A")[:8]
        conf = alert.get("confidence_score", 0.0)
        tier = alert.get("detector_tier", "TIER1_STATISTICAL")
        ts = alert.get("created_at", "")[:19]

        badge = get_severity_badge(sev)
        expander_title = f"{badge} | {hyp} | Conf: {conf:.0%} | Tier: {tier} ({ts})"

        with st.expander(expander_title, expanded=(sev in ("CRITICAL", "HIGH"))):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Alert ID:** `{alert.get('alert_id')}`")
            c2.markdown(f"**Tier:** `{tier}`")
            c3.markdown(f"**Score:** `{alert.get('anomaly_score', 0.0):.2f}` / Thresh: `{alert.get('decision_threshold', 0.0):.2f}`")
            c4.markdown(f"**Confidence:** `{conf:.1%}`")

            # Explanation narrative
            st.markdown(f"**AI Narrative:** {alert.get('explanation_narrative', 'No narrative generated.')}")

            # Top SHAP attributions preview
            attributions = alert.get("top_contributions", [])
            if attributions:
                st.markdown("**Top Anomalous Drivers (SHAP Attribution):**")
                cols = st.columns(min(4, len(attributions)))
                for idx, attr in enumerate(attributions[:4]):
                    with cols[idx]:
                        fname = attr.get("feature_name", "")
                        obs = attr.get("observed_value", 0.0)
                        exp = attr.get("baseline_expected", 0.0)
                        shap = attr.get("shap_value", 0.0)
                        st.caption(f"**{fname}**")
                        st.write(f"Obs: `{obs}` | Exp: `{exp}`")
                        st.progress(min(1.0, max(0.0, float(shap) * 2.0)))

            # Recommended mitigation playbook
            mitigation = alert.get("recommended_mitigation")
            if mitigation:
                st.success(f"🛡️ **Automated Playbook:** {mitigation}")


def render_xai_waterfall_chart(alert: Dict[str, Any]) -> None:
    """Render Plotly Waterfall chart representing SHAP feature contributions towards decision score."""
    attributions = alert.get("top_contributions", [])
    if not attributions:
        st.warning("No SHAP attribution records available for this incident.")
        return

    features = [a.get("feature_name", f"feat_{i}") for i, a in enumerate(attributions)]
    shap_vals = [a.get("shap_value", 0.0) for a in attributions]

    # Calculate cumulative anomaly contribution
    base_score = 0.05
    measures = ["relative"] * len(shap_vals) + ["total"]
    x_labels = features + ["Final Anomaly Score"]
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
        title=f"SHAP Feature Importance Attribution Waterfall — {alert.get('threat_hypothesis')}",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis_title="SHAP Importance Delta (Log-Odds Contribution)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_xai_deep_dive(alerts: List[Dict[str, Any]], client: Optional[DashboardApiClient] = None) -> None:
    """Render comprehensive XAI Explainability tab with interactive waterfall and mitigation plans."""
    st.subheader("Explainable AI (XAI) Attribution & Incident Response")

    display_alerts = alerts if alerts else [get_sample_xai_alert()]

    alert_options = {
        f"{a.get('alert_id', 'N/A')[:8]} - {a.get('threat_hypothesis', 'Threat')} ({a.get('created_at', '')[:19]})": a
        for a in display_alerts
    }

    selected_key = st.selectbox("Select Incident to Investigate:", list(alert_options.keys()))
    selected_alert = alert_options[selected_key]

    # Render Waterfall visualization
    render_xai_waterfall_chart(selected_alert)

    # Detailed Attribution Table
    st.markdown("#### Feature Deviation Breakdown")
    attributions = selected_alert.get("top_contributions", [])
    if attributions:
        df_attr = pd.DataFrame(attributions)
        st.dataframe(
            df_attr,
            use_container_width=True,
            column_config={
                "importance_rank": st.column_config.NumberColumn("Rank", width="small"),
                "feature_name": st.column_config.TextColumn("Telemetric Feature"),
                "observed_value": st.column_config.NumberColumn("Observed Telemetry"),
                "baseline_expected": st.column_config.NumberColumn("Diurnal Baseline Expected"),
                "shap_value": st.column_config.NumberColumn("SHAP Importance Score", format="%.4f"),
            },
            hide_index=True,
        )

    # Narrative & Incident Response Recommender Playbook
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Semantic AI Hypothesis")
        st.info(f"**Identified Vector:** {selected_alert.get('threat_hypothesis')}\n\n{selected_alert.get('explanation_narrative')}")

    with col_right:
        st.markdown("#### Cloud Mitigation Playbook")
        st.success(f"**Action Plan:**\n\n{selected_alert.get('recommended_mitigation')}")
