"""Interactive attack injection control panel and stress-testing simulator."""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st
from dashboard.api_client import DashboardApiClient


def render_attack_injector_panel(client: DashboardApiClient) -> None:
    """Render interactive attack injection triggers and parameter controls."""
    st.subheader("Interactive Attack Simulator & Traffic Injector")
    st.markdown(
        "Inject controlled synthetic cyber threats into the active telemetry queue "
        "to test real-time detection, XAI explanation, and mitigation playbooks."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 1. Select Attack Vector")
        attack_type = st.radio(
            "Attack Vector:",
            options=["ddos", "credential_stuffing", "off_hour_probing", "data_exfiltration"],
            format_func=lambda x: {
                "ddos": "🌊 High Volumetric HTTP Flood (DDoS)",
                "credential_stuffing": "🔑 Credential Stuffing & Auth Flooding",
                "off_hour_probing": "🕵️ Off-Hour Reconnaissance & Probing",
                "data_exfiltration": "💾 Massive Data Exfiltration",
            }[x],
        )

        burst_profile = st.selectbox(
            "Burst Multiplier Profile:",
            options=["sudden_spike", "gradual_ramp", "periodic_pulses", "constant"],
            format_func=lambda x: {
                "sudden_spike": "⚡ Sudden Spike (Step function)",
                "gradual_ramp": "📈 Gradual Ramp (Linear growth)",
                "periodic_pulses": "〰️ Periodic Pulses (Sinusoidal waves)",
                "constant": "➖ Constant Multiplier",
            }[x],
        )

        count = st.slider(
            "Event Batch Size (Telemetry logs to generate):",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
        )

    with col2:
        st.markdown("### 2. Threat Vector Description")
        vector_descriptions = {
            "ddos": (
                "**Attack Profile: Volumetric HTTP Flood**\n\n"
                "- **Impact:** Saturates web tiers with rapid-fire GET/POST requests.\n"
                "- **Expected Detection:** Tier 1 Z-Score flags request rate spike; Tier 2 flags IP dispersion.\n"
                "- **Expected XAI Attributions:** `request_count`, `request_rate_per_sec` dominant."
            ),
            "credential_stuffing": (
                "**Attack Profile: Credential Stuffing**\n\n"
                "- **Impact:** Repeated failed authentication attempts targeting `/api/v1/auth/login`.\n"
                "- **Expected Detection:** Tier 1 & Tier 2 flag abnormal 401 error ratios and auth failure spikes.\n"
                "- **Expected XAI Attributions:** `auth_failure_count`, `error_rate_4xx` dominant."
            ),
            "off_hour_probing": (
                "**Attack Profile: Off-Hour Reconnaissance**\n\n"
                "- **Impact:** Anomalous exploration of administrative endpoints during low-traffic night hours.\n"
                "- **Expected Detection:** Tier 2 Multivariate Isolation Forest catches off-hour ratio anomaly.\n"
                "- **Expected XAI Attributions:** `off_hour_ratio`, `mean_duration_ms` dominant."
            ),
            "data_exfiltration": (
                "**Attack Profile: Data Exfiltration**\n\n"
                "- **Impact:** Bulk download of sensitive datasets from `/api/v1/reports/export`.\n"
                "- **Expected Detection:** Tier 1 & Tier 2 flag massive response byte surges.\n"
                "- **Expected XAI Attributions:** `total_response_bytes`, `mean_response_bytes` dominant."
            ),
        }
        st.info(vector_descriptions[attack_type])

        st.markdown("### 3. Dispatch Threat")
        if st.button("🔥 INJECT THREAT INTO LIVE WORKLOAD", type="primary", use_container_width=True):
            if not client.is_available():
                st.error("Cannot inject attack: FastAPI backend is offline.")
            else:
                with st.spinner("Generating attack telemetry and publishing to queue..."):
                    result = client.inject_attack(
                        attack_type=attack_type,
                        burst_profile=burst_profile,
                        count=count,
                    )

                if result.get("status") in ("injected", "accepted"):
                    st.success(
                        f"✅ **Attack Injected Successfully!**\n\n"
                        f"- **Type:** `{attack_type}`\n"
                        f"- **Count:** `{count}` events dispatched\n"
                        f"- **Batch ID:** `{result.get('batch_id')}`\n\n"
                        "The background worker will aggregate these events into the sliding window and trigger alerts."
                    )
                    st.toast(f"Injected {count} {attack_type} events!", icon="🔥")
                else:
                    st.error(f"Failed to inject attack: {result.get('message', 'Unknown error')}")
