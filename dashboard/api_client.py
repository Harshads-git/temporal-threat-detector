"""Dashboard API client for communication with FastAPI Threat Detector backend.

Provides typed, resilient HTTP interactions with connection caching, timeouts,
and graceful offline fallback handling.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("dashboard.api_client")


class DashboardApiClient:
    """HTTP Client wrapping REST endpoints of Temporal Threat Detector API."""

    def __init__(self, base_url: Optional[str] = None, timeout_sec: float = 3.0) -> None:
        """Initialize API client with base URL and request timeout."""
        self.base_url = (base_url or os.getenv("DETECTOR_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        """Quick connectivity probe to check if backend API service is responsive."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout_sec)
            return resp.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Dict[str, Any]:
        """Fetch backend health status and subsystem driver information."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout_sec)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except Exception as err:
            return {"status": "offline", "error": str(err)}

    def get_metrics(self) -> Dict[str, Any]:
        """Fetch live operational pipeline KPIs and processing rates."""
        try:
            resp = requests.get(f"{self.base_url}/metrics", timeout=self.timeout_sec)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "error": f"HTTP {resp.status_code}"}
        except Exception as err:
            return {"status": "offline", "error": str(err)}

    def get_alerts(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
        detector_tier: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch recent threat alerts with optional severity and tier filters."""
        params: Dict[str, Any] = {"limit": limit}
        if severity:
            params["severity"] = severity
        if detector_tier:
            params["detector_tier"] = detector_tier

        try:
            resp = requests.get(f"{self.base_url}/alerts", params=params, timeout=self.timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("alerts", [])
            return []
        except Exception as err:
            logger.warning("Failed to fetch alerts from backend: %s", err)
            return []

    def get_alert_detail(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed record and XAI attribution for a specific alert."""
        try:
            resp = requests.get(f"{self.base_url}/alerts/{alert_id}", timeout=self.timeout_sec)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as err:
            logger.warning("Failed to fetch alert detail for %s: %s", alert_id, err)
            return None

    def get_xai_waterfall(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Fetch SHAP waterfall breakdown payload for an alert."""
        try:
            resp = requests.get(f"{self.base_url}/xai/waterfall/{alert_id}", timeout=self.timeout_sec)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as err:
            logger.warning("Failed to fetch XAI waterfall for %s: %s", alert_id, err)
            return None

    def inject_attack(
        self,
        attack_type: str,
        burst_profile: str = "sudden_spike",
        count: int = 200,
    ) -> Dict[str, Any]:
        """Trigger dynamic attack pattern injection into live ingestion queue."""
        payload = {
            "attack_type": attack_type,
            "burst_profile": burst_profile,
            "count": count,
        }
        try:
            resp = requests.post(f"{self.base_url}/inject", json=payload, timeout=self.timeout_sec * 2)
            if resp.status_code == 202:
                return resp.json()
            return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as err:
            return {"status": "offline", "message": str(err)}

    def detect_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a temporal window for immediate on-demand dual-tier evaluation."""
        payload = {"window": window_data}
        try:
            resp = requests.post(f"{self.base_url}/detect", json=payload, timeout=self.timeout_sec * 2)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as err:
            return {"status": "offline", "message": str(err)}
