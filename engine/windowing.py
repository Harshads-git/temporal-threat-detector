"""Temporal Windowing and Feature Aggregation Engine.

Provides sliding-window and tumbling-window time-series aggregation for cloud telemetry logs.
Transforms raw telemetry streams into multivariate temporal feature vectors (WindowAggregate)
consumed by Tier-1 statistical filters and Tier-2 unsupervised ML detectors.
"""

from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from api.schemas import (
    CloudTelemetryLog,
    UserAgentCategory,
    WindowAggregate,
)

# Standardized ordered feature column list for ML models & XAI
CANONICAL_FEATURE_NAMES: List[str] = [
    "request_count",
    "request_rate_per_sec",
    "error_rate_4xx",
    "error_rate_5xx",
    "mean_duration_ms",
    "p95_duration_ms",
    "unique_source_ips",
    "unique_users",
    "total_response_bytes",
    "mean_response_bytes",
    "auth_failure_count",
    "off_hour_ratio",
    "bot_agent_ratio",
]


class WindowType(str, Enum):
    """Temporal windowing interval strategy."""
    SLIDING = "SLIDING"      # Overlapping windows (e.g. 60s window advancing by 10s hop)
    TUMBLING = "TUMBLING"    # Non-overlapping contiguous windows (hop == window size)


def compute_window_features(
    events: List[CloudTelemetryLog],
    window_start: datetime,
    window_end: datetime,
) -> WindowAggregate:
    """Compute normalized behavioral and volumetric feature aggregates across an event slice.

    Args:
        events: Telemetry log records belonging to the window slice.
        window_start: UTC timestamp marking start of window interval.
        window_end: UTC timestamp marking end of window interval.

    Returns:
        WindowAggregate instance with calculated statistical metrics.
    """
    duration_seconds = max(0.001, (window_end - window_start).total_seconds())
    total_requests = len(events)

    if total_requests == 0:
        return WindowAggregate(
            window_start=window_start,
            window_end=window_end,
            duration_seconds=duration_seconds,
            request_count=0,
            request_rate_per_sec=0.0,
            error_count_4xx=0,
            error_count_5xx=0,
            error_rate_4xx=0.0,
            error_rate_5xx=0.0,
            mean_duration_ms=0.0,
            p95_duration_ms=0.0,
            unique_source_ips=0,
            unique_users=0,
            total_request_bytes=0,
            total_response_bytes=0,
            mean_response_bytes=0.0,
            auth_failure_count=0,
            off_hour_ratio=0.0,
            bot_agent_ratio=0.0,
        )

    # Volumetric and error calculations
    count_4xx = 0
    count_5xx = 0
    auth_failures = 0
    bot_agents = 0
    off_hour_count = 0
    total_req_bytes = 0
    total_resp_bytes = 0
    durations: List[float] = []
    source_ips = set()
    users = set()

    security_error_codes = {
        "AccessDenied",
        "InvalidCredentials",
        "AccountLocked",
        "UnauthorizedOperation",
        "AuthFailure",
    }

    for ev in events:
        # HTTP Error categorization
        if 400 <= ev.status_code < 500:
            count_4xx += 1
        elif 500 <= ev.status_code < 600:
            count_5xx += 1

        # Authentication / authorization failures
        if ev.status_code in (401, 403) or (ev.error_code in security_error_codes):
            auth_failures += 1

        # User-agent category
        if ev.user_agent_category == UserAgentCategory.AUTOMATED_BOT:
            bot_agents += 1

        # Off-hour detection (00:00 to 06:00 UTC)
        if 0 <= ev.timestamp.hour < 6:
            off_hour_count += 1

        # Cardinality & payloads
        source_ips.add(ev.source_ip)
        users.add(ev.user_identity)
        total_req_bytes += ev.request_bytes
        total_resp_bytes += ev.response_bytes
        durations.append(ev.duration_ms)

    # Latency percentiles
    mean_duration = float(np.mean(durations)) if durations else 0.0
    p95_duration = float(np.percentile(durations, 95)) if durations else 0.0

    return WindowAggregate(
        window_start=window_start,
        window_end=window_end,
        duration_seconds=duration_seconds,
        request_count=total_requests,
        request_rate_per_sec=round(total_requests / duration_seconds, 4),
        error_count_4xx=count_4xx,
        error_count_5xx=count_5xx,
        error_rate_4xx=round(count_4xx / total_requests, 4),
        error_rate_5xx=round(count_5xx / total_requests, 4),
        mean_duration_ms=round(mean_duration, 2),
        p95_duration_ms=round(p95_duration, 2),
        unique_source_ips=len(source_ips),
        unique_users=len(users),
        total_request_bytes=total_req_bytes,
        total_response_bytes=total_resp_bytes,
        mean_response_bytes=round(total_resp_bytes / total_requests, 2),
        auth_failure_count=auth_failures,
        off_hour_ratio=round(off_hour_count / total_requests, 4),
        bot_agent_ratio=round(bot_agents / total_requests, 4),
    )


class TemporalWindowAggregator:
    """Sliding and tumbling window aggregator for batch and streaming telemetry."""

    def __init__(
        self,
        window_size_seconds: float = 60.0,
        hop_seconds: float = 10.0,
        window_type: WindowType = WindowType.SLIDING,
    ) -> None:
        """Initialize temporal window aggregator.

        Args:
            window_size_seconds: Duration of each temporal window (default 60s).
            hop_seconds: Hop step / advancement interval for sliding windows (default 10s).
            window_type: SLIDING or TUMBLING. (If TUMBLING, hop_seconds equals window_size_seconds).
        """
        self.window_size_seconds = max(1.0, float(window_size_seconds))
        self.window_type = window_type

        if window_type == WindowType.TUMBLING:
            self.hop_seconds = self.window_size_seconds
        else:
            self.hop_seconds = max(1.0, min(self.window_size_seconds, float(hop_seconds)))

    def aggregate_batch(
        self,
        events: List[CloudTelemetryLog],
        min_events_per_window: int = 1,
    ) -> List[WindowAggregate]:
        """Aggregate a chronological collection of events into rolling time windows.

        Args:
            events: Telemetry events to slice and aggregate.
            min_events_per_window: Minimum events required to emit a window (default 1).

        Returns:
            List of WindowAggregate models ordered chronologically.
        """
        if not events:
            return []

        # Sort chronologically by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        first_time = sorted_events[0].timestamp
        last_time = sorted_events[-1].timestamp

        # Snap start time to aligned boundary
        start_epoch = first_time.timestamp()
        end_epoch = last_time.timestamp()

        windows: List[WindowAggregate] = []
        window_duration = timedelta(seconds=self.window_size_seconds)
        hop_duration = timedelta(seconds=self.hop_seconds)

        # Slide window from first_time to last_time
        curr_start = first_time
        event_idx = 0
        n_events = len(sorted_events)

        while curr_start <= last_time:
            curr_end = curr_start + window_duration

            # Collect events within [curr_start, curr_end)
            # Find matching events efficiently
            window_events = [
                ev for ev in sorted_events
                if curr_start <= ev.timestamp < curr_end
            ]

            if len(window_events) >= min_events_per_window:
                agg = compute_window_features(window_events, curr_start, curr_end)
                windows.append(agg)

            curr_start += hop_duration

        return windows

    @staticmethod
    def to_dataframe(windows: List[WindowAggregate]) -> pd.DataFrame:
        """Convert a list of WindowAggregate instances into a Pandas DataFrame."""
        if not windows:
            return pd.DataFrame(columns=CANONICAL_FEATURE_NAMES)
        records = [w.model_dump() for w in windows]
        df = pd.DataFrame(records)
        df["window_start"] = pd.to_datetime(df["window_start"])
        df["window_end"] = pd.to_datetime(df["window_end"])
        return df

    @staticmethod
    def to_feature_matrix(
        windows: List[WindowAggregate],
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract canonical 2D numeric feature matrix for ML models.

        Returns:
            Tuple of (numpy ndarray of shape [N, 13], list of feature column names).
        """
        if not windows:
            return np.empty((0, len(CANONICAL_FEATURE_NAMES)), dtype=np.float64), CANONICAL_FEATURE_NAMES

        rows = []
        for w in windows:
            fmap = w.to_feature_vector()
            row = [fmap[col] for col in CANONICAL_FEATURE_NAMES]
            rows.append(row)

        return np.array(rows, dtype=np.float64), list(CANONICAL_FEATURE_NAMES)


class StreamingWindowBuffer:
    """In-memory sliding temporal FIFO buffer for live streaming pipelines."""

    def __init__(
        self,
        window_size_seconds: float = 60.0,
        hop_seconds: float = 10.0,
    ) -> None:
        """Initialize streaming buffer.

        Args:
            window_size_seconds: Active window span.
            hop_seconds: Interval between consecutive emitted windows.
        """
        self.window_size_seconds = max(1.0, float(window_size_seconds))
        self.hop_seconds = max(1.0, min(self.window_size_seconds, float(hop_seconds)))
        self._buffer: deque[CloudTelemetryLog] = deque()
        self._last_emitted_window_end: Optional[datetime] = None

    def add_event(self, event: CloudTelemetryLog) -> None:
        """Ingest a new telemetry event into the buffer."""
        self._buffer.append(event)
        self._prune_expired()

    def add_events(self, events: List[CloudTelemetryLog]) -> None:
        """Ingest a batch of events into the buffer."""
        for ev in events:
            self._buffer.append(ev)
        self._prune_expired()

    def _prune_expired(self) -> None:
        """Prune records older than the maximum window retention watermark."""
        if not self._buffer:
            return

        latest_time = max(e.timestamp for e in self._buffer)
        watermark = latest_time - timedelta(seconds=self.window_size_seconds * 3.0)

        # Remove elements older than watermark from head of deque
        while self._buffer and self._buffer[0].timestamp < watermark:
            self._buffer.popleft()

    def check_and_emit_windows(self) -> List[WindowAggregate]:
        """Evaluate buffer contents and emit any mature sliding windows."""
        if not self._buffer:
            return []

        latest_time = max(e.timestamp for e in self._buffer)
        oldest_time = min(e.timestamp for e in self._buffer)

        # First window boundary initialization
        if self._last_emitted_window_end is None:
            self._last_emitted_window_end = oldest_time + timedelta(seconds=self.window_size_seconds)

        emitted: List[WindowAggregate] = []
        hop_duration = timedelta(seconds=self.hop_seconds)
        window_duration = timedelta(seconds=self.window_size_seconds)

        # Emit all complete windows whose window_end is <= latest_time
        while self._last_emitted_window_end <= latest_time:
            w_start = self._last_emitted_window_end - window_duration
            w_end = self._last_emitted_window_end

            # Gather events in [w_start, w_end)
            slice_events = [
                e for e in self._buffer
                if w_start <= e.timestamp < w_end
            ]

            if slice_events:
                agg = compute_window_features(slice_events, w_start, w_end)
                emitted.append(agg)

            self._last_emitted_window_end += hop_duration

        return emitted

    def clear(self) -> None:
        """Flush buffer state."""
        self._buffer.clear()
        self._last_emitted_window_end = None
