"""Temporal Threat Detector Engine Package."""

from engine.windowing import (
    CANONICAL_FEATURE_NAMES,
    FeatureMatrixVectorizer,
    StreamingWindowBuffer,
    TemporalWindowAggregator,
    WindowType,
    compute_window_features,
)
from engine.zscore_filter import (
    EWMAResult,
    EWMAVolatilityTracker,
    SlidingWindowZScoreFilter,
    StatisticalFilterEngine,
    ZScoreAnomalyResult,
)

__all__ = [
    "CANONICAL_FEATURE_NAMES",
    "FeatureMatrixVectorizer",
    "StreamingWindowBuffer",
    "TemporalWindowAggregator",
    "WindowType",
    "compute_window_features",
    "SlidingWindowZScoreFilter",
    "EWMAVolatilityTracker",
    "StatisticalFilterEngine",
    "ZScoreAnomalyResult",
    "EWMAResult",
]
