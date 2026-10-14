"""Temporal Threat Detector Engine Package."""

from engine.explainability import (
    ShapExplainerEngine,
)
from engine.isolation_forest import (
    HybridThreatEnsemble,
    IsolationForestDetector,
    IsolationForestResult,
    train_baseline_model,
)
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
    "IsolationForestDetector",
    "IsolationForestResult",
    "HybridThreatEnsemble",
    "train_baseline_model",
    "ShapExplainerEngine",
]
