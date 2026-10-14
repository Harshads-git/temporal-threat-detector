"""Explainable AI (XAI) Attribution Engine using SHAP TreeExplainer.

Novelty Component: Explains WHY an event/temporal window was flagged as anomalous.
Wraps SHAP (SHapley Additive exPlanations) for Isolation Forest ensembles to calculate
mathematically grounded feature attributions and base expected values.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import shap

from api.schemas import (
    FeatureAttribution,
    WindowAggregate,
)
from engine.windowing import CANONICAL_FEATURE_NAMES


class ShapExplainerEngine:
    """SHAP TreeExplainer wrapper for tree-based unsupervised anomaly detection models."""

    def __init__(
        self,
        model,
        scaler,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize SHAP TreeExplainer wrapper.

        Args:
            model: Trained Scikit-Learn IsolationForest model.
            scaler: Fitted StandardScaler used during model training.
            background_data: Unscaled or scaled background reference matrix [M, n_features].
            feature_names: Ordered canonical feature names.
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or list(CANONICAL_FEATURE_NAMES)

        # Scale background data if needed
        if background_data.ndim == 2:
            self.background_scaled = self.scaler.transform(background_data)
            self.baseline_unscaled_means = np.mean(background_data, axis=0)
        else:
            raise ValueError(f"background_data must be 2D array, got shape {background_data.shape}")

        # Initialize SHAP TreeExplainer on isolation forest
        self.explainer = shap.TreeExplainer(
            model=self.model,
            data=self.background_scaled,
            feature_perturbation="interventional",
        )
        self.expected_value = float(np.ravel(self.explainer.expected_value)[0])

    def compute_shap_values(
        self,
        window: WindowAggregate,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact SHAP feature attributions for a given temporal window.

        Returns:
            Tuple of:
            - 1D numpy array of signed SHAP values [n_features]
            - 1D numpy array of raw observed feature values [n_features]
        """
        fmap = window.to_feature_vector()
        raw_vec = np.array([[fmap.get(col, 0.0) for col in self.feature_names]], dtype=np.float64)
        scaled_vec = self.scaler.transform(raw_vec)

        # Compute TreeExplainer SHAP values
        # Positive values in TreeExplainer for IsolationForest indicate pushing towards shorter paths (anomaly)
        shap_matrix = self.explainer.shap_values(scaled_vec)
        shap_values_1d = np.ravel(shap_matrix)

        return shap_values_1d, raw_vec[0]

    def explain_window(
        self,
        window: WindowAggregate,
        top_k: int = 4,
    ) -> List[FeatureAttribution]:
        """Extract rank-ordered top-k feature importance attributions for a window.

        Args:
            window: Temporal window being investigated.
            top_k: Number of highest-magnitude contributing features to return.

        Returns:
            List of FeatureAttribution objects sorted by absolute impact.
        """
        shap_values, raw_vals = self.compute_shap_values(window)
        abs_shap = np.abs(shap_values)

        # Rank indices by descending absolute contribution
        ranked_indices = np.argsort(abs_shap)[::-1]

        attributions: List[FeatureAttribution] = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            feat_name = self.feature_names[idx]
            obs_val = float(raw_vals[idx])
            exp_val = float(self.baseline_unscaled_means[idx])
            shap_val = float(shap_values[idx])

            attributions.append(
                FeatureAttribution(
                    feature_name=feat_name,
                    observed_value=round(obs_val, 4),
                    baseline_expected=round(exp_val, 4),
                    shap_value=round(shap_val, 4),
                    importance_rank=rank,
                )
            )

        return attributions

    def get_waterfall_payload(
        self,
        window: WindowAggregate,
        top_k: int = 6,
    ) -> Dict[str, Union[float, List]]:
        """Generate structured visualization payload for Streamlit waterfall / bar graphs."""
        shap_values, raw_vals = self.compute_shap_values(window)
        abs_shap = np.abs(shap_values)
        ranked_indices = np.argsort(abs_shap)[::-1]

        features: List[str] = []
        values: List[float] = []
        observed: List[float] = []
        baseline: List[float] = []

        for idx in ranked_indices[:top_k]:
            features.append(self.feature_names[idx])
            values.append(round(float(shap_values[idx]), 4))
            observed.append(round(float(raw_vals[idx]), 4))
            baseline.append(round(float(self.baseline_unscaled_means[idx]), 4))

        return {
            "base_value": round(self.expected_value, 4),
            "features": features,
            "shap_values": values,
            "observed_values": observed,
            "baseline_expected": baseline,
        }
