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


class SemanticReasoningEngine:
    """Translates SHAP mathematical attributions and raw window metrics into semantic SOC threat hypotheses."""

    @staticmethod
    def infer_threat_semantics(
        attributions: List[FeatureAttribution],
        window: WindowAggregate,
    ) -> Tuple[str, str]:
        """Infer threat hypothesis category and plain-English narrative justification.

        Args:
            attributions: Rank-ordered SHAP feature importance attributions.
            window: Full raw window metrics snapshot.

        Returns:
            Tuple of (threat_hypothesis, explanation_narrative).
        """
        if not attributions:
            return (
                "Unspecified Multivariate Anomaly",
                "Workload feature vector deviated from normal multidimensional cluster, but individual feature impact was diffuse.",
            )

        top_feature = attributions[0].feature_name
        top_obs = attributions[0].observed_value
        top_exp = attributions[0].baseline_expected
        top_shap = attributions[0].shap_value

        # Calculate fold change / relative factor
        fold_change = (top_obs / max(0.01, top_exp)) if top_exp > 0 else 1.0

        # Secondary features set for contextual corroboration
        top_names = {a.feature_name for a in attributions[:3]}

        # 1. DDoS / Volumetric HTTP Flood
        if top_feature in ("request_count", "request_rate_per_sec"):
            hypothesis = "High Volumetric HTTP Flood (DDoS)"
            narrative = (
                f"SHAP attribution identifies {top_feature} as the primary driver (SHAP score +{top_shap:.3f}). "
                f"Request traffic escalated to {top_obs:.0f} reqs ({fold_change:.1f}x higher than baseline of {top_exp:.1f}), "
                f"originating across {window.unique_source_ips} distinct IPs with a {window.error_rate_5xx*100:.1f}% backend error rate."
            )
            return hypothesis, narrative

        # 2. Credential Stuffing / Brute Force
        if top_feature == "auth_failure_count" or ("error_rate_4xx" in top_names and window.auth_failure_count > 5):
            hypothesis = "Distributed Credential Stuffing Campaign"
            narrative = (
                f"SHAP attribution isolates authentication anomalies (SHAP score +{top_shap:.3f}). "
                f"Authorization failures surged to {window.auth_failure_count} (normal expected: {top_exp:.1f}), "
                f"exhibiting a {window.bot_agent_ratio*100:.1f}% automated bot agent signature across auth endpoints."
            )
            return hypothesis, narrative

        # 3. Off-Hour Reconnaissance / Admin Probing
        if top_feature == "off_hour_ratio" or (window.off_hour_ratio > 0.4 and "error_rate_4xx" in top_names):
            hypothesis = "Off-Hours Reconnaissance Probing"
            narrative = (
                f"SHAP TreeExplainer flagged temporal context anomaly (SHAP score +{top_shap:.3f}). "
                f"Unusual API access during off-peak hours ({window.off_hour_ratio*100:.1f}% off-hour activity vs {top_exp*100:.1f}% expected), "
                f"accompanied by elevated client errors ({window.error_count_4xx} 4xx status responses) targeting privileged paths."
            )
            return hypothesis, narrative

        # 4. Privileged Data Exfiltration Surge
        if top_feature in ("total_response_bytes", "mean_response_bytes"):
            megabytes = window.total_response_bytes / (1024 * 1024)
            hypothesis = "Privileged Data Exfiltration Surge"
            narrative = (
                f"SHAP attribution identified massive egress volume anomaly (SHAP score +{top_shap:.3f}). "
                f"Outbound byte throughput spiked to {megabytes:.2f} MB ({fold_change:.1f}x historical average), "
                f"suggesting bulk database dumping or unauthorized cloud storage exfiltration."
            )
            return hypothesis, narrative

        # 5. Latency Degradation / Resource Exhaustion
        if top_feature in ("mean_duration_ms", "p95_duration_ms"):
            hypothesis = "Workload Latency Degradation / Service Saturation"
            narrative = (
                f"SHAP attribution detected severe processing latency regression (SHAP score +{top_shap:.3f}). "
                f"P95 latency reached {window.p95_duration_ms:.1f}ms compared to normal baseline of {top_exp:.1f}ms, "
                f"indicating CPU starvation or upstream database lock contention."
            )
            return hypothesis, narrative

        # Default multivariate contextual fallback
        hypothesis = f"Multivariate Anomaly Dominance: {top_feature}"
        narrative = (
            f"Isolation Forest identified significant behavioral departure driven primarily by {top_feature} "
            f"(observed {top_obs:.2f} vs expected {top_exp:.2f}, SHAP score +{top_shap:.3f})."
        )
        return hypothesis, narrative


@dataclass
class MitigationPlaybook:
    """Actionable cloud incident response playbook recommendation."""

    playbook_id: str
    threat_hypothesis: str
    mitigation_summary: str
    aws_services: List[str]
    remediation_steps: List[str]
    cli_template: str


class IncidentResponseRecommender:
    """Maps identified threat hypotheses and XAI attributions to operational cloud playbooks."""

    PLAYBOOKS: Dict[str, MitigationPlaybook] = {
        "High Volumetric HTTP Flood (DDoS)": MitigationPlaybook(
            playbook_id="IR-AWS-DDOS-01",
            threat_hypothesis="High Volumetric HTTP Flood (DDoS)",
            mitigation_summary="Deploy AWS WAF rate-limiting rules and enable CloudFront edge caching.",
            aws_services=["AWS WAF", "Amazon CloudFront", "Application Load Balancer"],
            remediation_steps=[
                "Apply rate-based rule on regional ALB WAF to throttle callers exceeding 1,000 req/5min.",
                "Verify CloudFront DDoS protection (AWS Shield Standard/Advanced active).",
                "Scale backend ECS/EKS replica targets to absorb transient saturation.",
            ],
            cli_template="aws wafv2 update-web-acl --name IngressWafAcl --scope REGIONAL --default-action Block={} ...",
        ),
        "Distributed Credential Stuffing Campaign": MitigationPlaybook(
            playbook_id="IR-AWS-AUTH-02",
            threat_hypothesis="Distributed Credential Stuffing Campaign",
            mitigation_summary="Enforce adaptive challenge (CAPTCHA) on login endpoints and block abusive CIDRs.",
            aws_services=["AWS WAF Bot Control", "Amazon Cognito", "AWS IAM"],
            remediation_steps=[
                "Enable AWS WAF Fraud Control / Bot Control managed rule group on /auth/login.",
                "Enforce multi-factor authentication (MFA) step-up for repeatedly failing usernames.",
                "Invalidate active refresh tokens for compromised identities.",
            ],
            cli_template="aws wafv2 update-ip-set --name AbusiveAuthIPs --addresses 185.220.101.0/24 ...",
        ),
        "Off-Hours Reconnaissance Probing": MitigationPlaybook(
            playbook_id="IR-AWS-RECON-03",
            threat_hypothesis="Off-Hours Reconnaissance Probing",
            mitigation_summary="Restrict security group ingress to corporate VPN CIDRs and audit CloudTrail.",
            aws_services=["Amazon VPC Security Groups", "AWS CloudTrail", "Amazon GuardDuty"],
            remediation_steps=[
                "Revoke 0.0.0.0/0 ingress on administrative ports in VPC Security Groups.",
                "Search CloudTrail for GetCallerIdentity, ListRoles, and unauthorized Describe* API calls.",
                "Deactivate active AWS STS temporary session tokens for suspected roles.",
            ],
            cli_template="aws ec2 revoke-security-group-ingress --group-id sg-0123456789 --protocol tcp --port 8080 ...",
        ),
        "Privileged Data Exfiltration Surge": MitigationPlaybook(
            playbook_id="IR-AWS-EXFIL-04",
            threat_hypothesis="Privileged Data Exfiltration Surge",
            mitigation_summary="Apply emergency S3 Deny-All bucket policy and quarantine compromised IAM roles.",
            aws_services=["Amazon S3", "AWS IAM", "Amazon Macie", "VPC Flow Logs"],
            remediation_steps=[
                "Attach AWS IAM Deny-All policy boundary to affected IAM principal.",
                "Enable S3 Object Lock and review S3 Access Analyzer alerts for external exfiltration.",
                "Check VPC Flow Logs for high outbound byte egress to unfamiliar external IP CIDRs.",
            ],
            cli_template="aws iam attach-user-policy --user-name compromised-admin --policy-arn arn:aws:iam::aws:policy/AWSDenyAll",
        ),
        "Workload Latency Degradation / Service Saturation": MitigationPlaybook(
            playbook_id="IR-AWS-PERF-05",
            threat_hypothesis="Workload Latency Degradation / Service Saturation",
            mitigation_summary="Trigger emergency pod autoscaling and scale out RDS database read replicas.",
            aws_services=["Amazon ECS / EKS", "Amazon Aurora / RDS", "AWS Auto Scaling"],
            remediation_steps=[
                "Trigger horizontal pod autoscaler (HPA) to spin up additional task containers.",
                "Check Aurora/RDS CPU utilization and activate reader replica endpoints.",
                "Inspect slow database query logs for deadlock and table lock contention.",
            ],
            cli_template="aws ecs update-service --cluster app-cluster --service api-svc --desired-count 15",
        ),
    }

    @classmethod
    def get_playbook(cls, threat_hypothesis: str) -> MitigationPlaybook:
        """Fetch matching mitigation playbook or return generic containment fallback."""
        for name, playbook in cls.PLAYBOOKS.items():
            if name.lower() in threat_hypothesis.lower() or threat_hypothesis.lower() in name.lower():
                return playbook

        return MitigationPlaybook(
            playbook_id="IR-AWS-GENERIC-99",
            threat_hypothesis=threat_hypothesis,
            mitigation_summary="Isolate anomalous workload component and review correlated cloud telemetry.",
            aws_services=["AWS CloudWatch", "Amazon GuardDuty", "AWS Security Hub"],
            remediation_steps=[
                "Review correlated application and load balancer access logs.",
                "Check AWS GuardDuty for unauthorized activity findings.",
                "Apply rate limiting or temporary traffic throttling on ingress API gateway.",
            ],
            cli_template="aws guardduty get-findings --detector-id <detector-id> ...",
        )
