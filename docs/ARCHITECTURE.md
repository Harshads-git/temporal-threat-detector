# Architectural Specification: Temporal Threat Detector

## 1. Mathematical Formulation & Algorithmic Foundations

The Temporal Anomaly & Threat Detector addresses cloud workload threats through a tiered, multi-resolution temporal pipeline.

---

### 1.1 Temporal Window Aggregation
Telemetry stream events $e_i = (t_i, \text{ip}_i, \text{user}_i, \text{path}_i, \text{status}_i, d_i, b_{in,i}, b_{out,i})$ are partitioned into overlapping temporal windows $W_k = [t_{start}, t_{end})$ with length $L = 60\text{s}$ and hop size $H = 10\text{s}$.

Each window is transformed into a 13-dimensional standardized feature vector $\mathbf{x} \in \mathbb{R}^{13}$:
1. $x_1$: `request_count` $N = |W_k|$
2. $x_2$: `request_rate_per_sec` $r = N / L$
3. $x_3$: `error_rate_4xx` $E_{4xx} / \max(1, N)$
4. $x_4$: `error_rate_5xx` $E_{5xx} / \max(1, N)$
5. $x_5$: `mean_duration_ms` $\bar{d} = \frac{1}{N} \sum d_i$
6. $x_6$: `p95_duration_ms` $P_{95}(d)$
7. $x_7$: `unique_source_ips` $|\{\text{ip}_i\}|$
8. $x_8$: `unique_users` $|\{\text{user}_i\}|$
9. $x_9$: `total_response_bytes` $\sum b_{out,i}$
10. $x_{10}$: `mean_response_bytes` $\frac{1}{N} \sum b_{out,i}$
11. $x_{11}$: `auth_failure_count` $\sum \mathbf{1}(\text{status}_i \in \{401, 403\})$
12. $x_{12}$: `off_hour_ratio` $\frac{1}{N} \sum \mathbf{1}(\text{hour}(t_i) \in [0, 5])$
13. $x_{13}$: `bot_agent_ratio` $\frac{1}{N} \sum \mathbf{1}(\text{agent}_i \in \text{Bots})$

---

### 1.2 Tier 1: Real-Time Statistical Filter (MAD + EWMA)
To detect sudden volumetric floods in $O(1)$ time without ML overhead, Tier 1 uses **Median Absolute Deviation (MAD)** for scale estimation and **EWMA** for dynamic baseline volatility tracking.

#### Robust MAD Z-Score
Given a rolling buffer of window metrics $Y = \{y_1, \dots, y_m\}$:
$$\tilde{Y} = \text{median}(Y)$$
$$\text{MAD} = \text{median}(|y_i - \tilde{Y}|)$$
$$\hat{\sigma}_{MAD} = 1.4826 \cdot \max(0.5, \text{MAD})$$
$$Z_{MAD}(y) = \frac{y - \tilde{Y}}{\hat{\sigma}_{MAD}}$$

An anomaly is triggered if $Z_{MAD}(y) \ge \tau_z$ (where default $\tau_z = 3.0$).

#### EWMA Volatility Tracker
$$\mu_t = \alpha y_t + (1 - \alpha) \mu_{t-1}$$
$$\sigma^2_t = \beta (y_t - \mu_t)^2 + (1 - \beta) \sigma^2_{t-1}$$
Dynamic sensitivity scaling:
$$\tau_{\text{effective}} = \tau_z \cdot \left(1.0 + 0.5 \cdot \frac{\sigma_t}{\max(1.0, \mu_t)}\right)$$

---

### 1.3 Tier 2: Multivariate Unsupervised Isolation Forest
Contextual anomalies (e.g., low-volume off-hour scanning or gradual data exfiltration) evade 1D thresholds. Tier 2 uses an ensemble of $T = 50$ isolation trees.

#### Anomaly Score Formulation
For a window feature vector $\mathbf{x}$ evaluated across trees:
$$s(\mathbf{x}, n) = 2^{-\frac{E(h(\mathbf{x}))}{c(n)}}$$
where $h(\mathbf{x})$ is path length in an isolation tree and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree (BST):
$$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$

- $s \to 1.0$: Confirmed contextual anomaly.
- $s < 0.5$: Normal diurnal behavior.

---

### 1.4 Novelty: Explainable AI (Tree-SHAP) Formulation
To solve the "black-box" dilemma in SecOps, we apply Shapley Additive Explanations (Tree-SHAP) to attribute feature contributions:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{M} \phi_i(\mathbf{x})$$

The exact Shapley value $\phi_i$ for feature $i$ is calculated via:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

Tree-SHAP evaluates this in $O(T \cdot L^2)$ time ($3.3\text{ ms}$), providing instant per-incident feature rankings and baseline deltas.

---

### 1.5 Hybrid Ensemble Arbitration State Machine

```
              ┌──────────────────────────────┐
              │ Incoming Window Feature (x)  │
              └──────────────┬───────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       [ Tier 1 Filter ]             [ Tier 2 iForest ]
       (Z >= 3.0 or MAD)             (Score <= Thresh)
              │                             │
              ├──────────────┬──────────────┤
              │ Both Flag    │ Only T1 Flag │
              ▼              ▼              ▼ Only T2 Flag
      ┌───────────────┐ ┌─────────────┐ ┌──────────────┐
      │ HYBRID ENSEMBLE│ │ TIER 1 STAT │ │ TIER 2 AI    │
      │ Severity: CRIT│ │ Volumetric  │ │ Contextual   │
      │ Conf: +0.05   │ │ Fast-Path   │ │ Multi-Feature│
      └───────┬───────┘ └──────┬──────┘ └──────┬───────┘
              │                │               │
              └────────────────┼───────────────┘
                               ▼
                   [ SHAP TreeExplainer XAI ]
                               ▼
                   [ Cloud Mitigation Playbook ]
```
