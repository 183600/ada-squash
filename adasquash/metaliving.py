"""Cross-study cumulative meta-analysis infrastructure.

Science is cumulative. This module provides tools for:
* **Cumulative meta-analysis**: Sequentially adding studies to see how evidence accumulates.
* **Living meta-analysis**: Automatically updating as new studies are registered.
* **Prediction intervals**: For a *new* study, not just the mean.
* **Heterogeneity quantification**: I^2, tau^2, Cochran's Q with confidence intervals.
* **Meta-regression**: Exploring moderators (model size, dataset, compute budget).
* **GRADE-like certainty assessment**: Rating confidence in the cumulative evidence.

This builds on the existing `meta_analysis` in `metrics_meta.py` but adds
the longitudinal/cumulative perspective essential for a research program.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from .metrics_meta import _pool_meta_studies, meta_analysis

__all__ = [
    "CumulativeMetaAnalysis",
    "LivingMetaAnalysis",
    "StudyEffect",
    "grade_assessment",
    "living_meta_analysis",
    "meta_regression",
    "run_cumulative_meta_analysis",
]


@dataclass
class StudyEffect:
    """Effect size from a single study with full metadata."""
    study_id: str
    title: str
    effect_size: float
    std_err: float
    n_pairs: int
    model: str = ""
    dataset: str = ""
    compute_budget: str = ""
    design_type: str = ""  # "paired", "independent", "factorial"
    status: str = "completed"
    timestamp: float = 0.0
    # Optional: variance components for multi-level
    within_var: Optional[float] = None
    between_var: Optional[float] = None

    def weight(self) -> float:
        """Inverse variance weight."""
        return 1.0 / (self.std_err ** 2) if self.std_err > 0 else 0.0

    def to_meta_format(self) -> dict:
        """Convert to format expected by meta_analysis."""
        return {
            "effect": self.effect_size,
            "stderr": self.std_err,
            "n": self.n_pairs,
        }


@dataclass
class CumulativeMetaAnalysis:
    """Results of a cumulative meta-analysis."""
    schema: str = "adasquash.cumulative_meta.v1"
    studies: list[StudyEffect] = field(default_factory=list)
    cumulative: list[dict] = field(default_factory=list)
    overall: dict = field(default_factory=dict)
    heterogeneity: dict = field(default_factory=dict)
    prediction_interval: Optional[tuple[float, float]] = None
    grade: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "studies": [
                {
                    "study_id": s.study_id,
                    "title": s.title,
                    "effect_size": s.effect_size,
                    "std_err": s.std_err,
                    "n_pairs": s.n_pairs,
                    "model": s.model,
                    "dataset": s.dataset,
                    "compute_budget": s.compute_budget,
                    "design_type": s.design_type,
                    "status": s.status,
                    "timestamp": s.timestamp,
                }
                for s in self.studies
            ],
            "cumulative": self.cumulative,
            "overall": self.overall,
            "heterogeneity": self.heterogeneity,
            "prediction_interval": self.prediction_interval,
            "grade": self.grade,
        }


def run_cumulative_meta_analysis(
    studies: Sequence[StudyEffect],
    order_by: str = "timestamp",
    method: str = "REML",
) -> CumulativeMetaAnalysis:
    """Run cumulative meta-analysis adding studies one by one.

    Parameters
    ----------
    studies: Study effects to include.
    order_by: "timestamp" (chronological) or "precision" (largest first).
    method: "FE" (fixed effect), "REML", "DL" (DerSimonian-Laird).

    Returns
    -------
    CumulativeMetaAnalysis with step-by-step results.
    """
    if not studies:
        return CumulativeMetaAnalysis()

    # Sort studies
    sorted_studies = sorted(studies, key=lambda s: getattr(s, order_by, 0))

    cumulative_results = []
    for i in range(1, len(sorted_studies) + 1):
        subset = sorted_studies[:i]
        meta_inputs = [s.to_meta_format() for s in subset]

        # Run meta-analysis
        if method == "FE":
            # Fixed effect: simple inverse variance weighting
            weights = [1.0 / (m["stderr"] ** 2) for m in meta_inputs]
            total_w = sum(weights)
            pooled_effect = sum(m["effect"] * w for m, w in zip(meta_inputs, weights)) / total_w
            pooled_se = math.sqrt(1.0 / total_w)

            result = {
                "k": i,
                "pooled_effect": pooled_effect,
                "pooled_se": pooled_se,
                "pooled_ci_low": pooled_effect - 1.96 * pooled_se,
                "pooled_ci_high": pooled_effect + 1.96 * pooled_se,
                "method": "FE",
            }
        else:
            result = meta_analysis(meta_inputs, method=method)
            result["k"] = i

        # Add the latest study info
        latest = sorted_studies[i - 1]
        result["added_study"] = latest.study_id
        result["added_effect"] = latest.effect_size

        cumulative_results.append(result)

    # Overall analysis with all studies
    meta_inputs_all = [s.to_meta_format() for s in sorted_studies]
    if method == "FE":
        weights = [1.0 / (m["stderr"] ** 2) for m in meta_inputs_all]
        total_w = sum(weights)
        pooled_effect = sum(m["effect"] * w for m, w in zip(meta_inputs_all, weights)) / total_w
        pooled_se = math.sqrt(1.0 / total_w)
        overall = {
            "pooled_effect": pooled_effect,
            "pooled_se": pooled_se,
            "pooled_ci_low": pooled_effect - 1.96 * pooled_se,
            "pooled_ci_high": pooled_effect + 1.96 * pooled_se,
            "method": "FE",
            "k": len(sorted_studies),
        }
    else:
        overall = meta_analysis(meta_inputs_all, method=method)

    # Heterogeneity
    heterogeneity = _compute_heterogeneity(meta_inputs_all)

    # Prediction interval
    pred_interval = _compute_prediction_interval(overall, heterogeneity)

    return CumulativeMetaAnalysis(
        studies=list(sorted_studies),
        cumulative=cumulative_results,
        overall=overall,
        heterogeneity=heterogeneity,
        prediction_interval=pred_interval,
    )


def _compute_heterogeneity(meta_inputs: list[dict]) -> dict:
    """Compute Q, I^2, tau^2 with confidence intervals."""
    k = len(meta_inputs)
    if k < 2:
        return {"k": k, "Q": 0.0, "I2": 0.0, "tau2": 0.0, "Q_p": 1.0}

    effects = [m["effect"] for m in meta_inputs]
    ses = [m["stderr"] for m in meta_inputs]
    weights = [1.0 / (se ** 2) for se in ses]
    total_w = sum(weights)

    # Fixed effect pooled estimate
    fe_effect = sum(e * w for e, w in zip(effects, weights)) / total_w

    # Cochran's Q
    Q = sum(w * (e - fe_effect) ** 2 for e, w in zip(effects, weights))
    df = k - 1

    # p-value for Q (chi-squared approximation)
    if df > 0:
        Q_p = 1.0 - _chi2_cdf(Q, df)
    else:
        Q_p = 1.0

    # I^2
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0

    # tau^2 (DerSimonian-Laird)
    sum_w = sum(weights)
    sum_w2 = sum(w * w for w in weights)
    C = sum_w - sum_w2 / sum_w if sum_w > 0 else 0.0
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Confidence interval for tau^2 (approximate)
    if tau2 > 0 and k > 2:
        # Higgins & Thompson approximation
        se_tau2 = tau2 * math.sqrt(2.0 / (k - 1))
        tau2_ci_low = max(0.0, tau2 - 1.96 * se_tau2)
        tau2_ci_high = tau2 + 1.96 * se_tau2
    else:
        tau2_ci_low = tau2_ci_high = tau2

    return {
        "k": k,
        "Q": Q,
        "Q_df": df,
        "Q_p": Q_p,
        "I2": I2,
        "tau2": tau2,
        "tau2_ci": (tau2_ci_low, tau2_ci_high),
    }


def _chi2_cdf(x: float, df: int) -> float:
    """Approximate chi-squared CDF."""
    # Wilson-Hilferty approximation
    if df <= 0:
        return 1.0
    h = 2.0 / (9.0 * df)
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - h)) / math.sqrt(h)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _compute_prediction_interval(
    overall: dict,
    heterogeneity: dict,
) -> Optional[tuple[float, float]]:
    """Compute 95% prediction interval for a new study."""
    if "pooled_effect" not in overall or "pooled_se" not in overall:
        return None

    effect = overall["pooled_effect"]
    se = overall["pooled_se"]
    tau2 = heterogeneity.get("tau2", 0.0)

    # Prediction interval: effect ± t * sqrt(se^2 + tau2)
    # Use normal approximation for simplicity
    margin = 1.96 * math.sqrt(se * se + tau2)
    return (effect - margin, effect + margin)


def meta_regression(
    studies: Sequence[StudyEffect],
    moderators: Mapping[str, Sequence[Any]],
    method: str = "REML",
) -> dict:
    """Meta-regression: effect size ~ moderator(s).

    Parameters
    ----------
    studies: Study effects.
    moderators: Dict mapping moderator name to list of values (one per study).
    method: Estimation method.

    Returns
    -------
    Regression coefficients, SEs, p-values, and model fit.
    """
    k = len(studies)
    if k < 3:
        return {"error": "Need at least 3 studies for meta-regression"}

    # Check moderator lengths
    for name, values in moderators.items():
        if len(values) != k:
            return {"error": f"Moderator {name} has {len(values)} values, expected {k}"}

    # Prepare data
    effects = [s.effect_size for s in studies]
    variances = [s.std_err ** 2 for s in studies]
    weights = [1.0 / v for v in variances]

    # Design matrix: intercept + moderators
    mod_names = list(moderators.keys())
    n_mod = len(mod_names)
    X = [[1.0] + [moderators[name][i] for name in mod_names] for i in range(k)]

    # Weighted least squares
    # (X' W X)^-1 X' W y
    W = weights
    XtW = [[sum(X[i][j] * W[i] * X[i][k] for i in range(k)) for k in range(n_mod + 1)] for j in range(n_mod + 1)]
    XtWy = [sum(X[i][j] * W[i] * effects[i] for i in range(k)) for j in range(n_mod + 1)]

    # Solve linear system (simplified: use matrix inversion)
    # For simplicity, use a basic implementation
    # In practice, use numpy.linalg.solve

    # 2x2 case (intercept + one moderator)
    if n_mod == 1:
        w_sum = sum(W)
        wx_sum = sum(W[i] * X[i][1] for i in range(k))
        wx2_sum = sum(W[i] * X[i][1] * X[i][1] for i in range(k))
        wy_sum = sum(W[i] * effects[i] for i in range(k))
        wxy_sum = sum(W[i] * X[i][1] * effects[i] for i in range(k))

        det = w_sum * wx2_sum - wx_sum * wx_sum
        if det == 0:
            return {"error": "Singular matrix"}

        intercept = (wx2_sum * wy_sum - wx_sum * wxy_sum) / det
        slope = (w_sum * wxy_sum - wx_sum * wy_sum) / det

        # Residuals and residual variance
        residuals = [effects[i] - intercept - slope * X[i][1] for i in range(k)]
        # Weighted residual sum of squares
        wrss = sum(W[i] * r * r for i, r in enumerate(residuals))
        df_res = k - 2
        sigma2 = wrss / df_res if df_res > 0 else 0.0

        # SE of slope
        se_slope = math.sqrt(sigma2 / (wx2_sum - wx_sum * wx_sum / w_sum)) if wx2_sum > wx_sum * wx_sum / w_sum else 0.0
        se_intercept = math.sqrt(sigma2 * (1.0 / w_sum + (wx_sum / w_sum) ** 2 / (wx2_sum - wx_sum * wx_sum / w_sum)))

        # Test statistics
        t_intercept = intercept / se_intercept if se_intercept > 0 else 0.0
        t_slope = slope / se_slope if se_slope > 0 else 0.0

        p_intercept = 2 * (1 - 0.5 * (1 + math.erf(abs(t_intercept) / math.sqrt(2))))
        p_slope = 2 * (1 - 0.5 * (1 + math.erf(abs(t_slope) / math.sqrt(2))))

        return {
            "coefficients": {
                "intercept": {"estimate": intercept, "se": se_intercept, "t": t_intercept, "p": p_intercept},
                mod_names[0]: {"estimate": slope, "se": se_slope, "t": t_slope, "p": p_slope},
            },
            "residual_variance": sigma2,
            "df_residual": df_res,
            "moderators": mod_names,
            "method": method,
        }

    return {"error": "Multiple moderators not yet implemented; use statsmodels"}


@dataclass
class GradeAssessment:
    """GRADE-like certainty assessment for a body of evidence."""
    schema: str = "adasquash.grade_assessment.v1"
    domains: dict = field(default_factory=dict)
    overall_certainty: str = "very_low"  # high, moderate, low, very_low
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "domains": self.domains,
            "overall_certainty": self.overall_certainty,
            "summary": self.summary,
        }


def grade_assessment(
    cumulative: CumulativeMetaAnalysis,
    risk_of_bias: str = "serious",  # "none", "not_serious", "serious", "very_serious"
    inconsistency: str = "not_serious",
    indirectness: str = "not_serious",
    imprecision: str = "not_serious",
    publication_bias: str = "not_serious",
    large_effect: bool = False,
    dose_response: bool = False,
    plausible_confounding: bool = False,
) -> GradeAssessment:
    """Assess certainty of evidence using GRADE-inspired framework.

    Starts at "high" for randomized experiments, then downgrades for:
    - Risk of bias
    - Inconsistency (heterogeneity)
    - Indirectness
    - Imprecision
    - Publication bias

    And upgrades for:
    - Large effect
    - Dose-response gradient
    - Plausible confounding (would reduce effect)
    """
    # Start at high (these are randomized experiments)
    level = 3  # 3=high, 2=moderate, 1=low, 0=very_low

    # Downgrade
    downgrade_map = {"none": 0, "not_serious": 0, "serious": 1, "very_serious": 2}
    level -= downgrade_map.get(risk_of_bias, 1)
    level -= downgrade_map.get(inconsistency, 1)
    level -= downgrade_map.get(indirectness, 1)
    level -= downgrade_map.get(imprecision, 1)
    level -= downgrade_map.get(publication_bias, 1)

    # Upgrade
    if large_effect:
        level += 1
    if dose_response:
        level += 1
    if plausible_confounding:
        level += 1

    level = max(0, min(3, level))
    certainty_map = {3: "high", 2: "moderate", 1: "low", 0: "very_low"}
    overall = certainty_map[level]

    domains = {
        "risk_of_bias": {"rating": risk_of_bias, "downgrade": downgrade_map.get(risk_of_bias, 0)},
        "inconsistency": {"rating": inconsistency, "downgrade": downgrade_map.get(inconsistency, 0)},
        "indirectness": {"rating": indirectness, "downgrade": downgrade_map.get(indirectness, 0)},
        "imprecision": {"rating": imprecision, "downgrade": downgrade_map.get(imprecision, 0)},
        "publication_bias": {"rating": publication_bias, "downgrade": downgrade_map.get(publication_bias, 0)},
        "large_effect": {"present": large_effect, "upgrade": 1 if large_effect else 0},
        "dose_response": {"present": dose_response, "upgrade": 1 if dose_response else 0},
        "plausible_confounding": {"present": plausible_confounding, "upgrade": 1 if plausible_confounding else 0},
    }

    summary = f"Overall certainty: {overall}. "
    if level < 3:
        downgrades = [k for k, v in domains.items() if v.get("downgrade", 0) > 0]
        if downgrades:
            summary += f"Downgraded for: {', '.join(downgrades)}. "
    upgrades = [k for k, v in domains.items() if v.get("upgrade", 0) > 0]
    if upgrades:
        summary += f"Upgraded for: {', '.join(upgrades)}."

    return GradeAssessment(
        domains=domains,
        overall_certainty=overall,
        summary=summary,
    )


class LivingMetaAnalysis:
    """A living meta-analysis that updates automatically as studies are added."""

    def __init__(
        self,
        name: str,
        initial_studies: Optional[Sequence[StudyEffect]] = None,
        method: str = "REML",
        storage_dir: str = ".living_meta",
    ):
        self.name = name
        self.method = method
        self.storage_dir = storage_dir
        self.studies: list[StudyEffect] = list(initial_studies) if initial_studies else []
        self._current: Optional[CumulativeMetaAnalysis] = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.storage_dir):
            return
        filepath = os.path.join(self.storage_dir, f"{self.name}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                self.studies = [StudyEffect(**s) for s in data.get("studies", [])]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    def _save(self) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
        filepath = os.path.join(self.storage_dir, f"{self.name}.json")
        data = {
            "name": self.name,
            "method": self.method,
            "studies": [
                {
                    "study_id": s.study_id,
                    "title": s.title,
                    "effect_size": s.effect_size,
                    "std_err": s.std_err,
                    "n_pairs": s.n_pairs,
                    "model": s.model,
                    "dataset": s.dataset,
                    "compute_budget": s.compute_budget,
                    "design_type": s.design_type,
                    "status": s.status,
                    "timestamp": s.timestamp,
                }
                for s in self.studies
            ],
            "updated_at": time.time(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def add_study(self, study: StudyEffect) -> CumulativeMetaAnalysis:
        """Add a study and recompute the meta-analysis."""
        self.studies.append(study)
        self._current = run_cumulative_meta_analysis(self.studies, method=self.method)
        self._save()
        return self._current

    def get_current(self) -> CumulativeMetaAnalysis:
        """Get the current cumulative meta-analysis."""
        if self._current is None:
            self._current = run_cumulative_meta_analysis(self.studies, method=self.method)
        return self._current

    def get_grade(self, **kwargs) -> GradeAssessment:
        """Get GRADE assessment for current evidence."""
        return grade_assessment(self.get_current(), **kwargs)

    def living_report(self) -> dict:
        """Generate a living meta-analysis report."""
        current = self.get_current()
        return {
            "name": self.name,
            "n_studies": len(self.studies),
            "last_updated": time.time(),
            "cumulative": current.cumulative[-1] if current.cumulative else {},
            "overall": current.overall,
            "heterogeneity": current.heterogeneity,
            "prediction_interval": current.prediction_interval,
            "grade": self.get_grade().to_dict() if self.studies else None,
        }


def living_meta_analysis(
    name: str,
    studies: Sequence[StudyEffect],
    method: str = "REML",
) -> LivingMetaAnalysis:
    """Create a living meta-analysis from a set of studies."""
    return LivingMetaAnalysis(name, studies, method)