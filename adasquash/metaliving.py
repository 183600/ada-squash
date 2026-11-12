"""Cross-study cumulative meta-analysis infrastructure.

Science is cumulative. This module provides tools for:
* **Cumulative meta-analysis**: Sequentially adding studies to see how evidence accumulates.
* **Living meta-analysis**: Automatically updating as new studies are registered.
* **Prediction intervals**: For a *new* study, not just the mean.
* **Heterogeneity quantification**: I^2, tau^2, Cochran's Q with confidence intervals.
* **Meta-regression**: Exploring moderators (model size, dataset, compute budget).
* **GRADE-like certainty assessment**: Rating confidence in the cumulative evidence.

The pooling mathematics is **not** re-implemented here. Every pooled
estimate goes through `adasquash.metrics_meta.meta_analysis`, which is the
repository's calibrated implementation (exact chi-square tail, Student-t
intervals with `k - 1` degrees of freedom, a `k - 2` prediction interval and
a leave-one-scale-out influence block). A second, slightly different copy of
the same mathematics is the classic way for a cross-study report to disagree
with the single-study report it is supposed to extend, so this module only
adds the longitudinal layer: ordering, accumulation, persistence and grading.

Two consequences are worth stating explicitly, because they used to be
silent:

* the random-effects estimator here is DerSimonian-Laird, the one the
  repository uses everywhere else. A request for "REML" is answered with
  DerSimonian-Laird *and an explicit note*, never with a silent substitution;
* the cumulative curve is a *descriptive* trajectory. Looking at how the
  pooled estimate moves as studies arrive is not a repeated-significance test,
  and nothing in this module certifies a claim on its own.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .metrics_meta import meta_analysis
from .metrics_stats import chi2_ppf, chi2_sf, student_t_ppf, two_sided_p_from_t
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

__all__ = [
    "CumulativeMetaAnalysis",
    "LivingMetaAnalysis",
    "StudyEffect",
    "grade_assessment",
    "living_meta_analysis",
    "meta_regression",
    "run_cumulative_meta_analysis",
]

# Pooling methods this module understands. "RE" and "DL" are the same
# DerSimonian-Laird estimator the repository uses everywhere else; "REML" is
# accepted so an existing caller keeps working, but it is pooled with
# DerSimonian-Laird and the report says so.
_FIXED_EFFECT_METHOD = "FE"
_RANDOM_EFFECT_METHODS = ("RE", "DL", "REML")
_SUPPORTED_METHODS = (_FIXED_EFFECT_METHOD,) + _RANDOM_EFFECT_METHODS
# Orderings a cumulative curve can follow.
_SUPPORTED_ORDERINGS = ("timestamp", "precision", "effect_size", "std_err",
                        "study_id")


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

    @property
    def precision(self) -> float:
        """Inverse standard error; the funnel-plot precision of the study."""
        return 1.0 / self.std_err if self.std_err > 0 else 0.0

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


def _method_label(method: str) -> str:
    """Human-readable pooling method, with the substitution made explicit."""
    if method == _FIXED_EFFECT_METHOD:
        return "inverse-variance fixed effect"
    return "DerSimonian-Laird random effects"


def _sort_key(study: StudyEffect, order_by: str):
    """Ordering key for the cumulative curve.

    "precision" orders the studies from the most precise to the least
    precise one, which is the order in which a research programme usually
    accumulates evidence (bigger models first, then the cheap ones). Sorting
    by *ascending* precision would invert that reading, so the key is
    negated deliberately.
    """
    if order_by == "timestamp":
        return (study.timestamp, study.study_id)
    if order_by == "precision":
        return (-study.precision, study.study_id)
    if order_by == "effect_size":
        return (study.effect_size, study.study_id)
    if order_by == "std_err":
        return (study.std_err, study.study_id)
    return (study.study_id,)


def _tau_squared_interval(q: float, df: int, c: float,
                           alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval for tau^2 from Cochran's Q.

    The Higgins-Thompson-Spiegelhalter construction treats Q as
    chi-square distributed with `df` degrees of freedom and maps the
    chi-square quantiles back to tau^2 through the same
    ``tau^2 = (Q - df) / C`` relation the point estimate uses. The lower
    bound is truncated at zero because a negative tau^2 is not a variance.

    Args:
        q: Cochran's Q statistic.
        df: degrees of freedom of Q (k - 1).
        c: the DerSimonian-Laird denominator ``sum(w) - sum(w^2)/sum(w)``.
        alpha: two-sided error rate of the interval.

    Returns:
        The (low, high) tau^2 interval.
    """
    if c <= 0.0 or df <= 0:
        return (0.0, 0.0)
    low_q = chi2_ppf(alpha / 2.0, df)
    high_q = chi2_ppf(1.0 - alpha / 2.0, df)
    low = max(0.0, (low_q - df) / c)
    high = max(0.0, (high_q - df) / c)
    return (low, high)


def _pooled_block(pooled: dict, method: str) -> dict:
    """The fixed-effect or random-effects block of a pooled analysis."""
    if method == _FIXED_EFFECT_METHOD:
        return pooled["fixed_effect"]
    return pooled["random_effects"]


def run_cumulative_meta_analysis(
    studies: Sequence[StudyEffect],
    order_by: str = "timestamp",
    method: str = "RE",
    alpha: float = 0.05,
) -> CumulativeMetaAnalysis:
    """Run cumulative meta-analysis adding studies one by one.

    Parameters
    ----------
    studies: Study effects to include.
    order_by: "timestamp" (chronological), "precision" (most precise
        first), "effect_size", "std_err" or "study_id".
    method: "FE" (fixed effect) or "RE"/"DL"/"REML" (random effects). The
        random-effects estimator is DerSimonian-Laird, the one the rest of
        the repository uses; "REML" is pooled that way and the report says
        so instead of claiming a restricted maximum likelihood fit.
    alpha: two-sided error rate of the reported intervals.

    Returns
    -------
    CumulativeMetaAnalysis with step-by-step results.

    The curve is descriptive. Each step pools the studies seen so far with
    the repository's calibrated meta-analysis, so the last point of the
    curve and the single-shot `meta_analysis` of the same studies agree
    exactly - a consistency the previous implementation could not offer
    because it carried its own copy of the pooling arithmetic.
    """
    if order_by not in _SUPPORTED_ORDERINGS:
        raise ValueError(
            f"order_by must be one of {_SUPPORTED_ORDERINGS}, got {order_by!r}")
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {_SUPPORTED_METHODS}, got {method!r}")
    if not studies:
        return CumulativeMetaAnalysis()

    sorted_studies = sorted(studies, key=lambda s: _sort_key(s, order_by))

    cumulative_results = []
    for i in range(1, len(sorted_studies) + 1):
        subset = sorted_studies[:i]
        pooled = meta_analysis(
            [s.to_meta_format() for s in subset], alpha=alpha)
        block = _pooled_block(pooled, method)
        heterogeneity = pooled["heterogeneity"] or {}
        result = {
            "k": i,
            "pooled_effect": block.get("pooled_effect"),
            "stderr": block.get("stderr"),
            "ci_low": block.get("ci_low"),
            "ci_high": block.get("ci_high"),
            "t": block.get("t"),
            "df": block.get("df"),
            "p_one_sided": block.get("p_one_sided"),
            "p_two_sided": block.get("p_two_sided"),
            "tau_squared": heterogeneity.get("tau_squared"),
            "i_squared": heterogeneity.get("i_squared"),
            "q": heterogeneity.get("q"),
            "q_p": heterogeneity.get("p"),
            "method": _method_label(method),
        }
        latest = sorted_studies[i - 1]
        result["added_study"] = latest.study_id
        result["added_effect"] = latest.effect_size
        cumulative_results.append(result)

    overall_pooled = meta_analysis(
        [s.to_meta_format() for s in sorted_studies], alpha=alpha)
    overall = dict(overall_pooled)
    overall["k"] = len(sorted_studies)
    overall["method"] = _method_label(method)
    if method == "REML":
        overall["method_note"] = (
            "REML was requested; the estimator actually used is "
            "DerSimonian-Laird, the same one as the rest of the repository")
    overall["order_by"] = order_by

    heterogeneity = _compute_heterogeneity(overall_pooled, sorted_studies,
                                           alpha=alpha)

    prediction = _compute_prediction_interval(overall_pooled, heterogeneity)

    return CumulativeMetaAnalysis(
        studies=list(sorted_studies),
        cumulative=cumulative_results,
        overall=overall,
        heterogeneity=heterogeneity,
        prediction_interval=prediction,
    )


def _compute_heterogeneity(pooled: dict, studies: Sequence[StudyEffect],
                           alpha: float = 0.05) -> dict:
    """Cochran's Q, I^2, tau^2 and a tau^2 confidence interval.

    The Q tail comes from the calibrated chi-square implementation in
    `adasquash.metrics_stats`, not from a closed-form approximation: with
    k - 1 degrees of freedom of one to four, the Wilson-Hilferty
    approximation that the previous implementation used is visibly off, and
    the heterogeneity verdict is exactly the kind of number that must not be
    approximate.
    """
    block = dict(pooled.get("heterogeneity") or {})
    k = len(studies)
    block.setdefault("k", k)
    q = block.get("q")
    df = block.get("df")
    if q is not None and df is not None and df > 0 and k >= 2:
        weights = [1.0 / (study.std_err ** 2) for study in studies
                   if study.std_err > 0]
        total_weight = sum(weights)
        weight_square_sum = sum(weight * weight for weight in weights)
        c = (total_weight - weight_square_sum / max(total_weight, 1)
             if total_weight > 0 else 0.0)
        low, high = _tau_squared_interval(q, int(df), c, alpha=alpha)
        block["tau2_ci"] = (low, high)
        block["tau2_ci_method"] = (
            "chi-square quantiles of Cochran's Q mapped through "
            "tau^2 = (Q - (k - 1)) / C")
    else:
        block["tau2_ci"] = (None, None)
        block["tau2_ci_method"] = None
    return block


def _compute_prediction_interval(
    pooled: dict,
    heterogeneity: dict,
) -> Optional[tuple[float, float]]:
    """Compute the 95% prediction interval for a new study.

    The interval adds the between-study variance to the squared standard
    error of the pooled mean and uses `k - 2` degrees of freedom, so it is
    wider than the confidence interval for the mean and says where the
    effect at a *new* scale would plausibly land. It needs at least three
    studies; with fewer, the between-study variance is not estimable and
    the function returns None rather than a spuriously narrow interval.
    """
    block = pooled.get("prediction_interval") or {}
    low = block.get("low")
    high = block.get("high")
    if low is None or high is None or not block.get("meaningful"):
        return None
    return (float(low), float(high))


def _solve_linear_system(matrix: list[list[float]],
                         vector: list[float]) -> Optional[list[float]]:
    """Solve ``matrix @ x = vector`` by Gaussian elimination.

    Returns None when the matrix is singular (a perfectly collinear set of
    moderators), which is a modelling error the caller has to report rather
    than a numerical accident to paper over.
    """
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot_row = max(range(column, n),
                        key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot_row][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot_row] = (augmented[pivot_row],
                                                    augmented[column])
        pivot = augmented[column][column]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column] / max(pivot, 1e-10)
            if factor == 0.0:
                continue
            for col in range(column, n + 1):
                augmented[row][col] -= factor * augmented[column][col]
    return [augmented[i][n] / augmented[i][i] for i in range(n)]


def _invert_matrix(matrix: list[list[float]]) -> Optional[list[list[float]]]:
    """Invert a square matrix by solving against the unit vectors."""
    n = len(matrix)
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    columns = []
    for j in range(n):
        column = _solve_linear_system(matrix, [identity[i][j] for i in range(n)])
        if column is None:
            return None
        columns.append(column)
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def _numeric_moderator(values: Sequence[Any], name: str) -> Optional[list[float]]:
    """Moderator values as finite floats, or None when they are not numeric."""
    numeric = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        converted = float(value)
        if not math.isfinite(converted):
            return None
        numeric.append(converted)
    return numeric


def meta_regression(
    studies: Sequence[StudyEffect],
    moderators: Mapping[str, Sequence[Any]],
    method: str = "RE",
    alpha: float = 0.05,
) -> dict:
    """Meta-regression: effect size ~ moderator(s).

    The regression is a weighted least squares fit with weights
    ``1 / (v_i + tau^2)``: the between-study variance is estimated from the
    residual heterogeneity with the DerSimonian-Laird construction and added
    to each study's sampling variance, so a moderator is not credited with
    explaining noise that is really between-study spread. The standard
    errors use the estimated residual variance and a Student-t reference
    distribution with ``k - p`` degrees of freedom (``p`` = intercept plus
    moderators), which is the small-sample correction a handful of studies
    needs.

    Parameters
    ----------
    studies: Study effects.
    moderators: Dict mapping moderator name to list of numeric values (one
        per study). Categorical moderators must be expanded into numeric
        contrasts by the caller.
    method: "RE" (random effects, the default) or "FE" (fixed effect, the
        sampling variances alone).
    alpha: two-sided error rate of the reported intervals.

    Returns
    -------
    Regression coefficients, SEs, p-values, the residual heterogeneity and
    the model fit.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {_SUPPORTED_METHODS}, got {method!r}")
    k = len(studies)
    if k < 3:
        return {"error": "Need at least 3 studies for meta-regression"}
    if not moderators:
        return {"error": "At least one moderator is required"}

    for name, values in moderators.items():
        if len(values) != k:
            return {"error": f"Moderator {name} has {len(values)} values, expected {k}"}

    columns = []
    for name, values in moderators.items():
        numeric = _numeric_moderator(values, name)
        if numeric is None:
            return {"error": f"Moderator {name} must contain finite numeric values"}
        columns.append((name, numeric))

    effects = [study.effect_size for study in studies]
    variances = [study.std_err ** 2 for study in studies]
    if any(not math.isfinite(value) or value <= 0.0 for value in variances):
        return {"error": "every study needs a finite, positive standard error"}
    if any(not math.isfinite(value) for value in effects):
        return {"error": "every study needs a finite effect size"}

    design = [[1.0] + [column[1][i] for column in columns] for i in range(k)]
    n_parameters = len(design[0])

    def _fit(weights: list[float]) -> Optional[dict]:
        xtwx = [[sum(design[i][j] * weights[i] * design[i][l]
                     for i in range(k))
                 for l in range(n_parameters)] for j in range(n_parameters)]
        xtwy = [sum(design[i][j] * weights[i] * effects[i] for i in range(k))
                for j in range(n_parameters)]
        coefficients = _solve_linear_system(xtwx, xtwy)
        if coefficients is None:
            return None
        residuals = [effects[i] - sum(design[i][j] * coefficients[j]
                                      for j in range(n_parameters))
                     for i in range(k)]
        q = sum(weights[i] * residuals[i] ** 2 for i in range(k))
        return {"coefficients": coefficients, "residuals": residuals, "q": q,
                "xtwx": xtwx}

    fixed_weights = [1.0 / max(variance, 1e-10) for variance in variances]
    fixed = _fit(fixed_weights)
    if fixed is None:
        return {"error": "Singular moderator design (collinear moderators)"}

    tau_squared = 0.0
    if method != _FIXED_EFFECT_METHOD:
        df_q = k - 1
        total_weight = sum(fixed_weights)
        weight_square_sum = sum(weight * weight for weight in fixed_weights)
        c = (total_weight - weight_square_sum / max(total_weight, 1)
             if total_weight > 0 else 0.0)
        if c > 0.0:
            tau_squared = max(0.0, (fixed["q"] - df_q) / c)

    weights = (fixed_weights if method == _FIXED_EFFECT_METHOD
               else [1.0 / (variance + tau_squared)
                     for variance in variances])
    fitted = _fit(weights)
    if fitted is None:
        return {"error": "Singular moderator design (collinear moderators)"}

    df_residual = k - n_parameters
    if df_residual <= 0:
        return {"error": "not enough studies for the requested moderators"}
    sigma_squared = fitted["q"] / df_residual
    inverse = _invert_matrix(fitted["xtwx"])
    if inverse is None:
        return {"error": "Singular moderator design (collinear moderators)"}

    critical = student_t_ppf(1.0 - alpha / 2.0, float(df_residual))
    names = ["intercept"] + [name for name, _ in columns]
    coefficients = {}
    for index, name in enumerate(names):
        variance = sigma_squared * inverse[index][index]
        standard_error = math.sqrt(variance) if variance > 0.0 else 0.0
        estimate = fitted["coefficients"][index]
        t_statistic = estimate / standard_error if standard_error > 0.0 else 0.0
        coefficients[name] = {
            "estimate": estimate,
            "se": standard_error,
            "t": t_statistic,
            "p": two_sided_p_from_t(t_statistic, float(df_residual)),
            "ci_low": estimate - critical * standard_error,
            "ci_high": estimate + critical * standard_error,
        }

    mean_effect = sum(weights[i] * effects[i] for i in range(k)) / max(sum(weights), 1e-8)
    total_q = sum(weights[i] * (effects[i] - mean_effect) ** 2
                  for i in range(k))
    explained = max(0.0, 1.0 - fitted["q"] / total_q) if total_q > 0.0 else 0.0

    return {
        "coefficients": coefficients,
        "residual_variance": sigma_squared,
        "df_residual": df_residual,
        "moderators": [name for name, _ in columns],
        "method": method,
        "tau_squared": tau_squared,
        "residual_q": fitted["q"],
        "residual_q_p": chi2_sf(fitted["q"], float(df_residual)),
        "pseudo_r_squared": explained,
        "note": ("weighted least squares with the between-study variance "
                 "estimated by the DerSimonian-Laird construction; the "
                 "residual Q tests whether the moderators still leave "
                 "heterogeneity unexplained"),
    }


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


def _derive_inconsistency(cumulative: CumulativeMetaAnalysis,
                           alpha: float) -> str:
    """Inconsistency rating read off the measured heterogeneity."""
    heterogeneity = cumulative.heterogeneity or {}
    i_squared = heterogeneity.get("i_squared")
    q_p = heterogeneity.get("p")
    if i_squared is not None and i_squared >= 75.0:
        return "very_serious"
    if (i_squared is not None and i_squared >= 50.0) or (
            q_p is not None and q_p < alpha):
        return "serious"
    return "not_serious"


def _derive_imprecision(cumulative: CumulativeMetaAnalysis) -> str:
    """Imprecision rating read off the interval of the pooled effect.

    A confidence interval that spans both signs means the data do not pin
    down the direction of the effect; a prediction interval that spans both
    signs means the effect at the next study is not pinned down either. Both
    are imprecision in the GRADE sense, and the second is worse.
    """
    overall = cumulative.overall or {}
    random = overall.get("random_effects") or {}
    ci_low = random.get("ci_low")
    ci_high = random.get("ci_high")
    prediction = overall.get("prediction_interval") or {}
    p_low = prediction.get("low")
    p_high = prediction.get("high")
    if p_low is not None and p_high is not None and p_low < 0.0 < p_high:
        return "very_serious"
    if ci_low is not None and ci_high is not None and ci_low < 0.0 < ci_high:
        return "serious"
    return "not_serious"


def grade_assessment(
    cumulative: CumulativeMetaAnalysis,
    risk_of_bias: str = "serious",  # "none", "not_serious", "serious", "very_serious"
    inconsistency: str = "auto",
    indirectness: str = "not_serious",
    imprecision: str = "auto",
    publication_bias: str = "not_serious",
    large_effect: bool = False,
    dose_response: bool = False,
    plausible_confounding: bool = False,
    alpha: float = 0.05,
) -> GradeAssessment:
    """Assess certainty of evidence using a GRADE-inspired framework.

    Starts at "high" for randomized experiments, then downgrades for:
    - Risk of bias
    - Inconsistency (heterogeneity)
    - Indirectness
    - Imprecision
    - Publication bias

    And upgrades for:
    - Large effect
    - Dose-response gradient

    The inconsistency and imprecision domains default to "auto", which reads
    the measured heterogeneity and the interval of the pooled effect instead
    of assuming them to be absent. An explicit rating always wins, so a
    reviewer can override the automatic reading; the report states which
    value was used and where it came from.

    `plausible_confounding` is kept for API compatibility but does not
    change the rating: confounding is a reason to *lower* confidence, never
    a reason to raise it. It is reported as an informational field so the
    argument behind a rating stays visible.
    """
    downgrade_map = {"none": 0, "not_serious": 0, "serious": 1,
                     "very_serious": 2}
    level = 3  # 3=high, 2=moderate, 1=low, 0=very_low

    inconsistency_source = "caller"
    if inconsistency in (None, "auto"):
        inconsistency = _derive_inconsistency(cumulative, alpha)
        inconsistency_source = "derived from the measured heterogeneity"
    imprecision_source = "caller"
    if imprecision in (None, "auto"):
        imprecision = _derive_imprecision(cumulative)
        imprecision_source = "derived from the interval of the pooled effect"

    level -= downgrade_map.get(risk_of_bias, 1)
    level -= downgrade_map.get(inconsistency, 1)
    level -= downgrade_map.get(indirectness, 1)
    level -= downgrade_map.get(imprecision, 1)
    level -= downgrade_map.get(publication_bias, 1)

    if large_effect:
        level += 1
    if dose_response:
        level += 1

    level = max(0, min(3, level))
    certainty_map = {3: "high", 2: "moderate", 1: "low", 0: "very_low"}
    overall = certainty_map[level]

    domains = {
        "risk_of_bias": {"rating": risk_of_bias,
                         "downgrade": downgrade_map.get(risk_of_bias, 0),
                         "source": "caller"},
        "inconsistency": {"rating": inconsistency,
                          "downgrade": downgrade_map.get(inconsistency, 0),
                          "source": inconsistency_source},
        "indirectness": {"rating": indirectness,
                         "downgrade": downgrade_map.get(indirectness, 0),
                         "source": "caller"},
        "imprecision": {"rating": imprecision,
                        "downgrade": downgrade_map.get(imprecision, 0),
                        "source": imprecision_source},
        "publication_bias": {"rating": publication_bias,
                             "downgrade": downgrade_map.get(publication_bias, 0),
                             "source": "caller"},
        "large_effect": {"present": large_effect, "upgrade": 1 if large_effect else 0},
        "dose_response": {"present": dose_response, "upgrade": 1 if dose_response else 0},
        "plausible_confounding": {"present": plausible_confounding,
                                  "upgrade": 0,
                                  "note": ("confounding is a downgrade domain; "
                                           "it never raises the certainty "
                                           "rating")},
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
        method: str = "RE",
        storage_dir: str = ".living_meta",
    ):
        self.name = name
        self.method = method
        self.storage_dir = Path(storage_dir)
        self.studies: list[StudyEffect] = list(initial_studies) if initial_studies else []
        self._current: Optional[CumulativeMetaAnalysis] = None
        self._load()

    def _load(self) -> None:
        filepath = self.storage_dir / f"{self.name}.json"
        if not filepath.exists():
            return
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            self.studies = [StudyEffect(**s) for s in data.get("studies", [])]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            import logging as _logging; _logging.warning("Suppressed exception in metaliving.py:735")

    def _save(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.storage_dir / f"{self.name}.json"
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
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
    method: str = "RE",
) -> LivingMetaAnalysis:
    """Create a living meta-analysis from a set of studies."""
    return LivingMetaAnalysis(name, studies, method)
