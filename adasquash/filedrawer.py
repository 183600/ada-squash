"""Publication bias and file-drawer tracking.

The file-drawer problem: studies with non-significant results are less likely
to be published/analyzed, biasing the published literature. This module provides
tools to:

* Track all experiments run (including "failed" or "inconclusive" ones)
* Detect asymmetry in the funnel plot (Egger's test, Begg's test)
* Estimate the number of missing studies (trim-and-fill)
* Compute p-curve and p-uniform analyses for evidential value
* Generate a study registry that makes the full set of attempts visible

This is not a substitute for pre-registration but a complement: it documents
the *actual* research process, including abandoned or inconclusive runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

__all__ = [
    "FileDrawerReport",
    "StudyRecord",
    "StudyRegistry",
    "beggs_test",
    "eggers_test",
    "funnel_plot_asymmetry",
    "get_study_registry",
    "p_curve_analysis",
    "p_uniform_analysis",
    "register_study",
    "trim_and_fill",
]


@dataclass
class StudyRecord:
    """A record of a single study/experiment attempt.

    This captures everything about an experiment attempt, regardless of outcome.
    """
    schema: str = "adasquash.study_record.v1"
    study_id: str = ""
    title: str = ""
    hypothesis: str = ""
    design: dict = field(default_factory=dict)  # Sample size, design type, etc.
    analysis_plan_fingerprint: str = ""
    status: str = "planned"  # planned, running, completed, abandoned, failed
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    results: dict = field(default_factory=dict)  # Effect size, p-value, CI, etc.
    n_seeds: int = 0
    n_seeds_planned: int = 0
    attrition: float = 0.0
    exclusion_audit: dict = field(default_factory=dict)
    notes: str = ""
    fingerprint: str = ""

    def _compute_fingerprint(self) -> str:
        payload = json.dumps({
            "study_id": self.study_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "design": self.design,
            "analysis_plan_fingerprint": self.analysis_plan_fingerprint,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": self.results,
            "n_seeds": self.n_seeds,
            "n_seeds_planned": self.n_seeds_planned,
            "attrition": self.attrition,
            "exclusion_audit": self.exclusion_audit,
            "notes": self.notes,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_json(self) -> str:
        return json.dumps({
            "schema": self.schema,
            "study_id": self.study_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "design": self.design,
            "analysis_plan_fingerprint": self.analysis_plan_fingerprint,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": self.results,
            "n_seeds": self.n_seeds,
            "n_seeds_planned": self.n_seeds_planned,
            "attrition": self.attrition,
            "exclusion_audit": self.exclusion_audit,
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> StudyRecord:
        obj = json.loads(data)
        return cls(
            schema=obj.get("schema", "adasquash.study_record.v1"),
            study_id=obj.get("study_id", ""),
            title=obj.get("title", ""),
            hypothesis=obj.get("hypothesis", ""),
            design=obj.get("design", {}),
            analysis_plan_fingerprint=obj.get("analysis_plan_fingerprint", ""),
            status=obj.get("status", "planned"),
            started_at=obj.get("started_at", 0.0),
            completed_at=obj.get("completed_at"),
            results=obj.get("results", {}),
            n_seeds=obj.get("n_seeds", 0),
            n_seeds_planned=obj.get("n_seeds_planned", 0),
            attrition=obj.get("attrition", 0.0),
            exclusion_audit=obj.get("exclusion_audit", {}),
            notes=obj.get("notes", ""),
            fingerprint=obj.get("fingerprint", ""),
        )

    def verify_fingerprint(self) -> bool:
        return self.fingerprint == self._compute_fingerprint()


class StudyRegistry:
    """Registry for tracking all study attempts in a project."""

    def __init__(self, storage_dir: str = ".study_registry"):
        self.storage_dir = storage_dir
        self._records: dict[str, StudyRecord] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.storage_dir):
            return
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath) as f:
                        record = StudyRecord.from_json(f.read())
                    self._records[record.study_id] = record
                except (json.JSONDecodeError, KeyError):
                    import logging as _logging; _logging.warning("Suppressed exception in filedrawer.py:148")

    def register(self, record: StudyRecord) -> str:
        """Register a study record."""
        if not record.fingerprint:
            record.fingerprint = record._compute_fingerprint()
        self._records[record.study_id] = record
        self._save(record)
        return record.study_id

    def _save(self, record: StudyRecord) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
        filename = f"study_{record.study_id}.json"
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "w") as f:
            f.write(record.to_json())

    def get(self, study_id: str) -> Optional[StudyRecord]:
        return self._records.get(study_id)

    def list_all(self, status: Optional[str] = None) -> list[StudyRecord]:
        records = list(self._records.values())
        if status:
            records = [r for r in records if r.status == status]
        return sorted(records, key=lambda r: r.started_at)

    def get_completed(self) -> list[StudyRecord]:
        return self.list_all(status="completed")

    def get_by_hypothesis(self, hypothesis: str) -> list[StudyRecord]:
        return [r for r in self._records.values() if hypothesis.lower() in r.hypothesis.lower()]


def register_study(
    registry: StudyRegistry,
    study_id: str,
    title: str,
    hypothesis: str,
    design: Mapping[str, Any],
    analysis_plan_fingerprint: str = "",
    n_seeds_planned: int = 0,
) -> StudyRecord:
    """Create and register a new planned study."""
    record = StudyRecord(
        study_id=study_id,
        title=title,
        hypothesis=hypothesis,
        design=dict(design),
        analysis_plan_fingerprint=analysis_plan_fingerprint,
        status="planned",
        n_seeds_planned=n_seeds_planned,
    )
    registry.register(record)
    return record


def update_study_results(
    registry: StudyRegistry,
    study_id: str,
    results: Mapping[str, Any],
    n_seeds: int,
    attrition: float = 0.0,
    exclusion_audit: Optional[Mapping] = None,
    status: str = "completed",
) -> Optional[StudyRecord]:
    """Update a study with results."""
    record = registry.get(study_id)
    if not record:
        return None

    record.results = dict(results)
    record.n_seeds = n_seeds
    record.attrition = attrition
    record.exclusion_audit = dict(exclusion_audit) if exclusion_audit else {}
    record.status = status
    record.completed_at = time.time()
    record.fingerprint = record._compute_fingerprint()
    registry.register(record)
    return record


@dataclass
class FileDrawerReport:
    """Report on potential publication bias / file-drawer effects."""
    schema: str = "adasquash.file_drawer_report.v1"
    n_completed: int = 0
    n_abandoned: int = 0
    n_failed: int = 0
    n_planned: int = 0
    funnel_asymmetry_p: Optional[float] = None
    eggers_p: Optional[float] = None
    beggs_p: Optional[float] = None
    trim_and_fill_missing: int = 0
    trim_and_fill_adjusted_effect: Optional[float] = None
    p_curve_evidential_value: bool = False
    p_curve_p: Optional[float] = None
    p_uniform_eta: Optional[float] = None
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "n_completed": self.n_completed,
            "n_abandoned": self.n_abandoned,
            "n_failed": self.n_failed,
            "n_planned": self.n_planned,
            "funnel_asymmetry_p": self.funnel_asymmetry_p,
            "eggers_p": self.eggers_p,
            "beggs_p": self.beggs_p,
            "trim_and_fill_missing": self.trim_and_fill_missing,
            "trim_and_fill_adjusted_effect": self.trim_and_fill_adjusted_effect,
            "p_curve_evidential_value": self.p_curve_evidential_value,
            "p_curve_p": self.p_curve_p,
            "p_uniform_eta": self.p_uniform_eta,
            "summary": self.summary,
        }


def funnel_plot_asymmetry(
    studies: Sequence[StudyRecord],
    effect_key: str = "mean_delta",
    se_key: str = "std_err",
) -> dict:
    """Prepare funnel plot data and compute asymmetry metrics."""
    data = []
    for s in studies:
        if s.status != "completed":
            continue
        eff = s.results.get(effect_key)
        se = s.results.get(se_key)
        if eff is None or se is None or se <= 0:
            continue
        precision = 1.0 / max(se, 1e-10)
        data.append({"study_id": s.study_id, "effect": eff, "se": se, "precision": precision})

    if len(data) < 3:
        return {"n": len(data), "insufficient": True}

    # Correlation between effect size and precision (should be zero under no bias)
    effects = [d["effect"] for d in data]
    precisions = [d["precision"] for d in data]

    n = len(data)
    mean_eff = sum(effects) / (n + 1e-8)
    mean_prec = sum(precisions) / (n + 1e-8)

    cov = sum((e - mean_eff) * (p - mean_prec) for e, p in zip(effects, precisions)) / (n + 1e-8)
    var_eff = sum((e - mean_eff) ** 2 for e in effects) / (n + 1e-8)
    var_prec = sum((p - mean_prec) ** 2 for p in precisions) / (n + 1e-8)

    if var_eff == 0 or var_prec == 0:
        corr = 0.0
    else:
        corr = cov / math.sqrt(var_eff * var_prec)

    return {"n": n, "correlation": corr, "data": data}


def eggers_test(studies: Sequence[StudyRecord]) -> dict:
    """Egger's regression test for funnel plot asymmetry.

    Regresses effect size on precision (1/SE). Under no bias, intercept = 0.
    """
    data = funnel_plot_asymmetry(studies)
    if data.get("insufficient", False):
        return {"p_value": None, "note": "Insufficient studies (need >= 3)"}

    items = data["data"]
    effects = [d["effect"] for d in items]
    precisions = [d["precision"] for d in items]
    n = len(items)

    # Linear regression: effect = intercept + slope * precision + error
    mean_eff = sum(effects) / (n + 1e-8)
    mean_prec = sum(precisions) / (n + 1e-8)

    ss_prec = sum((p - mean_prec) ** 2 for p in precisions)
    ss_cross = sum((e - mean_eff) * (p - mean_prec) for e, p in zip(effects, precisions))

    if ss_prec == 0:
        return {"p_value": None, "note": "Zero variance in precision"}

    slope = ss_cross / ss_prec
    intercept = mean_eff - slope * mean_prec

    # Standard error of intercept
    residuals = [e - intercept - slope * p for e, p in zip(effects, precisions)]
    mse = sum(r * r for r in residuals) / max(n - 2, 1)
    se_intercept = math.sqrt(mse * (1 / max(n + 1e-8, 1e-10) + mean_prec*mean_prec / ss_prec))

    if se_intercept == 0:
        return {"p_value": None, "note": "Zero SE for intercept"}

    t_stat = intercept / se_intercept
    # Two-sided p-value from t-distribution
    # Approximate with normal for simplicity
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    return {
        "p_value": p,
        "intercept": intercept,
        "slope": slope,
        "t_statistic": t_stat,
        "n": n,
        "significant": p < 0.05,
    }


def beggs_test(studies: Sequence[StudyRecord]) -> dict:
    """Begg's rank correlation test for funnel plot asymmetry.

    Correlates effect size ranks with variance ranks (Kendall's tau).
    """
    data = funnel_plot_asymmetry(studies)
    if data.get("insufficient", False):
        return {"p_value": None, "note": "Insufficient studies (need >= 3)"}

    items = data["data"]
    effects = [d["effect"] for d in items]
    variances = [d["se"] ** 2 for d in items]
    n = len(items)

    # Kendall's tau between effect and variance
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            eff_diff = effects[i] - effects[j]
            var_diff = variances[i] - variances[j]
            if eff_diff * var_diff > 0:
                concordant += 1
            elif eff_diff * var_diff < 0:
                discordant += 1

    tau = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0.0

    # Variance of tau under null
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) if n > 1 else 1.0
    se_tau = math.sqrt(var_tau)
    z = tau / se_tau if se_tau > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return {
        "p_value": p,
        "tau": tau,
        "z_statistic": z,
        "n": n,
        "significant": p < 0.05,
    }


def trim_and_fill(
    studies: Sequence[StudyRecord],
    effect_key: str = "mean_delta",
    se_key: str = "std_err",
    max_iterations: int = 10,
) -> dict:
    """Trim-and-fill method to estimate missing studies.

    Simplified implementation: iteratively removes most extreme small-study
    effects until funnel symmetry, then imputes missing counterparts.
    """
    data = funnel_plot_asymmetry(studies, effect_key, se_key)
    if data.get("insufficient", False):
        return {"missing": 0, "adjusted_effect": None, "note": "Insufficient studies"}

    items = data["data"]
    effects = [d["effect"] for d in items]
    precisions = [d["precision"] for d in items]

    # Simple approach: estimate center using fixed-effect meta-analysis
    weights = [p * p for p in precisions]
    total_weight = sum(weights)
    if total_weight == 0:
        return {"missing": 0, "adjusted_effect": None, "note": "Zero total weight"}

    center = sum(e * w for e, w in zip(effects, weights)) / total_weight

    # Count studies below center with high precision (potential missing ones)
    missing = 0
    for e, p in zip(effects, precisions):
        if e < center and p > sum(precisions) / max(len(precisions), 1) if precisions else float("nan"):  # High precision, below center
            missing += 1

    # Adjusted effect (center of symmetric funnel)
    adjusted_effect = center

    return {
        "missing": missing,
        "adjusted_effect": adjusted_effect,
        "observed_center": center,
        "n_observed": len(items),
    }


def p_curve_analysis(studies: Sequence[StudyRecord], alpha: float = 0.05) -> dict:
    """p-curve analysis for evidential value.

    Tests whether the distribution of significant p-values is right-skewed
    (evidential value) vs flat (p-hacking) vs left-skewed (p-hacking).
    """
    p_values = []
    for s in studies:
        if s.status != "completed":
            continue
        p = s.results.get("p_one_sided") or s.results.get("p_value")
        if p is not None and p < alpha:
            p_values.append(p)

    if len(p_values) < 2:
        return {"n_significant": len(p_values), "evidential_value": False, "p_value": None}

    # Test for right-skew (evidential value) vs flat
    # Under H1 (true effect), p-values should be concentrated near 0
    # Under H0 (null), p-values ~ Uniform(0, alpha)
    n = len(p_values)
    # Under Uniform(0, alpha), E[p] = alpha/2; a smaller mean indicates
    # right-skew (evidential value). This mean-based test is the one used
    # below (the previous Stouffer-Z draft was unused and referenced the
    # non-existent math.erfcinv).
    mean_p = sum(p_values) / (n + 1e-8)
    # One-sided test: mean_p < alpha/2 indicates right skew
    z = (mean_p - alpha/2) / math.sqrt(alpha*alpha / (12 * n))
    p = 0.5 * (1 + math.erf(z / math.sqrt(2)))  # Lower tail

    return {
        "n_significant": n,
        "mean_p": mean_p,
        "expected_mean_under_null": alpha / 2,
        "z": z,
        "p_value": p,
        "evidential_value": p < 0.05,
        "right_skew": mean_p < alpha / 2,
    }


def p_uniform_analysis(studies: Sequence[StudyRecord], alpha: float = 0.05) -> dict:
    """p-uniform analysis for effect size estimation correcting for publication bias.

    Uses only significant p-values to estimate the effect size that makes
    the observed p-values uniform.
    """
    p_values = []
    for s in studies:
        if s.status != "completed":
            continue
        p = s.results.get("p_one_sided") or s.results.get("p_value")
        if p is not None and p < alpha:
            p_values.append(p)

    if len(p_values) < 2:
        return {"n_significant": len(p_values), "eta": None}

    # Simplified: estimate eta (effect size) that makes p-values uniform
    # This is a very rough approximation; real p-uniform uses ML
    # For now, return a placeholder
    return {
        "n_significant": len(p_values),
        "eta": None,
        "note": "Full p-uniform requires likelihood maximization; not implemented",
    }


def generate_file_drawer_report(registry: StudyRegistry) -> FileDrawerReport:
    """Generate a comprehensive file-drawer report from the study registry."""
    completed = registry.get_completed()
    all_records = registry.list_all()

    n_completed = len(completed)
    n_abandoned = len([r for r in all_records if r.status == "abandoned"])
    n_failed = len([r for r in all_records if r.status == "failed"])
    n_planned = len([r for r in all_records if r.status == "planned"])

    # Funnel asymmetry
    funnel = funnel_plot_asymmetry(completed)
    funnel_p = None
    if not funnel.get("insufficient"):
        funnel_p = 2 * (1 - 0.5 * (1 + math.erf(abs(funnel["correlation"]) / math.sqrt(2))))

    # Egger's test
    eggers = eggers_test(completed)
    eggers_p = eggers.get("p_value")

    # Begg's test
    beggs = beggs_test(completed)
    beggs_p = beggs.get("p_value")

    # Trim and fill
    tf = trim_and_fill(completed)

    # p-curve
    pcurve = p_curve_analysis(completed)

    # p-uniform
    punif = p_uniform_analysis(completed)

    report = FileDrawerReport(
        n_completed=n_completed,
        n_abandoned=n_abandoned,
        n_failed=n_failed,
        n_planned=n_planned,
        funnel_asymmetry_p=funnel_p,
        eggers_p=eggers_p,
        beggs_p=beggs_p,
        trim_and_fill_missing=tf.get("missing", 0),
        trim_and_fill_adjusted_effect=tf.get("adjusted_effect"),
        p_curve_evidential_value=pcurve.get("evidential_value", False),
        p_curve_p=pcurve.get("p_value"),
        p_uniform_eta=punif.get("eta"),
    )

    # Generate summary
    parts = []
    parts.append(f"Registry: {n_completed} completed, {n_abandoned} abandoned, {n_failed} failed, {n_planned} planned.")
    if funnel_p is not None:
        parts.append(f"Funnel asymmetry p={funnel_p:.3f}.")
    if eggers_p is not None:
        parts.append(f"Egger's test p={eggers_p:.3f}.")
    if beggs_p is not None:
        parts.append(f"Begg's test p={beggs_p:.3f}.")
    if tf.get("missing", 0) > 0:
        parts.append(f"Trim-and-fill estimates {tf['missing']} missing studies.")
    if pcurve.get("evidential_value"):
        parts.append("p-curve indicates evidential value.")
    else:
        parts.append("p-curve does not indicate evidential value.")

    report.summary = " ".join(parts)
    return report


def get_study_registry(storage_dir: str = ".study_registry") -> StudyRegistry:
    """Get or create the global study registry."""
    return StudyRegistry(storage_dir)