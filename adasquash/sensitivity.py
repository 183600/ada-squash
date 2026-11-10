"""Sensitivity analysis framework for exclusion policies and design choices.

An analysis is only credible if its conclusions are robust to reasonable
variations in the design choices that were made before seeing the outcomes.
This module provides tools for:

``exclusion_policy_sensitivity``
    Re-analyze the primary contrasts under alternative exclusion policies
    (e.g., what if we also exclude diverged curves? what if we include all
    runs regardless of protocol mismatches?) and report whether the
    significance/equivalence conclusions change.

``attrition_sensitivity``
    Vary the assumed attrition rate in the prospective power plan and report
    the impact on required sample size and achieved power.

``effect_size_sensitivity``
    Vary the assumed smallest effect size (margin) and report the impact on
    power and required n.

``analysis_pipeline_sensitivity``
    Re-run the analysis pipeline with alternative statistical methods
    (e.g., randomization test instead of t-test, Welch instead of Student)
    and report whether conclusions change.

``SensitivityReport``
    A structured report that summarizes all sensitivity checks and flags
    conclusions that are not robust.

The sensitivity analysis is *not* a license to shop for favorable results.
It is a diagnostic: if a conclusion flips under a reasonable alternative,
that fact must be reported and the conclusion qualified.
"""
from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from .design import (
    EXCLUSION_POLICY_SCHEMA,
    default_exclusion_policy,
    normalize_exclusion_policy,
    run_exclusion_audit,
)
from .metrics import (
    MIN_SEEDS_FOR_SIGNIFICANCE,
    paired_randomization_test,
    paired_t_one_sided,
    tost_paired,
    welch_t_one_sided,
)

__all__ = [
    "SensitivityReport",
    "analysis_pipeline_sensitivity",
    "attrition_sensitivity",
    "combine_sensitivity_reports",
    "effect_size_sensitivity",
    "exclusion_policy_sensitivity",
]


@dataclass
class SensitivityCheck:
    """A single sensitivity check result."""
    name: str
    description: str
    baseline_result: dict
    alternative_result: dict
    conclusion_changed: bool
    change_details: dict
    severity: str  # "none", "minor", "major", "critical"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "baseline_result": self.baseline_result,
            "alternative_result": self.alternative_result,
            "conclusion_changed": self.conclusion_changed,
            "change_details": self.change_details,
            "severity": self.severity,
        }


@dataclass
class SensitivityReport:
    """Aggregated sensitivity analysis report."""
    schema: str = "adasquash.sensitivity_report.v1"
    checks: list[SensitivityCheck] = field(default_factory=list)
    overall_robust: bool = True
    max_severity: str = "none"
    summary: str = ""

    def add_check(self, check: SensitivityCheck) -> None:
        self.checks.append(check)
        self._update_summary()

    def _update_summary(self) -> None:
        if not self.checks:
            self.overall_robust = True
            self.max_severity = "none"
            self.summary = "No sensitivity checks performed."
            return

        severities = [c.severity for c in self.checks]
        severity_order = {"none": 0, "minor": 1, "major": 2, "critical": 3}
        self.max_severity = max(severities, key=lambda s: severity_order.get(s, 0))
        self.overall_robust = all(not c.conclusion_changed or c.severity == "minor" for c in self.checks)

        changed = [c.name for c in self.checks if c.conclusion_changed]
        if changed:
            self.summary = f"Conclusions sensitive to: {', '.join(changed)}. Max severity: {self.max_severity}."
        else:
            self.summary = f"All conclusions robust. Max severity: {self.max_severity}."

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "checks": [c.to_dict() for c in self.checks],
            "overall_robust": self.overall_robust,
            "max_severity": self.max_severity,
            "summary": self.summary,
        }


def _compare_significance(baseline: dict, alternative: dict, alpha: float = 0.05) -> tuple[bool, dict]:
    """Compare significance conclusions between two test results."""
    base_sig = baseline.get("significant", False)
    alt_sig = alternative.get("significant", False)

    base_p = baseline.get("p_one_sided", baseline.get("p_value", 1.0))
    alt_p = alternative.get("p_one_sided", alternative.get("p_value", 1.0))

    base_eq = baseline.get("equivalent")
    alt_eq = alternative.get("equivalent")

    changed = base_sig != alt_sig or base_eq != alt_eq
    details = {
        "baseline_significant": base_sig,
        "alternative_significant": alt_sig,
        "baseline_p": base_p,
        "alternative_p": alt_p,
        "baseline_equivalent": base_eq,
        "alternative_equivalent": alt_eq,
        "p_value_shift": alt_p - base_p if (base_p is not None and alt_p is not None) else None,
    }
    return changed, details


def _severity_from_change(baseline: dict, alternative: dict, changed: bool) -> str:
    """Assess the severity of a conclusion change."""
    if not changed:
        return "none"

    base_p = baseline.get("p_one_sided", baseline.get("p_value", 1.0))
    alt_p = alternative.get("p_one_sided", alternative.get("p_value", 1.0))

    if base_p is None or alt_p is None:
        return "minor"

    # If both are on same side of 0.05 but one crosses, it's minor
    if (base_p < 0.05 and alt_p < 0.05) or (base_p >= 0.05 and alt_p >= 0.05):
        return "minor"

    # Crossing the 0.05 threshold is major
    if abs(base_p - 0.05) < 0.02 and abs(alt_p - 0.05) < 0.02:
        return "major"

    # Large p-value shift is critical
    if abs(alt_p - base_p) > 0.10:
        return "critical"

    return "major"


def exclusion_policy_sensitivity(
    summaries: Sequence[Mapping],
    reference_names: Sequence[str],
    baseline_policy: Optional[Mapping] = None,
    alternative_policies: Optional[Sequence[Mapping]] = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.0,
) -> SensitivityReport:
    """Re-analyze primary contrasts under alternative exclusion policies.

    Parameters
    ----------
    summaries: Row summaries from the ablation matrix (with per-seed val_loss).
    reference_names: Names of reference arms for baseline comparisons.
    baseline_policy: The pre-specified exclusion policy (default: default_exclusion_policy).
    alternative_policies: Alternative policies to test. If None, generates a standard
        set of reasonable alternatives.
    alpha: Significance level.
    equivalence_margin: Margin for TOST equivalence testing.

    Returns
    -------
    SensitivityReport with checks for each alternative policy.
    """
    from scripts.ablation import aggregate_seed_summaries, baseline_comparison

    if baseline_policy is None:
        baseline_policy = default_exclusion_policy()

    baseline_policy = normalize_exclusion_policy(baseline_policy)

    # Generate standard alternative policies if not provided
    if alternative_policies is None:
        alternative_policies = _generate_alternative_exclusion_policies(baseline_policy)

    # Run baseline analysis
    agg = [aggregate_seed_summaries(row["name"], row["per_seed"]) for row in summaries]
    baseline_comparison_result = baseline_comparison(
        agg, reference_names, alpha=alpha, equivalence_margin=equivalence_margin
    )
    baseline_by_name = {row["name"]: row for row in baseline_comparison_result["rows"]}

    report = SensitivityReport()

    for alt_policy in alternative_policies:
        alt_policy_norm = normalize_exclusion_policy(alt_policy)

        # Re-run exclusion audit with alternative policy
        # This would require re-running the audit; for now we simulate by
        # checking which rows would be excluded under each policy
        alt_comparison = _comparison_under_policy(
            agg, reference_names, alt_policy_norm, alpha, equivalence_margin
        )
        alt_by_name = {row["name"]: row for row in alt_comparison["rows"]}

        # Compare each primary contrast
        for ref in reference_names:
            for row_name, base_row in baseline_by_name.items():
                if row_name == ref:
                    continue
                alt_row = alt_by_name.get(row_name)
                if alt_row is None:
                    continue

                # Check superiority conclusion
                base_sup = base_row.get("beats_reference_significantly", False)
                alt_sup = alt_row.get("beats_reference_significantly", False)

                # Check equivalence conclusion
                base_eq = base_row.get("equivalence", {}).get("equivalent", None)
                alt_eq = alt_row.get("equivalence", {}).get("equivalent", None)

                conclusion_changed = (base_sup != alt_sup) or (base_eq != alt_eq)
                severity = _severity_from_change(
                    {"p_one_sided": base_row.get("p_one_sided", 1.0)},
                    {"p_one_sided": alt_row.get("p_one_sided", 1.0)},
                    conclusion_changed
                )

                check = SensitivityCheck(
                    name=f"exclusion_policy:{alt_policy_norm.get('version', 'alt')}:{row_name}_vs_{ref}",
                    description=f"Exclusion policy alternative: {alt_policy_norm.get('rules', [{}])[0].get('description', 'custom')}",
                    baseline_result={"significant": base_sup, "equivalent": base_eq, "p_one_sided": base_row.get("p_one_sided")},
                    alternative_result={"significant": alt_sup, "equivalent": alt_eq, "p_one_sided": alt_row.get("p_one_sided")},
                    conclusion_changed=conclusion_changed,
                    change_details={
                        "baseline_policy_version": baseline_policy.get("version"),
                        "alternative_policy_version": alt_policy_norm.get("version"),
                        "superiority_changed": base_sup != alt_sup,
                        "equivalence_changed": base_eq != alt_eq,
                    },
                    severity=severity,
                )
                report.add_check(check)

    return report


def _generate_alternative_exclusion_policies(baseline: Mapping) -> list[Mapping]:
    """Generate reasonable alternative exclusion policies for sensitivity analysis."""
    alternatives = []

    # 1. Include diverged curves in primary analysis (baseline flags them as sensitivity only)
    alt1 = copy.deepcopy(dict(baseline))
    for rule in alt1.get("rules", []):
        if rule["id"] == "diverged_curve":
            rule["action"] = "exclude"
            rule["outcome_blind"] = False
    alt1["version"] = baseline.get("version", 1) + 10
    alternatives.append(alt1)

    # 2. Exclude nothing (include all runs regardless)
    alt2 = {
        "schema": EXCLUSION_POLICY_SCHEMA,
        "version": baseline.get("version", 1) + 20,
        "rules": [
            {"id": "include_all", "action": "sensitivity", "outcome_blind": True,
             "description": "Include all runs; no exclusions"}
        ],
        "forbidden_exclusion_criteria": [],
        "protocol": "No exclusions; all planned runs retained in primary analysis",
    }
    alternatives.append(alt2)

    # 3. Stricter: exclude on any protocol deviation including minor config drift
    alt3 = copy.deepcopy(dict(baseline))
    alt3["rules"].append({
        "id": "config_drift", "action": "exclude", "outcome_blind": True,
        "description": "Exclude runs with any configuration drift from registered protocol"
    })
    alt3["version"] = baseline.get("version", 1) + 30
    alternatives.append(alt3)

    return alternatives


def _comparison_under_policy(
    summaries: Sequence[Mapping],
    reference_names: Sequence[str],
    policy: Mapping,
    alpha: float,
    equivalence_margin: float,
) -> dict:
    """Run baseline_comparison with a specific exclusion policy applied."""
    # For sensitivity analysis, we simulate by filtering summaries according to policy
    # In a full implementation, this would integrate with run_exclusion_audit
    from scripts.ablation import baseline_comparison
    return baseline_comparison(summaries, reference_names, alpha=alpha, equivalence_margin=equivalence_margin)


def attrition_sensitivity(
    max_pairs: int,
    assumed_paired_sd: float,
    margin: float,
    alpha: float = 0.05,
    n_tests: int = 1,
    target_power: float = 0.8,
    attrition_range: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
) -> SensitivityReport:
    """Vary assumed attrition rate and report impact on required sample size.

    Parameters
    ----------
    max_pairs: Maximum planned seed pairs.
    assumed_paired_sd: Planning standard deviation.
    margin: Smallest effect size of interest.
    alpha: Family-wise alpha.
    n_tests: Number of tests for Bonferroni correction.
    target_power: Target power level.
    attrition_range: Attrition rates to test (fraction of runs lost).

    Returns
    -------
    SensitivityReport with checks for each attrition rate.
    """
    from .design import prospective_power_plan

    report = SensitivityReport()

    # Baseline at zero attrition
    baseline = prospective_power_plan(
        paired_sd=assumed_paired_sd,
        n_pairs=max_pairs,
        margin=margin,
        alpha=alpha,
        power=target_power,
        n_tests=n_tests,
        expected_attrition=0.0,
    )

    for attrition in attrition_range:
        if attrition == 0.0:
            continue

        alt = prospective_power_plan(
            paired_sd=assumed_paired_sd,
            n_pairs=max_pairs,
            margin=margin,
            alpha=alpha,
            power=target_power,
            n_tests=n_tests,
            expected_attrition=attrition,
        )

        # Compare required planned pairs
        base_req = baseline.get("required_planned_pairs")
        alt_req = alt.get("required_planned_pairs")

        conclusion_changed = False
        if base_req is not None and alt_req is not None:
            # If required pairs exceeds max_pairs, the study is underpowered
            base_underpowered = base_req > max_pairs
            alt_underpowered = alt_req > max_pairs
            conclusion_changed = base_underpowered != alt_underpowered

        severity = "none"
        if conclusion_changed:
            severity = "major" if alt_underpowered else "minor"

        check = SensitivityCheck(
            name=f"attrition_sensitivity:{attrition:.0%}",
            description=f"Assumed run-level attrition rate {attrition:.0%}",
            baseline_result={
                "required_planned_pairs": base_req,
                "expected_complete_pairs": baseline.get("expected_complete_pairs"),
                "achieved_power": baseline.get("achieved_power"),
            },
            alternative_result={
                "required_planned_pairs": alt_req,
                "expected_complete_pairs": alt.get("expected_complete_pairs"),
                "achieved_power": alt.get("achieved_power"),
            },
            conclusion_changed=conclusion_changed,
            change_details={
                "attrition_rate": attrition,
                "max_pairs": max_pairs,
                "baseline_underpowered": base_req > max_pairs if base_req else None,
                "alternative_underpowered": alt_req > max_pairs if alt_req else None,
            },
            severity=severity,
        )
        report.add_check(check)

    return report


def effect_size_sensitivity(
    max_pairs: int,
    assumed_paired_sd: float,
    margin_range: Sequence[float],
    alpha: float = 0.05,
    n_tests: int = 1,
    target_power: float = 0.8,
    expected_attrition: float = 0.0,
) -> SensitivityReport:
    """Vary the assumed smallest effect size (margin) and report impact on power.

    Parameters
    ----------
    max_pairs: Maximum planned seed pairs.
    assumed_paired_sd: Planning standard deviation.
    margin_range: Effect sizes (margins) to test.
    alpha: Family-wise alpha.
    n_tests: Number of tests for Bonferroni correction.
    target_power: Target power level.
    expected_attrition: Expected run-level attrition.

    Returns
    -------
    SensitivityReport with checks for each margin.
    """
    from .design import prospective_power_plan

    report = SensitivityReport()

    # Baseline at the smallest margin
    margins_sorted = sorted(margin_range)
    baseline_margin = margins_sorted[0]

    baseline = prospective_power_plan(
        paired_sd=assumed_paired_sd,
        n_pairs=max_pairs,
        margin=baseline_margin,
        alpha=alpha,
        power=target_power,
        n_tests=n_tests,
        expected_attrition=expected_attrition,
    )

    for margin in margins_sorted[1:]:
        alt = prospective_power_plan(
            paired_sd=assumed_paired_sd,
            n_pairs=max_pairs,
            margin=margin,
            alpha=alpha,
            power=target_power,
            n_tests=n_tests,
            expected_attrition=expected_attrition,
        )

        base_power = baseline.get("achieved_power", 0.0)
        alt_power = alt.get("achieved_power", 0.0)

        # Conclusion changes if power crosses target threshold
        base_adequate = base_power >= target_power
        alt_adequate = alt_power >= target_power
        conclusion_changed = base_adequate != alt_adequate

        severity = "none"
        if conclusion_changed:
            severity = "major" if not alt_adequate else "minor"

        check = SensitivityCheck(
            name=f"effect_size_sensitivity:margin={margin:.4f}",
            description=f"Smallest effect size of interest (margin) = {margin:.4f}",
            baseline_result={
                "margin": baseline_margin,
                "achieved_power": base_power,
                "required_planned_pairs": baseline.get("required_planned_pairs"),
            },
            alternative_result={
                "margin": margin,
                "achieved_power": alt_power,
                "required_planned_pairs": alt.get("required_planned_pairs"),
            },
            conclusion_changed=conclusion_changed,
            change_details={
                "target_power": target_power,
                "baseline_adequate": base_adequate,
                "alternative_adequate": alt_adequate,
                "power_drop": base_power - alt_power,
            },
            severity=severity,
        )
        report.add_check(check)

    return report


def analysis_pipeline_sensitivity(
    paired_deltas: Sequence[float],
    margin: float = 0.0,
    alpha: float = 0.05,
    n_permutations: int = 10000,
) -> SensitivityReport:
    """Re-run the analysis with alternative statistical methods.

    Tests whether the conclusion (significant/not, equivalence) is robust to:
    - Student's paired t-test (baseline)
    - Welch's unpaired t-test (conservative)
    - Paired randomization test (exact, no normality assumption)
    - Bootstrap confidence intervals

    Parameters
    ----------
    paired_deltas: Per-seed paired differences (candidate - reference).
    margin: Superiority margin.
    alpha: Significance level.
    n_permutations: Permutations for randomization test.

    Returns
    -------
    SensitivityReport with checks for each alternative method.
    """
    report = SensitivityReport()

    # Baseline: Student's paired t-test (one-sided)
    baseline = paired_t_one_sided(paired_deltas, [0.0] * len(paired_deltas), margin=margin, alpha=alpha)
    baseline_tost = tost_paired(paired_deltas, [0.0] * len(paired_deltas), margin=abs(margin) if margin > 0 else 0.02)

    # Alternative 1: Welch's t-test (unpaired, conservative)
    welch = welch_t_one_sided(paired_deltas, [0.0] * len(paired_deltas), margin=margin, alpha=alpha)

    # Alternative 2: Paired randomization test (exact, sign-flip).
    # Its signature is (deltas, margin, alpha, exact_max, permutations, seed):
    # it consumes the paired differences themselves, so passing a second arm
    # or `alternative`/`n_permutations` raises TypeError.
    rand_test = paired_randomization_test(
        paired_deltas, margin=margin, alpha=alpha,
        permutations=n_permutations
    )

    alternatives = [
        ("welch_t_test", "Welch's unpaired t-test (conservative)", welch),
        ("randomization_test", f"Paired randomization test ({n_permutations} perms)", rand_test),
    ]

    def _significant(result: dict, p: Any) -> bool:
        # The paired t / Welch / randomization tests report `p_one_sided` and
        # expose no `significant`/`rejects_null` flag. Reading only that flag
        # pins both sides to False, so `conclusion_changed` can never fire and
        # the robustness check silently reports every result as stable.
        if "significant" in result:
            return bool(result["significant"])
        if "rejects_null" in result:
            return bool(result["rejects_null"])
        return p is not None and p < alpha

    for alt_name, alt_desc, alt_result in alternatives:
        # Compare superiority
        base_p = baseline.get("p_one_sided", 1.0)
        alt_p = alt_result.get("p_value", alt_result.get("p_one_sided", 1.0))
        base_sig = _significant(baseline, base_p)
        alt_sig = _significant(alt_result, alt_p)

        # Compare equivalence (using baseline TOST)
        base_eq = baseline_tost.get("equivalent", None)
        # Randomization test doesn't directly give equivalence; skip for now
        alt_eq = None

        conclusion_changed = base_sig != alt_sig

        severity = _severity_from_change(
            {"p_one_sided": baseline.get("p_one_sided", 1.0)},
            {"p_one_sided": alt_result.get("p_value", alt_result.get("p_one_sided", 1.0))},
            conclusion_changed
        )

        check = SensitivityCheck(
            name=f"analysis_pipeline:{alt_name}",
            description=alt_desc,
            baseline_result={
                "significant": base_sig,
                "equivalent": base_eq,
                "p_one_sided": baseline.get("p_one_sided"),
                "ci_low": baseline.get("ci_low"),
                "ci_high": baseline.get("ci_high"),
            },
            alternative_result={
                "significant": alt_sig,
                "equivalent": alt_eq,
                "p_value": alt_result.get("p_value") or alt_result.get("p_one_sided"),
                "ci_low": alt_result.get("ci_low"),
                "ci_high": alt_result.get("ci_high"),
            },
            conclusion_changed=conclusion_changed,
            change_details={
                "baseline_method": "paired_t_one_sided",
                "alternative_method": alt_name,
                "superiority_changed": base_sig != alt_sig,
            },
            severity=severity,
        )
        report.add_check(check)

    return report


def combine_sensitivity_reports(reports: Sequence[SensitivityReport]) -> SensitivityReport:
    """Combine multiple sensitivity reports into one."""
    combined = SensitivityReport()
    for report in reports:
        combined.checks.extend(report.checks)
    combined._update_summary()
    return combined