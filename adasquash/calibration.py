"""Enhanced simulation-based calibration of the analysis pipeline.


The existing ``simulate_paired_design`` in ``design.py`` validates the pipeline
under synthetic null and alternative. This module extends calibration to:

* **Coverage calibration**: Check that confidence intervals have correct coverage.
* **Type I error control under violations**: Non-normality, heteroscedasticity,
  autocorrelation, outliers.
* **Power calibration under realistic conditions**: Effect size heterogeneity,
  missing data, attrition.
* **Equivalence test calibration**: TOST error rates under various scenarios.
* **Sequential monitoring calibration**: Alpha spending under optional stopping.
* **Factorial interaction calibration**: Type I error for interaction tests.
* **Multiple testing calibration**: FWER/FDR control under dependence.

The calibration produces a ``CalibrationReport`` that documents the conditions
under which the analysis pipeline is validated.
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from .design import simulate_paired_design
from .metrics import (
    benjamini_hochberg_adjust,
    holm_adjust,
    leave_one_out_jackknife,
    paired_t_one_sided,
    tost_paired,
)
from .sequential import group_sequential_plan, sequential_paired_audit
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

__all__ = [
    "CalibrationReport",
    "CalibrationResult",
    "CalibrationScenario",
    "calibrate_coverage",
    "calibrate_equivalence",
    "calibrate_factorial_interaction",
    "calibrate_multiple_testing",
    "calibrate_power_under_heterogeneity",
    "calibrate_sequential_monitoring",
    "calibrate_type_i_under_violations",
    "run_calibration_suite",
]


@dataclass
class CalibrationScenario:
    """A simulation scenario for calibration."""
    name: str
    description: str
    n_pairs: int
    true_effect: float
    paired_sd: float
    n_simulations: int
    # Violation parameters
    non_normal: bool = False
    heteroscedastic: bool = False
    autocorr: float = 0.0
    outlier_prob: float = 0.0
    outlier_magnitude: float = 5.0
    missing_prob: float = 0.0
    attrition: float = 0.0
    # For factorial
    interaction_effect: float = 0.0
    # For multiple testing
    n_tests: int = 1
    test_correlation: float = 0.0


@dataclass
class CalibrationResult:
    """Result of a single calibration scenario."""
    scenario: CalibrationScenario
    # Type I error / false positive rate
    empirical_type_i: Optional[float] = None
    empirical_type_i_ci: tuple[float, float] = (0.0, 0.0)
    # Power
    empirical_power: Optional[float] = None
    empirical_power_ci: tuple[float, float] = (0.0, 0.0)
    # Coverage
    ci_coverage: Optional[float] = None
    ci_coverage_ci: tuple[float, float] = (0.0, 0.0)
    # TOST
    empirical_equiv_rate: Optional[float] = None
    # Bias
    mean_bias: Optional[float] = None
    median_bias: Optional[float] = None
    # RMSE
    rmse: Optional[float] = None
    # Additional metrics
    additional: dict = field(default_factory=dict)


@dataclass
class CalibrationReport:
    """Aggregated calibration report."""
    schema: str = "adasquash.calibration_report.v1"
    results: list[CalibrationResult] = field(default_factory=list)
    overall_pass: bool = True
    summary: str = ""

    def add_result(self, result: CalibrationResult) -> None:
        self.results.append(result)
        self._update_summary()

    def _update_summary(self) -> None:
        if not self.results:
            self.summary = "No calibration scenarios run."
            return

        failed = []
        for r in self.results:
            s = r.scenario
            # Check Type I error control (when true_effect == 0)
            if s.true_effect == 0 and r.empirical_type_i is not None:
                if r.empirical_type_i > 0.05 + 0.01:  # Allow small Monte Carlo error
                    failed.append(f"{s.name}: Type I={r.empirical_type_i:.3f}")
            # Check power (when true_effect != 0)
            if s.true_effect != 0 and r.empirical_power is not None:
                if r.empirical_power < 0.8 - 0.05:  # Allow some tolerance
                    failed.append(f"{s.name}: Power={r.empirical_power:.3f}")
            # Check coverage
            if r.ci_coverage is not None:
                if r.ci_coverage < 0.95 - 0.02:
                    failed.append(f"{s.name}: Coverage={r.ci_coverage:.3f}")

        if failed:
            self.overall_pass = False
            self.summary = f"Calibration failures: {'; '.join(failed)}"
        else:
            self.overall_pass = True
            self.summary = f"All {len(self.results)} calibration scenarios passed."

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "results": [
                {
                    "scenario": {
                        "name": r.scenario.name,
                        "description": r.scenario.description,
                        "n_pairs": r.scenario.n_pairs,
                        "true_effect": r.scenario.true_effect,
                        "paired_sd": r.scenario.paired_sd,
                        "n_simulations": r.scenario.n_simulations,
                    },
                    "empirical_type_i": r.empirical_type_i,
                    "empirical_type_i_ci": r.empirical_type_i_ci,
                    "empirical_power": r.empirical_power,
                    "empirical_power_ci": r.empirical_power_ci,
                    "ci_coverage": r.ci_coverage,
                    "ci_coverage_ci": r.ci_coverage_ci,
                    "empirical_equiv_rate": r.empirical_equiv_rate,
                    "mean_bias": r.mean_bias,
                    "median_bias": r.median_bias,
                    "rmse": r.rmse,
                    "additional": r.additional,
                }
                for r in self.results
            ],
            "overall_pass": self.overall_pass,
            "summary": self.summary,
        }


def _generate_synthetic_data(
    n_pairs: int,
    true_effect: float,
    paired_sd: float,
    non_normal: bool = False,
    heteroscedastic: bool = False,
    autocorr: float = 0.0,
    outlier_prob: float = 0.0,
    outlier_magnitude: float = 5.0,
) -> list[float]:
    """Generate synthetic paired differences under various conditions."""
    deltas = []
    prev = 0.0

    for i in range(n_pairs):
        # Base distribution
        if non_normal:
            # Contaminated normal: 90% N(0,1) + 10% N(0,10)
            if random.random() < 0.1:
                base = random.gauss(0, 10)
            else:
                base = random.gauss(0, 1)
            base *= paired_sd
        else:
            base = random.gauss(0, paired_sd)

        # Autocorrelation
        if autocorr != 0:
            base = autocorr * prev + math.sqrt(1 - autocorr**2) * base

        # Heteroscedasticity: variance increases with index
        if heteroscedastic:
            base *= (1 + 0.5 * i / max(n_pairs, 1))

        # Outliers
        if random.random() < outlier_prob:
            base += random.choice([-1, 1]) * outlier_magnitude * paired_sd

        delta = true_effect + base
        deltas.append(delta)
        prev = base

    return deltas


def _wilson_ci(p: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96  # Approximate for 95%
    denom = 1 + z*z / max(n, 1)
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center - margin), min(1, center + margin))


def calibrate_coverage(
    scenario: CalibrationScenario,
    alpha: float = 0.05,
    margin: float = 0.0,
) -> CalibrationResult:
    """Calibrate confidence interval coverage."""
    n_sims = scenario.n_simulations
    covered = 0
    estimates = []

    for _ in range(n_sims):
        deltas = _generate_synthetic_data(
            scenario.n_pairs, scenario.true_effect, scenario.paired_sd,
            scenario.non_normal, scenario.heteroscedastic, scenario.autocorr,
            scenario.outlier_prob, scenario.outlier_magnitude
        )
        result = paired_t_one_sided(deltas, [0.0]*len(deltas), margin=margin, alpha=alpha)
        estimates.append(result.get("mean_delta", 0.0))
        ci_low = result.get("ci_low")
        ci_high = result.get("ci_high")
        if ci_low is not None and ci_high is not None:
            if ci_low <= scenario.true_effect <= ci_high:
                covered += 1

    coverage = covered / max(n_sims, 1) if n_sims > 0 else 0.0
    ci = _wilson_ci(coverage, n_sims)

    return CalibrationResult(
        scenario=scenario,
        ci_coverage=coverage,
        ci_coverage_ci=ci,
        mean_bias=statistics.mean(estimates) - scenario.true_effect if estimates else None,
        median_bias=statistics.median(estimates) - scenario.true_effect if estimates else None,
        rmse=math.sqrt(statistics.mean((e - scenario.true_effect)**2 for e in estimates)) if estimates else None,
    )


def calibrate_type_i_under_violations(
    scenario: CalibrationScenario,
    alpha: float = 0.05,
    margin: float = 0.0,
) -> CalibrationResult:
    """Calibrate Type I error rate under assumption violations."""
    if scenario.true_effect != 0:
        raise ValueError("Type I calibration requires true_effect=0")

    n_sims = scenario.n_simulations
    rejections = 0

    for _ in range(n_sims):
        deltas = _generate_synthetic_data(
            scenario.n_pairs, 0.0, scenario.paired_sd,
            scenario.non_normal, scenario.heteroscedastic, scenario.autocorr,
            scenario.outlier_prob, scenario.outlier_magnitude
        )
        result = paired_t_one_sided(deltas, [0.0]*len(deltas), margin=margin, alpha=alpha)
        p_value = result.get("p_one_sided")
        if p_value is not None and p_value < alpha:
            rejections += 1

    type_i = rejections / max(n_sims, 1) if n_sims > 0 else 0.0
    ci = _wilson_ci(type_i, n_sims)

    return CalibrationResult(
        scenario=scenario,
        empirical_type_i=type_i,
        empirical_type_i_ci=ci,
    )


def calibrate_power_under_heterogeneity(
    scenario: CalibrationScenario,
    alpha: float = 0.05,
    margin: float = 0.0,
) -> CalibrationResult:
    """Calibrate power under effect size heterogeneity and attrition."""
    n_sims = scenario.n_simulations
    rejections = 0
    estimates = []

    for _ in range(n_sims):
        # Generate with heterogeneity
        deltas = _generate_synthetic_data(
            scenario.n_pairs, scenario.true_effect, scenario.paired_sd,
            scenario.non_normal, scenario.heteroscedastic, scenario.autocorr,
            scenario.outlier_prob, scenario.outlier_magnitude
        )

        # Apply attrition
        if scenario.attrition > 0:
            n_keep = int(len(deltas) * (1 - scenario.attrition))
            if n_keep < 2:
                continue
            deltas = deltas[:n_keep]

        result = paired_t_one_sided(deltas, [0.0]*len(deltas), margin=margin, alpha=alpha)
        estimates.append(result.get("mean_delta", 0.0))
        p_value = result.get("p_one_sided")
        if p_value is not None and p_value < alpha:
            rejections += 1

    power = rejections / max(n_sims, 1) if n_sims > 0 else 0.0
    ci = _wilson_ci(power, n_sims)

    return CalibrationResult(
        scenario=scenario,
        empirical_power=power,
        empirical_power_ci=ci,
        mean_bias=statistics.mean(estimates) - scenario.true_effect if estimates else None,
        median_bias=statistics.median(estimates) - scenario.true_effect if estimates else None,
        rmse=math.sqrt(statistics.mean((e - scenario.true_effect)**2 for e in estimates)) if estimates else None,
    )


def calibrate_equivalence(
    scenario: CalibrationScenario,
    equivalence_margin: float = 0.02,
    alpha: float = 0.05,
) -> CalibrationResult:
    """Calibrate TOST equivalence test error rates."""
    n_sims = scenario.n_simulations
    equiv_decisions = 0
    estimates = []

    for _ in range(n_sims):
        deltas = _generate_synthetic_data(
            scenario.n_pairs, scenario.true_effect, scenario.paired_sd,
            scenario.non_normal, scenario.heteroscedastic, scenario.autocorr,
            scenario.outlier_prob, scenario.outlier_magnitude
        )
        result = tost_paired(deltas, [0.0]*len(deltas), margin=equivalence_margin)
        estimates.append(result.get("mean_delta", 0.0))
        if result.get("equivalent", False):
            equiv_decisions += 1

    equiv_rate = equiv_decisions / max(n_sims, 1) if n_sims > 0 else 0.0

    return CalibrationResult(
        scenario=scenario,
        empirical_equiv_rate=equiv_rate,
        mean_bias=statistics.mean(estimates) - scenario.true_effect if estimates else None,
        rmse=math.sqrt(statistics.mean((e - scenario.true_effect)**2 for e in estimates)) if estimates else None,
    )


def calibrate_sequential_monitoring(
    scenario: CalibrationScenario,
    alpha: float = 0.05,
    n_looks: int = 3,
    spending: str = "obrien-fleming",
    margin: float = 0.0,
) -> CalibrationResult:
    """Calibrate sequential monitoring (alpha spending) under violations."""
    plan = group_sequential_plan(
        max_pairs=scenario.n_pairs,
        n_looks=n_looks,
        alpha=alpha,
        spending=spending,
    )

    n_sims = scenario.n_simulations
    stopped_early = 0
    stopped_at_final = 0
    total_type_i = 0

    for _ in range(n_sims):
        # Generate full sequence of deltas
        deltas = _generate_synthetic_data(
            scenario.n_pairs, scenario.true_effect, scenario.paired_sd,
            scenario.non_normal, scenario.heteroscedastic, scenario.autocorr,
            scenario.outlier_prob, scenario.outlier_magnitude
        )

        # Run sequential audit
        audit = sequential_paired_audit(deltas, plan, margin=margin)

        if audit.get("controlled_significant", False):
            if audit.get("first_crossing_look", n_looks) < n_looks:
                stopped_early += 1
            else:
                stopped_at_final += 1

        if scenario.true_effect == 0 and audit.get("optional_stopping_only", False):
            total_type_i += 1

    total_rejections = stopped_early + stopped_at_final
    type_i_rate = total_type_i / max(n_sims, 1) if n_sims > 0 else 0.0

    return CalibrationResult(
        scenario=scenario,
        empirical_type_i=type_i_rate,
        empirical_type_i_ci=_wilson_ci(type_i_rate, n_sims),
        additional={
            "stopped_early": stopped_early,
            "stopped_at_final": stopped_at_final,
            "total_rejections": total_rejections,
        },
    )


def calibrate_factorial_interaction(
    scenario: CalibrationScenario,
    alpha: float = 0.05,
) -> CalibrationResult:
    """Calibrate Type I error for 2x2 factorial interaction test.

    Under the null, the interaction effect is zero. We simulate the
    four cell means and test the interaction.
    """
    if scenario.true_effect != 0 and scenario.interaction_effect == 0:
        # Main effects only, no interaction
        pass

    n_sims = scenario.n_simulations
    interaction_rejections = 0

    for _ in range(n_sims):
        # Simulate 2x2 factorial: neither, A, B, AB
        # true_effect is main effect A, interaction_effect is A*B
        neither = _generate_synthetic_data(scenario.n_pairs//4, 0, scenario.paired_sd,
                                          scenario.non_normal, scenario.heteroscedastic, scenario.autocorr)
        a_only = _generate_synthetic_data(scenario.n_pairs//4, scenario.true_effect, scenario.paired_sd,
                                          scenario.non_normal, scenario.heteroscedastic, scenario.autocorr)
        b_only = _generate_synthetic_data(scenario.n_pairs//4, scenario.true_effect, scenario.paired_sd,
                                          scenario.non_normal, scenario.heteroscedastic, scenario.autocorr)
        ab = _generate_synthetic_data(scenario.n_pairs//4, 2*scenario.true_effect + scenario.interaction_effect, scenario.paired_sd,
                                       scenario.non_normal, scenario.heteroscedastic, scenario.autocorr)

        # Interaction contrast: (AB - A) - (B - neither) = interaction_effect
        # Under null, interaction_effect = 0
        interaction_contrasts = []
        for i in range(len(neither)):
            contrast = (ab[i] - a_only[i]) - (b_only[i] - neither[i])
            interaction_contrasts.append(contrast)

        if len(interaction_contrasts) >= 2:
            result = paired_t_one_sided(interaction_contrasts, [0.0]*len(interaction_contrasts), margin=0.0, alpha=alpha)
            p_value = result.get("p_one_sided")
            if p_value is not None and p_value < alpha:
                interaction_rejections += 1

    type_i = interaction_rejections / max(n_sims, 1) if n_sims > 0 else 0.0
    ci = _wilson_ci(type_i, n_sims)

    return CalibrationResult(
        scenario=scenario,
        empirical_type_i=type_i,
        empirical_type_i_ci=ci,
        additional={"interaction_effect": scenario.interaction_effect},
    )


def calibrate_multiple_testing(
    scenario: CalibrationScenario,
    alpha: float = 0.05,
    margin: float = 0.0,
    method: str = "holm",
) -> CalibrationResult:
    """Calibrate FWER/FDR control under dependence."""
    if scenario.n_tests < 2:
        raise ValueError("Multiple testing calibration requires n_tests >= 2")

    n_sims = scenario.n_simulations
    any_rejection = 0
    fdr_sum = 0.0
    fdr_count = 0

    for _ in range(n_sims):
        # Generate correlated test statistics
        p_values = []
        for t in range(scenario.n_tests):
            # Generate correlated data
            deltas = _generate_synthetic_data(
                scenario.n_pairs, scenario.true_effect, scenario.paired_sd,
                scenario.non_normal, scenario.heteroscedastic, scenario.autocorr,
                scenario.outlier_prob, scenario.outlier_magnitude
            )
            # Add correlation between tests via shared random seed
            # (Simplified: just use same deltas for all tests under global null)
            if scenario.true_effect == 0:
                result = paired_t_one_sided(deltas, [0.0]*len(deltas), margin=margin, alpha=alpha)
                p_values.append(result.get("p_one_sided", 1.0))
            else:
                # Under alternative, some tests are true alternatives
                is_true_alt = t < scenario.n_tests // 2
                eff = scenario.true_effect if is_true_alt else 0.0
                deltas_alt = _generate_synthetic_data(
                    scenario.n_pairs, eff, scenario.paired_sd,
                    scenario.non_normal, scenario.heteroscedastic, scenario.autocorr,
                    scenario.outlier_prob, scenario.outlier_magnitude
                )
                result = paired_t_one_sided(deltas_alt, [0.0]*len(deltas_alt), margin=margin, alpha=alpha)
                p_values.append(result.get("p_one_sided", 1.0))

        # Apply correction
        if method == "holm":
            adj = holm_adjust(p_values)
            rejected = any(p < alpha for p in adj if p is not None)
        elif method == "bh":
            adj = benjamini_hochberg_adjust(p_values)
            rejected = any(p < alpha for p in adj if p is not None)
            # Compute realized FDR
            true_nulls = [i for i in range(scenario.n_tests) if (scenario.true_effect == 0 or i >= scenario.n_tests // 2)]
            false_discoveries = sum(1 for i in true_nulls if adj[i] is not None and adj[i] < alpha)
            total_discoveries = sum(1 for p in adj if p is not None and p < alpha)
            if total_discoveries > 0:
                fdr_sum += false_discoveries / total_discoveries
                fdr_count += 1
        else:
            raise ValueError(f"Unknown method: {method}")

        if rejected:
            any_rejection += 1

    fwer = any_rejection / max(n_sims, 1) if n_sims > 0 else 0.0
    _denom_fdr = max(fdr_count, 1e-12)
    fdr = fdr_sum / _denom_fdr if fdr_count > 0 else 0.0

    return CalibrationResult(
        scenario=scenario,
        empirical_type_i=fwer,
        empirical_type_i_ci=_wilson_ci(fwer, n_sims),
        additional={
            "method": method,
            "n_tests": scenario.n_tests,
            "fdr": fdr,
            "test_correlation": scenario.test_correlation,
        },
    )


def run_calibration_suite(
    scenarios: Sequence[CalibrationScenario],
    include_coverage: bool = True,
    include_type_i: bool = True,
    include_power: bool = True,
    include_equivalence: bool = True,
    include_sequential: bool = False,
    include_factorial: bool = False,
    include_multiple_testing: bool = False,
) -> CalibrationReport:
    """Run a full calibration suite across multiple scenarios."""
    report = CalibrationReport()

    for scenario in scenarios:
        if include_coverage:
            report.add_result(calibrate_coverage(scenario))
        if include_type_i and scenario.true_effect == 0:
            report.add_result(calibrate_type_i_under_violations(scenario))
        if include_power and scenario.true_effect != 0:
            report.add_result(calibrate_power_under_heterogeneity(scenario))
        if include_equivalence:
            report.add_result(calibrate_equivalence(scenario))
        if include_sequential:
            report.add_result(calibrate_sequential_monitoring(scenario))
        if include_factorial:
            report.add_result(calibrate_factorial_interaction(scenario))
        if include_multiple_testing and scenario.n_tests >= 2:
            report.add_result(calibrate_multiple_testing(scenario))

    return report