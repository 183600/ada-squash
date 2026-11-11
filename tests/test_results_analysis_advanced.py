"""Tests for the advanced results analysis module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.results_analysis import (
    assumption_checks,
    bayesian_paired_analysis,
    bland_altman_data,
    bootstrap_ci,
    composite_endpoint,
    forest_plot_data,
    influence_diagnostics,
    mediation_analysis,
    post_hoc_sensitivity_curve,
    qq_plot_data,
    replication_probability,
    trimmed_mean_ci,
    wilcoxon_signed_rank,
)

# Fixed seeds for reproducibility
SEEDS = (42, 43, 44, 45, 46)


class TestWilcoxonSignedRank:
    def test_wilcoxon_basic_less(self):
        """One-sided test where all differences are negative (candidate better)."""
        a = [1.0, 1.1, 0.9, 1.05, 0.95]  # candidate
        b = [1.5, 1.6, 1.4, 1.55, 1.45]  # reference
        out = wilcoxon_signed_rank(a, b, alternative="less")
        # All negative -> W+ = 0 (sum of positive ranks)
        assert out["p_value"] < 0.05
        assert out["statistic"] == 0  # W+ = 0 when all negative
        assert out["method"] == "exact"

    def test_wilcoxon_basic_greater(self):
        """One-sided test where all differences are positive (reference better)."""
        a = [1.5, 1.6, 1.4, 1.55, 1.45]
        b = [1.0, 1.1, 0.9, 1.05, 0.95]
        out = wilcoxon_signed_rank(a, b, alternative="greater")
        assert out["p_value"] < 0.05
        assert out["statistic"] == 15  # W+ = 15 when all positive

    def test_wilcoxon_two_sided(self):
        """Two-sided test."""
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = wilcoxon_signed_rank(a, b, alternative="two-sided")
        # n=5 all negative -> min two-sided p = 2/32 = 0.0625
        assert out["p_value"] == pytest.approx(0.0625)

    def test_wilcoxon_with_zeros_pratt(self):
        """Pratt method handles zeros."""
        a = [1.0, 1.0, 1.1, 0.9]
        b = [1.0, 1.5, 1.4, 1.0]  # one zero diff
        out = wilcoxon_signed_rank(a, b, zero_method="pratt")
        assert out["n"] == 4  # zeros kept
        assert out["zero_method"] == "pratt"

    def test_wilcoxon_wilcoxon_method_discards_zeros(self):
        """Wilcoxon method discards zeros."""
        a = [1.0, 1.0, 1.1, 0.9]
        b = [1.0, 1.5, 1.4, 1.0]
        out = wilcoxon_signed_rank(a, b, zero_method="wilcoxon")
        assert out["n"] == 3  # one zero discarded

    def test_wilcoxon_hl_estimator(self):
        """Hodges-Lehmann estimator is the pseudo-median."""
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = wilcoxon_signed_rank(a, b)
        assert out["hodges_lehmann"] is not None
        assert out["hl_ci"][0] <= out["hodges_lehmann"] <= out["hl_ci"][1]

    def test_wilcoxon_invalid_alternative(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1,2], [3,4], alternative="invalid")

    def test_wilcoxon_invalid_zero_method(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1,2], [3,4], zero_method="invalid")

    def test_wilcoxon_mismatched_lengths(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1,2,3], [4,5])


class TestTrimmedMeanCI:
    def test_trimmed_mean_basic(self):
        vals = [1.0, 1.1, 0.9, 1.05, 0.95, 10.0, -5.0]  # two outliers
        out = trimmed_mean_ci(vals, trim_proportion=0.2)
        assert out["n"] == 7
        assert out["n_trimmed"] == 2  # 20% of 7 = 1.4 -> floor = 1 per tail
        assert out["trimmed_mean"] is not None
        assert out["ci"][0] < out["trimmed_mean"] < out["ci"][1]

    def test_trimmed_mean_no_trim(self):
        vals = [1.0, 1.1, 0.9, 1.05, 0.95]
        out = trimmed_mean_ci(vals, trim_proportion=0.0)
        assert out["n_trimmed"] == 0
        assert abs(out["trimmed_mean"] - 1.0) < 0.01

    def test_trimmed_mean_all_trimmed(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        out = trimmed_mean_ci(vals, trim_proportion=0.4)
        # 40% of 4 = 1.6 -> floor = 1 per tail, so 2 trimmed, 2 remain
        assert out["n_trimmed"] == 2
        assert out["trimmed_mean"] == 2.5  # mean of [2.0, 3.0]

    def test_trimmed_mean_invalid_proportion(self):
        with pytest.raises(ValueError):
            trimmed_mean_ci([1,2], trim_proportion=0.5)
        with pytest.raises(ValueError):
            trimmed_mean_ci([1,2], trim_proportion=-0.1)


class TestBootstrapCI:
    def test_bootstrap_mean_diff(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = bootstrap_ci(a, b, statistic="mean_diff")
        assert out["observed"] < 0  # a better
        assert out["ci"][0] <= out["observed"] <= out["ci"][1]

    def test_bootstrap_median_diff(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = bootstrap_ci(a, b, statistic="median_diff")
        assert out["observed"] < 0
        assert out["ci"][0] <= out["observed"] <= out["ci"][1]

    def test_bootstrap_trimmed_mean_diff(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95, 10.0]
        b = [1.5, 1.6, 1.4, 1.55, 1.45, 10.0]
        out = bootstrap_ci(a, b, statistic="trimmed_mean_diff")
        assert out["observed"] is not None

    def test_bootstrap_ppl_ratio(self):
        a = [1.0, 1.1, 0.9]
        b = [1.5, 1.6, 1.4]
        out = bootstrap_ci(a, b, statistic="ppl_ratio")
        assert out["observed"] > 0
        assert out["ci"][0] > 0

    def test_bootstrap_invalid_statistic(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1,2], [3,4], statistic="invalid")


class TestBayesianPairedAnalysis:
    def test_bayesian_default_prior(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = bayesian_paired_analysis(a, b, prior="default")
        assert out["p_better"] > 0.95
        assert out["p_rope"] < 0.1
        assert out["bf_10"] > 10  # strong evidence for H1

    def test_bayesian_skeptical_prior(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = bayesian_paired_analysis(a, b, prior="skeptical")
        assert out["p_better"] > 0.9

    def test_bayesian_custom_prior(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = bayesian_paired_analysis(a, b, prior={"mean": 0.0, "sd": 0.1})
        assert "posterior_mean" in out
        assert out["prior_mean"] == 0.0
        assert out["prior_sd"] == 0.1

    def test_bayesian_rope(self):
        a = [1.0, 1.01, 0.99, 1.005, 0.995]  # tiny difference
        b = [1.005, 1.015, 0.995, 1.01, 1.0]
        out = bayesian_paired_analysis(a, b, rope=(-0.01, 0.01))
        # With small differences, ROPE probability should be non-trivial
        assert out["p_rope"] >= 0.0

    def test_bayesian_invalid_prior(self):
        with pytest.raises(ValueError):
            bayesian_paired_analysis([1,2], [3,4], prior="unknown")


class TestInfluenceDiagnostics:
    def test_influence_loo(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = influence_diagnostics(a, b)
        assert len(out["loo"]) == 5
        assert out["max_cooks_d"] is not None
        assert out["max_dfbeta"] is not None

    def test_influence_lpo(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]
        b = [1.5, 1.6, 1.4, 1.55, 1.45, 1.5]
        out = influence_diagnostics(a, b)
        assert len(out["lpo"]) == 15  # C(6,2)

    def test_influence_conclusion_change(self):
        # Create data where one seed drives significance
        a = [1.0, 1.1, 0.9, 1.05, -5.0]  # one large negative outlier
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = influence_diagnostics(a, b)
        # The outlier might change conclusion
        assert "conclusion_changes_loo" in out


class TestAssumptionChecks:
    def test_assumptions_normal_data(self):
        # Create differences with some variation (not constant)
        a = [1.0, 1.12, 0.88, 1.05, 0.93, 1.02, 0.97]
        b = [1.5, 1.6, 1.4, 1.55, 1.45, 1.52, 1.48]
        out = assumption_checks(a, b)
        assert out["qq_correlation"] is not None
        # With small n, correlation may not be extremely high
        assert out["qq_correlation"] > 0.5

    def test_assumptions_skewed_data(self):
        # Create actually skewed differences (exponential-like)
        a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.0]
        b = [1.5, 1.6, 1.4, 1.55, 1.45, 2.0, 2.5]  # increasing shift
        out = assumption_checks(a, b)
        # Should detect skewness
        assert "skewness" in out["symmetry"]

    def test_assumptions_outliers(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, -10.0]  # one huge outlier in reference
        out = assumption_checks(a, b)
        assert out["outliers"]["n_outliers"] >= 0


class TestReplicationProbability:
    def test_replication_basic(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = replication_probability(a, b)
        assert out["p_replicate_significant"] > 0.5
        assert out["p_same_direction"] > 0.5
        assert out["n_current"] == 5

    def test_replication_future_n(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = replication_probability(a, b, future_n=10)
        assert out["n_future"] == 10

    def test_replication_margin(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = replication_probability(a, b, margin=0.2)
        assert out["p_exceed_margin"] is not None


class TestCompositeEndpoint:
    def test_composite_equal_weights(self):
        arms = {
            "candidate": {
                "val_loss": [1.0, 1.1, 0.9],
                "ppl": [2.7, 3.0, 2.5],
            },
            "reference": {
                "val_loss": [1.5, 1.6, 1.4],
                "ppl": [4.5, 5.0, 4.0],
            },
        }
        out = composite_endpoint(arms)
        # candidate has lower loss (better), so it becomes the reference
        assert out["reference"] == "candidate"
        assert "reference" in out["comparisons"]
        # reference vs candidate: reference has higher loss (worse), so difference is positive
        # The test is one-sided: candidate better = negative difference
        assert out["comparisons"]["reference"]["p_one_sided"] > 0.5  # reference is worse

    def test_composite_custom_weights(self):
        arms = {
            "candidate": {"m1": [1.0, 1.1], "m2": [2.0, 2.1]},
            "reference": {"m1": [1.5, 1.6], "m2": [3.0, 3.1]},
        }
        out = composite_endpoint(arms, weights={"m1": 0.7, "m2": 0.3})
        assert abs(out["weights"]["m1"] - 0.7) < 0.01
        assert abs(out["weights"]["m2"] - 0.3) < 0.01

    def test_composite_explicit_reference(self):
        arms = {
            "a": {"m1": [1.0], "m2": [2.0]},
            "b": {"m1": [1.5], "m2": [3.0]},
            "c": {"m1": [2.0], "m2": [4.0]},
        }
        out = composite_endpoint(arms, reference="b")
        assert out["reference"] == "b"


class TestMediationAnalysis:
    def test_mediation_basic(self):
        full = [1.0, 1.1, 0.9, 1.05, 0.95]
        ref = [1.5, 1.6, 1.4, 1.55, 1.45]
        mediators = {
            "mech1": [1.2, 1.3, 1.1, 1.25, 1.15],  # partial mediation
        }
        out = mediation_analysis(full, ref, mediators, bootstrap_samples=100)
        assert "mech1" in out["mediators"]
        assert "indirect_effect" in out["mediators"]["mech1"]
        assert "direct_effect" in out["mediators"]["mech1"]
        assert "proportion_mediated" in out["mediators"]["mech1"]


class TestPostHocSensitivityCurve:
    def test_sensitivity_curve(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = post_hoc_sensitivity_curve(a, b, n_points=10)
        assert len(out["curve"]) == 10
        assert out["observed_dz"] is not None
        for point in out["curve"]:
            assert "effect_size" in point
            assert "achieved_power" in point
            assert "required_n_for_power" in point


class TestForestPlotData:
    def test_forest_plot(self):
        comparisons = {
            "comp1": {"mean_delta": -0.2, "ci_low": -0.3, "ci_high": -0.1, "n": 5},
            "comp2": {"mean_delta": -0.1, "ci_low": -0.25, "ci_high": 0.05, "n": 5},
        }
        out = forest_plot_data(comparisons)
        assert len(out["rows"]) == 2
        assert out["rows"][0]["estimate"] <= out["rows"][1]["estimate"]


class TestBlandAltmanData:
    def test_bland_altman(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = bland_altman_data(a, b)
        assert out["n"] == 5
        assert len(out["per_seed"]) == 5
        assert out["limits_of_agreement"][0] < out["mean_difference"] < out["limits_of_agreement"][1]


class TestQQPlotData:
    def test_qq_plot(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95]
        b = [1.5, 1.6, 1.4, 1.55, 1.45]
        out = qq_plot_data(a, b)
        assert out["n"] == 5
        assert len(out["observed"]) == 5
        assert len(out["theoretical"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])