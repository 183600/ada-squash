"""Decision-theoretic framework for experiment analysis.

Statistical significance (p < 0.05) is not a decision rule. A decision requires:
- Explicit loss function for false positives vs false negatives
- Cost of running additional experiments
- Value of information from different outcomes
- Pre-specified decision thresholds

This module provides:
``LossFunction``
    Abstract base for loss functions (Type I/II error costs, estimation loss).
``DecisionRule``
    Maps statistical evidence to actions (adopt/reject/continue).
``expected_loss``
    Compute expected loss under a decision rule given prior and data.
``value_of_information``
    Compute the expected value of running additional seeds/experiments.
``DecisionAnalysis``
    Complete decision analysis for a study: optimal sample size, stopping rules,
    and final recommendation.

The framework makes the cost of errors explicit so that "p < 0.05" is
replaced by "the expected loss of adopting this variant is acceptable".
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "LossFunction",
    "TypeITypeIILoss",
    "EstimationLoss",
    "DecisionRule",
    "ThresholdDecisionRule",
    "BayesDecisionRule",
    "expected_loss",
    "value_of_information",
    "DecisionAnalysis",
    "run_decision_analysis",
]


class LossFunction(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def loss(self, decision: str, truth: str, **kwargs) -> float:
        """Compute loss for a decision given the true state.

        Parameters
        ----------
        decision: The action taken ("adopt", "reject", "continue").
        truth: The true state ("better", "equivalent", "worse").
        **kwargs: Additional context (effect size, sample size, cost, etc.).

        Returns
        -------
        Non-negative loss value.
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description."""
        pass


@dataclass
class TypeITypeIILoss(LossFunction):
    """Asymmetric loss for Type I and Type II errors.

    Parameters
    ----------
    type_i_cost: Cost of adopting a variant that is not better (false positive).
    type_ii_cost: Cost of rejecting a variant that is better (false negative).
    continue_cost: Cost per additional seed/experiment for "continue" decision.
    """
    type_i_cost: float = 1.0
    type_ii_cost: float = 1.0
    continue_cost: float = 0.1

    def loss(self, decision: str, truth: str, **kwargs) -> float:
        if decision == "adopt":
            if truth == "better":
                return 0.0
            elif truth == "equivalent":
                return self.type_i_cost * 0.5  # Partial cost for equivalent
            else:  # worse
                return self.type_i_cost
        elif decision == "reject":
            if truth == "better":
                return self.type_ii_cost
            else:
                return 0.0
        elif decision == "continue":
            return self.continue_cost
        else:
            raise ValueError(f"Unknown decision: {decision}")

    def describe(self) -> str:
        return (f"Type I/II Loss: Type I={self.type_i_cost}, Type II={self.type_ii_cost}, "
                f"Continue={self.continue_cost}")


@dataclass
class EstimationLoss(LossFunction):
    """Loss based on estimation error (squared error, absolute error).

    Useful when the decision is about parameter estimation rather than
    hypothesis testing.
    """
    loss_type: str = "squared"  # "squared", "absolute"

    def loss(self, decision: str, truth: str, **kwargs) -> float:
        # decision is an estimate, truth is the true value
        try:
            est = float(decision)
            true_val = float(truth)
        except (ValueError, TypeError):
            return float('inf')

        if self.loss_type == "squared":
            return (est - true_val) ** 2
        else:
            return abs(est - true_val)

    def describe(self) -> str:
        return f"Estimation Loss: {self.loss_type} error"


class DecisionRule(ABC):
    """Abstract base class for decision rules."""

    @abstractmethod
    def decide(self, evidence: Mapping[str, Any]) -> str:
        """Map statistical evidence to a decision."""
        pass

    @abstractmethod
    def describe(self) -> str:
        pass


@dataclass
class ThresholdDecisionRule(DecisionRule):
    """Decision rule based on p-value and equivalence thresholds.

    Parameters
    ----------
    alpha: Significance threshold for superiority.
    equivalence_margin: Margin for equivalence testing.
    require_both: If True, require both superiority AND non-equivalence to adopt.
    """
    alpha: float = 0.05
    equivalence_margin: float = 0.0
    require_both: bool = False

    def decide(self, evidence: Mapping[str, Any]) -> str:
        p_sup = evidence.get("p_one_sided", 1.0)
        equiv = evidence.get("equivalent", None)

        if p_sup < self.alpha:
            if self.require_both and equiv is not None:
                return "adopt" if not equiv else "continue"
            return "adopt"
        elif self.equivalence_margin > 0 and equiv is True:
            return "reject"  # Equivalent to reference, no need to adopt
        else:
            return "continue"

    def describe(self) -> str:
        return (f"Threshold Rule: alpha={self.alpha}, "
                f"equivalence_margin={self.equivalence_margin}, "
                f"require_both={self.require_both}")


@dataclass
class BayesDecisionRule(DecisionRule):
    """Decision rule based on posterior probabilities (requires prior).

    Parameters
    ----------
    prior_better: Prior probability that candidate is better.
    prior_equiv: Prior probability that candidate is equivalent.
    prior_worse: Prior probability that candidate is worse.
    loss_fn: Loss function to minimize expected loss.
    """
    prior_better: float = 0.33
    prior_equiv: float = 0.34
    prior_worse: float = 0.33
    loss_fn: LossFunction = field(default_factory=TypeITypeIILoss)

    def decide(self, evidence: Mapping[str, Any]) -> str:
        # Compute posterior probabilities (simplified: use p-value as likelihood)
        # In practice, use proper Bayesian updating with likelihood function
        p_sup = evidence.get("p_one_sided", 1.0)
        equiv = evidence.get("equivalent", None)

        # Rough mapping from p-value to posterior (very approximate)
        # This is a placeholder; real implementation needs full Bayes
        if p_sup < 0.05:
            post_better = 0.8
            post_equiv = 0.1
            post_worse = 0.1
        elif equiv is True:
            post_better = 0.1
            post_equiv = 0.8
            post_worse = 0.1
        else:
            post_better = 0.2
            post_equiv = 0.3
            post_worse = 0.5

        # Expected loss for each decision
        losses = {}
        for decision in ("adopt", "reject", "continue"):
            losses[decision] = (
                post_better * self.loss_fn.loss(decision, "better") +
                post_equiv * self.loss_fn.loss(decision, "equivalent") +
                post_worse * self.loss_fn.loss(decision, "worse")
            )

        return min(losses, key=losses.get)

    def describe(self) -> str:
        return (f"Bayes Rule: priors=(better={self.prior_better}, "
                f"equiv={self.prior_equiv}, worse={self.prior_worse}), "
                f"loss={self.loss_fn.describe()}")


def expected_loss(
    decision_rule: DecisionRule,
    evidence: Mapping[str, Any],
    loss_fn: LossFunction,
    truth_probs: Optional[Mapping[str, float]] = None,
) -> float:
    """Compute expected loss of a decision rule under given truth probabilities.

    If truth_probs not provided, uses uniform prior over {better, equivalent, worse}.
    """
    decision = decision_rule.decide(evidence)

    if truth_probs is None:
        truth_probs = {"better": 1/3, "equivalent": 1/3, "worse": 1/3}

    total_loss = 0.0
    for truth, prob in truth_probs.items():
        total_loss += prob * loss_fn.loss(decision, truth)

    return total_loss


def value_of_information(
    current_evidence: Mapping[str, Any],
    additional_seeds: int,
    decision_rule: DecisionRule,
    loss_fn: LossFunction,
    prior_sd: float,
    effect_size: float,
    truth_probs: Optional[Mapping[str, float]] = None,
    n_simulations: int = 1000,
) -> float:
    """Compute expected value of information from additional seeds.

    Uses Monte Carlo simulation to estimate how much additional data would
    reduce expected loss.
    """
    import random

    current_loss = expected_loss(decision_rule, current_evidence, loss_fn, truth_probs)

    # Simulate future evidence
    future_losses = []
    for _ in range(n_simulations):
        # Simulate additional paired differences
        # Assuming true effect = effect_size, noise = prior_sd
        n_total = len(current_evidence.get("paired_deltas", [])) + additional_seeds
        # For simplicity, assume current evidence is summarized by its mean/std
        current_mean = current_evidence.get("mean_delta", effect_size)
        current_n = current_evidence.get("n_pairs", 0)

        # Generate synthetic future data
        combined_deltas = [random.gauss(effect_size, prior_sd) for _ in range(n_total)]

        # Simulate analysis result
        future_evidence = {
            "p_one_sided": _simulate_p_value(combined_deltas, margin=0.0),
            "equivalent": _simulate_equivalence(combined_deltas, margin=0.02),
            "mean_delta": sum(combined_deltas) / len(combined_deltas),
            "n_pairs": n_total,
            "paired_deltas": combined_deltas,
        }

        loss = expected_loss(decision_rule, future_evidence, loss_fn, truth_probs)
        future_losses.append(loss)

    expected_future_loss = sum(future_losses) / len(future_losses)
    voi = current_loss - expected_future_loss

    return max(0.0, voi)  # VOI cannot be negative


def _simulate_p_value(deltas: Sequence[float], margin: float = 0.0) -> float:
    """Simulate a p-value from paired differences."""
    from .metrics import paired_t_one_sided
    if len(deltas) < 2:
        return 1.0
    result = paired_t_one_sided(list(deltas), [0.0] * len(deltas), margin=margin)
    return result.get("p_one_sided", 1.0)


def _simulate_equivalence(deltas: Sequence[float], margin: float) -> bool:
    """Simulate TOST equivalence result."""
    from .metrics import tost_paired
    if len(deltas) < 2:
        return False
    result = tost_paired(list(deltas), [0.0] * len(deltas), margin=margin)
    return result.get("equivalent", False)


@dataclass
class DecisionAnalysis:
    """Complete decision analysis for a study."""
    schema: str = "adasquash.decision_analysis.v1"
    loss_function: str = ""
    decision_rule: str = ""
    current_evidence: dict = field(default_factory=dict)
    expected_loss_current: float = 0.0
    optimal_additional_seeds: int = 0
    voi_per_seed: list[float] = field(default_factory=list)
    final_recommendation: str = ""
    sensitivity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "loss_function": self.loss_function,
            "decision_rule": self.decision_rule,
            "current_evidence": self.current_evidence,
            "expected_loss_current": self.expected_loss_current,
            "optimal_additional_seeds": self.optimal_additional_seeds,
            "voi_per_seed": self.voi_per_seed,
            "final_recommendation": self.final_recommendation,
            "sensitivity": self.sensitivity,
        }


def run_decision_analysis(
    paired_deltas: Sequence[float],
    loss_fn: Optional[LossFunction] = None,
    decision_rule: Optional[DecisionRule] = None,
    prior_sd: float = 0.1,
    effect_size: Optional[float] = None,
    max_additional_seeds: int = 10,
    truth_probs: Optional[Mapping[str, float]] = None,
) -> DecisionAnalysis:
    """Run a complete decision analysis for the current evidence.

    Parameters
    ----------
    paired_deltas: Current per-seed paired differences.
    loss_fn: Loss function (default: Type I/II with equal costs).
    decision_rule: Decision rule (default: Threshold with alpha=0.05).
    prior_sd: Prior standard deviation for effect size.
    effect_size: Estimated true effect size (default: current mean).
    max_additional_seeds: Maximum additional seeds to consider.
    truth_probs: Prior probabilities for true state.

    Returns
    -------
    DecisionAnalysis with optimal sample size and recommendation.
    """
    if loss_fn is None:
        loss_fn = TypeITypeIILoss(type_i_cost=1.0, type_ii_cost=1.0, continue_cost=0.1)
    if decision_rule is None:
        decision_rule = ThresholdDecisionRule(alpha=0.05, equivalence_margin=0.02)
    if effect_size is None:
        effect_size = sum(paired_deltas) / len(paired_deltas) if paired_deltas else 0.0

    current_evidence = {
        "paired_deltas": list(paired_deltas),
        "mean_delta": effect_size,
        "n_pairs": len(paired_deltas),
        "p_one_sided": _simulate_p_value(paired_deltas),
        "equivalent": _simulate_equivalence(paired_deltas, 0.02),
    }

    current_loss = expected_loss(decision_rule, current_evidence, loss_fn, truth_probs)

    # Evaluate VOI for additional seeds
    voi_per_seed = []
    for k in range(1, max_additional_seeds + 1):
        voi = value_of_information(
            current_evidence, k, decision_rule, loss_fn, prior_sd, effect_size, truth_probs
        )
        voi_per_seed.append(voi)

    # Optimal additional seeds: where marginal VOI < continue_cost
    continue_cost = loss_fn.continue_cost if isinstance(loss_fn, TypeITypeIILoss) else 0.1
    optimal = 0
    for i, voi in enumerate(voi_per_seed):
        if voi > continue_cost:
            optimal = i + 1
        else:
            break

    # Final recommendation
    final_decision = decision_rule.decide(current_evidence)
    if final_decision == "continue" and optimal > 0:
        final_recommendation = f"Continue with {optimal} more seeds (VOI > cost)"
    elif final_decision == "continue":
        final_recommendation = "Stop: VOI of additional seeds < cost"
    else:
        final_recommendation = f"{final_decision.capitalize()} based on current evidence"

    # Sensitivity to loss function parameters
    sensitivity = {}
    if isinstance(loss_fn, TypeITypeIILoss):
        for type_i_mult in [0.5, 1.0, 2.0, 5.0]:
            alt_loss = TypeITypeIILoss(
                type_i_cost=loss_fn.type_i_cost * type_i_mult,
                type_ii_cost=loss_fn.type_ii_cost,
                continue_cost=loss_fn.continue_cost
            )
            alt_loss_val = expected_loss(decision_rule, current_evidence, alt_loss, truth_probs)
            sensitivity[f"type_i_cost_x{type_i_mult}"] = alt_loss_val

    return DecisionAnalysis(
        loss_function=loss_fn.describe(),
        decision_rule=decision_rule.describe(),
        current_evidence=current_evidence,
        expected_loss_current=current_loss,
        optimal_additional_seeds=optimal,
        voi_per_seed=voi_per_seed,
        final_recommendation=final_recommendation,
        sensitivity=sensitivity,
    )