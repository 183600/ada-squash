from __future__ import annotations
"""Efficiency accounting: parameters, FLOPs and the accuracy/cost verdict.

* `parameter_breakdown` splits a model's parameters by role (embedding /
  attention / MLP / normalization / LayerScale / LM head), counting tied
  storage once;
* `analytic_flops_per_token` derives the matmul FLOPs per token from the
  config (forward and backward), including the quadratic T x T attention
  score/context matmuls - the norm layers are elementwise and add no matmul
  FLOPs;
* `efficiency_verdict` marks a gain as viable only when it clears the
  practical margin and the cost ratio stays within the allowed maximum.

The functions are pure (config/model in, dict out) so ablation and sweep
runners can attach the numbers to their summaries.
"""
import math
import numbers
from collections import defaultdict
from typing import Optional

from .metrics import student_t_ppf

__all__ = [
    "DEFAULT_MIN_LOSS_GAIN",
    "DEFAULT_MAX_COST_RATIO",
    "analytic_flops_per_token",
    "efficiency_verdict",
    "mlp_hidden_size",
    "parameter_breakdown",
]

DEFAULT_MIN_LOSS_GAIN = 0.01
DEFAULT_MAX_COST_RATIO = 1.25


def _is_boolean_value(value) -> bool:
    """Python and NumPy booleans (``isinstance(np.bool_, bool)`` is False)."""
    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == 'numpy' and cls.__name__ in ('bool_', 'bool')

# Two-sided 95% Student-t critical values, by degrees of freedom. Efficiency
# measurements usually have very few rounds (the benchmark defaults to 3), so
# the normal 1.96 would report an interval far narrower than the data support.
_T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def _positive_int(value, name: str) -> int:
    """A positive integer geometry field without truncating/coercing it."""
    if _is_boolean_value(value) or value is None:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if converted != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if converted < 1:
        raise ValueError(f"{name} must be >= 1, got {converted}")
    return converted


def _positive_real(value, name: str) -> float:
    """A finite strictly-positive real (a ratio/scale, not a count)."""
    if _is_boolean_value(value) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive number, got {value!r}") from None
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return converted


def _nonnegative_real(value, name: str) -> float:
    """A finite non-negative real threshold, without truthiness coercion."""
    if _is_boolean_value(value) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}") from None
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return converted


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df <= len(_T95):
        return _T95[df - 1]
    return student_t_ppf(0.975, float(df))


def mlp_hidden_size(dim: int, mlp_ratio: float) -> int:
    """Hidden width of the SwiGLU MLP, exactly as `model.MLP` computes it."""
    dim = _positive_int(dim, "dim")
    mlp_ratio = _positive_real(mlp_ratio, "mlp_ratio")
    hidden = int(int(dim * mlp_ratio) * 2 / 3)
    return hidden + 7 & ~7


def analytic_flops_per_token(cfg, backward: bool = False, seq_len: int | None = None) -> dict:
    """Matmul FLOPs per token of the GPT decoder defined by `cfg`.

    Counts every weight-matrix multiplication (attention qkv/proj, SwiGLU
    w1/w2/w3, LM head) as 2 x multiply-adds *plus* the attention score and
    context matmuls: ``Q @ K^T`` and ``softmax(QK^T) @ V`` contribute
    ``4 * d * T^2`` FLOPs per layer, i.e. ``4 * d * T`` per token.

    `seq_len` is the context length the model is actually run at (defaults to
    `cfg.max_seq_len`; pass the trained length when it differs). The score
    count is the dense T x T convention; the causal mask serves only the lower
    triangle, so a causal-exact count is about half - the dense convention is
    the conservative upper bound and is applied identically to every cell.

    `backward=True` returns the training-step estimate: forward plus backward,
    with the multiplier used recorded as ``backward_factor``.
    """
    d = _positive_int(getattr(cfg, "dim", None), "cfg.dim")
    n_layers = _positive_int(getattr(cfg, "n_layers", None), "cfg.n_layers")
    vocab = _positive_int(getattr(cfg, "vocab_size", None), "cfg.vocab_size")
    h = mlp_hidden_size(d, getattr(cfg, "mlp_ratio", None))
    # The default (`cfg.max_seq_len`) is the same kind of context length as an
    # explicit `seq_len`: a fractional/bool/negative table size cannot be
    # silently truncated or clamped to a different protocol. `None` means the
    # config did not record one and keeps the legacy zero-context fallback.
    if seq_len is None:
        raw_len = getattr(cfg, "max_seq_len", 0)
        if raw_len is None:
            raw_len = 0
        len_name = "cfg.max_seq_len"
    else:
        raw_len = seq_len
        len_name = "seq_len"
    if _is_boolean_value(raw_len):
        raise ValueError(f"{len_name} must be an integer, got bool {raw_len!r}")
    try:
        T = int(raw_len)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{len_name} must be an integer, got {raw_len!r}") from None
    if T != raw_len:
        raise ValueError(f"{len_name} must be an integer, got {raw_len!r}")
    if T < 0:
        raise ValueError(f"{len_name} must be >= 0, got {T}")
    attn = float(n_layers * 8 * d * d)          # 4 d x d matmuls/token/layer
    attn_scores = float(4 * n_layers * d * T)   # QK^T + AV, 4 d T^2 per layer / T
    mlp = float(n_layers * 6 * d * h)           # 3 d x h matmuls/token/layer
    lm_head = float(2 * d * vocab)
    forward = attn + attn_scores + mlp + lm_head
    training = 3.0 * forward
    # `backward` selects the reported estimate; both component estimates stay
    # available so callers can compare a forward-only run with a training run.
    per_token = training if backward else forward
    return {
        "attention": attn,
        "attention_scores": attn_scores,
        "mlp": mlp,
        "lm_head": lm_head,
        "forward_per_token": forward,
        "training_per_token": training,
        "per_token": per_token,
        "backward": bool(backward),
        "backward_factor": 3.0,
        "mlp_hidden": h,
        "seq_len": T,
        "note": (
            "matmul FLOPs per token (2 x MACs), including the quadratic attention "
            "score/context matmuls counted densely (causal masking roughly halves "
            "them; the dense convention is the conservative upper bound applied to "
            "every cell). Elementwise ops (softsign/tanh, LayerScale, softmax, "
            "SiLU) are not counted, so norm variants differ here only through the "
            "shared backbone"
        ),
    }


def _param_bucket(name: str) -> str:
    if name.startswith("_orig_mod."):  # torch.compile wrapper
        name = name[len("_orig_mod."):]
    if name.startswith("tok_emb"):
        return "embedding"
    if name.startswith("pos_emb"):
        return "positional_embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    if "norm" in name:
        return "norm"
    if ".ls1" in name or ".ls2" in name or name.endswith(".ls1") or name.endswith(".ls2"):
        return "layerscale"
    if ".attn." in name or name.startswith("attn."):
        return "attention"
    if ".mlp." in name or name.startswith("mlp."):
        return "mlp"
    return "other"


def parameter_breakdown(model) -> dict:
    """Per-role parameter counts, tied storage counted once.

    Returns ``{'total', 'by_role', 'norm_share_of_total', 'tied_storage_skipped'}``.
    Tied embeddings share one tensor between `tok_emb.weight` and
    `lm_head.weight`; the second reference is counted under
    `tied_storage_skipped` rather than double-counting it.
    """
    by_role: dict = defaultdict(int)
    seen: set = set()
    skipped = 0
    for name, p in model.named_parameters(remove_duplicate=False):
        key = p.data_ptr()
        if key == 0:
            key = id(p)
        if key in seen:
            skipped += int(p.numel())
            continue
        seen.add(key)
        by_role[_param_bucket(name)] += int(p.numel())
    total = sum(by_role.values())
    norm = by_role.get("norm", 0)
    return {
        "total": total,
        "by_role": dict(sorted(by_role.items())),
        "norm_share_of_total": (norm / total) if total else 0.0,
        "tied_storage_skipped": skipped,
    }


def efficiency_verdict(
    loss_improvement: Optional[float],
    cost_ratio: Optional[float],
    min_loss_gain: float = DEFAULT_MIN_LOSS_GAIN,
    max_cost_ratio: float = DEFAULT_MAX_COST_RATIO,
    cost_name: str = "compute",
    cost_components: Optional[dict] = None,
    max_component_ratio: Optional[float] = None,
    statistically_significant: Optional[bool] = None,
    cost_ratio_stderr: Optional[float] = None,
    cost_ratio_n: Optional[int] = None,
    cost_is_proxy: bool = False,
) -> dict:
    """Is `loss_improvement` worth `cost_ratio` times the baseline cost?

    `loss_improvement` is positive when the candidate's loss is lower than the
    reference's (a *better* model); `cost_ratio` is candidate/reference (e.g.
    latency, FLOPs or parameter ratio; 1.0 = free). A candidate is viable only
    when the gain clears `min_loss_gain` **and** the cost ratio stays within
    `max_cost_ratio`. `cost_per_loss_point` (cost_ratio per unit of gain) makes
    the exchange rate explicit.

    `cost_components` is an optional ``{name: candidate/reference}`` mapping
    for the other cost axes (parameters, FLOPs, forward latency, peak memory).
    Every present component must stay within `max_component_ratio`
    (default: `max_cost_ratio`), so a win cannot hide a 10x memory or
    parameter blow-up behind one cheap primary cost. Components are reported
    individually; a missing component is never treated as passing, and a value
    that cannot be read as a finite non-negative ratio is a reported failure
    rather than an exception.

    `cost_ratio_stderr` (with `cost_ratio_n`) turns the point estimate into a
    two-sided 95% Student-t interval. Certification then requires the
    interval's *upper* bound to stay within `max_cost_ratio`, not merely the
    point estimate: a noisy benchmark whose ratio could be either 1.01x or
    1.30x has not shown the cost is acceptable. Without a standard error the
    old point-estimate rule applies when no round count is supplied; if
    `cost_ratio_n` is supplied (a stochastic benchmark) but the stderr is
    missing, certification fails closed because a single timing has no
    resolution of its own. A supplied round count below 2 also fails closed
    even if a stderr is present: one round has no degrees of freedom, so a
    zero stderr cannot turn a single timing into a zero-width interval. A
    negative or non-finite standard error is not a measurement either, and a
    round count that cannot be interpreted as an integer (non-numeric,
    non-finite, or fractional) is rejected as an invalid measurement rather
    than raising. A non-finite loss gain or a negative cost ratio is rejected
    rather than treated as an infinite gain or free compute.

    `statistically_significant=None` means the significance gate was **not
    established** (no comparison row, or no significance test was run), not
    that it passed. Certification is fail-closed: only `True` can certify a
    gain; `None` and `False` both block viability. `cost_is_proxy=True` does
    the same for cost: a measurement taken on a different architecture cannot
    certify the candidate's cost.
    """
    if not _is_boolean_value(cost_is_proxy):
        raise ValueError(
            f"cost_is_proxy must be a boolean, got {cost_is_proxy!r}")
    cost_is_proxy = bool(cost_is_proxy)
    if _is_boolean_value(statistically_significant):
        statistically_significant = bool(statistically_significant)
    # Any other non-None value (1, "yes", a list) is not evidence that the
    # significance gate passed. It must be reported as malformed, not silently
    # treated like an absent test while the reason text still looks positive.
    significance_invalid = (statistically_significant is not None
                            and not isinstance(statistically_significant, bool))
    min_gain_value = _nonnegative_real(min_loss_gain, "min_loss_gain")
    max_cost_value = _nonnegative_real(max_cost_ratio, "max_cost_ratio")
    max_component_value = (max_cost_value if max_component_ratio is None
                           else _nonnegative_real(max_component_ratio,
                                                  "max_component_ratio"))
    verdict = {
        "loss_improvement": loss_improvement,
        "cost_ratio": cost_ratio,
        "cost_name": cost_name,
        "min_loss_gain": min_gain_value,
        "max_cost_ratio": max_cost_value,
        "max_component_ratio": max_component_value,
        "statistically_significant": statistically_significant,
        "significance_status": (
            "invalid" if significance_invalid
            else "certified" if statistically_significant is True
            else "not_significant" if statistically_significant is False
            else "no_evidence"),
        "cost_is_proxy": bool(cost_is_proxy),
        "component_costs": {},
        "component_failures": [],
        "gain_meaningful": None,
        "cost_ratio_stderr": None,
        "cost_ratio_n": None,
        "cost_ratio_ci_95": None,
        "cost_uncertainty_certified": None,
        "cost_confidently_acceptable": None,
        "cost_acceptable": None,
        "viable": None,
        "cost_per_loss_point": None,
        "reason": "",
    }
    if loss_improvement is None or cost_ratio is None:
        verdict["reason"] = "missing loss or cost measurement; nothing to judge"
        return verdict
    if _is_boolean_value(loss_improvement) or _is_boolean_value(cost_ratio):
        verdict["reason"] = (
            "loss improvement/cost ratio is a boolean, not a measurement; "
            "nothing to certify")
        return verdict
    try:
        gain = float(loss_improvement)
        cost = float(cost_ratio)
    except (TypeError, ValueError, OverflowError):
        verdict["reason"] = (
            "loss improvement/cost ratio cannot be read as a finite measurement; "
            "nothing to certify")
        return verdict
    if not math.isfinite(gain):
        verdict["reason"] = (
            f"loss improvement {gain!r} is not a finite measurement; nothing to certify")
        return verdict
    if not math.isfinite(cost) or cost < 0.0:
        verdict["reason"] = (
            f"cost ratio {cost!r} is not a finite non-negative measurement; "
            "nothing to certify")
        return verdict
    cost_se = None
    invalid_se = None
    if cost_ratio_stderr is not None:
        # A bool is not an uncertainty measurement: `float(False)` would turn
        # "no stderr recorded" into an exactly-zero interval and let a noisy
        # benchmark certify a cost it never resolved. Reject it like the
        # boolean gain / component measurements.
        if _is_boolean_value(cost_ratio_stderr):
            invalid_se = repr(cost_ratio_stderr)
            candidate_se = None
        else:
            try:
                candidate_se = float(cost_ratio_stderr)
            except (TypeError, ValueError, OverflowError):
                invalid_se = repr(cost_ratio_stderr)
                candidate_se = None
        if invalid_se is None:
            if math.isfinite(candidate_se) and candidate_se >= 0.0:
                cost_se = candidate_se
            else:
                invalid_se = repr(candidate_se)
    cost_n = None
    invalid_n = None
    if cost_ratio_n is not None:
        # Validate integrality on the numeric value, not only on Python float:
        # Decimal/Fraction/NumPy floats are real measurements too, and
        # int(2.5) would silently turn a fractional round count into 2 and
        # let it certify a t-interval.
        if _is_boolean_value(cost_ratio_n):
            invalid_n = repr(cost_ratio_n)
        else:
            try:
                candidate_n = float(cost_ratio_n)
            except (TypeError, ValueError, OverflowError):
                invalid_n = repr(cost_ratio_n)
            else:
                if not math.isfinite(candidate_n) or not candidate_n.is_integer():
                    invalid_n = repr(cost_ratio_n)
                else:
                    cost_n = int(candidate_n)
    enough_rounds = cost_n is None or cost_n >= 2
    if cost_se is not None:
        verdict["cost_ratio_stderr"] = cost_se
    if cost_n is not None:
        verdict["cost_ratio_n"] = cost_n
    # A standard error without a round count has no degrees of freedom, so a
    # claimed 95% interval would fall back to the anti-conservative normal
    # critical value. Report no interval and refuse to certify rather than
    # contradicting the module's small-sample rationale.
    if (cost_se is not None and cost_n is not None
            and enough_rounds and invalid_n is None):
        df = cost_n - 1
        t_crit = _t_critical_95(df)
        half = t_crit * cost_se
        verdict["cost_ratio_ci_95"] = [max(0.0, cost - half), cost + half]
        verdict["cost_ratio_ci_95_df"] = df
    # A round count without a standard error has no resolution, and a supplied
    # round count below 2 has no degrees of freedom: in either case the point
    # estimate cannot prove the cost stays within the cap, so certification
    # must fail closed. A supplied but invalid standard error is not the same
    # as no uncertainty estimate: it cannot certify anything either. The same
    # fail-closed rule applies to a standard error with no round count.
    verdict["cost_uncertainty_certified"] = (
        False if (invalid_se is not None or invalid_n is not None
                  or (cost_se is not None and cost_n is None))
        else None if cost_n is None
        else (cost_se is not None and enough_rounds))
    component_limit = verdict["max_component_ratio"]
    if invalid_se is not None:
        verdict["component_failures"].append(
            f"cost ratio standard error {invalid_se} is not a finite, "
            f"non-negative measurement; cannot certify the {cost_name} cost")
    elif invalid_n is not None:
        verdict["component_failures"].append(
            f"cost ratio round count {invalid_n} is not an integer measurement; "
            f"cannot certify the {cost_name} cost")
    elif verdict["cost_uncertainty_certified"] is False:
        if cost_se is not None and cost_n is None:
            failure = (
                f"cost ratio has a standard error but no round count; at least "
                f"2 paired rounds are needed to form a Student-t interval and "
                f"certify the {cost_name} cost")
        elif cost_se is not None:
            failure = (
                f"cost ratio was measured in {cost_n} round(s); at least 2 "
                f"paired rounds are needed to certify the {cost_name} cost")
        else:
            failure = (
                f"cost ratio has no uncertainty estimate from {cost_n} "
                f"round(s); at least 2 paired rounds are needed to certify the "
                f"{cost_name} cost")
        verdict["component_failures"].append(failure)
    for name, raw_ratio in (cost_components or {}).items():
        if _is_boolean_value(raw_ratio):
            verdict["component_costs"][str(name)] = {
                "ratio": None,
                "acceptable": False,
            }
            verdict["component_failures"].append(
                f"{name} ratio {raw_ratio!r} is a boolean, not a finite "
                "non-negative measurement")
            continue
        if raw_ratio is None:
            verdict["component_costs"][str(name)] = {
                "ratio": None,
                "acceptable": False,
            }
            verdict["component_failures"].append(
                f"{name} is unmeasured; cannot certify it stays within {component_limit:.3f}x")
            continue
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError, OverflowError):
            verdict["component_costs"][str(name)] = {
                "ratio": None,
                "acceptable": False,
            }
            verdict["component_failures"].append(
                f"{name} ratio {raw_ratio!r} is not a finite non-negative measurement")
            continue
        valid = math.isfinite(ratio) and ratio >= 0.0
        acceptable = valid and ratio <= component_limit
        verdict["component_costs"][str(name)] = {
            "ratio": ratio,
            "acceptable": bool(acceptable),
        }
        if not valid:
            verdict["component_failures"].append(
                f"{name} ratio {ratio!r} is not a finite non-negative measurement")
        elif not acceptable:
            verdict["component_failures"].append(
                f"{name} ratio {ratio:.3f}x exceeds {component_limit:.3f}x")
    verdict["gain_meaningful"] = gain >= float(min_loss_gain)
    ci_upper = (None if verdict["cost_ratio_ci_95"] is None
                else float(verdict["cost_ratio_ci_95"][1]))
    # A missing CI is only acceptable when no uncertainty measurement was
    # claimed at all (the legacy point-estimate path). If a round count or a
    # malformed stderr explicitly failed uncertainty certification, the cost
    # cannot simultaneously be reported as confidently acceptable.
    verdict["cost_confidently_acceptable"] = bool(
        not cost_is_proxy
        and verdict["cost_uncertainty_certified"] is not False
        and (ci_upper is None or ci_upper <= float(max_cost_ratio)))
    verdict["cost_acceptable"] = bool(
        not cost_is_proxy
        and cost <= float(max_cost_ratio)
        and verdict["cost_confidently_acceptable"]
        and not verdict["component_failures"])
    # A gain that the seed-level comparison could not call significant is weak
    # evidence: a micro point estimate is not a win on its own, so it cannot be
    # certified as viable regardless of how cheap it is. `None` is not a pass:
    # it means no significance evidence was produced, and the fail-closed rule
    # must not turn missing evidence into certification.
    if statistically_significant is False:
        verdict["component_failures"] = list(verdict["component_failures"]) + [
            "gain is not statistically significant; a micro point estimate is "
            "not convincing evidence of an improvement"]
    elif statistically_significant is None:
        verdict["component_failures"] = list(verdict["component_failures"]) + [
            "no significance evidence is available (no comparison row or no "
            "seed-replicated comparison was run); a point estimate cannot "
            "certify the gain"]
    elif significance_invalid:
        verdict["component_failures"] = list(verdict["component_failures"]) + [
            f"significance flag {statistically_significant!r} is neither a "
            "boolean nor None; it cannot certify the gain"]
    if cost_is_proxy:
        verdict["component_failures"] = list(verdict["component_failures"]) + [
            "cost is a proxy measured on a different architecture; it cannot "
            "certify this candidate's cost"]
    verdict["viable"] = bool(
        verdict["gain_meaningful"] and verdict["cost_acceptable"]
        and statistically_significant is True and not cost_is_proxy)
    if gain > 0:
        verdict["cost_per_loss_point"] = cost / gain
    reasons = []
    if not verdict["gain_meaningful"]:
        reasons.append(
            f"loss gain {gain:+.4f} is below the practical margin {min_loss_gain:g}"
        )
    primary_ok = cost <= float(max_cost_ratio)
    if not primary_ok:
        reasons.append(
            f"cost ratio {cost:.3f}x exceeds the allowed maximum {max_cost_ratio:.3f}x"
        )
    elif ci_upper is not None and ci_upper > float(max_cost_ratio):
        reasons.append(
            f"cost ratio {cost:.3f}x has a 95% upper bound {ci_upper:.3f}x above "
            f"the allowed maximum {max_cost_ratio:.3f}x; the measurement cannot "
            "certify the cost is acceptable"
        )
    reasons.extend(verdict["component_failures"])
    verdict["reason"] = "; ".join(reasons) if reasons else (
        f"gain {gain:+.4f} clears {min_loss_gain:g} at cost {cost:.3f}x within "
        f"{max_cost_ratio:.3f}x"
        + (f" (all {len(verdict['component_costs'])} component ratios within "
           f"{component_limit:.3f}x)" if verdict["component_costs"] else "")
    )
    return verdict
