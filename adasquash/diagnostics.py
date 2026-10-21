from __future__ import annotations
"""Training-curve diagnosis and the ordered NaN/Inf triage.

Why this module exists
----------------------
Two failure modes are easy to *log* but hard to *read*:

* **A loss curve is not a final number.** A final training loss that fell
  beautifully says nothing about generalization. The decision "overfitted /
  underfitted / diverged" is a property of the *pair* of curves, so
  `diagnose_train_val_curve` reads the recent train and validation points
  together and returns a classification with the evidence attached, plus the
  concrete repo tool that addresses it (early stopping, `scripts/tune.py`,
  `scripts/audit_data.py`, `scripts/sanity_overfit.py`).
* **A NaN is not a verdict.** `triage_nonfinite` walks the failure back
  through the fixed investigation order - (1) data NaN/Inf or invalid token
  ids, (2) fp32 forward / softmax / division pathologies, (3) mixed-precision
  overflow, (4) gradient explosion or too-large learning rate - and reports
  the *first* stage that actually reproduces the non-finiteness, instead of
  guessing.

Everything here is import-light (torch only) so the training entrypoints can
call it at the moment of failure; the pure classification functions take
plain dicts/lists and are unit-testable without a model.
"""
import bisect
import math
from typing import Optional, Sequence

import torch

__all__ = [
    "NAN_TRIAGE_ORDER",
    "DEFAULT_CURVE_THRESHOLDS",
    "classify_nan_stage",
    "diagnose_train_val_curve",
    "first_nonfinite_module",
    "gradient_health_report",
    "parameter_finite_report",
    "relative_change",
    "tensor_finite_report",
    "triage_nonfinite",
]

# ---------------------------------------------------------------------------
# Curve diagnosis
# ---------------------------------------------------------------------------

DEFAULT_CURVE_THRESHOLDS = {
    # A relative move smaller than this over the diagnosis window is "flat".
    "min_rel_improvement": 0.01,
    # A relative move larger than this in the bad direction is "worsening".
    "worsen_rel_tolerance": 0.002,
    # Gap growth (val - train) larger than this over the window is
    # overfitting pressure even when the validation curve is merely flat.
    "gap_widen_rel": 0.01,
    # Number of trailing validation points the verdict rests on.
    "window": 3,
    # A validation move is only called a trend when it exceeds this many
    # standard errors of the difference. Two validation passes are independent
    # sample means, so their difference has se = sqrt(se_first^2 + se_last^2).
    "noise_z": 2.0,
}


def relative_change(values: Sequence[float], window: int = 3) -> Optional[float]:
    """(last - first) / max(|first|, eps) over the trailing `window` points.

    Relative instead of absolute so the same threshold means the same thing
    at loss 10 (random init) and loss 2 (late training). Non-finite points are
    excluded here as a last-resort guard; `diagnose_train_val_curve` reports
    them explicitly before this helper is reached.
    """
    tail = [float(v) for v in values
            if v is not None and math.isfinite(float(v))][-max(2, int(window)) :]
    if len(tail) < 2:
        return None
    first, last = tail[0], tail[-1]
    return (last - first) / max(abs(first), 1e-8)


def _step_aligned_gap_points(train_losses, val_losses, train_steps, val_steps):
    """Gap at each validation pass, paired with the latest train loss at/before its step.

    Train and validation are often logged at different cadences, so zipping
    the two lists by position compares losses from different steps. When both
    step arrays are available and parallel to their loss arrays, align them;
    otherwise return ``None`` so the caller keeps the positional fallback.
    """
    if (train_steps is None or val_steps is None
            or len(train_steps) != len(train_losses)
            or len(val_steps) != len(val_losses)):
        return None
    train_pairs = sorted(
        (float(step), float(loss)) for step, loss in zip(train_steps, train_losses)
        if loss is not None and math.isfinite(float(loss))
    )
    val_pairs = sorted(
        (float(step), float(loss)) for step, loss in zip(val_steps, val_losses)
        if loss is not None and math.isfinite(float(loss))
    )
    if not train_pairs or not val_pairs:
        return None
    train_steps_sorted = [step for step, _ in train_pairs]
    gaps = []
    for val_step, val_loss in val_pairs:
        index = bisect.bisect_right(train_steps_sorted, val_step) - 1
        if index >= 0:
            gaps.append(val_loss - train_pairs[index][1])
    return gaps or None

def _trend(rel: Optional[float], min_rel_improvement: float, worsen_rel_tolerance: float) -> str:
    if rel is None:
        return "unknown"
    if rel < -min_rel_improvement:
        return "falling"
    if rel > worsen_rel_tolerance:
        return "rising"
    return "flat"


def diagnose_train_val_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_steps: Optional[Sequence[int]] = None,
    val_steps: Optional[Sequence[int]] = None,
    thresholds: Optional[dict] = None,
    log_vocab_size: Optional[float] = None,
    val_stderr: Optional[Sequence[float]] = None,
) -> dict:
    """Classify the current train/val regime from the recent points.

    Reads the *pair* of curves, never the final number alone:

    * train falling + val falling  -> ``healthy``;
    * train falling + val rising (or gap widening) -> ``overfitting``
      (regularize / add data / early-stop - the run already keeps
      ``ckpt_best.pt`` selected on validation);
    * both flat -> ``underfitting_or_stalled``: under-capacity, a learning
      rate that is not moving the model, or a data pipeline with no learnable
      signal - the three hypotheses have separate repo tools
      (`scripts/tune.py`, `scripts/sanity_overfit.py`,
      `scripts/audit_data.py`) and must not be guessed between;
    * both rising -> ``diverging`` (learning rate too large, gradient
      explosion, or corrupted data).

    `log_vocab_size` (ln of the vocabulary) is optional context: a flat
    curve far below ``ln(V)`` may simply be the corpus entropy floor rather
    than underfitting, so the flat verdict text says which case it may be.

    `val_stderr` (same length as `val_losses`) is the per-pass standard error
    of the held-out mean. The validation curve is itself an estimate: a rise
    smaller than `noise_z * sqrt(se_first^2 + se_last^2)` is reported as
    ``noise_limited`` rather than ``overfitting``/``diverging``, because the
    measured change cannot support the stronger claim. Without stderr the
    classification is unchanged (the caller did not measure the noise).

    Returns a JSON-serializable dict; every number the verdict used is kept
    under ``evidence``.
    """
    thresholds = {**DEFAULT_CURVE_THRESHOLDS, **(thresholds or {})}
    window = int(thresholds["window"])
    min_imp = float(thresholds["min_rel_improvement"])
    worsen = float(thresholds["worsen_rel_tolerance"])
    gap_widen = float(thresholds["gap_widen_rel"])
    noise_z = float(thresholds["noise_z"])

    n_train_input = len(train_losses)
    n_val_input = len(val_losses)
    raw_train = [float(v) for v in train_losses if v is not None]
    raw_val = [float(v) for v in val_losses if v is not None]
    # `val_stderr` is parallel to the caller's (unfiltered) `val_losses`: a
    # `None`/NaN slot still occupies its position. Filter it with the same
    # finite mask as `val_losses` so the surviving stderr entries stay aligned
    # with the surviving losses.
    val_stderr_series = None
    if val_stderr is not None:
        if len(val_stderr) != len(val_losses):
            raise ValueError("val_stderr must be parallel to val_losses")
        val_stderr_series = [
            None if se is None else float(se)
            for v, se in zip(val_losses, val_stderr)
            if v is not None and math.isfinite(float(v))
        ]
    # Indices are into the caller's arrays (counting any None entries), so the
    # report points at the curve position the reader can look up.
    bad_train = [i for i, v in enumerate(train_losses)
                 if v is not None and not math.isfinite(float(v))]
    bad_val = [i for i, v in enumerate(val_losses)
               if v is not None and not math.isfinite(float(v))]
    # Keep each surviving loss paired with its original step. Filtering only
    # the loss list (while passing the unfiltered step list) makes the arrays
    # unequal in length, and `_step_aligned_gap_points` silently falls back to
    # positional pairing - exactly the train/val cadence bug it exists to
    # prevent. None means "no point logged": drop the step with its loss.
    finite_train_idx = [i for i, v in enumerate(train_losses)
                        if v is not None and math.isfinite(float(v))]
    finite_val_idx = [i for i, v in enumerate(val_losses)
                      if v is not None and math.isfinite(float(v))]
    train_losses = [float(train_losses[i]) for i in finite_train_idx]
    val_losses = [float(val_losses[i]) for i in finite_val_idx]
    train_steps = ([] if train_steps is None or len(train_steps) != n_train_input
                   else [train_steps[i] for i in finite_train_idx])
    val_steps = ([] if val_steps is None or len(val_steps) != n_val_input
                 else [val_steps[i] for i in finite_val_idx])

    out = {
        "classification": "insufficient_data",
        "train_trend": "unknown",
        "val_trend": "unknown",
        "train_relative_change": None,
        "val_relative_change": None,
        "generalization_gap": None,
        "generalization_gap_change": None,
        "recommendations": [
            "at least 2 validation passes are needed before the curves can be "
            "read; keep --val-every small enough to get them"
        ],
        "evidence": {
            "n_train_points": len(raw_train),
            "n_val_points": len(raw_val),
            "window": window,
            "thresholds": dict(thresholds),
            "n_nonfinite_train_points": len(bad_train),
            "n_nonfinite_val_points": len(bad_val),
            "nonfinite_train_indices": bad_train,
            "nonfinite_val_indices": bad_val,
        },
    }
    if bad_train or bad_val:
        # A NaN/Inf on either curve is a numerical failure, not a trend.
        # Returning "flat"/"underfitting" here would send the reader to the
        # wrong remedy (lr/data) when the answer is the ordered NaN triage.
        out.update(
            classification="nonfinite_loss",
            recommendations=[
                "a train or validation loss on this curve is NaN/Inf: run the "
                "ordered triage (data -> fp32 math -> mixed precision -> "
                "gradients) on the first non-finite point; adasquash.diagnostics."
                "triage_nonfinite is the same path the trainer uses",
                "do not compare or checkpoint from this curve until the "
                "non-finite point is explained; a finite later value does not "
                "make the earlier corruption harmless",
            ],
        )
        return out
    if len(train_losses) < 2 or len(val_losses) < 2:
        return out

    train_rel = relative_change(train_losses, window)
    val_rel = relative_change(val_losses, window)
    gap_points = _step_aligned_gap_points(
        train_losses, val_losses, train_steps, val_steps)
    gap_alignment = "steps" if gap_points is not None else "position"
    if gap_points is None:
        gap_points = ([v - t for t, v in zip(train_losses[-len(val_losses):], val_losses)]
                      if len(train_losses) >= len(val_losses) else [])
    gap = gap_points[-1] if gap_points else None
    gap_change = relative_change(gap_points, window) if len(gap_points) >= 2 else None
    train_trend = _trend(train_rel, min_imp, worsen)
    val_trend = _trend(val_rel, min_imp, worsen)

    tail_len = min(window, len(val_losses))
    val_change = (val_losses[-1] - val_losses[-tail_len]) if tail_len >= 2 else None
    val_noise = None
    if (val_stderr_series is not None and tail_len >= 2
            and val_stderr_series[-1] is not None
            and val_stderr_series[-tail_len] is not None):
        se_last = val_stderr_series[-1]
        se_first = val_stderr_series[-tail_len]
        if (math.isfinite(se_last) and math.isfinite(se_first)
                and se_last >= 0.0 and se_first >= 0.0):
            val_noise = math.hypot(se_first, se_last)
    val_noise_threshold = (noise_z * val_noise) if val_noise is not None else None
    val_change_within_noise = (
        None if val_noise_threshold is None or val_change is None
        else abs(val_change) <= val_noise_threshold
    )
    # A widening gap is `val_change - train_change`. The train mean is much
    # tighter than a small held-out pass, so the gap change carries roughly
    # the validation-difference noise. Without this check a flat val curve
    # with a wide stderr can produce an "overfitting" verdict from a gap move
    # the held-out sample cannot resolve.
    gap_tail_len = min(window, len(gap_points))
    gap_change_abs = (gap_points[-1] - gap_points[-gap_tail_len]
                      if gap_tail_len >= 2 else None)
    gap_change_within_noise = (
        None if val_noise_threshold is None or gap_change_abs is None
        else abs(gap_change_abs) <= val_noise_threshold
    )

    recommendations: list[str] = []
    if val_trend == "rising" and val_change_within_noise:
        classification = "noise_limited"
        recommendations = [
            f"the validation change {val_change:+.4f} is within {noise_z:g} "
            f"standard errors ({val_noise_threshold:.4f}) of the held-out "
            "sample mean, so it cannot support an overfitting/divergence "
            "verdict yet",
            "keep training and read the next validation passes; increase "
            "--val-batches if the interval stays wider than the effect you "
            "need to detect",
        ]
    elif train_trend == "rising" and val_trend == "rising":
        classification = "diverging"
        recommendations = [
            "both curves are rising: suspect a learning rate that is too large "
            "(re-run the lr grid with scripts/tune.py), missing/weak warmup, or a "
            "data batch that is itself corrupted - the NaN triage path "
            "(`--nonfinite-policy abort`) checks the batch first",
            "log and plot the pre-clip gradient norm: a spike there localizes "
            "the step where the optimizer left the stable region",
        ]
    elif train_trend == "falling" and val_trend == "rising":
        classification = "overfitting"
        recommendations = [
            "train loss is still falling while validation rises: use "
            "--early-stop-patience (the best-validation checkpoint ckpt_best.pt "
            "is already the one the final test evaluation restores)",
            "add regularization (weight decay / dropout) or more unique data; "
            "verify the split with scripts/audit_data.py before trusting the gap",
            "a small loss win bought with a much larger model is not automatically "
            "a win: compare the efficiency numbers before claiming the trade",
        ]
    elif train_trend == "falling" and val_trend == "falling":
        classification = "healthy"
        recommendations = [
            "both curves are falling with the gap roughly stable; keep this "
            "recipe as the comparison point for the next change",
        ]
    elif train_trend == "falling" and val_trend == "flat" and gap_change is not None and gap_change > gap_widen:
        if gap_change_within_noise:
            # The gap widened by less than the held-out mean's resolution.
            # Calling this overfitting would spend regularization/data on a
            # difference the measurement cannot resolve.
            classification = "noise_limited"
            recommendations = [
                f"the train-validation gap widened by {gap_change_abs:+.4f}, "
                f"within {noise_z:g} standard errors ({val_noise_threshold:.4f}) "
                "of the held-out mean; the curve cannot yet distinguish "
                "overfitting from held-out noise",
                "keep training and read the next validation passes; increase "
                "--val-batches if the interval stays wider than the effect you "
                "need to detect",
            ]
        else:
            classification = "overfitting"
            recommendations = [
                "training keeps improving while the validation curve has gone flat "
                "and the train-validation gap is widening: the extra training "
                "capacity is fitting noise; use --early-stop-patience and add data "
                "or regularization",
                "compare the efficiency numbers before claiming the deeper/wider "
                "model is the better trade",
            ]
    elif val_trend == "falling" and train_trend == "flat":
        classification = "healthy"
        recommendations = [
            "validation still falling while the training window is flat is fine "
            "(the train window is a single-batch noisy estimate); keep going",
        ]
    elif train_trend == "flat" and val_trend == "flat":
        classification = "underfitting_or_stalled"
        floor_note = ""
        if log_vocab_size is not None and val_losses:
            floor_note = (
                f" (validation loss {val_losses[-1]:.4f} vs uniform-guess "
                f"ln(V)={float(log_vocab_size):.4f}: "
                + ("far below the uniform floor, so this may be the corpus "
                   "entropy floor rather than underfitting"
                   if val_losses[-1] < 0.7 * float(log_vocab_size)
                   else "still near the uniform-guess floor, so the model has "
                        "not learned the signal yet")
                + ")"
            )
        recommendations = [
            "both curves are flat: distinguish the three causes, do not guess - "
            "(a) optimization/lr: sweep it with scripts/tune.py; "
            "(b) plumbing/labels: the small-batch overfit check "
            "scripts/sanity_overfit.py must pass; "
            "(c) data: scripts/audit_data.py reports duplicates, leakage and "
            "train/val distribution shift" + floor_note,
            "if the loss is far below ln(V) the run may already sit at the "
            "corpus entropy floor; a bigger model / lower lr will not move it",
        ]
    elif val_trend == "rising":
        classification = "diverging"
        recommendations = [
            "validation is rising while training is not improving: reduce the "
            "learning rate, increase warmup, or inspect the batch the first bad "
            "step consumed (NaN triage order: data -> fp32 math -> precision -> "
            "gradients)",
        ]
    else:
        classification = "indeterminate"
        recommendations = [
            "curves do not match a known regime; keep logging and re-check after "
            "the next validation pass",
        ]

    out.update(
        classification=classification,
        train_trend=train_trend,
        val_trend=val_trend,
        train_relative_change=train_rel,
        val_relative_change=val_rel,
        generalization_gap=gap,
        generalization_gap_change=gap_change,
        recommendations=recommendations,
    )
    out["evidence"].update(
        {
            "train_steps_tail": train_steps[-window:],
            "val_steps_tail": val_steps[-window:],
            "train_losses_tail": train_losses[-window:],
            "val_losses_tail": val_losses[-window:],
            "log_vocab_size": log_vocab_size,
            "gap_alignment": gap_alignment,
            "gap_points": gap_points,
            "val_change": val_change,
            "val_stderr_tail": (None if val_stderr_series is None
                                else val_stderr_series[-tail_len:]),
            "val_noise_se": val_noise,
            "val_noise_threshold": val_noise_threshold,
            "val_change_within_noise": val_change_within_noise,
            "gap_change_abs": gap_change_abs,
            "gap_change_within_noise": gap_change_within_noise,
        }
    )
    return out


# ---------------------------------------------------------------------------
# NaN / Inf triage
# ---------------------------------------------------------------------------

# The fixed investigation order. Each stage's fix is different, so the triage
# returns the *first* stage that reproduces the non-finite value and never
# jumps ahead of it.
NAN_TRIAGE_ORDER = (
    ("data", "input batch contains NaN/Inf, tokens outside the vocabulary, or an empty loss mask"),
    ("fp32_math", "the fp32 forward is already non-finite (log(0)/1/0/softmax overflow)"),
    ("mixed_precision", "the fp32 forward is finite but the autocast forward overflows the low-precision range"),
    ("gradients", "the backward produces non-finite/huge gradients (explosion or too-large lr)"),
    ("unreproduced", "the non-finiteness did not reproduce in any isolated stage"),
)


def tensor_finite_report(t: torch.Tensor) -> dict:
    """NaN/Inf census of one tensor (floating or integer) plus its range."""
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "n_elements": int(t.numel()),
        "finite": True,
        "n_nan": 0,
        "n_inf": 0,
    }
    if t.numel() == 0:
        return out
    if t.is_complex():
        finite = torch.isfinite(t)
        n_bad = int((~finite).sum().item())
        out["finite"] = n_bad == 0
        if n_bad:
            out["n_nan"] = int(torch.isnan(t).sum().item())
            out["n_inf"] = n_bad - out["n_nan"]
        else:
            out["abs_max"] = float(t.detach().abs().max().item())
    elif t.is_floating_point():
        # NaN propagates into both extrema and +Inf / -Inf land in max / min,
        # so a finite extremum pair gives the range directly and a non-finite
        # one marks the tensor and needs the per-symbol NaN/Inf census below.
        td = t.detach()
        t_min, t_max = (float(v) for v in torch.aminmax(td))
        if math.isfinite(t_min) and math.isfinite(t_max):
            out["abs_max"] = max(abs(t_min), abs(t_max))
        else:
            n_nan = int(torch.isnan(t).sum().item())
            n_inf = int(torch.isinf(t).sum().item())
            out["finite"] = False
            out["n_nan"] = n_nan
            out["n_inf"] = n_inf
    return out


def _batch_data_report(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    vocab_size = int(getattr(getattr(model, "cfg", None), "vocab_size", 0) or 0)
    ignore_index = int(getattr(getattr(model, "cfg", None), "ignore_index", -100))
    x_report = tensor_finite_report(x)
    y_report = tensor_finite_report(y)

    def _integer_ids(t: torch.Tensor) -> bool:
        # Embedding and cross-entropy both index with these tensors; a float
        # tensor (even one holding integral values) is a data-pipeline bug that
        # would otherwise surface as a downstream model/CE exception and be
        # misdiagnosed as fp32 math.
        return not (t.is_floating_point() or t.is_complex() or t.dtype == torch.bool)

    input_integer_ids = _integer_ids(x)
    target_integer_ids = _integer_ids(y)
    targets_valid = target_integer_ids
    if y.numel() and target_integer_ids:
        in_range = (y == ignore_index) | ((y >= 0) & (y < vocab_size))
        targets_valid = bool(in_range.all().item())
    if x.numel() and input_integer_ids:
        tokens_valid = bool(((x >= 0) & (x < vocab_size)).all().item())
    else:
        tokens_valid = input_integer_ids
    mask_count = int((y != ignore_index).sum().item()) if target_integer_ids and y.numel() else 0
    return {
        "input": x_report,
        "targets": y_report,
        "vocab_size": vocab_size,
        "input_integer_ids": bool(input_integer_ids),
        "target_integer_ids": bool(target_integer_ids),
        "targets_valid": bool(targets_valid),
        "input_token_ids_valid": bool(tokens_valid),
        "loss_mask_nonempty": mask_count > 0,
        "loss_mask_tokens": mask_count,
        "loss_denominator_positive": mask_count > 0,
        "data_finite": bool(x_report["finite"] and y_report["finite"] and targets_valid and tokens_valid and mask_count > 0),
    }


def first_nonfinite_module(model, run_forward, max_modules: int = 8) -> list[dict]:
    """Forward `run_forward()` once and name the first modules with bad outputs.

    Used by the fp32 stage: when the forward is already non-finite, the first
    module whose tensor output contains NaN/Inf is where the corruption is
    created (hook order = execution order). Returns a list of
    ``{'module', 'class', 'n_nan', 'n_inf'}``, execution order, capped.
    """
    names = {id(m): name or m.__class__.__name__ for name, m in model.named_modules()}
    found: list[dict] = []
    seen: set[int] = set()
    handles = []

    def hook(module, inp, out):
        if id(module) in seen or len(found) >= max_modules:
            return
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        for t in tensors:
            if torch.is_tensor(t) and (t.is_floating_point() or t.is_complex()) and t.numel():
                finite = torch.isfinite(t)
                n_bad = int((~finite).sum().item())
                if n_bad:
                    seen.add(id(module))
                    n_nan = int(torch.isnan(t).sum().item())
                    found.append(
                        {
                            "module": names.get(id(module), module.__class__.__name__),
                            "class": module.__class__.__name__,
                            "n_nan": n_nan,
                            "n_inf": n_bad - n_nan,
                        }
                    )
                    break

    for module in model.modules():
        handles.append(module.register_forward_hook(hook))
    try:
        run_forward()
    finally:
        for h in handles:
            h.remove()
    return found


_GRAD_CENSUS_CHUNK = 1 << 22


def gradient_health_report(model) -> dict:
    """Non-finite/zero/huge gradient census over every parameter with a grad.

    Each parameter is scanned in bounded chunks: chunked extrema
    (`max(g.max(), -g.min())`, no `abs` copy), a chunked fp64 sum of squares,
    and a chunked finiteness reduction.
    """
    missing, nonfinite, zero = [], [], []
    n_with_grad = 0
    names = []
    finite_t = []
    abs_t = []
    sq_t = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
            continue
        g = p.grad.detach()
        n_with_grad += 1
        names.append(name)
        flat = g.reshape(-1)
        n = flat.numel()
        is_finite = torch.ones((), dtype=torch.bool, device=g.device)
        g_abs = torch.zeros((), dtype=torch.float64, device=g.device)
        g_sq = torch.zeros((), dtype=torch.float64, device=g.device)
        for start in range(0, n, _GRAD_CENSUS_CHUNK):
            blk = flat[start:start + _GRAD_CENSUS_CHUNK]
            lo, hi = torch.aminmax(blk)
            # NaN propagates into both extrema and a non-finite value lands in
            # one of them, so the two scalars certify the whole chunk.
            is_finite = is_finite & torch.isfinite(lo) & torch.isfinite(hi)
            cur_abs = torch.maximum(hi, lo.neg_()).to(torch.float64)
            g_abs = cur_abs if start == 0 else torch.maximum(g_abs, cur_abs)
            g_sq = g_sq + blk.to(torch.float64).square().sum()
        finite_t.append(is_finite)
        abs_t.append(g_abs)
        sq_t.append(g_sq)
    finite = torch.stack(finite_t).tolist() if finite_t else []
    abs_vals = torch.stack(abs_t).tolist() if abs_t else []
    sq_vals = torch.stack(sq_t).tolist() if sq_t else []
    max_abs = 0.0
    total_sq = 0.0
    for name, is_finite, g_abs, g_sq in zip(names, finite, abs_vals, sq_vals):
        if not is_finite:
            nonfinite.append(name)
            continue
        max_abs = max(max_abs, g_abs)
        total_sq += g_sq
        if g_abs == 0.0:
            zero.append(name)
    return {
        "total_norm": math.sqrt(total_sq),
        "max_abs_grad": max_abs,
        "params_with_grad": n_with_grad,
        "n_missing_grad": len(missing),
        "n_nonfinite_grad": len(nonfinite),
        "n_zero_grad": len(zero),
        "missing_grad": missing[:20],
        "nonfinite_grad": nonfinite[:20],
        "zero_grad": zero[:20],
        "gradients_finite": not nonfinite,
    }


def parameter_finite_report(model) -> dict:
    """NaN/Inf census over a model's parameters and floating buffers.

    The evaluation-side counterpart of `gradient_health_report`: before
    *reading* results out of a checkpoint, confirm the weights themselves are
    finite; the first offenders are named so the triage can start at the
    module that went bad. Integer buffers (position ids, masks) are skipped:
    NaN/Inf are floating-point states.
    """
    bad_params: list = []
    bad_buffers: list = []
    n_params = 0
    n_buffers = 0

    def _inspect(name: str, t: torch.Tensor, bucket: list) -> None:
        if not (t.is_floating_point() or t.is_complex()) or t.numel() == 0:
            return
        report = tensor_finite_report(t)
        if not report["finite"]:
            bucket.append(
                {
                    "name": name,
                    "shape": report["shape"],
                    "dtype": report["dtype"],
                    "n_nan": report["n_nan"],
                    "n_inf": report["n_inf"],
                }
            )

    for name, p in model.named_parameters():
        n_params += 1
        _inspect(name, p.detach(), bad_params)
    for name, b in model.named_buffers():
        n_buffers += 1
        _inspect(name, b.detach(), bad_buffers)

    finite = not bad_params and not bad_buffers
    return {
        "finite": finite,
        "n_parameters": n_params,
        "n_buffers": n_buffers,
        "n_nonfinite_parameters": len(bad_params),
        "nonfinite_parameters": bad_params[:20],
        "n_nonfinite_buffers": len(bad_buffers),
        "nonfinite_buffers": bad_buffers[:20],
        "recommendation": (
            "all parameters and floating buffers are finite"
            if finite else
            "the checkpoint's weights are already non-finite: do not trust any "
            "metric computed from it. Re-run the NaN triage on the training "
            "step that produced it (data -> fp32 math -> mixed precision -> "
            "gradients) and restore the last finite ckpt_best.pt instead"
        ),
    }


def classify_nan_stage(evidence: dict) -> dict:
    """Return the first failing stage of the fixed triage order.

    `evidence` keys are produced by `triage_nonfinite` (or by hand in tests);
    a missing key is treated as "this stage did not fail", so the classifier
    is conservative: it only blames a stage with positive evidence.
    """
    stages = {name: (i, why) for i, (name, why) in enumerate(NAN_TRIAGE_ORDER)}

    def stage(name: str, diagnosis: str, recommendations: list[str]) -> dict:
        idx, why = stages[name]
        return {
            "stage": name,
            "stage_index": idx,
            "stage_means": why,
            "diagnosis": diagnosis,
            "recommendations": recommendations,
        }

    if (not evidence.get("data_finite", True)
            or evidence.get("loss_denominator_positive") is False
            or evidence.get("loss_mask_nonempty") is False):
        dtype_bad = (evidence.get("input_integer_ids") is False
                     or evidence.get("target_integer_ids") is False)
        if dtype_bad:
            diagnosis = ("the batch's token ids/targets are not an integer dtype "
                         "(floating or boolean tensors cannot index an embedding "
                         "or be used as cross-entropy targets)")
        else:
            diagnosis = "the batch itself is invalid before it ever reaches the model"
        return stage(
            "data",
            diagnosis,
            [
                "inspect the first bad batch: which field is NaN/Inf, which token "
                "id is out of vocabulary, or why the loss mask is empty",
                "cast input_ids/targets to an integer dtype (int64) at the "
                "tokenizer/packer boundary; do not let float tensors reach "
                "embedding() or cross_entropy()",
                "re-run scripts/audit_data.py (it counts missing/NaN text, "
                "out-of-range ids and tokenizer failures) and check the packing "
                "pipeline in scripts/data_pipeline.py",
            ],
        )
    if evidence.get("parameters_finite") is False:
        bad = evidence.get("nonfinite_parameters") or []
        where = ", ".join(p["name"] for p in bad) or "unknown parameter"
        return stage(
            "gradients",
            "the weights are already non-finite before this batch reaches the "
            "model, so a previous optimizer update (or the loaded checkpoint) "
            f"corrupted them; the current data/loss math did not create it. "
            f"First bad parameter(s): {where}",
            [
                "restore the last finite ckpt_best.pt instead of resuming from "
                "the corrupted step",
                "inspect the pre-clip gradient norm and optimizer state around "
                "the update that produced the non-finite weight (a finite loss "
                "can still be followed by an overflow in the optimizer step)",
                "lower the learning rate / strengthen warmup and keep the "
                "optimizer step in fp32; do not keep training this state",
            ],
        )
    training_path_bad = (
        evidence.get("fp32_fused_loss_finite") is False
        or evidence.get("fused_forward_error")
        or evidence.get("first_nonfinite_modules_training_path")
    )
    if (evidence.get("fp32_forward_finite") is False
            or evidence.get("fp32_logits_finite") is False
            or evidence.get("first_nonfinite_modules")
            or evidence.get("forward_error")
            or training_path_bad):
        first = ((evidence.get("first_nonfinite_modules") or [])
                 + (evidence.get("first_nonfinite_modules_training_path") or []))
        where = ", ".join(f"{m['module']}({m['class']})" for m in first) or "unknown module"
        if evidence.get("forward_error"):
            diagnosis = ("the fp32 forward raised before a finite loss could be measured "
                         f"({evidence['forward_error']})")
        elif evidence.get("fused_forward_error"):
            diagnosis = ("the fp32 logits path raised "
                         f"({evidence['fused_forward_error']})")
        elif evidence.get("fp32_fused_loss_finite") is False:
            diagnosis = ("the fp32 training-path (fused cross-entropy) loss is "
                         "non-finite while the logits path is finite: the bug is "
                         "in the fused CE path (log-sum-exp/denominator/ignore-index "
                         f"handling), not in the backbone; first bad module(s): {where}")
        else:
            diagnosis = ("the fp32 forward is already non-finite (softmax/log/division "
                         f"pathology), first bad module(s): {where}")
        return stage(
            "fp32_math",
            diagnosis,
            [
                "if the named module computes log/softmax/division, check its "
                "denominator: log(0) -> -inf, x/0 -> inf; a fully masked batch "
                "makes the cross-entropy denominator zero",
                "for a fused-CE-only failure, check adasquash/model.py's "
                "fused_cross_entropy_mean/_FusedLinearCrossEntropy: the same "
                "batch must be finite through both return_logits=True and "
                "return_logits=False (the trainer uses the latter)",
                "re-run the forward in float64 on a tiny batch to confirm the "
                "value is truly non-finite and not a dtype artifact",
            ],
        )
    if evidence.get("autocast_forward_finite") is False or evidence.get("autocast_error"):
        dtype = evidence.get("autocast_dtype")
        if evidence.get("autocast_error"):
            diagnosis = (f"the fp32 forward is finite but the {dtype} autocast forward "
                         f"raised ({evidence['autocast_error']})")
        else:
            diagnosis = (f"the fp32 forward is finite but the {dtype} autocast forward "
                         "overflows the low-precision range")
        return stage(
            "mixed_precision",
            diagnosis,
            [
                "keep the optimizer step in fp32 and reduce the activation scale "
                "(lower lr / stronger warmup), or run without autocast "
                "(--no-bf16) until the run is stable",
                "for FP8 targets run scripts/check_fp8_stability.py, which "
                "reports the first overflow layer and the overflow rate",
            ],
        )
    if evidence.get("gradients_finite") is False or evidence.get("gradient_explosion", False):
        norm = evidence.get("grad_total_norm")
        norm_s = "n/a" if norm is None else f"{norm:.3e}"
        return stage(
            "gradients",
            f"the backward is the first non-finite/huge stage (total grad norm {norm_s})",
            [
                "lower the learning rate and/or increase warmup; re-tune with "
                "scripts/tune.py rather than guessing",
                "lower --grad-clip (or enable it) and watch the logged pre-clip "
                "gradient norm: a spike one step before the NaN localizes it",
                "if the loss had already fallen, the run may simply have left the "
                "stable region: resume from ckpt_best.pt with a lower lr",
            ],
        )
    error_note = ""
    if evidence.get("forward_error"):
        error_note += f"; forward raised {evidence['forward_error']}"
    if evidence.get("backward_error"):
        error_note += f"; backward raised {evidence['backward_error']}"
    return stage(
        "unreproduced",
        "the non-finite value did not reproduce in the isolated stages; it was "
        "likely transient (a single bad batch) or hardware/library specific"
        + error_note,
        [
            "keep the batch that produced it (log its index/step) and re-run it "
            "in isolation; if it reproduces only there, the data stage applies",
            "check the run for grad-norm spikes around that step, and whether "
            "the batch counter wrapped (a reused generator on resume)",
        ],
    )


def triage_nonfinite(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    autocast_dtype: Optional[torch.dtype] = None,
    device_type: str = "cuda",
    lr: Optional[float] = None,
    grad_clip: Optional[float] = None,
    explosion_factor: float = 100.0,
) -> dict:
    """Run the ordered NaN/Inf investigation and return stage + full evidence.

    Order (never reordered - the user-facing path is data -> math -> precision
    -> gradients):

    1. data validity: NaN/Inf, token-id range, empty loss mask, zero
       cross-entropy denominator;
    2. fp32 forward, on both the logits path (`return_logits=True`) and the
       fused-CE training path (`return_logits=False`, what the trainer uses),
       with module hooks: if non-finite, name the first module whose output
       goes bad (log/softmax/division pathologies live there);
    3. autocast (bf16/fp16) forward on the training path: fp32 finite +
       autocast non-finite pinpoints low-precision overflow;
    4. fp32 backward: non-finite or exploding gradients point at the learning
       rate / clipping rather than the data or the math.

    The function never mutates parameters; it zeroes all gradients before and
    after so the caller can safely continue (or abort) whichever policy it
    chose. `explosion_factor` is relative to `grad_clip` when that is given.
    """
    was_training = model.training
    evidence: dict = {"lr": lr, "grad_clip": grad_clip, "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype)}
    evidence.update(_batch_data_report(model, x, y))

    # Stage 1 first, always: an invalid batch is the answer the moment it is
    # found, and running the model on it would only crash (or diagnose a
    # consequence of the data, not the cause).
    if not evidence["data_finite"]:
        verdict = classify_nan_stage(evidence)
        return {"evidence": evidence, **verdict}

    # The data is valid, but the model may already be corrupted from an earlier
    # optimizer step: a forward on non-finite weights would look like an
    # fp32-math failure and send the reader after the batch instead of the
    # update that actually produced the bad weights.
    evidence["parameters_finite"] = None
    try:
        param_report = parameter_finite_report(model)
        evidence.update(
            parameters_finite=param_report["finite"],
            n_nonfinite_parameters=param_report["n_nonfinite_parameters"],
            nonfinite_parameters=param_report["nonfinite_parameters"],
            n_nonfinite_buffers=param_report["n_nonfinite_buffers"],
            nonfinite_buffers=param_report["nonfinite_buffers"],
        )
    except Exception as exc:  # diagnosis must never mask the original failure
        evidence["parameter_report_error"] = repr(exc)
    if evidence.get("parameters_finite") is False:
        evidence["autocast_forward_finite"] = None
        evidence["gradients_finite"] = None
        evidence["stages_skipped"] = ["fp32_math", "mixed_precision", "gradients"]
        verdict = classify_nan_stage(evidence)
        return {"evidence": evidence, **verdict}

    def _forward(loss_only: bool):
        if loss_only:
            return model(x, y, return_logits=False)[1]
        out = model(x, y, return_logits=True)
        return out[1], out[0]

    # Stage 2: fp32 forward, no autocast, hooks to name the first bad module.
    # One forward serves both the hook census and the loss/logits check.
    captured: dict = {}

    def _capture_forward():
        out = model(x, y, return_logits=True)
        captured["logits"], captured["loss"] = out[0], out[1]
        return out

    evidence["fp32_forward_finite"] = None
    evidence["fp32_fused_loss_finite"] = None
    try:
        with torch.no_grad():
            evidence["first_nonfinite_modules"] = first_nonfinite_module(model, _capture_forward)
            logits, loss32 = captured.get("logits"), captured.get("loss")
            evidence["fp32_forward_finite"] = bool(torch.isfinite(loss32).all().item())
            evidence["fp32_loss"] = float(loss32.item())
            evidence["fp32_logits_finite"] = tensor_finite_report(logits)["finite"] if logits is not None else None
    except Exception as exc:  # diagnosis must never mask the original failure
        evidence["forward_error"] = repr(exc)

    # The trainer consumes `return_logits=False` (fused CE), which never
    # materializes the logits tensor the block above probes. Both paths must be
    # checked: a log-sum-exp/denominator bug can be non-finite only in the
    # fused path, and diagnosing that as "unreproduced" sends the reader to
    # hardware/transience instead of the actual CE implementation.
    try:
        with torch.no_grad():
            fused_loss = _forward(loss_only=True)
            evidence["fp32_fused_loss_finite"] = bool(
                torch.isfinite(fused_loss).all().item())
            evidence["fp32_fused_loss"] = float(fused_loss.detach().item())
    except Exception as exc:  # diagnosis must never mask the original failure
        evidence["fused_forward_error"] = repr(exc)
    if (evidence.get("fp32_fused_loss_finite") is False
            and not evidence.get("first_nonfinite_modules")
            and "forward_error" not in evidence):
        # The backbone itself was finite on the logits path, so the logits-path
        # hook census cannot locate a fused-only failure. Re-run the census on
        # the actual training path to name the first hidden state that is bad
        # before the fused CE consumes it.
        try:
            with torch.no_grad():
                evidence["first_nonfinite_modules_training_path"] = first_nonfinite_module(
                    model, lambda: _forward(loss_only=True))
        except Exception as exc:  # diagnosis must never mask the original failure
            evidence["fused_hook_error"] = repr(exc)

    # The first stage that went bad already settles the diagnosis: either fp32
    # loss path is non-finite (or could not run at all), so stages 3-4 are
    # recorded as skipped instead.
    stage2_bad = (
        evidence.get("fp32_forward_finite") is False
        or evidence.get("fp32_logits_finite") is False
        or evidence.get("fp32_fused_loss_finite") is False
        or evidence.get("first_nonfinite_modules")
        or evidence.get("first_nonfinite_modules_training_path")
        or "forward_error" in evidence
        or "fused_forward_error" in evidence
    )
    if stage2_bad:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        evidence["autocast_forward_finite"] = None
        evidence["gradients_finite"] = None
        evidence["post_stage2_skipped"] = ["mixed_precision", "gradients"]
        verdict = classify_nan_stage(evidence)
        return {"evidence": evidence, **verdict}

    # Stage 3: autocast forward (only meaningful where autocast is supported).
    evidence["autocast_forward_finite"] = None
    if autocast_dtype is not None and device_type == "cuda" and torch.cuda.is_available():
        try:
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
                loss_ac = _forward(loss_only=True)
            evidence["autocast_forward_finite"] = bool(torch.isfinite(loss_ac).all().item())
            evidence["autocast_loss"] = float(loss_ac.item())
        except Exception as exc:  # diagnosis must never mask the original failure
            evidence["autocast_error"] = repr(exc)

    # Stage 4: fp32 backward. Zero before and after so the caller's optimizer
    # state is not polluted by the probe.
    evidence["gradients_finite"] = None
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            loss_grad = _forward(loss_only=True)
            if math.isfinite(float(loss_grad.detach().item())):
                loss_grad.backward()
                report = gradient_health_report(model)
                evidence.update(
                    gradients_finite=report["gradients_finite"],
                    grad_total_norm=report["total_norm"],
                    grad_max_abs=report["max_abs_grad"],
                    grad_n_nonfinite=report["n_nonfinite_grad"],
                    grad_nonfinite_params=report["nonfinite_grad"],
                    grad_n_missing=report["n_missing_grad"],
                    grad_n_zero=report["n_zero_grad"],
                )
                threshold = max(1e4, explosion_factor * float(grad_clip)) if grad_clip else 1e4
                evidence["gradient_explosion"] = bool(report["total_norm"] > threshold)
                evidence["gradient_explosion_threshold"] = threshold
            else:
                evidence["grad_backward_skipped"] = "fp32 forward loss non-finite"
    except Exception as exc:  # diagnosis must never mask the original failure
        evidence["backward_error"] = repr(exc)
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    verdict = classify_nan_stage(evidence)
    return {"evidence": evidence, **verdict}
