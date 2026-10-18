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
}


def relative_change(values: Sequence[float], window: int = 3) -> Optional[float]:
    """(last - first) / max(|first|, eps) over the trailing `window` points.

    Relative instead of absolute so the same threshold means the same thing
    at loss 10 (random init) and loss 2 (late training).
    """
    tail = [float(v) for v in values][-max(2, int(window)) :]
    if len(tail) < 2:
        return None
    first, last = tail[0], tail[-1]
    return (last - first) / max(abs(first), 1e-8)


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
    Returns a JSON-serializable dict; every number the verdict used is kept
    under ``evidence``.
    """
    thresholds = {**DEFAULT_CURVE_THRESHOLDS, **(thresholds or {})}
    window = int(thresholds["window"])
    min_imp = float(thresholds["min_rel_improvement"])
    worsen = float(thresholds["worsen_rel_tolerance"])
    gap_widen = float(thresholds["gap_widen_rel"])

    train_losses = [float(v) for v in train_losses if v is not None]
    val_losses = [float(v) for v in val_losses if v is not None]
    train_steps = list(train_steps) if train_steps is not None else []
    val_steps = list(val_steps) if val_steps is not None else []

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
            "n_train_points": len(train_losses),
            "n_val_points": len(val_losses),
            "window": window,
            "thresholds": dict(thresholds),
        },
    }
    if len(train_losses) < 2 or len(val_losses) < 2:
        return out

    train_rel = relative_change(train_losses, window)
    val_rel = relative_change(val_losses, window)
    gap_points = [v - t for t, v in zip(train_losses[-len(val_losses):], val_losses)] if len(train_losses) >= len(val_losses) else []
    gap = gap_points[-1] if gap_points else None
    gap_change = relative_change(gap_points, window) if len(gap_points) >= 2 else None
    train_trend = _trend(train_rel, min_imp, worsen)
    val_trend = _trend(val_rel, min_imp, worsen)

    recommendations: list[str] = []
    if train_trend == "rising" and val_trend == "rising":
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
            "a 0.5% loss win bought with a much larger model is not automatically "
            "a win: check params/flops/latency with scripts/benchmark_efficiency.py",
        ]
    elif train_trend == "falling" and val_trend == "falling":
        classification = "healthy"
        recommendations = [
            "both curves are falling with the gap roughly stable; keep this "
            "recipe as the comparison point for the next change",
        ]
    elif train_trend == "falling" and val_trend == "flat" and gap_change is not None and gap_change > gap_widen:
        classification = "overfitting"
        recommendations = [
            "training keeps improving while the validation curve has gone flat "
            "and the train-validation gap is widening: the extra training "
            "capacity is fitting noise; use --early-stop-patience and add data "
            "or regularization",
            "check params/flops/latency with scripts/benchmark_efficiency.py "
            "before claiming the deeper/wider model is the better trade",
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
    if t.is_floating_point() or t.is_complex():
        finite = torch.isfinite(t)
        n_bad = int((~finite).sum().item())
        out["finite"] = n_bad == 0
        if n_bad:
            out["n_nan"] = int(torch.isnan(t).sum().item())
            out["n_inf"] = n_bad - out["n_nan"]
        else:
            out["abs_max"] = float(t.detach().abs().max().item())
    return out


def _batch_data_report(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    vocab_size = int(getattr(getattr(model, "cfg", None), "vocab_size", 0) or 0)
    ignore_index = int(getattr(getattr(model, "cfg", None), "ignore_index", -100))
    x_report = tensor_finite_report(x)
    y_report = tensor_finite_report(y)
    targets_valid = True
    if y.numel():
        in_range = (y == ignore_index) | ((y >= 0) & (y < vocab_size))
        targets_valid = bool(in_range.all().item())
    if x.is_floating_point() or x.is_complex():
        tokens_valid = x_report["finite"]
    else:
        tokens_valid = bool(((x >= 0) & (x < vocab_size)).all().item()) if x.numel() else True
    mask_count = int((y != ignore_index).sum().item())
    return {
        "input": x_report,
        "targets": y_report,
        "vocab_size": vocab_size,
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
                    found.append(
                        {
                            "module": names.get(id(module), module.__class__.__name__),
                            "class": module.__class__.__name__,
                            "n_nan": int(torch.isnan(t).sum().item()),
                            "n_inf": n_bad - int(torch.isnan(t).sum().item()),
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


def gradient_health_report(model) -> dict:
    """Non-finite/zero/huge gradient census over every parameter with a grad."""
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
        finite_t.append(torch.isfinite(g).all())
        abs_t.append(g.abs().amax().to(torch.float64) if g.numel()
                    else torch.zeros((), dtype=torch.float64, device=g.device))
        sq_t.append(g.float().pow(2).sum().to(torch.float64))
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

    if not evidence.get("data_finite", True):
        return stage(
            "data",
            "the batch itself is invalid before it ever reaches the model",
            [
                "inspect the first bad batch: which field is NaN/Inf, which token "
                "id is out of vocabulary, or why the loss mask is empty",
                "re-run scripts/audit_data.py (it counts missing/NaN text, "
                "out-of-range ids and tokenizer failures) and check the packing "
                "pipeline in scripts/data_pipeline.py",
            ],
        )
    if evidence.get("fp32_forward_finite") is False or evidence.get("loss_denominator_positive") is False:
        first = evidence.get("first_nonfinite_modules") or []
        where = ", ".join(f"{m['module']}({m['class']})" for m in first) or "unknown module"
        return stage(
            "fp32_math",
            "the fp32 forward is already non-finite (softmax/log/division pathology), "
            f"first bad module(s): {where}",
            [
                "if the named module computes log/softmax/division, check its "
                "denominator: log(0) -> -inf, x/0 -> inf; a fully masked batch "
                "makes the cross-entropy denominator zero",
                "re-run the forward in float64 on a tiny batch to confirm the "
                "value is truly non-finite and not a dtype artifact",
            ],
        )
    if evidence.get("autocast_forward_finite") is False:
        dtype = evidence.get("autocast_dtype")
        return stage(
            "mixed_precision",
            f"the fp32 forward is finite but the {dtype} autocast forward overflows "
            "the low-precision range",
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
    2. fp32 forward with module hooks: if non-finite, name the first module
       whose output goes bad (log/softmax/division pathologies live there);
    3. autocast (bf16/fp16) forward: fp32 finite + autocast non-finite pinpoints
       low-precision overflow;
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
    try:
        with torch.no_grad():
            evidence["first_nonfinite_modules"] = first_nonfinite_module(model, _capture_forward)
            logits, loss32 = captured.get("logits"), captured.get("loss")
            evidence["fp32_forward_finite"] = bool(torch.isfinite(loss32).all().item())
            evidence["fp32_loss"] = float(loss32.item())
            evidence["fp32_logits_finite"] = tensor_finite_report(logits)["finite"] if logits is not None else None
    except Exception as exc:  # diagnosis must never mask the original failure
        evidence["forward_error"] = repr(exc)

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
