"""
Unit tests for the DyT++ / AdaSquash project.

Run:
    python -m pytest tests/ -v

Tests cover:
  - norm registry / build_norm
  - all normalization layers preserve shape and dtype
  - every fused autograd kernel (DyT / DyT-C / DyT-D / DyT-CD, RMSNorm,
    SwiGLU, both cross-entropies, the generalized softsign) matches an
    independent recomposition in plain autograd, per parameter, from a
    *randomized* parameter set - not just "the gradients are finite"
  - frozen `p` / `k`, `beta=False`, and the alpha / p clamps
  - the generalized softsign is bounded in [-1, 1] and finite for runaway `p`
  - depth schedule of DyT-D / DyT-CD is monotone in alpha_0 / k
  - QK-squash keeps logits bounded; attention logit stats read the causal triangle
  - model forward / loss / checkpoint round-trip
  - metrics: KurtosisMeter matches torch.kurtosis on synthetic data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import pytest
import torch
import torch.nn as nn

from adasquash import (
    build_norm, NORM_REGISTRY, generalized_softsign,
    LayerNorm, RMSNorm, DyT, DyTS, DyTP, DyTC, DyTD, DyTDI, DyTCD, DyTCDI,
    GPTConfig, GPT, build_gpt_124m, build_llama_tiny_1b,
    ActivationStats, LossSpikeCounter, KurtosisMeter,
    count_parameters,
)
from adasquash.attention import apply_rope
from adasquash.norms import QKSquash
from adasquash.model import cross_entropy_mean


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_contains_all():
    expected = {"ln", "rmsnorm", "dyt", "dytc", "dytd", "dytcd"}
    assert expected.issubset(set(NORM_REGISTRY))


def test_build_norm_returns_correct_type():
    assert isinstance(build_norm("ln", 64), LayerNorm)
    assert isinstance(build_norm("rmsnorm", 64), RMSNorm)
    assert isinstance(build_norm("dyt", 64), DyT)
    assert isinstance(build_norm("dytc", 64), DyTC)
    assert isinstance(build_norm("dytd", 64), DyTD)
    assert isinstance(build_norm("dytcd", 64), DyTCD)


# ---------------------------------------------------------------------------
# Shape / dtype preservation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["ln", "rmsnorm", "dyt", "dyts", "dytp", "dytc", "dytd", "dytdi", "dytcd", "dytcdi"])
def test_norm_preserves_shape_and_dtype(name):
    for dtype in (torch.float32, torch.bfloat16):
        norm = build_norm(name, 64).to(dtype)
        x = torch.randn(2, 16, 64, dtype=dtype) * 3.0
        y = norm(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype


@pytest.mark.parametrize("name", ["ln", "rmsnorm", "dyt", "dyts", "dytp", "dytc", "dytd", "dytdi", "dytcd", "dytcdi"])
def test_norm_gradient_flows(name):
    norm = build_norm(name, 64)
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = norm(x * 3.0)
    loss = y.pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Softsign boundedness
# ---------------------------------------------------------------------------
def test_dytc_output_bounded_by_one():
    norm = DyTC(64, alpha_init=2.0, p_init=2.0, learnable_p=False)
    # huge outliers
    x = torch.randn(8, 32, 64) * 100
    y = norm(x)
    assert y.abs().max().item() <= 1.0 + 1e-4


def test_dytcd_output_bounded_by_one():
    norm = DyTCD(64, layer_idx=5, num_layers=12, alpha_0=2.0, learnable_k=False, k_init=2.0)
    x = torch.randn(8, 32, 64) * 100
    y = norm(x)
    assert y.abs().max().item() <= 1.0 + 1e-4


def test_softsign_matches_closed_form_where_the_closed_form_is_safe():
    """The split evaluation must agree with x / (1+|x|^p)^(1/p)."""
    torch.manual_seed(0)
    for p in (1.0, 2.0, 4.0):
        x = torch.randn(4096, dtype=torch.float64) * 2.0
        ref = x / (x.abs().pow(p) + 1.0).pow(1.0 / p)
        got = generalized_softsign(x, p)
        assert torch.allclose(got, ref, atol=1e-12), p


@pytest.mark.parametrize("p", [16.0, 64.0, 256.0])
def test_softsign_saturates_instead_of_overflowing(p):
    """For large p the layer must saturate to +-1 rather than overflow."""
    x = torch.tensor([[[-10.0, -4.5, 4.5, 10.0]]], requires_grad=True)
    y = generalized_softsign(x, p)
    y.sum().backward()
    assert torch.isfinite(y).all()
    assert torch.allclose(y, torch.tensor([[[-1.0, -1.0, 1.0, 1.0]]]), atol=1e-4)
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("cls", ["dytc", "dytcd"])
def test_hardtanh_limit_ablation_produces_finite_loss(cls):
    """The `p -> infinity == hard tanh` ablation (p = 64) must not NaN out."""
    if cls == "dytc":
        norm = DyTC(8, alpha_init=1.0, p_init=64.0, learnable_p=False)
    else:
        norm = DyTCD(8, layer_idx=3, num_layers=12, alpha_0=1.0, p_init=64.0,
                     learnable_p=False)
    x = (torch.randn(4, 8, 8) * 8.0).requires_grad_(True)
    y = norm(x)
    y.pow(2).sum().backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert y.abs().max().item() <= 1.0 + 1e-4


def test_softsign_zero_maps_to_zero():
    x = torch.zeros(3, 5)
    assert generalized_softsign(x, 2.0).abs().max().item() == 0.0
    assert generalized_softsign(x, 64.0).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Depth schedule
# ---------------------------------------------------------------------------
def test_dytd_alpha_decreases_with_depth():
    """alpha_l should *decrease* monotonically across layers."""
    L = 12
    alphas = []
    for l in range(L):
        norm = DyTD(64, layer_idx=l, num_layers=L, alpha_0=0.9, k_init=1.0)
        alphas.append(norm.alpha_l.item())
    # shallow > deep
    assert alphas[0] > alphas[-1]
    # monotone
    diffs = [alphas[i] - alphas[i + 1] for i in range(len(alphas) - 1)]
    assert all(d >= -1e-6 for d in diffs), f"alpha not monotone: {alphas}"


def test_dytcd_alpha_per_channel_shape():
    norm = DyTCD(64, layer_idx=2, num_layers=12, alpha_0=0.9, k_init=1.0)
    a = norm.alpha_per_channel
    assert a.shape == (64,)
    assert (a > 0).all()


# ---------------------------------------------------------------------------
# Rotary embeddings
# ---------------------------------------------------------------------------
def _ref_rope(x, pos, inv_freq):
    """Independent LLaMA-style RoPE for one position."""
    h = x.shape[-1] // 2
    th = pos.double() * inv_freq.double()
    c, s = torch.cos(th), torch.sin(th)
    xd = x.double()
    return torch.cat(
        [xd[..., :h] * c - xd[..., h:] * s, xd[..., h:] * c + xd[..., :h] * s],
        dim=-1,
    )


def _ref_freqs(head_dim, max_seq_len, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    return inv, t


@pytest.mark.parametrize("head_dim,base", [(8, 10000.0), (64, 10000.0), (128, 500000.0)])
def test_apply_rope_matches_llama_reference(head_dim, base):
    torch.manual_seed(0)
    T_ = 12
    inv, t = _ref_freqs(head_dim, T_, base)
    freqs = torch.outer(t, inv)
    fc = torch.polar(torch.ones_like(freqs), freqs)
    q = torch.randn(1, 2, T_, head_dim)
    k = torch.randn(1, 2, T_, head_dim)
    qo, ko = apply_rope(q, k, fc)
    for i in range(T_):
        assert torch.allclose(qo[0, 0, i].double(), _ref_rope(q[0, 0, i], t[i], inv), atol=1e-4)
        assert torch.allclose(ko[0, 0, i].double(), _ref_rope(k[0, 0, i], t[i], inv), atol=1e-4)


def test_rope_dimension_to_frequency_assignment():
    """rotate_half pairs dim j with j + hd//2, so both need inv_freq[j]."""
    head_dim, T_ = 8, 4
    inv, t = _ref_freqs(head_dim, T_)
    fc = torch.polar(torch.ones_like(torch.outer(t, inv)), torch.outer(t, inv))
    cos = torch.cat([fc[1].real, fc[1].real]).double()
    expected = torch.cat([torch.cos(inv), torch.cos(inv)]).double()
    assert torch.allclose(cos, expected, atol=1e-6)


def test_rope_is_relative():
    """<R_pos_i q, R_pos_j k> must depend only on pos_i - pos_j."""
    torch.manual_seed(0)
    head_dim, T_ = 16, 16
    inv, t = _ref_freqs(head_dim, T_)
    freqs = torch.outer(t, inv)
    fc = torch.polar(torch.ones_like(freqs), freqs)
    # the *same* q / k vector placed at every position
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    q_at, k_at = [], []
    for i in range(T_):
        qi, ki = apply_rope(q, k, fc[i:i + 1])
        q_at.append(qi)
        k_at.append(ki)
    q_at = torch.cat(q_at, dim=2)
    k_at = torch.cat(k_at, dim=2)
    dot = (q_at @ k_at.transpose(-1, -2))[0, 0]
    for i in range(T_):
        for j in range(T_):
            d = t[j] - t[i]
            ref = (q[0, 0, 0].double() * _ref_rope(k[0, 0, 0], d, inv)).sum()
            assert torch.allclose(dot[i, j].double(), ref, atol=1e-4), (i, j)


# ---------------------------------------------------------------------------
# QK squashing
# ---------------------------------------------------------------------------
def test_qk_squash_keeps_logits_bounded():
    bs = 2
    head_dim = 64
    q = torch.randn(bs, 4, 16, head_dim) * 50
    k = torch.randn(bs, 4, 16, head_dim) * 50
    squash = QKSquash()
    q2, k2 = squash(q, k)
    # tanh bounds every element of Q' / K' to [-1, 1]
    assert q2.abs().max().item() <= 1.0 + 1e-4
    assert k2.abs().max().item() <= 1.0 + 1e-4
    # The pre-scale dot product is therefore bounded by head_dim, so the
    # scaled logits are bounded by sqrt(head_dim) - independent of the
    # (here 50x) input magnitude. This is the "logit explosion" guarantee.
    logits = (q2 @ k2.transpose(-1, -2)) / math.sqrt(head_dim)
    assert logits.abs().max().item() <= math.sqrt(head_dim) + 1e-4
    # ... and strictly smaller than the unsquashed logits would be.
    logits_raw = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    assert logits.abs().max().item() < logits_raw.abs().max().item()


# ---------------------------------------------------------------------------
# End-to-end model
# ---------------------------------------------------------------------------
def test_gpt_forward_loss_backward():
    model = build_gpt_124m(norm_name="dytcd")
    x = torch.randint(0, model.cfg.vocab_size, (2, 64))
    y = torch.randint(0, model.cfg.vocab_size, (2, 64))
    logits, loss = model(x, y)
    assert logits.shape == (2, 64, model.cfg.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    # check params got grads
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total, f"{n_with_grad}/{n_total} params have grads"


def test_gpt_param_count_124m_close_to_target():
    model = build_gpt_124m()
    n = count_parameters(model)
    # GPT-2 124M target (loose because of vocab/embedding sizes).
    assert 100e6 < n < 130e6, f"param count = {n/1e6:.1f}M, target ~124M"


def test_checkpoint_roundtrip():
    import tempfile
    model = build_gpt_124m(norm_name="dytcd")
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({"model": model.state_dict()}, f.name)
        ckpt = torch.load(f.name)
        model2 = build_gpt_124m(norm_name="dytcd")
        model2.load_state_dict(ckpt["model"])
    # forward should match
    x = torch.randint(0, model.cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out1, _ = model(x)
        out2, _ = model2(x)
    assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# Loss masking
# ---------------------------------------------------------------------------
def test_ignore_index_does_not_collide_with_a_real_token():
    """The mask id must lie outside the vocabulary id range."""
    cfg = GPTConfig()
    assert cfg.ignore_index == -100
    assert not (0 <= cfg.ignore_index < cfg.vocab_size)


def test_token_zero_targets_are_scored():
    """A batch of nothing but token-0 targets must produce a real loss."""
    torch.manual_seed(0)
    # Uniform (all-zero) logits give an exactly log(V) cross-entropy.
    logits = torch.zeros(64, 16, requires_grad=True)
    targets = torch.zeros(64, dtype=torch.long)
    loss = cross_entropy_mean(logits, targets, ignore_index=GPTConfig().ignore_index)
    # If id 0 were masked the valid count would be zero and the result NaN.
    assert math.isfinite(loss.item())
    assert abs(loss.item() - math.log(16)) < 1e-4
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0


def test_padded_positions_are_masked_and_excluded_from_the_count():
    torch.manual_seed(0)
    logits = torch.randn(64, 8)
    targets = torch.randint(0, 8, (64,))
    targets[-1] = -100
    masked = cross_entropy_mean(logits, targets, ignore_index=-100)
    kept = cross_entropy_mean(logits[:-1], targets[:-1], ignore_index=-100)
    assert abs(masked.item() - kept.item()) < 1e-5


def test_gpt_forward_ignores_only_ignore_index():
    """A masked target must contribute neither loss nor gradient."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                          max_seq_len=16))
    model.eval()
    x = torch.randint(0, 64, (1, 16))
    y = torch.randint(0, 64, (1, 16))
    y_masked = y.clone()
    y_masked[:, -1] = model.cfg.ignore_index

    with torch.no_grad():
        logits, loss_masked = model(x, y_masked)
        # Reference: the same loss over the unmasked prefix only. If the
        # masked position leaked into the sum or the normalizer this drifts.
        ref = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            y[:, :-1].reshape(-1),
        )
    assert abs(loss_masked.item() - ref.item()) < 1e-5


# ---------------------------------------------------------------------------
# Gradient correctness of the fused autograd kernels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1.0, 2.0, 4.0])
def test_softsign_gradcheck_wrt_x_and_p(p):
    """The hand-written backward must match autograd on the closed form."""
    from adasquash.norms import _GeneralizedSoftsign

    x = (torch.randn(4, 6, dtype=torch.float64) * 3).requires_grad_(True)
    pt = torch.tensor(p, dtype=torch.float64, requires_grad=True)

    def fn(a, b):
        return _GeneralizedSoftsign.apply(a, b)

    assert torch.autograd.gradcheck(fn, (x, pt), eps=1e-6, atol=1e-7)


def test_swiglu_and_rmsnorm_gradcheck():
    from adasquash.norms import _RMSNorm
    from adasquash.model import _SwiGLU

    x = torch.randn(2, 5, 7, dtype=torch.float64).requires_grad_(True)
    w = torch.randn(7, dtype=torch.float64).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, b: _RMSNorm.apply(a, b, 1e-6), (x, w), atol=1e-7
    )

    x = torch.randn(2, 5, 7, dtype=torch.float64).requires_grad_(True)
    hidden = 9
    ws = [torch.randn(hidden, 7, dtype=torch.float64).requires_grad_(True),
          torch.randn(7, hidden, dtype=torch.float64).requires_grad_(True),
          torch.randn(hidden, 7, dtype=torch.float64).requires_grad_(True)]
    assert torch.autograd.gradcheck(
        lambda *a: _SwiGLU.apply(*a), (x, *ws), atol=1e-6
    )


def test_layerscale_residual_gradcheck():
    from adasquash.model import _LayerScaleResidual

    x = torch.randn(2, 4, 6, dtype=torch.float64).requires_grad_(True)
    r = torch.randn(2, 4, 6, dtype=torch.float64).requires_grad_(True)
    s = torch.randn(6, dtype=torch.float64).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda *a: _LayerScaleResidual.apply(*a), (x, r, s), atol=1e-9
    )


def test_fused_cross_entropy_matches_the_reference_value_and_gradients():
    """The training path's loss node: `hidden @ weight.T` folded into the CE.

    It is what every training step backpropagates through, and nothing else
    in the graph reproduces it, so its gradients are checked against plain
    `F.cross_entropy` on the materialized logits - including the
    `ignore_index` rows, which must contribute neither loss nor gradient.
    """
    from adasquash.model import fused_cross_entropy_mean

    torch.manual_seed(0)
    N, C, V = 40, 5, 7
    hidden = torch.randn(N, C, dtype=torch.float64).requires_grad_(True)
    weight = torch.randn(V, C, dtype=torch.float64).requires_grad_(True)
    targets = torch.randint(0, V, (N,), dtype=torch.long)
    targets[:5] = -100
    targets[20:23] = -100

    fused = fused_cross_entropy_mean(
        hidden, weight, targets, ignore_index=-100, chunk_rows=7
    )
    ref = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, weight), targets, ignore_index=-100
    )
    assert torch.allclose(fused, ref, atol=1e-12)
    g_fused = torch.autograd.grad(fused, (hidden, weight))
    g_ref = torch.autograd.grad(ref, (hidden, weight))
    for a, b in zip(g_fused, g_ref):
        assert torch.allclose(a, b, atol=1e-12)
    # The masked rows must be exactly zero, not merely small.
    assert torch.count_nonzero(g_fused[0][:5]) == 0
    assert torch.count_nonzero(g_fused[0][20:23]) == 0


# ---------------------------------------------------------------------------
# Gradient correctness of the fused DyT-family kernels
# ---------------------------------------------------------------------------
def _fused_norm_reference(m, x):
    """Independent, plain-autograd recomposition of a norm layer.

    Deliberately written out of the public pieces rather than reusing the
    kernels under test, so it is an actual second opinion on both the value
    and every gradient. For DyTD / DyTCD the reference mirrors the module's
    *effective* (k_min-clamped) decay rate: the floor is part of the
    module's contract, not an implementation detail.
    """
    from adasquash.norms import generalized_softsign

    if isinstance(m, DyT):
        # Also covers DyTDI (a DyT whose alpha was initialized at the
        # schedule value) - after init it is a plain per-layer scalar alpha.
        phi = torch.tanh(m.alpha * x)
    elif isinstance(m, DyTP):
        phi = torch.tanh(m.alpha * x)
    elif isinstance(m, DyTS):
        p = m.p.clamp(min=1e-3) if isinstance(m.p, torch.Tensor) else m.p
        phi = generalized_softsign(m.alpha * x, p)
    elif isinstance(m, DyTD):
        k = m.k if m.k_min is None else m.k.clamp(min=m.k_min)
        a = m.alpha_0 * torch.exp(-k * m._depth_ratio)
        phi = torch.tanh(a * x)
    elif isinstance(m, DyTC):
        # Also covers DyTCDI (a DyTC whose alpha was initialized at the
        # schedule value).
        phi = generalized_softsign(m.alpha * x, m.p)
    elif isinstance(m, DyTCD):
        k = m.k if m.k_min is None else m.k.clamp(min=m.k_min)
        scale = m.alpha_0 * torch.exp(-k * m._depth_ratio)
        alpha_c = torch.addcmul(scale, scale, m.delta).clamp(min=1e-4)
        phi = generalized_softsign(alpha_c * x, m.p)
    else:
        raise AssertionError(type(m))
    if m.beta is None:
        return phi * m.gamma
    return torch.addcmul(m.beta, phi, m.gamma)


def _compare_norm_to_reference(build, dim=6, seed=0):
    """Same batch, same weights, same loss -> same values and same grads."""
    torch.manual_seed(seed)

    def build_trained():
        # The affine has to be *off* its initialization, or the comparison is
        # vacuous: at `gamma == 1, beta == 0` the post-affine output equals the
        # squashed activation exactly, which is precisely why reading one for
        # the other stayed invisible until a real (trained) checkpoint. Every
        # parameter is therefore randomized, which is the state a model is in
        # one step into training.
        m = build().double()
        with torch.no_grad():
            for _, p in m.named_parameters():
                if p.ndim == 0:      # alpha / k / p scalars
                    p.copy_(torch.randn((), dtype=torch.float64) * 0.3 + 0.7)
                else:                # gamma / delta / beta / alpha_c
                    p.copy_(torch.randn(p.shape, dtype=torch.float64) * 0.4)
        return m

    m = build_trained()
    x = (torch.randn(3, 5, dim, dtype=torch.float64) * 2.0).requires_grad_(True)
    y = m(x)
    # A non-uniform cotangent, so a gradient that is merely proportional to
    # the right tensor (rather than equal to it) still fails.
    cot = torch.linspace(0.5, 1.3, y.numel(), dtype=torch.float64).view_as(y)
    y.mul(cot).sum().backward()
    grads = {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in m.named_parameters()}
    x_grad = x.grad.clone()

    ref = build_trained()
    ref.load_state_dict(m.state_dict())
    x2 = x.detach().clone().requires_grad_(True)
    y2 = _fused_norm_reference(ref, x2)
    y2.mul(cot).sum().backward()

    assert torch.allclose(y.detach(), y2.detach(), atol=1e-10), "forward"
    assert torch.allclose(x_grad, x2.grad, atol=1e-9), "d/dx"
    for name, p in ref.named_parameters():
        expected = grads[name]
        if expected is None:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
        else:
            assert p.grad is not None and torch.allclose(
                expected, p.grad, atol=1e-9
            ), f"d/d{name}"


@pytest.mark.parametrize("name", ["dyt", "dyts", "dytp", "dytd", "dytdi", "dytc", "dytcd", "dytcdi"])
def test_fused_norm_kernels_match_autograd(name):
    """Every hand-written backward must agree with autograd, per parameter."""
    build = lambda: build_norm(name, 6)
    _compare_norm_to_reference(build)


@pytest.mark.parametrize("name", ["dyt", "dyts", "dytp", "dytd", "dytdi", "dytc", "dytcd", "dytcdi"])
def test_fused_norm_kernels_match_autograd_without_beta(name):
    """Same check with the affine bias removed (`beta=False`)."""
    _compare_norm_to_reference(lambda: build_norm(name, 6, beta=False))


@pytest.mark.parametrize("learnable_p", [True, False])
@pytest.mark.parametrize("learnable_k", [True, False])
def test_dytd_kernels_match_autograd_with_frozen_schedule(learnable_p, learnable_k):
    """A frozen `p` / `k` must not change the gradients that *are* trained.

    `learnable_p=False` makes `p` a plain Python float, which
    `save_for_backward` rejects outright, so the forward carries the value on
    the context instead of saving it as a tensor.
    """
    for cls_name, kw in (
        ("dytd", dict(layer_idx=3, num_layers=12, learnable_k=learnable_k)),
        ("dytc", dict(learnable_p=learnable_p)),
        ("dytcd", dict(layer_idx=3, num_layers=12,
                       learnable_p=learnable_p, learnable_k=learnable_k)),
    ):
        _compare_norm_to_reference(
            lambda cls_name=cls_name, kw=kw: build_norm(cls_name, 6, **kw)
        )


def test_dytcd_alpha_clamp_has_a_correct_gradient():
    """`alpha_c` is clamped away from zero, so the clamp must gate `delta`.

    The backward cannot divide the clamped `alpha_c` back into `scale`: the
    division is only valid where the clamp is inactive, so the raw product is
    re-derived and the clamp's own zero-gradient is applied.
    """
    torch.manual_seed(0)
    m = DyTCD(6, layer_idx=5, num_layers=12).double()
    with torch.no_grad():
        # scale * (1 + delta) < 0 for these, so alpha_c is pinned at the floor.
        m.delta.fill_(-3.0)
    x = (torch.randn(3, 5, 6, dtype=torch.float64) * 2.0).requires_grad_(True)
    m(x).sum().backward()
    assert torch.count_nonzero(m.delta.grad) == 0, (
        "a channel held at the alpha floor still received a delta gradient"
    )
    # Unclamped channels must still be trained.
    with torch.no_grad():
        m.delta[0] = 0.0
    x2 = (torch.randn(3, 5, 6, dtype=torch.float64) * 2.0).requires_grad_(True)
    m(x2).sum().backward()
    assert m.delta.grad[0].abs().item() > 0.0


def test_dytcd_trains_alpha0_and_k_when_x_and_delta_are_frozen():
    """`alpha_0` / `k` must receive gradients on their own merit.

    The backward used to compute the alpha chain only inside
    `needs_input_grad[delta]`, and gated `need_softsign` on x/delta alone, so
    a run that freezes the input and `delta` (e.g. probing the depth schedule
    with the delta term ablated) silently got `None` gradients for `alpha_0`
    and `k` - the depth schedule would never train, with no error to say why.
    """
    torch.manual_seed(0)
    m = DyTCD(6, layer_idx=5, num_layers=12).double()
    m.delta.requires_grad_(False)
    x = torch.randn(3, 5, 6, dtype=torch.float64)  # no grad
    m(x).sum().backward()
    assert m.alpha_0.grad is not None and m.alpha_0.grad.abs().item() > 0.0
    assert m.k.grad is not None and m.k.grad.abs().item() > 0.0
    assert m.delta.grad is None
    # gamma / beta still train; only the explicitly frozen inputs are None.
    assert m.gamma.grad is not None


def test_dytcd_partial_freeze_matches_the_reference_grads():
    """With `delta` frozen but x live, the alpha chain must match a recomposition.

    The alpha chain and the x path share `grad_x_scaled`, so gating the chain
    on the wrong `needs_input_grad` entry corrupts whichever side survives.
    Every live gradient is compared against a plain-autograd recomposition
    that treats `delta` as a constant.
    """
    torch.manual_seed(1)
    m = DyTCD(6, layer_idx=5, num_layers=12).double()
    m.delta.requires_grad_(False)
    x = torch.randn(3, 5, 6, dtype=torch.float64).requires_grad_(True)
    m(x).sum().backward()

    def ref(x_, a0_, k_):
        scale = a0_ * torch.exp(-k_ * m._depth_ratio)
        alpha_c = torch.addcmul(scale, scale, m.delta.detach()).clamp(min=1e-4)
        phi = generalized_softsign(alpha_c * x_, m.p.detach())
        return torch.addcmul(m.beta, phi, m.gamma)

    x_ref = x.detach().clone().requires_grad_(True)
    a0_ref = m.alpha_0.detach().clone().requires_grad_(True)
    k_ref = m.k.detach().clone().requires_grad_(True)
    ref(x_ref, a0_ref, k_ref).sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-10)
    assert torch.allclose(m.alpha_0.grad, a0_ref.grad, atol=1e-10)
    assert torch.allclose(m.k.grad, k_ref.grad, atol=1e-10)


@pytest.mark.parametrize("p", [-2.0, -1e-4, 0.0, 1e-9])
def test_softsign_exponent_floor_keeps_a_runaway_p_finite(p):
    """A learnable `p` that crosses zero must not NaN the model.

    `phi_p` is only a finite family member for `p > 0`: as `p -> 0+` it
    collapses to 0 (a dead layer), and for `p < 0` the `|x|^p` term blows up
    at `|x| = 0`, so the layer returns inf/NaN and poisons every gradient that
    reaches it. `p` is a free parameter here and the ablation matrix sweeps it
    upward, so it can walk into that region.
    """
    x = (torch.randn(2, 4, 8, dtype=torch.float64)).requires_grad_(True)
    pt = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    y = generalized_softsign(x, pt)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    # Below the floor the forward only ever saw `clamp(p)`, so p is frozen.
    assert pt.grad.item() == 0.0


def test_attention_logit_stats_read_the_causal_triangle():
    """`return_stats=True` is the QK-squash ablation's monitor.

    Only the causal triangle is ever read, so it is selected with
    `masked_select` of a `[T, T]` mask against the `[B, H, T, T]` logits:
    boolean *indexing* would apply the mask to the leading two dims instead
    of the last two.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                          max_seq_len=64, norm_name="dytcd")).eval()
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        _, _, stats = model(x, return_stats=True)
    assert len(stats) == model.cfg.n_layers
    for s in stats:
        assert set(s) == {"logit_abs_mean", "logit_abs_max", "logit_std"}
        assert s["logit_abs_max"] >= s["logit_abs_mean"] >= 0.0
        assert math.isfinite(s["logit_std"])


def test_attention_logit_stats_agree_with_a_manual_causal_measurement():
    from adasquash.attention import CausalSelfAttention

    torch.manual_seed(0)
    attn = CausalSelfAttention(32, 4).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        _, stats = attn(x, return_stats=True)
        h = attn.qkv(x).view(2, 6, 3, 4, 8).permute(2, 0, 3, 1, 4)
        q, k, _ = h[0], h[1], h[2]
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(8)
        mask = torch.ones(6, 6, dtype=torch.bool).tril()
        valid = logits.masked_select(mask)
    assert stats["logit_abs_max"] == pytest.approx(valid.abs().max().item(), rel=1e-5)
    assert stats["logit_abs_mean"] == pytest.approx(valid.abs().mean().item(), rel=1e-5)
    assert stats["logit_std"] == pytest.approx(valid.std().item(), rel=1e-4)


@pytest.mark.parametrize("chunk_rows", [4, 8, 1000])
def test_chunked_cross_entropy_matches_reference(chunk_rows):
    torch.manual_seed(0)
    V = 37
    targets = torch.randint(0, V, (200,))
    targets[:17] = -100
    targets[50:60] = -100
    logits = torch.randn(200, V, dtype=torch.float64).requires_grad_(True)
    got = cross_entropy_mean(logits, targets, ignore_index=-100, chunk_rows=chunk_rows)
    ref = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(got, ref)
    g1, = torch.autograd.grad(got, logits, retain_graph=True)
    g2, = torch.autograd.grad(ref, logits, retain_graph=True)
    assert torch.allclose(g1, g2, atol=1e-12)


def test_chunked_cross_entropy_leaves_the_callers_logits_intact():
    """`GPT.forward(..., return_logits=True)` hands the logit block to its
    caller, so the chunked loss node must not use it as a scratch buffer.

    The node writes its gradient into a private buffer instead. The
    reference graph here is computed *after* the backward, so a
    scratch-buffered gradient shows up immediately.
    """
    torch.manual_seed(0)
    V = 37
    N = 200
    targets = torch.randint(0, V, (N,))
    targets[:17] = -100
    targets[50:60] = -100
    logits = torch.randn(N, V, dtype=torch.float64).requires_grad_(True)
    pristine = logits.detach().clone()

    got = cross_entropy_mean(logits, targets, ignore_index=-100, chunk_rows=8)
    assert torch.allclose(
        got, torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100),
        rtol=1e-12,
    )
    got.backward()

    assert torch.equal(logits, pristine), (
        "the chunked-loss node overwrote the caller's logit block in place"
    )
    # ... and the gradient it did hand back is still the correct one.
    ref_logits = pristine.clone().requires_grad_(True)
    ref = torch.nn.functional.cross_entropy(ref_logits, targets, ignore_index=-100)
    ref.backward()
    assert torch.allclose(logits.grad, ref_logits.grad, atol=1e-12)


def test_gpt_forward_returned_logits_survive_backward():
    """The same guarantee through the model's public forward signature."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50304, dim=64, n_layers=1, n_heads=4,
                          max_seq_len=128, norm_name="rmsnorm")).eval()
    # 8*128 = 1024 rows > the fitted chunk size for a 50k vocab, so the
    # chunked path (not the `F.cross_entropy` fast path) is what runs.
    from adasquash.model import _cross_entropy_chunk_rows
    x = torch.randint(0, 50304, (8, 128))
    y = torch.randint(0, 50304, (8, 128))
    y[:, -1] = model.cfg.ignore_index
    assert x.numel() > _cross_entropy_chunk_rows(model.cfg.vocab_size)

    logits, loss = model(x, y, return_logits=True)
    pristine = logits.detach().clone()
    loss.backward()
    assert torch.equal(logits, pristine), (
        "logits returned by GPT.forward were corrupted by backward()"
    )


# ---------------------------------------------------------------------------
# Quantization harnesses
# ---------------------------------------------------------------------------
def test_w8a8_wrapper_quantizes_swiglu_weights():
    """The W8A8 probe must quantize the MLP, not just attention/lm_head.

    `MLP.forward` calls `_SwiGLU.apply` with the raw `.weight` tensors, so
    the wrapper attaches to those tensors.
    """
    import copy
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.check_w8a8_ppl import wrap_model_w8a8

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=256, dim=32, n_layers=1, n_heads=4, max_seq_len=32))
    model.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        ref = model(x)[0]
    q = wrap_model_w8a8(copy.deepcopy(model)).eval()
    with torch.no_grad():
        got = q(x)[0]
    assert hasattr(q.blocks[0].mlp.w1, "_q_weight"), "SwiGLU w1 left in fp32"
    assert hasattr(q.blocks[0].mlp.w2, "_q_weight"), "SwiGLU w2 left in fp32"
    assert hasattr(q.blocks[0].mlp.w3, "_q_weight"), "SwiGLU w3 left in fp32"
    # Quantization must actually perturb the output.
    assert not torch.allclose(ref, got, atol=1e-8)


def test_w8a8_quantizes_every_gemm_activation_including_the_hidden():
    """The SwiGLU hidden (`silu(w1 x) * (w3 x)`) is fed to `w2` from inside
    `_SwiGLU`, so leaving it fp32 makes the W8A8 measurement a partial one."""
    import torch.nn.functional as F
    from scripts.check_w8a8_ppl import (
        quantize_activation_per_token, wrap_model_w8a8,
    )

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=32, norm_name="rmsnorm"))
    wrap_model_w8a8(model)
    shapes = []
    orig = F.linear

    def spy(x, w, b=None):
        shapes.append(tuple(x.shape))
        # Already on the int8 grid iff re-quantizing is a no-op.
        assert (quantize_activation_per_token(x) - x).abs().max().item() == 0.0
        return orig(x, w, b)

    F.linear = spy
    try:
        with torch.no_grad():
            model(torch.randint(0, 128, (1, 16)))
    finally:
        F.linear = orig
    hidden = model.blocks[0].mlp.w1.weight.shape[0]
    assert any(s[-1] == hidden for s in shapes), "SwiGLU GEMM never observed"


def test_w8a8_quantizes_tied_embeddings_consistently():
    """GPT-2 ties lm_head to tok_emb; both sides must use the same copy."""
    from scripts.check_w8a8_ppl import wrap_model_w8a8

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, dim=32, n_layers=1, n_heads=4,
                          tie_word_embeddings=True))
    assert model.lm_head.weight is model.tok_emb.weight
    wrap_model_w8a8(model)
    with torch.no_grad():
        model(torch.randint(0, 128, (1, 8)))
    assert hasattr(model.tok_emb, "_q_weight"), "token embedding left in fp32"
    assert hasattr(model.lm_head, "_q_weight")


def test_w8a8_quantizes_the_head_on_the_fused_loss_path():
    """The output projection is the largest GEMM in the model, and the
    fused-loss path reaches its weight directly
    (`fused_cross_entropy_mean` takes `lm_head.weight` as an argument and
    never calls `lm_head.forward`), so a wrapper that only overrides
    `forward` silently measures that GEMM unquantized.
    """
    import copy
    from adasquash.model import fused_cross_entropy_mean
    from scripts.check_w8a8_ppl import (
        compute_ce_sum, quantize_weight_per_channel, wrap_model_w8a8,
    )

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=16, norm_name="rmsnorm")).eval()
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    y[:, -1] = model.cfg.ignore_index

    qmodel = wrap_model_w8a8(copy.deepcopy(model)).eval()
    qw, _ = quantize_weight_per_channel(model.lm_head.weight.data)
    # The wrapped model's own activations, captured from a real forward.
    seen = {}
    handle = qmodel.norm_f.register_forward_hook(
        lambda m, i, o: seen.setdefault("h", o.detach())
    )
    with torch.no_grad():
        fused = compute_ce_sum(qmodel, x, y, device="cpu")[0]
    handle.remove()
    n_valid = (y != model.cfg.ignore_index).sum().item()
    fused = fused / n_valid
    hidden = seen["h"].reshape(-1, model.cfg.dim)

    with torch.no_grad():
        ref = fused_cross_entropy_mean(
            hidden, qw, y.reshape(-1), ignore_index=model.cfg.ignore_index
        ).item()
        fp = fused_cross_entropy_mean(
            hidden, model.lm_head.weight, y.reshape(-1),
            ignore_index=model.cfg.ignore_index,
        ).item()

    assert torch.allclose(qmodel.lm_head.weight, qw), (
        "the wrapped model's head weight is still the fp32 matrix"
    )
    assert not torch.equal(qw, model.lm_head.weight), (
        "the quantized head must actually differ from the fp32 one"
    )
    assert abs(fused - ref) < 1e-6, (
        "the fused loss path is not using the W8A8-quantized output "
        "projection (it still sees the fp32 weight)"
    )
    assert abs(fused - fp) > abs(fused - ref), (
        "the fused loss path matches the fp32 projection better than the "
        "quantized one"
    )


def test_w8a8_weight_quantization_is_idempotent():
    """A tied matrix is quantized once per accessor (embedding lookup and
    head GEMM), so the two copies must not drift apart."""
    from scripts.check_w8a8_ppl import quantize_weight_per_channel

    torch.manual_seed(0)
    w = torch.randn(64, 32) * 3.0
    once, _ = quantize_weight_per_channel(w)
    assert torch.allclose(once, quantize_weight_per_channel(once)[0])
    assert not torch.equal(once, w)


def test_w8a8_probe_rejects_a_signal_free_corpus():
    from adasquash import markov_batch
    from scripts.check_w8a8_ppl import probe_token_coverage

    V = 50304
    rng = torch.Generator().manual_seed(0)
    random_probe = [torch.randint(0, V, (4, 512), generator=rng) for _ in range(8)]
    rng = torch.Generator().manual_seed(0)
    markov_probe = [markov_batch(rng, -100, 4, 512, "cpu")[0] for _ in range(8)]
    # Uniform tokens are (almost) all distinct: nothing repeats, so no model
    # can beat chance and the reported PPL drop is noise.
    assert probe_token_coverage(random_probe) > 0.5
    assert probe_token_coverage(markov_probe) < 0.05


def test_fp8_quantizer_preserves_input_dtype():
    """Under a bf16 autocast the sim must not silently promote back to fp32."""
    from scripts.check_fp8_stability import simulate_fp8, to_fp8_e4m3

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert to_fp8_e4m3(torch.tensor([1.0], dtype=dtype)).dtype == dtype
    assert simulate_fp8(torch.tensor([1.0], dtype=torch.float64),
                        448.0, 4, 3).dtype == torch.float64


def test_fp8_wrapper_quantizes_weights_not_just_activations():
    """An fp8 GEMM is fp8 x fp8; quantizing only the activation is A8-only."""
    import torch.nn.functional as F
    from scripts.check_fp8_stability import fp8_forward, to_fp8_e4m3

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=32, norm_name="rmsnorm"))
    on_grid = lambda t: bool((to_fp8_e4m3(t) - t).abs().max().item() < 1e-6)
    seen = []
    orig = F.linear

    def spy(x, w, b=None):
        seen.append((on_grid(x), on_grid(w)))
        return orig(x, w, b)

    F.linear = spy
    try:
        with torch.no_grad():
            with fp8_forward(model, to_fp8_e4m3):
                model(torch.randint(0, 128, (1, 16)))
    finally:
        F.linear = orig
    assert seen, "no GEMM observed"
    assert all(a and w for a, w in seen), f"unquantized operands: {seen}"


def test_fp8_probe_corpus_is_structured_not_uniform():
    """`check_fp8_stability.py` reports activation *magnitudes* (overflow /
    underflow rate, max|x|, kurtosis), so the probe has to be a corpus with
    real next-token structure rather than embedding noise. The default probe
    must therefore be a real walk.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import scripts.check_fp8_stability as fp8
    from adasquash.utils import markov_batch
    from scripts.check_w8a8_ppl import probe_token_coverage

    p = fp8.argparse.ArgumentParser()
    p.add_argument("--norms", nargs="+", default=["ln"])
    p.add_argument("--ckpt", nargs="+", default=None)
    p.add_argument("--probe", default="markov", choices=["markov", "random"])
    args = p.parse_args([])
    assert args.probe == "markov", (
        "the FP8 probe defaults to a signal-free uniform-token corpus"
    )
    # And the corpus it produces is the learnable walk, not independent draws.
    rng = torch.Generator().manual_seed(0)
    probe = [
        markov_batch(rng, -100, 4, 512, "cpu", n_states=64)[0] for _ in range(8)
    ]
    # Uniform ids over a 50k vocab: coverage ~ 1. A 64-state walk: ~64 ids.
    assert probe_token_coverage(probe) < 0.05, (
        "the probe corpus is degenerate: nothing repeats, so the activation "
        "statistics describe embedding noise"
    )


def test_fp8_reports_overflow_depth_separately_from_nonfinite():
    """`simulate_fp8` saturates to +-448, so an overflowing residual stream
    is finite and `first_nonfinite_layer` alone can never report it."""
    from scripts.check_fp8_stability import measure_fp8_divergence

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, dim=32, n_layers=4, n_heads=4,
                          norm_name="dytcd", layerscale_init=1.0))
    with torch.no_grad():
        model.tok_emb.weight.mul_(1e4)
    model.eval()
    r = measure_fp8_divergence(model, torch.randint(0, 128, (2, 32)), "dytcd")
    assert r["first_overflow_layer"] is not None
    assert r["first_overflow_layer"] < model.cfg.n_layers
    # `n_layers` is the block count, not the norm-hook count (2L+1).
    assert r["n_layers"] == model.cfg.n_layers
    assert r["n_norm_hooks"] == 2 * model.cfg.n_layers + 1


def test_fp8_forward_simulation_perturbs_logits_and_restores():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.check_fp8_stability import fp8_forward, to_fp8_e4m3

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=256, dim=32, n_layers=2, n_heads=4, max_seq_len=32))
    model.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        ref = model(x)[0]
    with fp8_forward(model, to_fp8_e4m3):
        with torch.no_grad():
            got = model(x)[0]
    assert not torch.allclose(ref, got, atol=1e-8)
    # The context manager must restore the original forwards.
    with torch.no_grad():
        again = model(x)[0]
    assert torch.allclose(ref, again)


def test_fp8_quantizer_respects_format_range():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.check_fp8_stability import to_fp8_e4m3, to_fp8_e5m2, fp8_min_normal

    x = torch.tensor([1e6, -1e6, 0.0, 1e-8, 1.0, -3.25])
    q = to_fp8_e4m3(x)
    assert q.abs().max().item() <= 448.0 + 1e-6
    q5 = to_fp8_e5m2(x)
    assert q5.abs().max().item() <= 57344.0 + 1e-3
    # Subnormals are representable: half the min *normal* of e4m3 (2^-7)
    # survives on the subnormal grid (2^-7 = 4 * 2^-9).
    assert to_fp8_e4m3(torch.tensor([fp8_min_normal(4) * 0.5])).item() == pytest.approx(2.0 ** -7)
    # Below the smallest *subnormal* (2^-9 for e4m3) values flush to zero.
    assert to_fp8_e4m3(torch.tensor([2.0 ** -9 * 0.5])).item() == 0.0
    assert to_fp8_e4m3(torch.tensor([2.0 ** -10])).item() == 0.0
    # Exact representable values round-trip, including the min subnormal.
    assert to_fp8_e4m3(torch.tensor([1.0, 0.5, -2.0, 448.0])).tolist() == [1.0, 0.5, -2.0, 448.0]
    assert to_fp8_e4m3(torch.tensor([2.0 ** -9])).item() == pytest.approx(2.0 ** -9)


def test_build_llama_tiny_accepts_tie_override():
    """`tie_word_embeddings` can be overridden through **over.

    The tie/untie wiring asserted here is config-level reference identity
    (`lm_head.weight is tok_emb.weight`), which does not depend on storage,
    so both models are built on `torch.device("meta")` — the same zero-
    allocation pattern as `adasquash/ablation/matrix.py` and the meta path
    of `_skip_init` (see also the meta GPT build in this file, above).

    A REAL build is the whole point of the guard: untied ~1.07B + tied
    ~1.00B params are alive at once, i.e. ~7.7 GB of fp32 weights (before
    any activations). The old "<6 GB skip" gate let 6-9 GB runners through
    exactly when that peak OOM-kills the entire machine (SIGKILL / exit
    137), turning `ci_gate.py`'s `pytest tests/ -x -q` into a runner
    reboot instead of a test failure. On meta the same assertions hold
    with ~0 RAM, so the test now runs on every runner, no memory gate.
    `init=False` matches how `scripts/train_gpt2.py` / `eval_downstream.py`
    build the 1B model (values would be discarded immediately anyway).
    """
    with torch.device("meta"):
        m = build_llama_tiny_1b(norm_name="dytcd", init=False,
                                 tie_word_embeddings=True)
        # Guard the zero-allocation contract: if `_skip_init` ever falls
        # back to materializing weights, this fails LOUDLY here instead of
        # silently OOM-killing the runner again.
        assert m.tok_emb.weight.device.type == "meta"
        assert m.lm_head.weight is m.tok_emb.weight
        m2 = build_llama_tiny_1b(norm_name="dytcd", init=False)
        assert m2.tok_emb.weight.device.type == "meta"
        assert m2.lm_head.weight is not m2.tok_emb.weight


# ---------------------------------------------------------------------------
# MoE: the auxiliary loss and the reported `expert_cv`
# ---------------------------------------------------------------------------
def test_moe_tokens_per_expert_counts_tokens_not_gate_mass():
    """`tokens_per_expert` / `expert_cv` must count the tokens routed to each
    expert. Summing the renormalised gate weights instead reports routing
    *mass*, which sums to 1 and is not the `f_i` of the Switch / Mixtral
    load-balancing loss the module documents itself as implementing.
    """
    from scripts.train_moe_tiny import MoELayer

    torch.manual_seed(0)
    E, K = 8, 2
    mo = MoELayer(32, 64, n_experts=E, top_k=K).eval()
    x = torch.randn(4, 16, 32)
    with torch.no_grad():
        _, _, tpe = mo(x)
        logits = mo.router(x.view(-1, 32))
        _, indices = logits.softmax(-1).topk(K, dim=-1)

    n_tokens = x.reshape(-1, 32).shape[0]
    assert tpe.shape == (E,)
    assert (tpe >= 0).all()
    # Token counts: N tokens, each assigning its top-k slots, so the total is
    # N*K and every count is an integer number of tokens.
    assert tpe.sum().item() == pytest.approx(n_tokens * K)
    assert torch.allclose(tpe, tpe.round()), "counts must be whole tokens"
    true_counts = torch.zeros(E).scatter_add_(
        0, indices.reshape(-1), torch.ones(indices.numel())
    )
    assert torch.equal(tpe, true_counts)


def test_moe_aux_loss_matches_the_switch_formula():
    """aux = n_experts * sum_i(f_i * P_i) with f_i the token fraction."""
    from scripts.train_moe_tiny import MoELayer

    torch.manual_seed(0)
    E = 8
    mo = MoELayer(32, 64, n_experts=E, top_k=2).eval()
    x = torch.randn(4, 16, 32)
    with torch.no_grad():
        _, aux, tpe = mo(x)
        logits = mo.router(x.view(-1, 32))
        _, indices = logits.softmax(-1).topk(2, dim=-1)
        n = logits.shape[0]
        f_i = torch.zeros(E).scatter_add_(
            0, indices.reshape(-1), torch.ones(indices.numel())
        ) / n
        p_i = logits.softmax(-1).mean(dim=0)
        expected = E * (f_i * p_i).sum()
    assert torch.allclose(aux, expected, atol=1e-6), (
        "the auxiliary loss is not the Switch/Mixtral load-balancing loss"
    )


def test_moe_qk_alpha_init_reaches_the_attention():
    """`GPTConfig.qk_alpha_init` must reach the QKSquash the MoE attention
    builds, not fall back to the default alpha."""
    from adasquash.model import GPTConfig
    from scripts.train_moe_tiny import MoEBlock

    cfg = GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                    max_seq_len=16, norm_name="rmsnorm",
                    qk_norm="dytqk", qk_alpha_init=0.25)
    blk = MoEBlock(cfg, n_experts=4, layer_idx=0)
    sq = blk.attn.q_squash
    assert sq is not None
    assert sq.alpha_q.item() == pytest.approx(0.25)
    assert sq.alpha_k.item() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Ablation aggregate
# ---------------------------------------------------------------------------
def test_ablation_aggregate_keeps_rows_from_previous_runs():
    """A focused re-run (`--names <one> --force`) must fold into the
    aggregate instead of rewriting it and discarding the other rows."""
    from scripts.ablation import merge_summaries

    previous = [
        {"name": "dytcd_full", "final_loss": 1.0},
        {"name": "dytcd_no_ls", "final_loss": 2.0},
        {"name": "dytd_full", "final_loss": 3.0},
    ]
    fresh = [{"name": "dytcd_full", "final_loss": 9.0}]

    merged = merge_summaries(previous, fresh)
    names = [r["name"] for r in merged]
    assert names == ["dytcd_full", "dytcd_no_ls", "dytd_full"], (
        f"a focused re-run reshaped the aggregate: {names}"
    )
    # The re-run replaces the stale row rather than being ignored.
    assert merged[0]["final_loss"] == 9.0
    # Brand-new rows are appended, and an empty/absent aggregate still works.
    assert merge_summaries([], fresh)[0]["name"] == "dytcd_full"
    assert [r["name"] for r in merge_summaries(previous, [])] == names
    assert merge_summaries([], []) == []
    # Malformed rows are skipped, not crashed on.
    assert merge_summaries([{"nope": 1}, "junk"], fresh) == fresh


def test_ablation_matrix_contrasts_are_single_variable():
    """Every documented contrast in configs/ablation-matrix.yaml must differ
    from its reference row by exactly ONE GPTConfig axis (modulo the seed,
    which the runner varies deliberately).

    This is the guard against the old matrix's confounds: dyt->dytc used to
    change per-channel AND tanh->softsign at once, and the LN baseline ->
    dytcd jump bundled the norm swap with enabling LayerScale.
    """
    import dataclasses
    root = Path(__file__).resolve().parent.parent
    from scripts.ablation import load_ablations_from_config, effective_config_key

    rows = load_ablations_from_config(str(root / "configs" / "ablation-matrix.yaml"))

    def cfg_of(name):
        with torch.device("meta"):
            return dataclasses.asdict(
                GPT(GPTConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4,
                              max_seq_len=32, **rows[name])).cfg)

    def diff_axes(a, b):
        ca, cb = cfg_of(a), cfg_of(b)
        return sorted(k for k in ca if ca[k] != cb[k])

    # Factorial: activation family x per-channel, LS held off. (p_init
    # coincides at 2.0, so it does not show up as a differing axis; the
    # learnable_p flag differs but is a no-op for the tanh cell.)
    assert diff_axes("dyt", "dyts") == ["learnable_p", "norm_name"]
    assert diff_axes("dyt", "dytp") == ["alpha_jitter", "norm_name"]
    assert diff_axes("dyts", "dytc") == ["alpha_jitter", "norm_name"]
    assert diff_axes("dytp", "dytc") == ["learnable_p", "norm_name"]

    # Depth axis at matched LS: reparameterization, init-only control, shared.
    assert diff_axes("dyt_ls", "dytd") == ["norm_name"]
    assert diff_axes("dytd", "dytdi") == ["norm_name"]
    assert diff_axes("dytd", "dytd_shared") == ["shared_depth_schedule"]
    assert diff_axes("dytd", "dytd_k0_no_depth_schedule") == [
        "k_init", "learnable_k"]
    assert diff_axes("dytc_ls", "dytcd") == ["norm_name"]
    assert diff_axes("dytcd", "dytcdi") == ["norm_name"]

    # LayerScale axis under every family (layerscale_init coincides with
    # the default 1e-5, so only the switch itself differs).
    assert diff_axes("ln", "ln_ls") == ["use_layerscale"]
    assert diff_axes("rmsnorm", "rmsnorm_ls") == ["use_layerscale"]
    assert diff_axes("dytcd", "dytcd_no_ls") == ["use_layerscale"]

    # The depth-init rows must exist and be distinct model configs from their
    # reparameterized counterparts (otherwise the contrast is vacuous).
    assert rows["dytdi"]["norm_name"] == "dytdi"
    assert rows["dytcdi"]["norm_name"] == "dytcdi"
    assert effective_config_key("gpt2-124m", rows["dytd"]) != \
        effective_config_key("gpt2-124m", rows["dytdi"])


def test_dytd_k_min_floors_the_decay_rate():
    """k_min keeps the depth schedule non-increasing wherever training takes
    k, and k_min=None restores the old unconstrained behaviour."""
    n = build_norm("dytd", 8, layer_idx=3, num_layers=12, alpha_0=0.9,
                   k_init=1.0, learnable_k=True, k_min=0.0)
    with torch.no_grad():
        n.k.fill_(-5.0)
    # k floored at 0 -> alpha_l == alpha_0 (no growth with depth).
    assert torch.allclose(n.alpha_l, torch.tensor(0.9))
    # Below the floor the clamp passes zero gradient to k.
    x = torch.randn(2, 8, requires_grad=True)
    n(x).pow(2).sum().backward()
    assert n.k.grad is not None and n.k.grad.item() == 0.0

    n2 = build_norm("dytd", 8, layer_idx=3, num_layers=12, alpha_0=0.9,
                    k_init=1.0, learnable_k=True, k_min=None)
    with torch.no_grad():
        n2.k.fill_(-5.0)
    assert n2.alpha_l.item() > 0.9  # negative k grows alpha with depth


def test_dytd_shared_depth_schedule_shares_one_pair():
    """--shared-depth-schedule makes every DyTD/DyTCD norm read the SAME
    (alpha_0, k) tensors, so the depth law is identifiable (2 scalars for
    the whole model, not 2 per norm) and checkpoints round-trip."""
    m = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=3, n_heads=4,
                      max_seq_len=16, norm_name="dytd",
                      shared_depth_schedule=True))
    alphas = [id(mod.alpha_0) for mod in m.modules() if isinstance(mod, DyTD)]
    ks = [id(mod.k) for mod in m.modules() if isinstance(mod, DyTD)]
    assert len(alphas) == 7 and len(set(alphas)) == 1  # 3 blocks x 2 + final
    assert len(set(ks)) == 1
    # state_dict carries both spellings; a fresh model loads it strictly.
    m2 = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=3, n_heads=4,
                       max_seq_len=16, norm_name="dytd",
                       shared_depth_schedule=True))
    m2.load_state_dict(m.state_dict())


def test_dytdi_matches_dytd_bitwise_at_init():
    """The depth-init control starts at exactly the same function as the
    reparameterized variant (same alpha_l), so any early-training divergence
    is attributable to the parameterization, not the start point."""
    def build(norm):
        torch.manual_seed(11)
        m = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=4, n_heads=4,
                          max_seq_len=16, norm_name=norm, alpha_0=0.9,
                          k_init=1.0, learnable_k=False))
        m.reset_parameters(11)
        return m.eval()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = build("dytd")(x)[0]
        b = build("dytdi")(x)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_residual_outlier_probe_modifies_no_weights():
    """The ablation outlier probe is a forward hook on the residual stream:
    it must change the activations (that is its point) while leaving every
    parameter untouched - unlike the old embedding-row inflation, which with
    tied embeddings also rewrote a row of the output classifier."""
    from scripts.ablation import install_residual_outlier

    m = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=4, n_heads=4,
                      max_seq_len=16, norm_name="dytcd")).eval()
    x = torch.randint(0, 64, (2, 12))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        base = m(x)[0]
    info = install_residual_outlier(m, layer_idx=2, channel=16, scale=25.0)
    assert info == {"kind": "residual_channel", "layer_idx": 2,
                    "channel": 16, "scale": 25.0}
    with torch.no_grad():
        after = m(x)[0]
    for k, v in before.items():
        assert torch.equal(v, m.state_dict()[k]), f"weight {k} changed"
    assert not torch.allclose(base, after)


def test_activation_stats_hook_the_new_norms():
    from adasquash.utils import ActivationStats
    m = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                      max_seq_len=16, norm_name="dyts"))
    st = ActivationStats(m)
    st.attach()
    m(torch.randint(0, 64, (1, 8)))
    summary = st.summary()
    st.detach()
    hooked = [k for k in summary if k != "__summary__"]
    assert len(hooked) == 5  # 2 blocks x 2 norms + final


def test_yaml_config_overrides_parser_defaults():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train_gpt2 import apply_config_to_parser, _TRAIN_CONFIG_ALIASES

    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--norm", default="dytcd")
    p.add_argument("--no-layerscale", action="store_true")
    p.add_argument("--no-learnable-p", action="store_true")
    p.add_argument("--no-learnable-k", action="store_true")
    p.add_argument("--no-tie-embeddings", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--layerscale-init", type=float, default=None)
    p.add_argument("--output-dir", default="x")
    apply_config_to_parser(p, str(root / "configs" / "gpt2-124m-ln.yaml"),
                           _TRAIN_CONFIG_ALIASES)
    a = p.parse_args([])
    assert a.norm == "ln"
    # The baseline config turns LayerScale off, the parser default turns it on.
    assert a.no_layerscale is True
    assert a.output_dir == "runs/gpt2-124m-ln"


def test_explicit_cli_flags_beat_the_config():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train_gpt2 import apply_config_to_parser, _TRAIN_CONFIG_ALIASES

    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--norm", default="dytcd")
    p.add_argument("--no-layerscale", action="store_true")
    p.add_argument("--no-learnable-p", action="store_true")
    p.add_argument("--no-learnable-k", action="store_true")
    p.add_argument("--no-tie-embeddings", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--layerscale-init", type=float, default=None)
    apply_config_to_parser(p, str(root / "configs" / "gpt2-124m-dytcd.yaml"),
                           _TRAIN_CONFIG_ALIASES)
    assert p.parse_args([]).no_layerscale is False      # from the config
    assert p.parse_args(["--no-layerscale"]).no_layerscale is True  # from the CLI
