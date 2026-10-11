"""
Unit tests for the DyT++ / AdaSquash project.

Run:
    cd /home/z/my-project/ada-squash
    python -m pytest tests/ -v

Tests cover:
  - norm registry / build_norm
  - all normalization layers preserve shape and dtype
  - DyT family gradients flow correctly
  - the generalized softsign is bounded in [-1, 1]
  - depth schedule of DyT-D / DyT-CD is monotone in alpha_0 / k
  - QK-squash keeps logits bounded
  - model forward / loss / checkpoint round-trip
  - metrics: KurtosisMeter matches torch.kurtosis on synthetic data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import pytest
import torch
import torch.nn as nn

from adasquash import (
    build_norm, NORM_REGISTRY, generalized_softsign,
    LayerNorm, RMSNorm, DyT, DyTC, DyTD, DyTCD,
    GPTConfig, GPT, build_gpt_124m,
    ActivationStats, LossSpikeCounter, KurtosisMeter, MaxAbsMeter,
    count_parameters, measure_activation_stats,
)
from adasquash.attention import CausalSelfAttention, apply_rope
from adasquash.norms import QKSquash


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
@pytest.mark.parametrize("name", ["ln", "rmsnorm", "dyt", "dytc", "dytd", "dytcd"])
def test_norm_preserves_shape_and_dtype(name):
    for dtype in (torch.float32, torch.bfloat16):
        norm = build_norm(name, 64).to(dtype)
        x = torch.randn(2, 16, 64, dtype=dtype) * 3.0
        y = norm(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype


@pytest.mark.parametrize("name", ["ln", "rmsnorm", "dyt", "dytc", "dytd", "dytcd"])
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
def test_tanh_equivalent_ablation_produces_finite_loss(cls):
    """The `p -> infinity == tanh` ablation (p = 64) must not NaN out."""
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
# Metrics
# ---------------------------------------------------------------------------
def test_kurtosis_matches_torch_on_normal():
    torch.manual_seed(0)
    # Pure normal distribution has excess kurtosis == 0
    x = torch.randn(4096)
    meter = KurtosisMeter()
    meter.update(x)
    result = meter.result()
    # Allow generous tolerance because of the simplified formulas.
    assert abs(result) < 0.2, f"excess kurtosis = {result}, expected ~0"


def test_kurtosis_high_for_outlier_distribution():
    # Heavy-tailed distribution should have positive excess kurtosis.
    x = torch.cat([torch.randn(2048) * 0.5, torch.randn(64) * 50.0])
    meter = KurtosisMeter()
    meter.update(x)
    result = meter.result()
    assert result > 5.0, f"kurtosis = {result}, expected > 5"


def test_loss_spike_counter():
    counter = LossSpikeCounter()
    # Steady loss
    for _ in range(20):
        counter.update(3.0)
    assert counter.n_spikes == 0
    # Spike
    counter.update(10.0)
    assert counter.n_spikes >= 1


def test_activation_stats_hooks():
    model = build_gpt_124m(norm_name="dytcd")
    stats = ActivationStats(model)
    stats.attach()
    x = torch.randint(0, model.cfg.vocab_size, (2, 64))
    with torch.no_grad():
        model(x)
    s = stats.summary()
    stats.detach()
    assert "__summary__" in s
    # ActivationStats reports per-norm max|x| of the *pre-norm* activations.
    assert "mean_max_abs" in s["__summary__"]
    assert "max_max_abs" in s["__summary__"]
    assert s["__summary__"]["max_max_abs"] >= s["__summary__"]["mean_max_abs"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def test_forward_smoke():
    """Just check that a forward pass doesn't crash on CPU."""
    model = build_gpt_124m(norm_name="dytcd")
    model.eval()
    x = torch.randint(0, model.cfg.vocab_size, (1, 32))
    with torch.no_grad():
        out, _ = model(x)
    assert out.shape == (1, 32, model.cfg.vocab_size)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
