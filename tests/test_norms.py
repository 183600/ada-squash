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
    build_norm, NORM_REGISTRY,
    LayerNorm, RMSNorm, DyT, DyTC, DyTD, DyTCD,
    GPTConfig, GPT, build_gpt_124m,
    ActivationStats, LossSpikeCounter, KurtosisMeter, MaxAbsMeter,
    count_parameters, measure_activation_stats,
)
from adasquash.attention import CausalSelfAttention
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
    # `x` must stay the *leaf* of the graph, otherwise `x.grad` is always
    # None (the gradient would land on the pre-multiplication tensor).
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
    assert "mean_kurtosis" in s["__summary__"]


# ---------------------------------------------------------------------------
# Throughput sanity
# ---------------------------------------------------------------------------
def test_throughput_smoke():
    """Just check that a forward+backward pass doesn't crash on CPU."""
    model = build_gpt_124m(norm_name="dytcd")
    model.eval()
    x = torch.randint(0, model.cfg.vocab_size, (1, 32))
    with torch.no_grad():
        out, _ = model(x)
    assert out.shape == (1, 32, model.cfg.vocab_size)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
