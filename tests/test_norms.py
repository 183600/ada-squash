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

import argparse
import math
import pytest
import torch
import torch.nn as nn

from adasquash import (
    build_norm, NORM_REGISTRY, generalized_softsign,
    LayerNorm, RMSNorm, DyT, DyTC, DyTD, DyTCD,
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
    # Below the smallest normal magnitude both formats flush to zero.
    assert to_fp8_e4m3(torch.tensor([fp8_min_normal(4) * 0.5])).item() == 0.0
    # Exact representable values round-trip.
    assert to_fp8_e4m3(torch.tensor([1.0, 0.5, -2.0, 448.0])).tolist() == [1.0, 0.5, -2.0, 448.0]


def test_build_llama_tiny_accepts_tie_override():
    """`tie_word_embeddings` can be overridden through **over."""
    m = build_llama_tiny_1b(norm_name="dytcd", tie_word_embeddings=True)
    assert m.lm_head.weight is m.tok_emb.weight
    m2 = build_llama_tiny_1b(norm_name="dytcd")
    assert m2.lm_head.weight is not m2.tok_emb.weight


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
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
    # README metric-table name for the same quantity.
    assert s["__summary__"]["max_abs_pre_norm"] == s["__summary__"]["max_max_abs"]
    assert "mean_excess_kurtosis_pre_norm" in s["__summary__"]
    # Kurtosis is documented on the class and must actually be emitted.
    assert "excess_kurtosis" in s["blocks.0.norm1"]


def test_activation_stats_can_pool_across_sampling_windows():
    """`attach(reset=False)` must accumulate, so run-level stats are possible.

    The ablation runner re-attaches on a sampling window; without this the
    reported statistics describe only the last window while `final_loss`
    describes the whole run.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=16, norm_name="dytcd"))
    stats = ActivationStats(model)
    probe = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        # Window 1 carries the larger activation.
        model.tok_emb.weight[0].fill_(100.0)
        stats.attach(reset=True)
        model(probe)
        stats.detach()
        first = stats.summary()["__summary__"]["max_max_abs"]
        # Window 2 is much smaller; a pooled report must still show window 1's
        # peak, while a resetting one drops back to the smaller value.
        model.tok_emb.weight.zero_()
        stats.attach(reset=False)
        model(probe)
        stats.detach()
        pooled = stats.summary()["__summary__"]["max_max_abs"]
    assert first > 100.0, "the first window should carry the outlier"
    assert pooled > 100.0, "reset=False must not discard earlier meters"
    # A fresh attach() does reset.
    with torch.no_grad():
        stats.attach(reset=True)
        model(probe)
        stats.detach()
        assert stats.summary()["__summary__"]["max_max_abs"] <= 1.0


# ---------------------------------------------------------------------------
# Synthetic corpus: the ablations / MoE / W8A8 probe depend on it having
# real next-token signal.
# ---------------------------------------------------------------------------
def _mutual_information(x, y, n_states):
    pairs = torch.stack([x[:, :-1].reshape(-1), y[:, :-1].reshape(-1)], 1)
    joint = torch.zeros(n_states, n_states, dtype=torch.double)
    joint[pairs[:, 0].long(), pairs[:, 1].long()] += 1
    p = joint / joint.sum()
    px, py = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    tiny = 1e-30
    return (
        p * ((p.clamp(min=tiny).log()) - px.clamp(min=tiny).log()
             - py.clamp(min=tiny).log())
    ).sum().item()


def test_markov_corpus_has_learnable_next_token_signal():
    from adasquash import markov_batch

    rng = torch.Generator().manual_seed(0)
    x, y = markov_batch(rng, -100, 256, 512, "cpu", n_states=64)
    mi = _mutual_information(x, y, 64)
    # A per-position independent draw has I == 0 and pins the loss at
    # log(vocab_size) for every variant, making the ablation matrix
    # unrankable. A real walk over 64 states with jumps in {0,1,2} carries
    # log(64) - log(3) ~ 3.06 nats.
    assert mi > 2.5, f"I(x;y) = {mi:.3f} nats, the corpus has no signal"
    assert abs(mi - (math.log(64) - math.log(3))) < 0.2


def test_markov_corpus_shapes_and_masking():
    from adasquash import markov_batch

    rng = torch.Generator().manual_seed(0)
    x, y = markov_batch(rng, -100, 4, 16, "cpu", n_states=64)
    assert x.shape == y.shape == (4, 16)
    # Target is the shifted successor, final position masked out.
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert (y[:, -1] == -100).all()
    assert x.min() >= 0 and x.max() < 64


def test_markov_outlier_channel_does_not_destroy_the_targets():
    from adasquash import markov_batch

    rng = torch.Generator().manual_seed(0)
    x, y = markov_batch(rng, -100, 8, 512, "cpu", outlier_id=50303)
    frac = (x == 50303).float().mean().item()
    assert 0.01 < frac < 0.06, f"outlier channel fraction = {frac}"
    # The spike perturbs the *input* only; the targets stay on the clean
    # walk, so the corpus is still learnable.
    assert 50303 not in y.unique().tolist()


def test_depth_schedule_ablation_row_actually_disables_the_schedule():
    """`k_init=0.0` alone leaves `k` learnable, so the schedule comes back."""
    from scripts.ablation import load_ablations_from_config

    root = Path(__file__).resolve().parent.parent
    rows = load_ablations_from_config(
        str(root / "configs" / "ablation-matrix.yaml")
    )
    row = rows["dytd_k0_no_depth_schedule"]
    assert row["norm_name"] == "dytd"
    assert row["k_init"] == 0.0
    assert row.get("learnable_k") is False, (
        "with learnable_k=True the optimizer learns k straight back off zero "
        "and the row stops isolating the depth schedule"
    )


def test_depth_sweep_shape_holds_the_parameter_count_roughly_fixed():
    """The depth axis must not be confounded with model size."""
    from scripts.train_llama_depth_sweep import shape_for_depth

    V = 32000

    def params(depth, dim):
        hidden = int(dim * (8 / 3.0))
        hidden = int(hidden * 2 / 3)
        hidden = (hidden + 7) & ~7
        return depth * (4 * dim * dim + 3 * hidden * dim + 4 * dim) + 2 * V * dim

    counts = []
    for depth in (12, 24, 48, 96):
        dim, heads = shape_for_depth(depth, 2048, 16)
        assert dim % heads == 0
        assert dim // heads == 128, "head_dim must be constant across the sweep"
        counts.append(params(depth, dim))
    spread = max(counts) / min(counts)
    assert spread < 1.3, (
        f"parameter count varies {spread:.2f}x across the depth axis; a deep "
        f"run that is also a bigger model makes 'RMSNorm diverges at depth' "
        f"unfalsifiable"
    )


def test_depth_sweep_shape_honours_a_non_llama_base():
    from scripts.train_llama_depth_sweep import _BASE_SHAPE, shape_for_depth

    # The documented table is only valid for the LLaMA-tiny base shape; a
    # gpt2-124m sweep must not silently be built at LLaMA-tiny width.
    # head_dim is still held at the gpt2 base's 64, but dim/heads follow the
    # gpt2 base, not the table.
    dim, heads = shape_for_depth(12, 768, 12)
    assert dim // heads == 768 // 12
    assert (dim, heads) not in [(2560, 20), (2048, 16), (1536, 12), (1280, 10)]
    # The base depth is unchanged, so a gpt2 sweep still lands on the
    # builder's own shape there.
    assert shape_for_depth(24, 768, 12) == (768, 12)
    assert shape_for_depth(24, *_BASE_SHAPE) == (2048, 16)


# ---------------------------------------------------------------------------
# Training-loop invariants
# ---------------------------------------------------------------------------
def test_lr_schedule_has_no_dead_first_step():
    """`LambdaLR` evaluates step 0 at construction, so the first update
    would otherwise run at lr == 0 and warmup end one step late."""
    import argparse as _argparse
    from scripts.train_gpt2 import build_lr_lambda
    from torch.optim.lr_scheduler import LambdaLR

    args = _argparse.Namespace(max_steps=100, warmup=10, min_lr_ratio=0.1)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=6e-4)
    sched = LambdaLR(opt, build_lr_lambda(args))
    assert opt.param_groups[0]["lr"] > 0.0, "first update would be a no-op"
    # The lambda is evaluated at `last_epoch`, which starts at 0, so
    # `last_epoch = 9` is the last index whose `s + 1` lands on `warmup`.
    for _ in range(9):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 6e-4) < 1e-9, "peak not reached at end of warmup"
    for _ in range(90):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 6e-5) < 1e-9, "min_lr_ratio not reached"


def test_resume_restores_the_schedule_defining_flags():
    import argparse as _argparse
    from scripts.train_gpt2 import restore_schedule_args

    ckpt_args = dict(max_steps=10000, warmup=500, min_lr_ratio=0.1, lr=6e-4,
                     weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0)
    args = _argparse.Namespace(max_steps=30000, warmup=123, min_lr_ratio=0.1,
                               lr=6e-4, weight_decay=0.1, beta1=0.9, beta2=0.95,
                               grad_clip=1.0)
    changed = restore_schedule_args(args, ckpt_args)
    assert set(changed) == {"max_steps", "warmup"}
    assert args.max_steps == 10000 and args.warmup == 500
    # Identical flags are a no-op; a checkpoint without `args` keeps the CLI.
    same = _argparse.Namespace(**ckpt_args)
    assert restore_schedule_args(same, ckpt_args) == []
    assert restore_schedule_args(same, {}) == []


def test_qk_alpha_init_reaches_the_attention():
    """The QKSquash init used to be a constructor argument nothing could set."""
    cfg = GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                    norm_name="rmsnorm", qk_norm="dytqk", qk_alpha_init=0.25)
    model = GPT(cfg)
    sq = model.blocks[0].attn.q_squash
    assert sq is not None and sq is model.blocks[0].attn.k_squash
    assert sq.alpha_q.item() == pytest.approx(0.25)
    assert sq.alpha_k.item() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Needle-in-a-haystack protocol
# ---------------------------------------------------------------------------
class _CharTokenizer:
    """Char-level tokenizer, so a secret spans several tokens."""

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) % 60 + 1 for c in s]}


class _ScriptedModel:
    """Emits a scripted token sequence, one per call."""

    def __init__(self, ids, cfg):
        self.ids, self.cfg, self.base = ids, cfg, None

    def __call__(self, x, y=None):
        if self.base is None:
            self.base = x.shape[1]
        k = min(x.shape[1] - self.base, len(self.ids) - 1)
        logits = torch.full((x.shape[0], x.shape[1], 64), -20.0)
        logits[:, -1, self.ids[k]] = 20.0
        return logits, None


def test_needle_scores_the_whole_secret_not_just_its_first_token():
    """GPT-2 splits " 12345" into several BPE tokens; matching only the
    first one accepts a model that recites 3 of the 5 digits."""
    from scripts.eval_downstream import needle_in_haystack

    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=256, norm_name="rmsnorm")).eval()
    tok = _CharTokenizer()
    needle = "The magic number is 12345."
    target = tok(" 12345")["input_ids"]
    assert len(target) > 1, "the secret must span several tokens for this test"

    exact = needle_in_haystack(
        _ScriptedModel(target, model.cfg), tok, needle=needle,
        context_len=256, n_trials=2, device="cpu")
    assert exact == 1.0

    wrong_tail = target[:-1] + [(target[-1] % 60) + 1]
    partial = needle_in_haystack(
        _ScriptedModel(wrong_tail, model.cfg), tok, needle=needle,
        context_len=256, n_trials=2, device="cpu")
    assert partial == 0.0


def test_needle_rejects_contexts_it_cannot_evaluate():
    """A context past `max_seq_len` used to crash inside the forward pass,
    and one too short to hold a single haystack repeat silently scored 0.0."""
    from scripts.eval_downstream import needle_in_haystack

    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=256, norm_name="rmsnorm")).eval()
    tok = _CharTokenizer()
    with pytest.raises(ValueError, match="max_seq_len"):
        needle_in_haystack(model, tok, context_len=4096, n_trials=1, device="cpu")
    with pytest.raises(ValueError, match="too short"):
        needle_in_haystack(model, tok, context_len=20, n_trials=1, device="cpu")


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
