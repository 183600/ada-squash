#!/usr/bin/env python3
"""CI gate for the DyT++ / AdaSquash continuous loop.

Exit 0 = pass. Any other exit code = the calling workflow must discard the
phase's changes (git reset --hard).

Layers
------
1. Syntax: every .py file parses (always run, no dependencies).
2. Design pins: docs/design_notes.md 【固定】 decisions D1-D6 enforced by
   source inspection.
3. Full pytest suite (tests/), including the per-parameter comparison of
   every fused autograd kernel against an independent autograd
   recomputation. Skipped (with a warning) when torch/pytest are missing so
   a network hiccup can never brick the loop.

--syntax-only runs layer 1 only (used by the shutdown trap).
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[gate] FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"[gate] ok: {msg}")


# --------------------------------------------------------------------------
# Layer 1: syntax
# --------------------------------------------------------------------------
def check_syntax() -> None:
    n = 0
    for py in sorted(ROOT.rglob("*.py")):
        if any(part in (".git", "__pycache__", ".venv", "venv") for part in py.parts):
            continue
        try:
            ast.parse(py.read_text(encoding="utf-8"))
            n += 1
        except SyntaxError as e:
            fail(f"syntax error in {py.relative_to(ROOT)}: {e}")
    if n:
        ok(f"syntax: {n} python files parse")


# --------------------------------------------------------------------------
# Layer 2: design pins (docs/design_notes.md 【固定】 decisions)
# --------------------------------------------------------------------------
def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def check_pins() -> None:
    attn = read("adasquash/attention.py")
    norms = read("adasquash/norms.py")
    model = read("adasquash/model.py")
    tests = read("tests/test_norms.py")

    if not (attn and norms and model):
        fail("adasquash/{attention,norms,model}.py must all exist")
        return

    # D1: attention stays the standard SDPA baseline (it is infrastructure,
    # not a research variable in this repo).
    if "scaled_dot_product_attention" not in attn:
        fail("D1: CausalSelfAttention must keep F.scaled_dot_product_attention "
             "(is_causal=True) as the fixed baseline path")
    if "def apply_rope" not in attn or "class CausalSelfAttention" not in attn:
        fail("D1: apply_rope / CausalSelfAttention disappeared from attention.py")

    # D2: numerical-stability floors and the reciprocal-on-max formulation.
    if "_MIN_P = 1e-3" not in norms:
        fail("D2: the generalized-softsign p floor must stay _MIN_P = 1e-3")
    if "_MIN_ALPHA = 1e-4" not in norms:
        fail("D2: the DyT-CD alpha floor must stay _MIN_ALPHA = 1e-4")
    if "clamp_(min=1.0).reciprocal_()" not in norms:
        fail("D2: the softsign core must keep the reciprocal-on-max(|x|,1) form")
    if "copysign_" not in norms:
        fail("D2: the softsign core must keep the copysign sign handling")
    if "def generalized_softsign" not in norms:
        fail("D2: generalized_softsign public entry point is missing")

    # D3: the fused-kernel family (research core).
    for sym in ("class _GeneralizedSoftsign", "class _RMSNorm",
                "class _DyTForward", "class _DyTCForward",
                "class _DyTDForward", "class _DyTCDForward"):
        if sym not in norms:
            fail(f"D3: {sym} is missing from norms.py")
    for sym in ("class _ChunkedCrossEntropy", "class _FusedLinearCrossEntropy",
                "class _LayerScaleResidual", "class _SwiGLU",
                "def cross_entropy_mean", "def fused_cross_entropy_mean"):
        if sym not in model:
            fail(f"D3: {sym} is missing from model.py")
    if not tests:
        fail("D3: tests/test_norms.py must exist (per-parameter kernel checks)")

    # D4: QKSquash stays tanh.
    if "class QKSquash" not in norms or "torch.tanh" not in norms:
        fail("D4: QKSquash must keep the tanh(alpha * Q/K) formulation")

    # D5: registry + hyper-parameter dispatch.
    for name in ('"ln"', '"rmsnorm"', '"dyt"', '"dytc"', '"dytd"', '"dytcd"'):
        if name not in norms:
            fail(f"D5: NORM_REGISTRY lost the {name} entry")
    if "def build_norm_kwargs" not in model:
        fail("D5: build_norm_kwargs hyper-parameter dispatch is missing")

    # D6: cache key semantics (no per-call rebuild, no id()-keyed cache).
    if "_tril_cache" not in attn or "_rope_cache" not in attn:
        fail("D6: the tril/rope caches in attention.py are missing")
    if "data_ptr()" not in attn:
        fail("D6: the rope cache must keep the storage-pointer key")

    ok("design pins D1-D6 hold")


# --------------------------------------------------------------------------
# Layer 3: full pytest suite (needs torch+numpy+PyYAML+pytest)
# --------------------------------------------------------------------------
def check_pytest() -> None:
    try:
        import torch  # noqa: F401
        import pytest  # noqa: F401
    except Exception as e:
        print(f"[gate] warn: torch/pytest unavailable ({e}); pytest layer skipped")
        return
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    tail = (r.stdout or r.stderr).strip().splitlines()
    summary = tail[-1] if tail else "(no output)"
    if r.returncode != 0:
        fail(f"pytest suite failed ({summary})")
        for line in tail[-15:]:
            print(f"    {line}")
    else:
        ok(f"pytest: {summary}")


def main() -> int:
    syntax_only = "--syntax-only" in sys.argv
    check_syntax()
    if not syntax_only:
        check_pins()
        check_pytest()
    if FAILURES:
        print(f"\n[gate] REJECTED — {len(FAILURES)} failure(s)")
        return 1
    print("\n[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
