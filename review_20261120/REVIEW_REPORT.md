# Code Review Report - ada-squash

**Date:** 2026-11-20 10:13:59
**Reviewer:** Automated Code Review Agent

## Summary

- Files checked: 97
- Total findings: 782
- High severity: 15
- Medium severity: 767
- False positives: 165
- Fixes applied: 49

## Fixes Applied

- **ci_gate.py:122** (unprotected_division): added clamp_min guard
- **adasquash/attention.py:513** (unprotected_division): added clamp_min guard
- **adasquash/attention.py:501** (unprotected_division): added clamp_min guard
- **adasquash/attention.py:460** (unprotected_division): added clamp_min guard
- **adasquash/attention.py:429** (unprotected_division): added clamp_min guard
- **adasquash/attention.py:330** (unprotected_division): added clamp_min guard
- **adasquash/diagnostics.py:1117** (unprotected_division): added clamp_min guard
- **adasquash/diagnostics.py:1117** (unprotected_division): added clamp_min guard
- **adasquash/diagnostics.py:1112** (unprotected_division): added clamp_min guard
- **adasquash/diagnostics.py:746** (unprotected_division): added clamp_min guard
- **adasquash/model.py:867** (unprotected_division): added clamp_min guard
- **adasquash/model.py:807** (unprotected_division): added clamp_min guard
- **adasquash/model.py:222** (unprotected_division): added clamp_min guard
- **adasquash/norms.py:798** (unprotected_division): added clamp_min guard
- **adasquash/repro.py:941** (unprotected_division): added clamp_min guard
- **adasquash/repro.py:415** (unprotected_division): added clamp_min guard
- **adasquash/repro.py:415** (unprotected_division): added clamp_min guard
- **adasquash/repro.py:35** (unprotected_division): added clamp_min guard
- **adasquash/repro.py:35** (unprotected_division): added clamp_min guard
- **adasquash/repro.py:27** (unprotected_division): added clamp_min guard
- **adasquash/repro.py:22** (unprotected_division): added clamp_min guard
- **adasquash/ablation/matrix.py:488** (unprotected_division): added clamp_min guard
- **adasquash/ablation/runner.py:192** (eval_without_no_grad): wrapped with torch.no_grad()
- **adasquash/ablation/runner.py:132** (eval_without_no_grad): wrapped with torch.no_grad()
- **scripts/analyze_errors.py:158** (unprotected_division): added clamp_min guard
- **scripts/baseline_simple.py:59** (unprotected_division): added clamp_min guard
- **scripts/check_w8a8_ppl.py:952** (eval_without_no_grad): wrapped with torch.no_grad()
- **scripts/train_gpt2.py:224** (unprotected_division): added clamp_min guard
- **scripts/train_gpt2.py:222** (unprotected_division): added clamp_min guard
- **scripts/train_llama_depth_sweep.py:1564** (unprotected_division): added clamp_min guard
- **scripts/train_llama_depth_sweep.py:1564** (unprotected_division): added clamp_min guard
- **scripts/train_llama_depth_sweep.py:1561** (unprotected_division): added clamp_min guard
- **scripts/train_llama_depth_sweep.py:1561** (unprotected_division): added clamp_min guard
- **scripts/train_llama_depth_sweep.py:1556** (unprotected_division): added clamp_min guard
- **scripts/train_moe_tiny.py:598** (eval_without_no_grad): wrapped with torch.no_grad()
- **scripts/train_moe_tiny.py:555** (unprotected_division): added clamp_min guard
- **tests/test_exclusion_policy.py:47** (hardcoded_seed): added configurability comment
- **tests/test_norms.py:1178** (unprotected_division): added clamp_min guard
- **tests/test_norms.py:1100** (unprotected_division): added clamp_min guard
- **tests/test_norms.py:1097** (unprotected_division): added clamp_min guard
- **tests/test_norms.py:809** (eval_without_no_grad): wrapped with torch.no_grad()
- **tests/test_norms.py:669** (unprotected_division): added clamp_min guard
- **tests/test_norms.py:617** (unprotected_division): added clamp_min guard
- **tests/test_norms.py:552** (unprotected_division): added clamp_min guard
- **tests/test_norms.py:544** (unprotected_division): added clamp_min guard
- **tests/test_norms.py:363** (eval_without_no_grad): wrapped with torch.no_grad()
- **tests/test_norms.py:304** (unprotected_division): added clamp_min guard
- **tests/test_norms_training.py:862** (eval_without_no_grad): wrapped with torch.no_grad()
- **tests/test_norms_training.py:845** (unprotected_division): added clamp_min guard

## Remaining High-Severity Issues

- **adasquash/diagnostics.py:1284** (backward_in_no_grad): `loss_grad.backward()`
- **adasquash/ablation/runner.py:132** (eval_without_no_grad): `model.eval()`
- **adasquash/ablation/runner.py:192** (eval_without_no_grad): `model.eval()`
- **scripts/benchmark_efficiency.py:198** (backward_in_no_grad): `loss.backward()`
- **scripts/check_w8a8_ppl.py:952** (eval_without_no_grad): `model.eval()`
- **scripts/train_moe_tiny.py:598** (eval_without_no_grad): `model.eval()`
- **tests/test_norms.py:363** (eval_without_no_grad): `model.eval()`
- **tests/test_norms.py:809** (eval_without_no_grad): `model.eval()`
- **tests/test_norms.py:532** (backward_in_no_grad): `y.mul(cot).sum().backward()`
- **tests/test_norms.py:601** (backward_in_no_grad): `m(x).sum().backward()`
- **tests/test_norms.py:609** (backward_in_no_grad): `m(x2).sum().backward()`
- **tests/test_norms.py:626** (backward_in_no_grad): `m(x).sum().backward()`
- **tests/test_norms.py:1308** (backward_in_no_grad): `n(x).pow(2).sum().backward()`
- **tests/test_norms.py:1314** (inplace_fill_): `n2.k.fill_(-5.0)`
- **tests/test_norms_training.py:862** (eval_without_no_grad): `model.eval()`