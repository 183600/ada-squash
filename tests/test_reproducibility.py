"""Reproducibility-infrastructure checks that must stay green.

The same-seed and resume-equivalence properties of the training driver live
next to `_tiny_train_run` in ``test_norms_training.py``; this module covers
the reusable RNG/manifest/split infrastructure and its command-line entry
point. Running the module's self-test in a subprocess keeps the global RNG,
cuDNN and hash-seed mutations it deliberately performs out of the main pytest
process.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_repro_self_test_passes_via_cli():
    result = subprocess.run(
        [sys.executable, "-m", "adasquash.repro", "--self-test"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[repro-self-test] PASS" in result.stdout
    assert "FAIL:" not in result.stdout
