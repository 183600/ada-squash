from __future__ import annotations
"""Reproducibility and experiment-tracking helpers.

`set_all_seeds` fixes every RNG a run touches (Python, NumPy, torch, CUDA,
cuDNN) and `seed_worker` extends that to DataLoader worker processes.
`collect_run_metadata` / `save_run_manifest` record the code revision, the
seed, the environment and any data fingerprints next to the run outputs so
every experiment is traceable after the fact.

The contract is same-machine reproducibility: bit-identical results across
different hardware/frameworks are not promised, but a rerun on the same
machine with the same manifest must reproduce its own curve and final
metrics. `environment_info` therefore also records the *RNG/determinism
configuration* the run actually used (cuDNN flags, TF32 flags, thread
counts, hash seeds), so a manifest mismatch can be diagnosed from the
record instead of from memory.

TensorFlow is not a dependency of this repository; the training stack is
pure PyTorch, so no TF RNG is seeded here.
"""
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_all_seeds(seed: int, deterministic: bool = True) -> int:
    """Seed the Python, NumPy, torch and CUDA RNGs from `seed`.

    `deterministic=True` additionally pins cuDNN to its deterministic
    algorithms (same-machine reproducibility). Torch workers spawned later
    (DataLoader, subprocess launches) inherit the seed through `seed_worker`
    / `PYTHONHASHSEED`.

    Note on `PYTHONHASHSEED`: it is exported here so child processes
    (DataLoader workers, subprocess launches) get a fixed string-hash seed
    from interpreter startup. The *current* process already picked its hash
    seed at startup, so a launcher that spawns training subprocesses (e.g.
    the depth sweep) should also put `PYTHONHASHSEED=<seed>` in the child
    environment explicitly - that is the only point where it takes effect.
    """
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS needs a fixed workspace to run deterministic reductions once
        # `torch.use_deterministic_algorithms(True)` is enabled; pinning it
        # here keeps the environment reproducible without forcing the
        # deterministic-algorithm mode on.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed % (2 ** 32))
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: derive each worker's RNGs from its seed.

    `torch.initial_seed()` in a worker is the loader's base seed (from the
    explicit `generator` the pipeline passes) plus the worker id, so worker
    randomness is a deterministic function of the run seed. Python and NumPy
    are re-seeded from that same value; torch is re-seeded as well (idempotent
    with the loader's own worker seeding) so the worker contract does not
    depend on torch-version internals.
    """
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)


def _run_git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_revision() -> dict:
    """Code version: commit, branch and whether the tree has uncommitted changes.

    The branch and dirty flag come from one `git status --porcelain=v1
    --branch` call (the default porcelain is already v1; the branch header
    line names the checked-out branch) instead of a separate `rev-parse
    --abbrev-ref` spawn; only the commit hash needs its own call.
    """
    status = _run_git("status", "--porcelain=v1", "--branch")
    branch = None
    dirty = None
    if status is not None:
        lines = status.splitlines()
        dirty = False
        if lines and lines[0].startswith("## "):
            header = lines[0][3:].strip()
            name = header.split("...")[0].split(" ")[0]
            branch = name or None
            dirty = len(lines) > 1
        else:
            # No branch header (e.g. unborn repository): fall back to the
            # tracked-only answer rather than dropping the fields.
            branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
            dirty = bool(status)
    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": branch,
        "dirty": dirty,
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def file_fingerprint(path) -> dict:
    """Content fingerprint of a data file (e.g. a hash ledger): sha256 + size + mtime."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "sha256": h.hexdigest(),
        "bytes": st.st_size,
        "mtime": st.st_mtime,
    }


def environment_info() -> dict:
    """Software/hardware fingerprint of the process that produced a run.

    Besides library versions it records the reproducibility configuration:
    the cuDNN/TF32/deterministic-algorithm flags and the thread counts that
    decide the reduction order of CPU/GPU kernels, plus the hash seeds. Two
    manifests with different values here are not expected to match even on
    the same machine.
    """
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": _module_version("torch"),
        "numpy": _module_version("numpy"),
        "transformers": _module_version("transformers"),
        "datasets": _module_version("datasets"),
        "pyyaml": _module_version("yaml"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        # Reproducibility configuration actually in effect for this process.
        "rng_config": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_algorithms_warn_only": bool(
                getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
            ),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "hash_randomization": bool(sys.flags.hash_randomization),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    if torch.cuda.is_available():
        info["device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return info


def collect_run_metadata(seed: int | None = None, extra: dict | None = None) -> dict:
    """Metadata block attached to every run manifest."""
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": None if seed is None else int(seed),
        "git": git_revision(),
        "environment": environment_info(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_run_manifest(path, metadata: dict) -> Path:
    """Atomically write a run manifest (tmp + rename, so readers never see a partial file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


__all__ = [
    "REPO_ROOT",
    "collect_run_metadata",
    "environment_info",
    "file_fingerprint",
    "git_revision",
    "save_run_manifest",
    "seed_worker",
    "set_all_seeds",
]
