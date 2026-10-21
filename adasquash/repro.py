from __future__ import annotations
"""Reproducibility and experiment-tracking helpers.

`set_all_seeds` fixes every RNG a run touches (Python, NumPy, torch, CUDA,
cuDNN) and `seed_worker` extends that to DataLoader worker processes.
`collect_run_metadata` / `save_run_manifest` record the code revision, the
seed, the environment and any data fingerprints next to the run outputs so
every experiment is traceable after the fact.

The contract is same-machine reproducibility: bit-identical results across
different hardware/frameworks are not promised, but a rerun on the same
machine with the same manifest should reproduce its own curve and final
metrics. Some CUDA kernels (flash-attention backward, `index_add_`,
scatter reductions) are nondeterministic by implementation, so bit-identity
is only guaranteed once those are routed to deterministic kernels: set
`ADASQUASH_DETERMINISTIC_ALGORITHMS=warn` (warn but keep running) or
`=strict` (raise on a nondeterministic op) to request
`torch.use_deterministic_algorithms`; `set_all_seeds` picks the mode up.
The manifest records whether that opt-in was active, so a mismatch can be
diagnosed from the record instead of from memory. `environment_info` also
records the rest of the *RNG/determinism configuration* the run actually
used (cuDNN flags, TF32 flags, thread counts, hash seeds).

TensorFlow is not a dependency of this repository; the training stack is
pure PyTorch, so no TF RNG is seeded here.

`python3 -m adasquash.repro --self-test` runs the in-process regression
check for this module (seed determinism across Python/NumPy/torch, worker
seeding, DataLoader reproducibility, manifest round-trip, split-boundary
invariants).
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

# `PYTHONHASHSEED` is only read by the interpreter at startup. Snapshotting
# the environment here (import time, before `set_all_seeds` can overwrite it)
# lets `environment_info` distinguish "this process really started with a
# fixed hash seed" from "this process is randomizing and only exports a seed
# for its children".
_HASHSEED_AT_IMPORT = os.environ.get("PYTHONHASHSEED")
_DETERMINISTIC_ALGO_ENV = "ADASQUASH_DETERMINISTIC_ALGORITHMS"


def _requested_deterministic_algorithms() -> str | None:
    """Parse `ADASQUASH_DETERMINISTIC_ALGORITHMS` into None / 'warn' / 'strict'."""
    raw = os.environ.get(_DETERMINISTIC_ALGO_ENV)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("", "0", "false", "off", "none", "no"):
        return None
    if value in ("warn", "warn-only", "warn_only"):
        return "warn"
    if value in ("1", "true", "yes", "on", "strict"):
        return "strict"
    raise ValueError(
        f"{_DETERMINISTIC_ALGO_ENV}={raw!r} is not understood; use 'warn' "
        f"(warn on nondeterministic ops but keep running), 'strict' (raise), "
        f"or '0' to leave torch's setting untouched"
    )


def apply_deterministic_algorithms(mode: str | None = None) -> str | None:
    """Enable `torch.use_deterministic_algorithms` when the run asks for it.

    The default (`mode=None` and no `ADASQUASH_DETERMINISTIC_ALGORITHMS`)
    leaves torch's setting untouched: forcing the mode on would make known
    nondeterministic kernels (flash-attention backward, `index_add_`) either
    warn on every step or abort training on hardware where they are the only
    implementation. When requested, 'warn' routes every op that has a
    deterministic implementation to it and warns otherwise; 'strict' raises.
    Returns the mode actually applied (None = not applied).
    """
    if mode is None:
        mode = _requested_deterministic_algorithms()
    if mode == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif mode == "strict":
        torch.use_deterministic_algorithms(True, warn_only=False)
    elif mode is not None:
        raise ValueError(f"unknown deterministic-algorithms mode {mode!r}")
    return mode


def deterministic_algorithms_context(mode: str | None = None):
    """Context manager to temporarily enable deterministic algorithms.

    Usage:
        with deterministic_algorithms_context("warn"):
            # training code here
            pass
        # original setting restored

    The mode can be 'warn' (route to deterministic kernels where available,
    warn otherwise) or 'strict' (raise on nondeterministic ops). If None,
    reads from `ADASQUASH_DETERMINISTIC_ALGORITHMS` env var.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        previous_enabled = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = bool(
            getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
        )
        applied = apply_deterministic_algorithms(mode)
        try:
            yield applied
        finally:
            torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)

    return _ctx()


def set_all_seeds(seed: int, deterministic: bool = True) -> int:
    """Seed the Python, NumPy, torch and CUDA RNGs from `seed`.

    `deterministic=True` additionally pins cuDNN to its deterministic
    algorithms (same-machine reproducibility). Torch workers spawned later
    (DataLoader, subprocess launches) inherit the seed through `seed_worker`
    / `PYTHONHASHSEED`.

    When `ADASQUASH_DETERMINISTIC_ALGORITHMS` is set, the corresponding
    `torch.use_deterministic_algorithms` mode is applied here as well, and
    `environment_info` records it in the manifest.

    Note on `PYTHONHASHSEED` (IMPORTANT): it is exported here so child processes
    (DataLoader workers, subprocess launches) get a fixed string-hash seed
    from interpreter startup. The *current* process already picked its hash
    seed at startup (recorded as `pythonhashseed_at_process_start` in the
    manifest), so **setting `PYTHONHASHSEED` here does NOT affect the current
    process's hash randomization**. A launcher that spawns training subprocesses
    (e.g. the depth sweep, multi-seed launcher) MUST also put
    `PYTHONHASHSEED=<seed>` in the child environment explicitly - that is the
    only point where it takes effect. This is handled automatically by
    `launch_multi_seed` in `train_gpt2.py` and `_child_env` in
    `train_llama_depth_sweep.py`.
    """
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2 ** 32))
    # TensorFlow is an optional dependency of the training stack; seed it when
    # it is importable so a run that mixes torch and tf code (or a downstream
    # eval that does) reproduces. Failures to import are expected and ignored.
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Honor an explicit ADASQUASH_DETERMINISTIC_ALGORITHMS request regardless
    # of the cuDNN flag, so a caller can opt into deterministic kernels
    # without forcing cuDNN determinism (and vice versa).
    apply_deterministic_algorithms()
    # cuBLAS needs a fixed workspace configuration for deterministic
    # reductions to be available at all; pinning it unconditionally (rather
    # than only under the deterministic flag) keeps the environment
    # reproducible even when cuDNN determinism is left off, and is a no-op
    # when the variable was already set by the launcher.
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass


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
    line names the checked-out branch); only the commit hash needs its own
    call.
    """
    status = _run_git("status", "--porcelain=v1", "--branch")
    branch = None
    dirty = None
    if status is not None:
        lines = status.splitlines()
        dirty = False
        if lines and lines[0].startswith("## "):
            header = lines[0][3:].strip()
            if header.startswith("No commits yet"):
                # Unborn repository: the header names the branch-to-be
                # (`## No commits yet on master`) but no branch exists yet,
                # so ask `rev-parse --abbrev-ref HEAD`.
                branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
            else:
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


# Training variables that must be identical when two arms are compared: a
# different optimizer/schedule/step count/data slice makes the delta a recipe
# effect, not an effect of the module. Architecture/initialization knobs are
# deliberately absent (they are the treatment in an ablation); model shape is
# checked separately by the callers that load checkpoints.
TRAINING_RECIPE_KEYS = (
    "model", "dataset", "dataset_config", "dataset_revision",
    "tokenizer", "tokenizer_revision", "seq_len", "batch_size", "grad_accum",
    "max_steps", "lr", "min_lr_ratio", "warmup", "weight_decay",
    "beta1", "beta2", "grad_clip", "no_bf16", "gradient_checkpointing",
    "early_stop_patience", "nonfinite_policy", "num_workers",
    "val_frac", "test_frac", "num_documents", "val_every", "val_batches",
    "min_doc_chars", "max_doc_chars", "dedup", "train_shuffle_buffer",
    "exclude_hash_file", "exclude_test_hash_file", "group_key",
    "exclude_group_file", "exclude_test_group_file",
)


def training_recipe_mismatches(reference: dict, other: dict,
                               keys=TRAINING_RECIPE_KEYS) -> dict:
    """Fields whose recorded training variables differ between two runs.

    A missing key is treated as unknown, never as equal: an old record that
    does not say which step count it used cannot establish that it used the
    same one. Empty dict means every requested field is present and equal.
    """
    if not isinstance(reference, dict) or not isinstance(other, dict):
        return {"hyperparameters": ["missing in one run", "missing in the other"]}
    missing = object()
    out = {}
    for key in keys:
        a = reference.get(key, missing)
        b = other.get(key, missing)
        if a is missing or b is missing or a != b:
            out[key] = ["<missing>" if a is missing else a,
                        "<missing>" if b is missing else b]
    return out


def artifact_record(path) -> dict:
    """Cheap pointer to a run artifact: path, existence, size and mtime.

    Size+mtime detects a changed/replaced artifact without reading it; use
    `file_fingerprint` when a content hash of a small file (ledger, report)
    is what pins the version.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": st.st_size,
        "mtime": st.st_mtime,
    }


def finish_metadata(metadata: dict, results: dict | None = None,
                    artifacts: dict | None = None) -> dict:
    """Return a copy of `metadata` marked as a finished run.

    One place for the `status` / `finished_utc` contract every report
    manifest carries, so a reader can tell a completed experiment from an
    interrupted one without re-deriving it from the artifact timestamps.
    """
    finished = dict(metadata)
    finished["status"] = "finished"
    finished["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if results is not None:
        finished["results"] = results
    if artifacts is not None:
        finished["artifacts"] = artifacts
    return finished


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
        "tensorflow": _module_version("tensorflow"),
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
            "deterministic_algorithms_env": os.environ.get(_DETERMINISTIC_ALGO_ENV),
            # What the interpreter actually started with (None = randomizing,
            # `hash_randomization` is then true) versus the value exported for
            # child processes. `set_all_seeds` overwrites the env var but
            # cannot change this process's own hash seed.
            "pythonhashseed_at_process_start": _HASHSEED_AT_IMPORT,
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


def collect_run_metadata(seed: int | None = None, extra: dict | None = None,
                        hyperparameters: dict | None = None,
                        data_version=None, training_curves=None,
                        results: dict | None = None) -> dict:
    """Metadata block attached to every run manifest.

    `hyperparameters`, `data_version`, `training_curves` and `results` are
    optional convenience fields so a caller can record the full experiment
    (code version, hyperparameters, seed, data version, training curves and
    final results) in one call instead of mutating the dict by hand. They are
    merged into the metadata under the same keys the manifests already use,
    and `extra` (if given) is applied last, so callers can still override.
    """
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": None if seed is None else int(seed),
        "git": git_revision(),
        "environment": environment_info(),
    }
    if hyperparameters is not None:
        metadata["hyperparameters"] = hyperparameters
    if data_version is not None:
        metadata["data_version"] = data_version
    if training_curves is not None:
        metadata["training_curves"] = training_curves
    if results is not None:
        metadata["results"] = results
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


def _self_test_seed_determinism() -> bool:
    import numpy as np

    def draw():
        return (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(3).sum()),
            float(torch.randn(3).sum()),
        )

    set_all_seeds(1234)
    first = draw()
    set_all_seeds(1234)
    return draw() == first


def _self_test_worker_seeding() -> bool:
    original_initial_seed = torch.initial_seed
    try:
        # Same base seed -> same worker stream.
        torch.initial_seed = lambda: 7
        seed_worker(0)
        first = (random.random(), float(torch.rand(1)))
        seed_worker(0)
        second = (random.random(), float(torch.rand(1)))
        # Different base seeds (what DataLoader hands worker i as
        # base_seed + worker_id) must produce different streams: sharing one
        # stream across workers would make the loader's worker count change
        # the data.
        torch.initial_seed = lambda: 8
        seed_worker(1)
        third = (random.random(), float(torch.rand(1)))
    finally:
        torch.initial_seed = original_initial_seed
    return first == second and first != third


def _self_test_dataloader_reproducibility() -> bool:
    from torch.utils.data import DataLoader, TensorDataset

    data = TensorDataset(torch.arange(64))
    orders = []
    for _ in range(2):
        g = torch.Generator()
        g.manual_seed(99)
        loader = DataLoader(data, batch_size=8, shuffle=True, generator=g,
                            worker_init_fn=seed_worker, num_workers=0)
        orders.append([int(batch[0][0]) for batch in loader])
    return orders[0] == orders[1]


def _self_test_manifest_roundtrip() -> bool:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'nested' / 'run_manifest.json'
        metadata = collect_run_metadata(seed=5)
        save_run_manifest(path, metadata)
        loaded = json.loads(path.read_text())
        if not (path.exists() and loaded.get('seed') == 5):
            return False
        if (path.parent / (path.name + '.tmp')).exists():
            return False
        rng = loaded.get('environment', {}).get('rng_config', {})
        required = (
            'torch_num_threads', 'cudnn_deterministic', 'tf32_matmul_allowed',
            'deterministic_algorithms', 'pythonhashseed_at_process_start',
            'pythonhashseed', 'hash_randomization',
        )
        return all(key in rng for key in required)


def _self_test_split_boundaries() -> bool:
    from scripts.data_pipeline import split_document_boundaries

    bounds = split_document_boundaries(1000, 0.005, 0.005)
    train, val, test = bounds['train'], bounds['validation'], bounds['test']
    contiguous = (train[0] == 0 and val[0] == train[0] + train[1]
                  and test[0] == val[0] + val[1]
                  and test[0] + test[1] == 1000)
    nonempty = all(count > 0 for (_, count) in (train, val, test))
    try:
        split_document_boundaries(10, 0.6, 0.6)
        rejects_impossible = False
    except ValueError:
        rejects_impossible = True
    return bool(contiguous and nonempty and rejects_impossible)


def _self_test_worker_numpy_seeding() -> bool:
    import numpy as np

    original_initial_seed = torch.initial_seed
    try:
        torch.initial_seed = lambda: 11
        seed_worker(0)
        first = float(np.random.rand())
        seed_worker(0)
        second = float(np.random.rand())
    finally:
        torch.initial_seed = original_initial_seed
    return first == second


def _self_test_seed_contract_flags() -> bool:
    """`set_all_seeds` must also pin the flags/manifest inputs, not only streams."""
    import numpy as np

    previous = (
        bool(torch.backends.cudnn.deterministic),
        bool(torch.backends.cudnn.benchmark),
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        os.environ.get("PYTHONHASHSEED"),
    )
    try:
        set_all_seeds(21)
        first = float(np.random.rand())
        set_all_seeds(21)
        np_ok = float(np.random.rand()) == first
        return bool(
            torch.backends.cudnn.deterministic
            and not torch.backends.cudnn.benchmark
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and os.environ.get("PYTHONHASHSEED") == str(21)
            and np_ok
        )
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[0], previous[1]
        if previous[2] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous[2]
        if previous[3] is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = previous[3]


def _self_test_deterministic_algorithms_optin() -> bool:
    if torch.are_deterministic_algorithms_enabled():
        return True
    previous_warn_only = bool(
        getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)())
    was_enabled = torch.are_deterministic_algorithms_enabled()
    os.environ[_DETERMINISTIC_ALGO_ENV] = "warn"
    try:
        mode = apply_deterministic_algorithms()
        ok = (mode == "warn" and torch.are_deterministic_algorithms_enabled()
              and bool(getattr(torch, "is_deterministic_algorithms_warn_only_enabled",
                               lambda: False)()))
    finally:
        del os.environ[_DETERMINISTIC_ALGO_ENV]
        torch.use_deterministic_algorithms(was_enabled, warn_only=previous_warn_only)
    return ok


def self_test(verbose: bool = True) -> dict:
    """Run the in-process reproducibility regression checks.

    Covers the guarantees the experiments rely on: identical RNG streams for
    a reused seed across Python/NumPy/torch, per-worker stream derivation,
    deterministic DataLoader ordering under an explicit generator +
    `seed_worker`, atomic manifest round-trip with the determinism fields,
    the chronological/disjoint split boundaries, and the deterministic-
    algorithms opt-in. Meant to be run from CI or by hand
    (`python3 -m adasquash.repro --self-test`).
    """
    checks = {
        'seed_determinism_python_numpy_torch': _self_test_seed_determinism(),
        'worker_seeding_derived_and_distinct': _self_test_worker_seeding(),
        'worker_numpy_seeding': _self_test_worker_numpy_seeding(),
        'seed_contract_flags_and_hashseed': _self_test_seed_contract_flags(),
        'dataloader_reproducible': _self_test_dataloader_reproducibility(),
        'manifest_atomic_roundtrip_with_determinism_fields': _self_test_manifest_roundtrip(),
        'split_boundaries_chronological_and_disjoint': _self_test_split_boundaries(),
        'deterministic_algorithms_optin': _self_test_deterministic_algorithms_optin(),
    }
    if verbose:
        for name, ok in checks.items():
            print(f"[repro-self-test] {'ok' if ok else 'FAIL'}: {name}")
    checks['passed'] = all(v for v in checks.values())
    return checks


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true',
                        help='run the reproducibility-infrastructure regression checks')
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.print_help()
        return 2
    result = self_test()
    print(f"[repro-self-test] {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result['passed'] else 1


__all__ = [
    "REPO_ROOT",
    "apply_deterministic_algorithms",
    "artifact_record",
    "collect_run_metadata",
    "deterministic_algorithms_context",
    "environment_info",
    "file_fingerprint",
    "finish_metadata",
    "git_revision",
    "save_run_manifest",
    "seed_worker",
    "self_test",
    "training_recipe_mismatches",
    "TRAINING_RECIPE_KEYS",
    "set_all_seeds",
]

if __name__ == "__main__":
    sys.exit(_main())
