from __future__ import annotations
"""Fail-fast overfit sanity check: can the model memorize one small batch?

Run this before any full-scale training. It freezes one small batch (a few
sequences of `--seq-len` tokens), turns regularization off (dropout stays 0,
weight decay is 0), and trains on exactly that batch. If the plumbing is
correct the training loss must fall toward 0 and next-token accuracy must
approach 1 (for the deterministic `copy` task); a loss that stalls near its
initial value, a missing/NaN gradient or a mismatch between the logits loss
path and the fused/no-logits loss path means a real bug (gradients not
backpropagated, wrong loss, broken labels) - do not launch the full run.

This check is necessary but not sufficient: passing it does not prove that
data augmentation, feature alignment or label mapping are correct on the
real corpus, so pair it with the pipeline audit (`scripts/audit_data.py`)
and the intermediate-statistics hooks before scaling up.

Tasks
-----
* `copy`    - synthetic deterministic batch: target = (input + 1) mod V;
              must reach near-zero loss / ~100% accuracy on the frozen batch.
* `markov`  - the repo's synthetic Markov walk; a learnable signal with a
              non-trivial floor instead of the trivially memorizable copy map.
* `dataset` - one real batch from the leakage-safe pipeline; natural text
              never reaches ~0 loss, so pass `--loss-threshold` /
              `--acc-threshold` explicitly and read them as progress bounds.

Models
------
`--model gpt` (default) exercises the dense GPT; `--model moe` exercises the
MoE whole-model path (router, top-k dispatch and the auxiliary load-balancing
loss) used by `scripts/train_moe_tiny.py`. `--repro-check` additionally
retrains the same configuration a second time from the same seed and requires
a bit-identical curve, so seed stability itself is checked, not assumed.

Metrics
-------
Accuracy alone can be passed by a strawman predictor, so every run also
reports (and the dataset task gates on) the trivial baselines it must beat:

* `majority_class_accuracy` - always predicting the most frequent target;
  the check `accuracy_beats_majority_baseline` requires a margin over it, so
  an imbalanced batch cannot be "solved" by collapsing to the common class;
* `uniform_random_accuracy` = 1 / vocab_size;
* `copy_input_accuracy` - the "always predict the current token" rule. On the
  markov walk this rule is exactly at the next-token accuracy ceiling
  (target = current + jump, jump uniform in {0,1,2}), which is why the markov
  gate is carried by the loss, not by accuracy;
* `macro_f1`, `weighted_f1`, `micro_f1` - all three averaging modes are named
  explicitly because macro and micro can disagree under class imbalance
  (`micro_f1 == accuracy` for a single-label argmax prediction);
* `topk_accuracy` (k=3 by default) - a ceiling-free signal for tasks whose
  top-1 accuracy is capped by label ambiguity;
* one-vs-rest `macro_average_precision` / `weighted_average_precision`
  (PR-AUC) and `macro_roc_auc` / `weighted_roc_auc` over the target classes
  present in the batch - the ranking metrics that expose a collapse accuracy
  hides. The two averaging modes are named for the same reason as the F1
  ones, and when `prefer_pr_auc` is set (>= 1000 negatives per positive on
  some class) PR-AUC is the one to read: ROC-AUC's false-positive rate has a
  huge negative denominator and therefore stays optimistic.
"""
import argparse
import math
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from adasquash import GPT, GPTConfig, NORM_REGISTRY, markov_batch
from adasquash.metrics import binary_auc_report, f1_from_counts
from adasquash.repro import collect_run_metadata, finish_metadata, save_run_manifest, set_all_seeds

DEFAULT_THRESHOLDS = {
    "copy": {"loss": 0.05, "acc": 0.99},
    "markov": {"loss": 1.25, "acc": 0.30},
    "dataset": {"loss": None, "acc": None},
}


def run_preflight(norms, seed: int, output_dir, model: str = "gpt", n_experts: int = 8,
                  repro_check: bool = False, arch_args: list | None = None) -> None:
    """Run this script as a subprocess; raise CalledProcessError when it fails.

    Training entrypoints use this behind a `--sanity-check` flag so a broken
    gradient/loss path aborts the run before full-scale training starts.
    The default synthetic `copy` task is used, so no dataset is required.
    `model='moe'` exercises the MoE whole-model path (router + aux loss)
    instead of the dense GPT. `arch_args` forwards the caller's architecture
    flags (depth/width/heads/context/batch), so the preflight exercises the
    actual configuration instead of the tiny default model.
    """
    import subprocess

    if isinstance(norms, str):
        norms = [norms]
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--norms", *norms, "--seed", str(seed),
           "--model", model, "--n-experts", str(int(n_experts)),
           "--output", str(Path(output_dir) / "sanity_overfit.json")]
    if arch_args:
        cmd += [str(a) for a in arch_args]
    if repro_check:
        cmd.append("--repro-check")
    print(f"[sanity] pre-flight overfit check: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def build_model(args, cfg: GPTConfig):
    """Construct the model under test: the dense GPT or the tiny MoE GPT."""
    if args.model == "moe":
        from scripts.train_moe_tiny import MoEGPT

        return MoEGPT(cfg, n_experts=args.n_experts)
    # Callers reset with the run seed right after building; skip the
    # constructor's init pass so the parameters are initialized exactly once.
    return GPT(cfg, init=False)


def make_batch(task: str, args, cfg: GPTConfig, device: str):
    if task == "dataset":
        from scripts.train_gpt2 import open_data_split

        data_args = argparse.Namespace(
            val_frac=0.005, test_frac=0.005, num_documents=args.num_documents,
            dedup=True, min_doc_chars=0, max_doc_chars=0, exclude_hash_file=None,
            exclude_test_hash_file=None,
            train_shuffle_buffer=0, seed=args.seed,
            dataset=args.dataset, dataset_config=args.dataset_config,
            tokenizer=args.tokenizer, seq_len=args.seq_len,
            batch_size=args.batch_size,
            dataset_revision=args.dataset_revision,
            tokenizer_revision=args.tokenizer_revision,
        )
        loader = open_data_split("train", data_args)
        x, y = next(iter(loader))
        x = x.to(device)
        y = y.to(device)
        if int(x.max()) >= cfg.vocab_size:
            raise SystemExit(
                f"dataset batch contains token id {int(x.max())} >= vocab_size="
                f"{cfg.vocab_size}; raise --vocab-size (GPT-2 needs 50304)"
            )
        return x, y
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    if task == "markov":
        return markov_batch(generator, cfg.ignore_index, args.batch_size, args.seq_len, device)
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), generator=generator, device=device)
    y = torch.full_like(x, cfg.ignore_index)
    y[:, :-1] = (x[:, :-1] + 1) % cfg.vocab_size
    return x, y


@torch.no_grad()
def next_token_metrics(model, x, y, topk: int = 3, ctx=None) -> dict:
    """Classification metrics for next-token prediction, with their baselines.

    All quantities are computed on the unmasked targets only. `micro_f1` is
    accuracy for a single-label argmax prediction; `macro_f1` averages over
    classes present (see `adasquash.metrics.macro_weighted_f1`). The trivial
    baselines are reported so a pass cannot hide behind class imbalance or a
    degenerate label mapping. `ctx` is the run's autocast context (or None),
    so the evaluation is scored under the same numeric format as training.
    """
    with (ctx if ctx is not None else nullcontext()):
        logits, _ = model(x, return_logits=True)
    mask = y.ne(model.cfg.ignore_index)
    vocab = int(logits.size(-1))
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return {
            "accuracy": 0.0, "topk_accuracy": 0.0, "topk": int(topk),
            "macro_f1": 0.0, "weighted_f1": 0.0, "micro_f1": 0.0,
            "n_classes_present": 0, "n_valid_tokens": 0,
            "majority_class_accuracy": 0.0,
            "uniform_random_accuracy": 1.0 / vocab,
            "copy_input_accuracy": None,
            "macro_average_precision": None, "weighted_average_precision": None,
            "macro_roc_auc": None, "weighted_roc_auc": None,
            "prefer_pr_auc": False, "auc_report": None,
        }
    gold = y[mask]
    pred = logits.argmax(-1)[mask]
    accuracy = float((pred == gold).float().mean().item())
    counts = torch.bincount(gold, minlength=vocab)
    majority = float(counts.max().item() / n_valid)
    # Count-based F1: the three confusion marginals come from device
    # bincounts, so the |valid| predictions/targets are never round-tripped
    # through Python lists (the metric is identical to `macro_weighted_f1`).
    f1 = f1_from_counts(
        counts.tolist(),
        torch.bincount(pred, minlength=vocab).tolist(),
        torch.bincount(gold[pred == gold], minlength=vocab).tolist(),
    )
    k = max(1, min(int(topk), vocab))
    top = logits.topk(k, dim=-1).indices[mask]
    topk_accuracy = float((top == gold.unsqueeze(-1)).any(dim=-1).float().mean().item())
    copy_input = None
    if tuple(x.shape) == tuple(y.shape):
        copy_input = float((x[mask] == gold).float().mean().item())
    # One-vs-rest ranking metrics over the classes that actually occur in this
    # batch (at most one column per valid token). The scores must be the
    # row-wise softmax, not raw logits: the softmax normalizes over the whole
    # vocabulary, so its per-row factor differs across rows and it does change
    # the cross-row ordering that the per-class ROC-AUC / PR-AUC compare. A
    # raw-logit column is not an equivalent re-basing.
    present = torch.unique(gold).tolist()
    probs = torch.softmax(logits[mask].float(), dim=-1)[:, present]
    auc = binary_auc_report(probs.tolist(), gold.tolist(), present)
    return {
        "accuracy": accuracy,
        "topk_accuracy": topk_accuracy,
        "topk": k,
        "macro_f1": f1["macro_f1"],
        "weighted_f1": f1["weighted_f1"],
        "micro_f1": f1["micro_f1"],
        "n_classes_present": f1["n_classes_present"],
        "n_valid_tokens": n_valid,
        "majority_class_accuracy": majority,
        "uniform_random_accuracy": 1.0 / vocab,
        "copy_input_accuracy": copy_input,
        "macro_average_precision": auc["macro_average_precision"],
        "weighted_average_precision": auc["weighted_average_precision"],
        "macro_roc_auc": auc["macro_roc_auc"],
        "weighted_roc_auc": auc["weighted_roc_auc"],
        "prefer_pr_auc": auc["prefer_pr_auc"],
        "auc_report": auc,
    }


@torch.no_grad()
def next_token_accuracy(model, x, y) -> float:
    return next_token_metrics(model, x, y)["accuracy"]


def _structurally_zero_grad_names(model) -> set:
    """Parameter names whose gradient is exactly zero *by construction*.

    DyT-D / DyT-CD depth conditioning uses ``alpha_l = alpha_0 * exp(-k * l/L)``:
    at depth ratio 0 (the first layer) the k parameter does not enter the
    computation at all, so its gradient is exactly zero by design. Treating
    that as a wiring bug would reject a correct model.
    """
    out = set()
    for mod_name, mod in model.named_modules():
        depth_ratio = getattr(mod, "_depth_ratio", None)
        if depth_ratio is None or float(depth_ratio) != 0.0:
            continue
        if "k" in dict(mod.named_parameters(recurse=False)):
            out.add(f"{mod_name}.k" if mod_name else "k")
    return out


def gradient_report(model) -> dict:
    missing, nonfinite, zero = [], [], []
    names = []
    finite_t = []
    norm_t = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
            continue
        g = p.grad.detach()
        names.append(name)
        finite_t.append(torch.isfinite(g).all())
        norm_t.append(g.float().norm().to(torch.float64))
    finite = torch.stack(finite_t).tolist() if finite_t else []
    norms = torch.stack(norm_t).tolist() if norm_t else []
    sq = 0.0
    n_used = 0
    for name, is_finite, norm in zip(names, finite, norms):
        if not is_finite:
            nonfinite.append(name)
            continue
        n_used += 1
        sq += norm * norm
        if norm == 0.0:
            zero.append(name)
    structural = _structurally_zero_grad_names(model)
    return {
        "total_norm": math.sqrt(sq),
        "params_with_grad": n_used,
        "missing_grad": missing[:20],
        "nonfinite_grad": nonfinite[:20],
        "zero_grad": zero[:20],
        "zero_grad_unaccounted": [n for n in zero if n not in structural][:20],
        "structurally_zero_grad": sorted(n for n in zero if n in structural)[:20],
    }


def run_one(norm: str, args, device: str) -> dict:
    vocab = args.vocab_size
    if vocab is None:
        vocab = 50304 if args.task == "dataset" else 64
    if args.task == "markov" and vocab < 64:
        raise SystemExit("the markov task draws 64 states; pass --vocab-size >= 64")
    thresholds = dict(DEFAULT_THRESHOLDS[args.task])
    if args.loss_threshold is not None:
        thresholds["loss"] = args.loss_threshold
    if args.acc_threshold is not None:
        thresholds["acc"] = args.acc_threshold
    if args.task == "dataset" and (thresholds["loss"] is None or thresholds["acc"] is None):
        raise SystemExit(
            "the dataset task cannot use the synthetic pass thresholds: pass "
            "--loss-threshold and --acc-threshold for the real batch"
        )
    set_all_seeds(args.seed)
    # Mirror the training entrypoints' numeric format: TF32 matmuls are on for
    # every path that has them, and autocast runs whenever the run being
    # checked would run under bf16. A failure that only shows up under the
    # reduced-precision path must fail the preflight, not the full run.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = (not args.no_bf16) and torch.device(device).type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.amp.autocast(device_type="cuda", dtype=dtype)
           if use_bf16 else nullcontext())
    cfg = GPTConfig(
        vocab_size=vocab, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=args.seq_len, mlp_ratio=4.0, dropout=0.0, norm_name=norm,
        qk_norm=args.qk_norm, qk_alpha_init=args.qk_alpha_init,
        use_layerscale=args.use_layerscale, layerscale_init=args.layerscale_init,
        alpha_0=args.alpha_0, k_init=args.k_init, alpha_init=args.alpha_init,
        p_init=args.p_init, learnable_p=not args.no_learnable_p,
        learnable_k=not args.no_learnable_k,
        learnable_delta=not args.no_learnable_delta, beta=args.beta,
        alpha_jitter=args.alpha_jitter,
        gradient_checkpointing=args.gradient_checkpointing, rotary=args.rotary,
    )
    if args.tie_embeddings is not None:
        cfg.tie_word_embeddings = args.tie_embeddings
    model = build_model(args, cfg)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters(args.seed)
    model.to(device)
    x, y = make_batch(args.task, args, cfg, device)
    from scripts.train_gpt2 import build_optimizer
    opt = build_optimizer(model, args.lr, 0.0, 0.9, 0.95)

    # The logits path computes full logits, so the equality check between the
    # two loss paths runs on a bounded prefix slice: causal attention makes
    # every kept position's logits independent of the dropped suffix, so the
    # comparison is exact.
    b_ref = min(int(x.shape[0]), 2)
    t_ref = min(int(x.shape[1]), 64)
    x_ref, y_ref = x[:b_ref, :t_ref], y[:b_ref, :t_ref]
    with torch.no_grad(), ctx:
        _, loss_ref = model(x_ref, y_ref, return_logits=True)
        loss_fused = model(x_ref, y_ref, return_logits=False)[1]
    loss_path_delta = abs(float(loss_fused.item()) - float(loss_ref.item()))
    loss_path_tol = max(1e-5, 1e-4 * abs(float(loss_ref.item())))

    history = []
    initial_loss = None
    final_metrics = next_token_metrics(model, x, y, ctx=ctx)
    grad_info = {}
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        with ctx:
            loss = model(x, y, return_logits=False)[1]
        loss_val = float(loss.detach().item())
        if initial_loss is None:
            initial_loss = loss_val
        loss.backward()
        if step == 1:
            grad_info = gradient_report(model)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            final_metrics = next_token_metrics(model, x, y, ctx=ctx)
            history.append({"step": step, "loss": loss_val,
                            "accuracy": final_metrics["accuracy"],
                            "macro_f1": final_metrics["macro_f1"]})
    # Final loss and metrics are measured on the final parameters.
    with torch.no_grad(), ctx:
        final_loss = float(model(x, y, return_logits=False)[1].detach().item())
    final_metrics = next_token_metrics(model, x, y, ctx=ctx)
    elapsed = time.time() - t0
    final_acc = final_metrics["accuracy"]

    checks = OrderedDict()
    checks["finite_loss"] = bool(math.isfinite(final_loss))
    checks["gradients_present_and_finite"] = (
        not grad_info.get("missing_grad") and not grad_info.get("nonfinite_grad")
        and grad_info.get("total_norm", 0.0) > 0.0
    )
    if args.model == "gpt":
        # Every dense-model parameter must participate in the loss: a trainable
        # tensor whose gradient is exactly zero after the first backward is
        # silently untrained (an unwired norm parameter, a dead branch), and the
        # overfit check alone would not notice. Layer-0 k of the depth-conditioned
        # norms is exempt ("structurally zero": depth ratio 0, so k never enters).
        # MoE is not gated here at all: an expert that receives no routed tokens
        # in this batch legitimately has zero gradient.
        checks["no_zero_grad_parameters"] = not grad_info.get("zero_grad_unaccounted")
    checks["loss_paths_agree"] = bool(loss_path_delta <= loss_path_tol)
    checks["initial_loss_as_expected"] = bool(
        initial_loss is not None and initial_loss >= math.log(vocab) * 0.5
    )
    checks["loss_decreased"] = bool(final_loss < initial_loss - 1e-4)
    checks["loss_below_threshold"] = bool(final_loss <= thresholds["loss"])
    # Strawman guards: an accuracy the majority-class predictor already reaches
    # (imbalance) or a loss no better than guessing uniformly over the vocab
    # means the run learned nothing, whatever the threshold says.
    checks["loss_below_uniform_baseline"] = bool(final_loss < math.log(vocab))
    checks["accuracy_above_threshold"] = bool(final_acc >= thresholds["acc"])
    checks["accuracy_beats_majority_baseline"] = bool(
        final_acc > final_metrics["majority_class_accuracy"] + args.baseline_margin
    )
    passed = all(checks.values())
    return {
        "norm": norm,
        "passed": passed,
        "checks": checks,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "final_macro_f1": final_metrics["macro_f1"],
        "final_metrics": final_metrics,
        "loss_path_delta": loss_path_delta,
        "loss_path_tol": loss_path_tol,
        "gradients": grad_info,
        "history": history,
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--norms", nargs="+", default=["dytcd"], choices=sorted(NORM_REGISTRY))
    p.add_argument("--model", choices=["gpt", "moe"], default="gpt",
                   help="which whole-model path to exercise: the dense GPT or the tiny MoE GPT (router + aux loss)")
    p.add_argument("--n-experts", type=int, default=8, help="experts per MoE layer (only for --model moe)")
    p.add_argument("--task", choices=["copy", "markov", "dataset"], default="copy")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=None, help="defaults to 64 for copy/markov, 50304 for dataset")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-layerscale", action="store_true")
    p.add_argument("--layerscale-init", type=float, default=1e-5)
    p.add_argument("--qk-norm", default=None, choices=[None, "qknorm", "dytqk"],
                   help="attention Q/K treatment of the run being checked (QKSquash etc.)")
    p.add_argument("--qk-alpha-init", type=float, default=1.0, help="initial alpha_q / alpha_k for --qk-norm dytqk")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="enable gradient checkpointing in the checked model")
    p.add_argument("--no-learnable-p", action="store_true")
    p.add_argument("--no-learnable-k", action="store_true")
    p.add_argument("--no-learnable-delta", action="store_true",
                   help="freeze the DyT-CD per-channel deviation at zero (controlled per-channel off)")
    p.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true", default=None,
                   help="tie input/output embeddings (default: the config's own default)")
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false")
    p.add_argument("--no-bf16", action="store_true",
                   help="disable bf16 autocast; by default the check mirrors the training runs and uses bf16 on CUDA")
    p.add_argument("--rotary", action="store_true",
                   help="RoPE positions (llama-style) instead of learned positional embeddings")
    p.add_argument("--no-beta", dest="beta", action="store_false")
    p.add_argument("--alpha-0", type=float, default=0.9)
    p.add_argument("--alpha-init", type=float, default=0.5)
    p.add_argument("--k-init", type=float, default=1.0)
    p.add_argument("--p-init", type=float, default=2.0)
    p.add_argument("--alpha-jitter", type=float, default=0.0)
    p.add_argument("--loss-threshold", type=float, default=None)
    p.add_argument("--acc-threshold", type=float, default=None)
    p.add_argument("--baseline-margin", type=float, default=0.05, help="required accuracy margin over the majority-class predictor (guards the metric, not the model)")
    p.add_argument("--repro-check", action="store_true",
                   help="retrain the same configuration a second time in-process and require a bit-identical loss curve / accuracy: a direct same-machine seed-stability check (fails if the RNGs, thread counts or kernels introduce run-to-run variance)")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default=None, help="dataset commit/revision to pin (recorded in the report; None = upstream default branch)")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--tokenizer-revision", default=None, help="tokenizer commit/revision to pin (recorded in the report; None = upstream default branch)")
    p.add_argument("--num-documents", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="runs/sanity_overfit.json")
    args = p.parse_args()
    if args.dim % args.n_heads:
        p.error(f"--dim {args.dim} is not divisible by --n-heads {args.n_heads}")
    # Step 1 is the initial-loss reference of the whole check, so a run
    # without at least one step has nothing to compare against (`checks
    # ["loss_decreased"]` would compare against None).
    if args.max_steps < 1:
        p.error(f"--max-steps {args.max_steps} must be >= 1: the sanity check "
                "needs at least one optimizer step")
    if args.log_every < 1:
        p.error(f"--log-every {args.log_every} must be >= 1")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for norm in args.norms:
        print(f"[sanity] task={args.task} model={args.model} norm={norm} device={device}")
        result = run_one(norm, args, device)
        if args.repro_check:
            # Independent rerun from the same seed: the second `run_one`
            # re-seeds every RNG and rebuilds model + batch from scratch, so
            # an identical curve is direct evidence that same-machine
            # reproduction holds for this configuration.
            rerun = run_one(norm, args, device)
            history_matched = rerun["history"] == result["history"]
            loss_matched = rerun["final_loss"] == result["final_loss"]
            acc_matched = rerun["final_accuracy"] == result["final_accuracy"]
            result["repro_check"] = {
                "history_matched": bool(history_matched),
                "final_loss_matched": bool(loss_matched),
                "final_accuracy_matched": bool(acc_matched),
                "rerun_final_loss": rerun["final_loss"],
                "rerun_final_accuracy": rerun["final_accuracy"],
            }
            result["checks"]["deterministic_rerun"] = bool(
                history_matched and loss_matched and acc_matched)
            result["passed"] = all(result["checks"].values())
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        m = result["final_metrics"]
        ap = m["macro_average_precision"]
        roc = m["macro_roc_auc"]
        ap_str = "n/a" if ap is None else f"{ap:.4f}"
        roc_str = "n/a" if roc is None else f"{roc:.4f}"
        print(
            f"[sanity] {status} {norm}: initial_loss={result['initial_loss']:.4f} "
            f"final_loss={result['final_loss']:.4f} accuracy={result['final_accuracy']:.4f} "
            f"macro_f1={result['final_macro_f1']:.4f} "
            f"macro_pr_auc={ap_str} macro_roc_auc={roc_str} "
            f"(majority_baseline={m['majority_class_accuracy']:.4f}, "
            f"copy_input_baseline={m['copy_input_accuracy']}, "
            f"top{m['topk']}_accuracy={m['topk_accuracy']:.4f}) "
            f"grad_norm={result['gradients'].get('total_norm', 0.0):.3e}"
        )
        if m["prefer_pr_auc"]:
            print(
                "[sanity]   note: >=1000 negatives per positive on some class: read "
                "macro_pr_auc (PR-AUC) over ROC-AUC, whose FPR denominator is huge"
            )
        for name, ok in result["checks"].items():
            print(f"[sanity]   {'ok' if ok else 'FAIL'}: {name}")
        if not result["passed"]:
            if result["gradients"].get("missing_grad"):
                print(f"[sanity]   missing grads: {result['gradients']['missing_grad']}")
            if result["gradients"].get("nonfinite_grad"):
                print(f"[sanity]   non-finite grads: {result['gradients']['nonfinite_grad']}")
            if result["gradients"].get("zero_grad_unaccounted"):
                print(f"[sanity]   exactly-zero grads (silently untrained parameters): "
                      f"{result['gradients']['zero_grad_unaccounted']}")
    report = {
        "passed": all(r["passed"] for r in results),
        "run_metadata": collect_run_metadata(seed=args.seed),
        "config": vars(args),
        "results": results,
    }
    report = finish_metadata(report)
    out_path = save_run_manifest(args.output, report)
    print(f"[sanity] report -> {out_path}")
    if not report["passed"]:
        print(
            "[sanity] REJECTED: the model cannot overfit one small batch; fix the "
            "gradient/loss plumbing before launching the full run.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[sanity] PASS")


if __name__ == "__main__":
    main()
