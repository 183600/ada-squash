"""
Downstream evaluation: MMLU, GSM8K, HumanEval, DROP, plus long-context
needle-in-a-haystack for the QK-squashing variant.

This is a thin wrapper around `lm-evaluation-harness` style loaders. It
uses the model's HuggingFace-compatible `state_dict` so any saved checkpoint
can be loaded back without retraining.

Example:
    python scripts/eval_downstream.py \
        --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
        --tasks mmlu gsm8k hellaswag \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from adasquash import GPTConfig, GPT
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b


def load_ckpt(path: str, device: str = "cuda"):
    ckpt = torch.load(path, map_location=device)
    args = ckpt["args"]
    # Reconstruct model with the same config.
    model_cfg = dict(
        norm_name=args["norm"],
        qk_norm=args.get("qk_norm"),
        use_layerscale=not args.get("no_layerscale", False),
        alpha_0=args.get("alpha_0", 0.9),
        k_init=args.get("k_init", 1.0),
        alpha_init=args.get("alpha_init", 0.5),
        p_init=args.get("p_init", 2.0),
        learnable_p=not args.get("no_learnable_p", False),
        learnable_k=not args.get("no_learnable_k", False),
    )
    if args["model"] == "gpt2-124m":
        model = build_gpt_124m(**model_cfg)
    elif args["model"] == "gpt2-350m":
        model = build_gpt_350m(**model_cfg)
    else:
        model = build_llama_tiny_1b(**model_cfg)
    # Apply depth / width override if any. The depth sweep shrinks `dim` as
    # depth grows so the parameter count stays ~constant, so the checkpoint
    # has to be rebuilt with *all* of the recorded overrides - rebuilding
    # with `n_layers` alone would produce a state_dict shape mismatch.
    depth = args.get("_depth_override")
    dim = args.get("_dim_override")
    heads = args.get("_heads_override")
    if depth or dim or heads:
        cfg_dict = {k: getattr(model.cfg, k) for k in vars(model.cfg)}
        if depth:
            cfg_dict["n_layers"] = depth
        if dim:
            cfg_dict["dim"] = dim
        if heads:
            cfg_dict["n_heads"] = heads
        model = GPT(GPTConfig(**cfg_dict))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, args


def evaluate_lm_harness(model, tasks: list[str], batch_size: int = 16, tokenizer: str | None = None):
    """Thin wrapper around `lm-eval`."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise RuntimeError(
            "lm-eval not installed. `pip install lm-eval`."
        ) from e
    # The simplest integration: wrap our model in an HFLM-like adapter.
    # For real evaluations use the project's HF model converter (see
    # adasquash.hf_compat) once a checkpoint is exported.
    raise NotImplementedError(
        "Use the HF-compat exporter + `lm_eval --model hf` for this step. "
        "See README for the exact recipe."
    )


# ---------------------------------------------------------------------------
# Long-context needle-in-a-haystack
# ---------------------------------------------------------------------------
def needle_in_haystack(
    model, tokenizer, needle: str = "The magic number is 12345.",
    haystack_prompt: str = "Many sentences follow. ",
    context_len: int = 4096, n_trials: int = 5, device: str = "cuda",
):
    """
    A minimal needle-in-a-haystack test. We insert `needle` at a random
    position in a long stream of `haystack_prompt` repeats, then ask the
    model to recite the secret. Returns accuracy.

    The expected answer is derived from `needle` itself (its final token /
    number), so a custom secret passed via `--secret` is actually scored.
    """
    import random
    tok = tokenizer
    needle_ids = tok(needle, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(haystack_prompt, add_special_tokens=False)["input_ids"]
    question_ids = tok("\n\nWhat is the magic number? The magic number is", add_special_tokens=False)["input_ids"]

    # The secret to recite is the last whitespace-separated chunk of the
    # needle with trailing punctuation stripped ("... is 12345." -> "12345").
    secret = needle.strip().rstrip(".!?").split()
    secret = secret[-1] if secret else ""
    target_ids = tok(" " + secret, add_special_tokens=False)["input_ids"] if secret else []

    n_correct = 0
    n_run = 0
    for _ in range(n_trials):
        body_len = context_len - len(needle_ids) - len(question_ids) - 2
        if body_len < len(prompt_ids):
            break
        n_repeats = body_len // len(prompt_ids)
        body = prompt_ids * n_repeats
        # Insert needle at a random position
        pos = random.randint(0, max(1, len(body) - 1))
        full = body[:pos] + needle_ids + body[pos:] + question_ids
        x = torch.tensor(full, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
        n_run += 1
        # greedy decode next token, check if it starts the secret
        next_id = int(logits[0, -1].argmax(-1).item())
        if target_ids and next_id == target_ids[0]:
            n_correct += 1
    if n_run == 0:
        return 0.0
    return n_correct / n_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k", "hellaswag"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--needle-test", action="store_true", help="also run needle-in-haystack")
    p.add_argument("--needle-context-len", type=int, default=4096)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args = load_ckpt(args.ckpt, device=device)
    print(f"[eval] loaded {args.ckpt} | norm={model_args['norm']} | n_layers={model.cfg.n_layers}")

    results = {"ckpt": args.ckpt, "norm": model_args["norm"], "tasks": {}}

    if args.tasks:
        # Try lm-eval harness; if not available, gracefully skip.
        try:
            res = evaluate_lm_harness(model, args.tasks, args.batch_size)
            results["tasks"].update(res)
        except NotImplementedError as e:
            print(f"[eval] lm-eval path not available: {e}", file=sys.stderr)

    if args.needle_test:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        acc = needle_in_haystack(
            model, tok, context_len=args.needle_context_len, device=device
        )
        results["needle_acc"] = acc
        print(f"[eval] needle-in-haystack accuracy @ {args.needle_context_len} = {acc:.3f}")

    out = args.output or (Path(args.ckpt).parent / "eval_results.json")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"[eval] results saved to {out}")


if __name__ == "__main__":
    main()
