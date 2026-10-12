"""
Long-context needle-in-a-haystack sweep across context lengths.

For each context length in --ctx-lens, we:
  - generate a haystack by repeating a fixed prompt,
  - insert the needle at a random position,
  - prompt the model to recite the secret,
  - check that *every* token of the secret is emitted, in order.

The scoring is the whole-secret match implemented by
`scripts.eval_downstream.needle_in_haystack`: the secret is a multi-token
string under any BPE (" 12345" is `" 123"` + `"45"` for GPT-2), so scoring
only the first token would accept a model that recites 3 of the 5 digits.

This is the bare-bones version that does not need external eval
harnesses. For a proper needle-in-a-haystack plot use the
`needleeval` package against the HF-exported checkpoint.

Usage:
    python scripts/run_needle.py --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
        --ctx-lens 512 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.eval_downstream import load_ckpt, needle_in_haystack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ctx-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--secret", default="The magic number is 98765.")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_ckpt(args.ckpt, device=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    results = []
    for L in args.ctx_lens:
        if L > model.cfg.max_seq_len:
            print(f"[skip] L={L} > max_seq_len={model.cfg.max_seq_len}, "
                  "RoPE scaling needed - skipping for now.")
            continue
        acc = needle_in_haystack(
            model, tok, needle=args.secret, context_len=L,
            n_trials=args.trials, device=device, seed=args.seed,
        )
        print(f"[needle] L={L:5d} acc={acc:.3f}")
        results.append({"ctx_len": L, "acc": acc})

    out = args.output or (Path(args.ckpt).parent / "needle_results.json")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"[done] saved to {out}")


if __name__ == "__main__":
    main()
