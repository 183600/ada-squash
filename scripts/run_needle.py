from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from scripts.eval_downstream import load_ckpt, needle_decode_overhead, needle_in_haystack

# Coverage beyond a single secret: different digits, different phrasings, so a
# result is not "can the model copy one memorized sentence".
DEFAULT_NEEDLES = ['The magic number is 98765.', 'The secret code is 4242.', 'The hidden word is pineapple.', 'The pass code is 77341.']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ctx-lens', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--needles', nargs='+', default=DEFAULT_NEEDLES, help='one or more needle sentences; results are reported per needle and aggregated')
    p.add_argument('--secret', default=None, help='(legacy single-needle flag; overrides --needles)')
    p.add_argument('--output', default=None)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    needles = [args.secret] if args.secret else list(args.needles)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_ckpt(args.ckpt, device=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    results = []
    for needle in needles:
        overhead = needle_decode_overhead(tok, needle)
        for L in args.ctx_lens:
            if L + overhead > model.cfg.max_seq_len:
                print(f'[skip] L={L} (+{overhead} scoring positions for the secret) > max_seq_len={model.cfg.max_seq_len}, RoPE scaling needed - skipping for now.')
                continue
            acc, details = needle_in_haystack(model, tok, needle=needle, context_len=L, n_trials=args.trials, device=device, seed=args.seed, return_details=True)
            by_bucket = {}
            for d in details:
                b = by_bucket.setdefault(d['bucket'], {'n': 0, 'correct': 0})
                b['n'] += 1
                b['correct'] += int(d['correct'])
            bucket_acc = {b: round(v['correct'] / v['n'], 4) for b, v in by_bucket.items() if v['n']}
            print(f'[needle] {needle!r} L={L:5d} acc={acc:.3f} ' + ' '.join(f'{b}={a:.2f}(n={by_bucket[b]["n"]})' for b, a in bucket_acc.items()))
            results.append({'needle': needle, 'ctx_len': L, 'acc': acc, 'acc_by_position': bucket_acc, 'trials': details})
    if not results:
        print('[done] no context length fit the model - nothing to save')
        return
    out = args.output or Path(args.ckpt).parent / 'needle_results.json'
    Path(out).write_text(json.dumps(results, indent=2))
    print(f'[done] saved to {out}')
if __name__ == '__main__':
    main()
