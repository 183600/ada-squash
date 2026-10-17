from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash.repro import collect_run_metadata, save_run_manifest, set_all_seeds
from scripts.eval_downstream import load_ckpt, needle_decode_overhead, needle_in_haystack

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ctx-lens', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--secret', default='The magic number is 98765.')
    p.add_argument('--output', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this evaluation tests (e.g. "DyT-CD retains needle accuracy at 4k tokens"); recorded in the manifest so a pass/fail is auditable as a test rather than a fishing trip')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(args.seed)
    model, _ = load_ckpt(args.ckpt, device=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    overhead = needle_decode_overhead(tok, args.secret)
    results = []
    for L in args.ctx_lens:
        if L + overhead > model.cfg.max_seq_len:
            print(f'[skip] L={L} (+{overhead} scoring positions for the secret) > max_seq_len={model.cfg.max_seq_len}, RoPE scaling needed - skipping for now.')
            continue
        res = needle_in_haystack(model, tok, needle=args.secret, context_len=L, n_trials=args.trials, device=device, seed=args.seed, return_counts=True)
        print(f"[needle] L={L:5d} acc={res['accuracy']:.3f} "
              f"({res['n_correct']}/{res['n_trials']}, 95% CI [{res['ci95_low']:.3f}, {res['ci95_high']:.3f}])")
        results.append({'ctx_len': L, 'acc': res['accuracy'],
                        'n_correct': res['n_correct'], 'n_trials': res['n_trials'],
                        'ci95_low': res['ci95_low'], 'ci95_high': res['ci95_high']})
    out = args.output or Path(args.ckpt).parent / 'needle_results.json'
    Path(out).write_text(json.dumps(results, indent=2))
    manifest = collect_run_metadata(seed=args.seed)
    manifest['hyperparameters'] = vars(args)
    if args.hypothesis:
        manifest['hypothesis'] = args.hypothesis
        print(f'[hypothesis] {args.hypothesis}')
    else:
        print('[hypothesis] none stated: pass --hypothesis "..." so this evaluation is auditable as a test of a claim, not a sweep')
    manifest['results_file'] = str(out)
    save_run_manifest(Path(out).with_name('needle_manifest.json'), manifest)
    print(f'[done] saved to {out}')
if __name__ == '__main__':
    main()
