from __future__ import annotations
import argparse
import sys
import json
import random
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash import GPTConfig, GPT
from adasquash.metrics import wilson_interval
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b
from adasquash.repro import collect_run_metadata, save_run_manifest, set_all_seeds

def _strip_compile_prefix(state):
    prefix = '_orig_mod.'
    if not any(k.startswith(prefix) for k in state):
        return state
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}

def load_ckpt(path: str, device: str='cuda'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt['args']
    model_cfg = dict(norm_name=args['norm'], qk_norm=args.get('qk_norm'), qk_alpha_init=args.get('qk_alpha_init', 1.0), use_layerscale=not args.get('no_layerscale', False), alpha_0=args.get('alpha_0', 0.9), k_init=args.get('k_init', 1.0), alpha_init=args.get('alpha_init', 0.5), p_init=args.get('p_init', 2.0), learnable_p=not args.get('no_learnable_p', False), learnable_k=not args.get('no_learnable_k', False), learnable_delta=not args.get('no_learnable_delta', False), beta=args.get('beta', True))
    if args.get('layerscale_init') is not None:
        model_cfg['layerscale_init'] = args['layerscale_init']
    if args.get('tie_embeddings') is not None:
        model_cfg['tie_word_embeddings'] = args['tie_embeddings']
    elif 'no_tie_embeddings' in args:
        model_cfg['tie_word_embeddings'] = not args['no_tie_embeddings']
    if args['model'] == 'gpt2-124m':
        model = build_gpt_124m(**model_cfg)
    elif args['model'] == 'gpt2-350m':
        model = build_gpt_350m(**model_cfg)
    else:
        model = build_llama_tiny_1b(**model_cfg)
    depth = args.get('_depth_override')
    dim = args.get('_dim_override')
    heads = args.get('_heads_override')
    if depth or dim or heads:
        cfg_dict = {k: getattr(model.cfg, k) for k in vars(model.cfg)}
        if depth:
            cfg_dict['n_layers'] = depth
        if dim:
            cfg_dict['dim'] = dim
        if heads:
            cfg_dict['n_heads'] = heads
        model = GPT(GPTConfig(**cfg_dict))
    model.load_state_dict(_strip_compile_prefix(ckpt['model']))
    model.to(device)
    model.eval()
    return (model, args)

def evaluate_lm_harness(model, tasks: list[str], batch_size: int=16, tokenizer: str | None=None):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise RuntimeError('lm-eval not installed. `pip install lm-eval`.') from e
    raise NotImplementedError('Use the HF-compat exporter + `lm_eval --model hf` for this step. See README for the exact recipe.')
DEFAULT_NEEDLE = 'The magic number is 12345.'
DEFAULT_QUESTION = '\n\nWhat is the magic number? The magic number is'
DEFAULT_HAYSTACK = 'Many sentences follow. '

def needle_target_ids(tokenizer, needle: str=DEFAULT_NEEDLE) -> list:
    secret = needle.strip().rstrip('.!?').split()
    secret = secret[-1] if secret else ''
    if not secret:
        return []
    return tokenizer(' ' + secret, add_special_tokens=False)['input_ids']

def needle_decode_overhead(tokenizer, needle: str=DEFAULT_NEEDLE) -> int:
    return max(0, len(needle_target_ids(tokenizer, needle)) - 1)

def needle_in_haystack(model, tokenizer, needle: str=DEFAULT_NEEDLE, haystack_prompt: str=DEFAULT_HAYSTACK, context_len: int=4096, n_trials: int=5, device: str='cuda', seed: int=0, return_counts: bool=False):
    """Exact-match accuracy of the secret over `n_trials` independent placements.

    A trial is pass/fail, so the point estimate alone hides its resolution:
    with `return_counts=True` the result is
    `{'accuracy', 'n_correct', 'n_trials', 'ci95_low', 'ci95_high', 'failures',
    'failure_causes', 'failure_confidence'}` with a Wilson 95% interval over
    the trials. The default return (a float) is kept for callers that only
    need the accuracy.

    Every failed trial also yields a *bad-case record* (token ids and logit
    statistics only, no text: tokenizer mocks in tests have no decode): where
    the needle sat, at which decode step the model first diverged from the
    secret, what it predicted instead, the expected token's rank and
    probability, the logit margin, whether the wrong prediction merely copied
    a context token, and a cause/confidence label. `failure_causes` aggregates
    those labels as
    `retrieval_miss` (failed on the first secret token),
    `context_copy` (drifted, predicted a token present in the context) or
    `decode_drift` (drifted otherwise); `failure_confidence` separates
    `near_miss` (expected in top-5), `confident_wrong` (top-1 prob >= 0.5)
    from `diffuse`. A reported accuracy of 0.0 with `retrieval_miss` for every
    trial means something different from 0.0 with `decode_drift`: the first
    says the model never located the needle, the second says it found it and
    lost it while decoding.
    """
    rng = random.Random(seed)
    tok = tokenizer
    needle_ids = tok(needle, add_special_tokens=False)['input_ids']
    prompt_ids = tok(haystack_prompt, add_special_tokens=False)['input_ids']
    question_ids = tok(DEFAULT_QUESTION, add_special_tokens=False)['input_ids']
    target_ids = needle_target_ids(tok, needle)
    model_limit = getattr(getattr(model, 'cfg', None), 'max_seq_len', None)
    if model_limit is not None and context_len > model_limit:
        raise ValueError(f"context_len={context_len} exceeds the model's max_seq_len={model_limit}; raise max_seq_len (and retrain) or lower the context length - there is no valid result at this length")
    overhead = max(0, len(target_ids) - 1)
    if model_limit is not None and context_len + overhead > model_limit:
        raise ValueError(f"context_len={context_len} + {overhead} scoring positions for the {len(target_ids)}-token secret exceeds the model's max_seq_len={model_limit}; scoring the last token would index past the positional table - lower the context length by at least {context_len + overhead - model_limit}")
    if not target_ids:
        raise ValueError(f'could not derive a secret to score from needle={needle!r}')
    if len(prompt_ids) == 0:
        raise ValueError('haystack_prompt tokenizes to zero tokens')
    n_correct = 0
    n_run = 0
    failures: list = []
    last_logits = getattr(model, 'last_position_logits', None)
    for trial_idx in range(n_trials):
        body_len = context_len - len(needle_ids) - len(question_ids) - 2
        if body_len < len(prompt_ids):
            raise ValueError(f'context_len={context_len} is too short to hold the needle ({len(needle_ids)} tok) + question ({len(question_ids)} tok) + one haystack repeat ({len(prompt_ids)} tok)')
        n_repeats = body_len // len(prompt_ids)
        body = prompt_ids * n_repeats
        pos = rng.randint(0, max(1, len(body) - 1))
        full = body[:pos] + needle_ids + body[pos:] + question_ids
        n_run += 1
        generated: list[int] = []
        correct = True
        buf = torch.empty((1, len(full) + len(target_ids)), dtype=torch.long, device=device)
        buf[0, :len(full)] = torch.tensor(full, dtype=torch.long, device=device)
        n = len(full)
        with torch.no_grad():
            for want in target_ids:
                ctx = buf[:, :n]
                if last_logits is not None:
                    step_logits = last_logits(ctx)
                else:
                    step_logits, _ = model(ctx)
                    step_logits = step_logits[:, -1]
                nxt = int(step_logits[0].argmax(-1).item())
                generated.append(nxt)
                if nxt != want:
                    correct = False
                    # Bad-case record: enough statistics to say *how* the
                    # decode failed, without decoding any token id to text
                    # (tokenizer mocks in tests have no decode method).
                    probs = torch.softmax(step_logits[0].float(), dim=-1)
                    rank = int((torch.argsort(step_logits[0], descending=True) == want)
                               .nonzero(as_tuple=True)[0][0].item())
                    top_prob = float(probs[nxt].item())
                    in_context = bool((buf[0, :n] == nxt).any().item())
                    prefix_ok = len(generated) - 1
                    if prefix_ok == 0:
                        cause = 'retrieval_miss'
                    elif in_context:
                        cause = 'context_copy'
                    else:
                        cause = 'decode_drift'
                    if rank <= 5:
                        confidence = 'near_miss'
                    elif top_prob >= 0.5:
                        confidence = 'confident_wrong'
                    else:
                        confidence = 'diffuse'
                    failures.append({
                        'trial': trial_idx,
                        'needle_position': pos,
                        'context_len': len(full),
                        'failure_step': prefix_ok,
                        'n_target_tokens': len(target_ids),
                        'generated_ids': list(generated),
                        'expected_id': int(want),
                        'predicted_id': nxt,
                        'expected_rank': rank,
                        'expected_prob': float(probs[want].item()),
                        'predicted_prob': top_prob,
                        'logit_margin_over_expected': float(
                            (step_logits[0, nxt] - step_logits[0, want]).item()),
                        'predicted_in_context': in_context,
                        'cause': cause,
                        'confidence': confidence,
                    })
                    break
                buf[0, n] = nxt
                n += 1
        n_correct += int(correct)
    if n_run == 0:
        raise ValueError('needle test produced no trials')
    acc = n_correct / n_run
    if return_counts:
        low, high = wilson_interval(n_correct, n_run)
        return {'accuracy': acc, 'n_correct': n_correct, 'n_trials': n_run,
                'ci95_low': low, 'ci95_high': high,
                'failures': failures,
                'failure_causes': dict(Counter(f['cause'] for f in failures)),
                'failure_confidence': dict(Counter(f['confidence'] for f in failures))}
    return acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tasks', nargs='+', default=['mmlu', 'gsm8k', 'hellaswag'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--needle-test', action='store_true', help='also run needle-in-haystack')
    p.add_argument('--needle-context-len', type=int, default=None, help="needle context length (default: the checkpoint's max_seq_len minus the positions the secret's decode needs - the longest the model can actually be scored at; asking for more raises)")
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--seed', type=int, default=0, help='seed for the needle-position and generation RNGs')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(args.seed)
    model, model_args = load_ckpt(args.ckpt, device=device)
    print(f"[eval] loaded {args.ckpt} | norm={model_args['norm']} | n_layers={model.cfg.n_layers}")
    results = {'ckpt': args.ckpt, 'norm': model_args['norm'], 'tasks': {}}
    if args.tasks:
        try:
            res = evaluate_lm_harness(model, args.tasks, args.batch_size)
            results['tasks'].update(res)
        except (NotImplementedError, RuntimeError) as e:
            print(f'[eval] lm-eval path not available: {e}', file=sys.stderr)
    if args.needle_test:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        ctx_len = args.needle_context_len or max(1, model.cfg.max_seq_len - needle_decode_overhead(tok))
        needle = needle_in_haystack(model, tok, context_len=ctx_len, device=device, seed=args.seed, return_counts=True)
        results['needle_acc'] = needle['accuracy']
        results['needle'] = needle
        results['needle_ctx_len'] = ctx_len
        print(f"[eval] needle-in-haystack accuracy @ {ctx_len} = {needle['accuracy']:.3f} "
              f"({needle['n_correct']}/{needle['n_trials']}, 95% CI [{needle['ci95_low']:.3f}, {needle['ci95_high']:.3f}])")
        if needle['failure_causes']:
            print(f"[eval] needle failure causes: {needle['failure_causes']} "
                  f"confidence: {needle['failure_confidence']} "
                  f"(per-trial bad-case records in the results JSON)")
    results['seed'] = args.seed
    results['run_metadata'] = collect_run_metadata(seed=args.seed)
    out = args.output or Path(args.ckpt).parent / 'eval_results.json'
    Path(out).write_text(json.dumps(results, indent=2, default=str))
    save_run_manifest(Path(out).with_name('eval_manifest.json'), results['run_metadata'])
    print(f'[eval] results saved to {out}')
if __name__ == '__main__':
    main()
