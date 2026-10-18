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
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache
from adasquash.repro import collect_run_metadata, file_fingerprint, save_run_manifest, set_all_seeds

def _strip_compile_prefix(state):
    prefix = '_orig_mod.'
    if not any(k.startswith(prefix) for k in state):
        return state
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}

def _load_checkpoint_cpu(path: str) -> dict:
    """Load a checkpoint on CPU, memory-mapped when the file format allows it.

    Training checkpoints carry the optimizer state (AdamW keeps two fp32
    moments per parameter, ~2x the model bytes); `map_location='cuda'` would
    deserialize all of that straight into VRAM and keep it alive through the
    returned dict even though the evaluation paths only read weights. The
    file is mapped read-only and the optimizer/scheduler entries are dropped
    as soon as `load_state_dict` has copied what is needed.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        return torch.load(path, map_location='cpu', weights_only=False)

def load_ckpt(path: str, device: str='cuda'):
    ckpt = _load_checkpoint_cpu(path)
    args = ckpt['args']
    model_cfg = dict(norm_name=args['norm'], qk_norm=args.get('qk_norm'), qk_alpha_init=args.get('qk_alpha_init', 1.0), use_layerscale=not args.get('no_layerscale', False), alpha_0=args.get('alpha_0', 0.9), k_init=args.get('k_init', 1.0), alpha_init=args.get('alpha_init', 0.5), p_init=args.get('p_init', 2.0), learnable_p=not args.get('no_learnable_p', False), learnable_k=not args.get('no_learnable_k', False), learnable_delta=not args.get('no_learnable_delta', False), beta=args.get('beta', True))
    if args.get('layerscale_init') is not None:
        model_cfg['layerscale_init'] = args['layerscale_init']
    if args.get('tie_embeddings') is not None:
        model_cfg['tie_word_embeddings'] = args['tie_embeddings']
    elif 'no_tie_embeddings' in args:
        model_cfg['tie_word_embeddings'] = not args['no_tie_embeddings']
    # The checkpoint's weights overwrite every parameter, and the model is
    # built straight on the target device: the constructor's init pass and the
    # full-model H2D copy are both skipped.
    with torch.device(device):
        if args['model'] == 'gpt2-124m':
            model = build_gpt_124m(init=False, **model_cfg)
        elif args['model'] == 'gpt2-350m':
            model = build_gpt_350m(init=False, **model_cfg)
        else:
            model = build_llama_tiny_1b(init=False, **model_cfg)
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
        with torch.device(device):
            model = GPT(GPTConfig(**cfg_dict), init=False)
    model.load_state_dict(_strip_compile_prefix(ckpt['model']))
    # The backing file is released as soon as the heavyweight entries are
    # dropped; only `args` travels back to the caller.
    ckpt.pop('model', None)
    ckpt.pop('optimizer', None)
    ckpt.pop('scheduler', None)
    model.to(device)
    model.eval()
    # Evaluation switches between models in-process (per-task loops, sweeps),
    # so the shape-keyed CE scratch cache is dropped between models; nothing
    # downstream depends on cache residency.
    clear_ce_scratch_cache()
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

def needle_in_haystack(model, tokenizer, needle: str=DEFAULT_NEEDLE, haystack_prompt: str=DEFAULT_HAYSTACK, context_len: int=4096, n_trials: int=5, device: str='cuda', seed: int=0, return_counts: bool=False, decode_batch: int=8):
    """Exact-match accuracy of the secret over `n_trials` independent placements.

    Cost: every trial generates `len(target_ids)` steps and each step runs one
    full forward over the whole prefix, so the flops are
    ``O(n_trials * len(target_ids) * prompt_len)``. There is no KV cache (the
    SDPA baseline path is fixed, see docs/design_notes.md D1); `decode_batch`
    only batches independent trials together so the shared part of the prefix
    is forwarded in one pass instead of `n_trials` sequential passes.

    A trial is pass/fail, so the point estimate alone hides its resolution:
    with `return_counts=True` the result is
    `{'accuracy', 'n_correct', 'n_trials', 'trial_correct', 'ci95_low',
    'ci95_high', 'failures', 'failure_causes', 'failure_confidence'}` with a
    Wilson 95% interval over the trials. The Wilson interval describes the
    trial noise *within one placement draw*: it does not cover run-to-run or
    placement-to-placement variability, so callers comparing arms must recur
    over several `seed` values. `trial_correct` is the per-trial pass/fail
    vector in trial order; since the placement RNG is seeded with `seed`, two
    arms evaluated with the same seed see the *same* placements, so their
    `trial_correct` vectors can be paired (McNemar) - the same pairing the
    ablation runner does across seeds. The default return (a float) is kept
    for callers that only need the accuracy.

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
    # Every trial uses the same fix-up RNG order as the sequential version, so
    # batching (which only groups independent trials) changes no placement.
    contexts: list = []
    for _ in range(n_trials):
        body_len = context_len - len(needle_ids) - len(question_ids) - 2
        if body_len < len(prompt_ids):
            raise ValueError(f'context_len={context_len} is too short to hold the needle ({len(needle_ids)} tok) + question ({len(question_ids)} tok) + one haystack repeat ({len(prompt_ids)} tok)')
        n_repeats = body_len // len(prompt_ids)
        body = prompt_ids * n_repeats
        pos = rng.randint(0, max(1, len(body) - 1))
        full = body[:pos] + needle_ids + body[pos:] + question_ids
        contexts.append((pos, full))
    n_run = len(contexts)
    if n_run == 0:
        raise ValueError('needle test produced no trials')
    n_correct = 0
    trial_correct: list = []
    failures: list = []
    last_logits = getattr(model, 'last_position_logits', None)
    decode_batch = max(1, int(decode_batch))
    with torch.inference_mode():
        for b0 in range(0, n_run, decode_batch):
            group = contexts[b0:b0 + decode_batch]
            n_rows = len(group)
            seq_len = len(group[0][1])
            buf = torch.empty((n_rows, seq_len + len(target_ids)), dtype=torch.long, device=device)
            # One host construction + one H2D for the whole decode batch
            # instead of a per-row `torch.tensor(full, device=...)`.
            host = torch.tensor([full for _, full in group], dtype=torch.long)
            buf[:, :seq_len].copy_(host, non_blocking=True)
            generated: list[list] = [[] for _ in range(n_rows)]
            failed = [False] * n_rows
            n = seq_len
            for want in target_ids:
                ctx = buf[:, :n]
                if last_logits is not None:
                    step_logits = last_logits(ctx)
                else:
                    step_logits, _ = model(ctx)
                    step_logits = step_logits[:, -1]
                nxt = step_logits.argmax(-1)
                # Device-side comparison: the common all-correct step pays one
                # tiny scalar sync and no int list round-trip; only the rows
                # that actually diverged get materialized.
                neq = nxt != want
                if bool(neq.any().item()):
                    bad_rows = neq.nonzero(as_tuple=True)[0].tolist()
                    nxt_list = nxt.tolist()
                    for i in bad_rows:
                        if failed[i]:
                            continue
                        # Bad-case record: enough statistics to say *how* the
                        # decode failed, without decoding any token id to text
                        # (tokenizer mocks in tests have no decode method).
                        nxt_i = nxt_list[i]
                        generated[i].append(nxt_i)
                        row_logits = step_logits[i]
                        probs = torch.softmax(row_logits.float(), dim=-1)
                        # O(V) rank: how many logits beat the expected token
                        # (0-based); no full argsort of the 50k-way row.
                        rank = int((row_logits > row_logits[want]).sum().item())
                        top_prob = float(probs[nxt_i].item())
                        in_context = bool((buf[i, :n] == nxt_i).any().item())
                        prefix_ok = len(generated[i]) - 1
                        if prefix_ok == 0:
                            cause = 'retrieval_miss'
                        elif in_context:
                            cause = 'context_copy'
                        else:
                            cause = 'decode_drift'
                        if rank < 5:
                            confidence = 'near_miss'
                        elif top_prob >= 0.5:
                            confidence = 'confident_wrong'
                        else:
                            confidence = 'diffuse'
                        failures.append({
                            'trial': b0 + i,
                            'needle_position': group[i][0],
                            'context_len': seq_len,
                            'failure_step': prefix_ok,
                            'n_target_tokens': len(target_ids),
                            'generated_ids': [int(v) for v in generated[i]],
                            'expected_id': int(want),
                            'predicted_id': nxt_i,
                            'expected_rank': rank,
                            'expected_prob': float(probs[want].item()),
                            'predicted_prob': top_prob,
                            'logit_margin_over_expected': float(
                                (row_logits[nxt_i] - row_logits[want]).item()),
                            'predicted_in_context': in_context,
                            'cause': cause,
                            'confidence': confidence,
                        })
                        failed[i] = True
                    for i in range(n_rows):
                        if not failed[i]:
                            generated[i].append(nxt_list[i])
                else:
                    # All rows correct: keep the device scalars; they are only
                    # read back if a later step fails.
                    for i in range(n_rows):
                        generated[i].append(nxt[i])
                buf[:, n] = nxt
                n += 1
                if all(failed):
                    break
            n_correct += sum(not f for f in failed)
            trial_correct.extend(not f for f in failed)
    if n_run == 0:
        raise ValueError('needle test produced no trials')
    acc = n_correct / n_run
    if return_counts:
        low, high = wilson_interval(n_correct, n_run)
        return {'accuracy': acc, 'n_correct': n_correct, 'n_trials': n_run,
                'trial_correct': trial_correct,
                'ci95_low': low, 'ci95_high': high,
                'failures': failures,
                'failure_causes': dict(Counter(f['cause'] for f in failures)),
                'failure_confidence': dict(Counter(f['confidence'] for f in failures))}
    return acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tasks', nargs='+', default=[], help='lm-eval tasks to request (none by default: the in-script harness is a NotImplementedError placeholder, so requesting tasks without the documented HF-export + `lm_eval --model hf` recipe is a hard error, never an empty "successful" result)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--needle-test', action='store_true', help='also run needle-in-haystack')
    p.add_argument('--needle-context-len', type=int, default=None, help="needle context length (default: the checkpoint's trained seq_len, or its max_seq_len minus the positions the secret's decode needs when the checkpoint records no seq_len - so a model trained short is not silently evaluated at an extrapolated length; asking for more than max_seq_len raises)")
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--decode-batch', type=int, default=8, help='independent needle trials decoded per forward pass (bounds peak memory; each step still recomputes the whole prefix)')
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
            results['tasks_ran'] = True
        except (NotImplementedError, RuntimeError) as e:
            # A requested benchmark that did not run must not be written out as
            # an empty-but-successful result: that is how "evaluated on MMLU"
            # gets claimed with zero scores behind it.
            print(f'[eval] lm-eval path not available: {e}', file=sys.stderr)
            results['tasks_ran'] = False
            results['tasks_error'] = str(e)
    else:
        results['tasks_ran'] = None
    if args.needle_test:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        trained_seq_len = model_args.get('seq_len')
        trained_seq_len = int(trained_seq_len) if isinstance(trained_seq_len, int) else None
        max_scoreable = max(1, model.cfg.max_seq_len - needle_decode_overhead(tok))
        # The longest length the model can be *scored* at is not the same as
        # the length it was *trained* at: evaluating at max_seq_len after
        # training at a shorter context measures out-of-distribution positional
        # extrapolation, so the trained length is the honest default.
        if args.needle_context_len is not None:
            ctx_len = args.needle_context_len
            ctx_source = 'cli'
        elif trained_seq_len is not None:
            ctx_len = max(1, min(trained_seq_len, max_scoreable))
            ctx_source = 'trained_seq_len'
        else:
            ctx_len = max_scoreable
            ctx_source = 'max_seq_len'
        results['needle_trained_seq_len'] = trained_seq_len
        results['needle_ctx_source'] = ctx_source
        if trained_seq_len is not None and ctx_len > trained_seq_len:
            results['needle_ctx_exceeds_trained_seq_len'] = True
            print(f'[eval][warn] needle context length {ctx_len} exceeds the trained seq_len='
                  f'{trained_seq_len}: the result measures positional extrapolation, not '
                  f'in-distribution long-context accuracy', file=sys.stderr)
        needle = needle_in_haystack(model, tok, context_len=ctx_len, device=device, seed=args.seed, return_counts=True, decode_batch=args.decode_batch)
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
    results['hyperparameters'] = vars(args)
    results['run_metadata'] = collect_run_metadata(seed=args.seed)
    out = args.output or Path(args.ckpt).parent / 'eval_results.json'
    Path(out).write_text(json.dumps(results, indent=2, default=str))
    # The manifest mirrors what the run needs to be auditable: which checkpoint
    # (hashed), with which args, and what came out. Task scores live in the
    # results file, which is fingerprinted here rather than duplicated.
    manifest = dict(results['run_metadata'])
    manifest['hyperparameters'] = vars(args)
    manifest['checkpoint'] = file_fingerprint(args.ckpt)
    manifest['results_file'] = file_fingerprint(out)
    manifest['summary'] = {k: results[k] for k in
                           ('norm', 'tasks_ran', 'needle_acc', 'needle_ctx_len',
                            'needle_ctx_source', 'needle_trained_seq_len') if k in results}
    save_run_manifest(Path(out).with_name('eval_manifest.json'), manifest)
    print(f'[eval] results saved to {out}')
    if args.tasks and not results.get('tasks_ran', False):
        print(f'[eval] ERROR: tasks {args.tasks} were requested but lm-eval did not run them; '
              f'nothing about {args.tasks} was measured (see tasks_error in {out})', file=sys.stderr)
        raise SystemExit(2)
if __name__ == '__main__':
    main()
