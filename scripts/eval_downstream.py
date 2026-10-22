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
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache, clear_cast_weight_cache
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds

def _strip_compile_prefix(state):
    prefix = '_orig_mod.'
    if not any(k.startswith(prefix) for k in state):
        return state
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}

def _load_checkpoint_cpu(path: str) -> dict:
    """Load a checkpoint on CPU, memory-mapped when the file format allows it.

    Training checkpoints carry optimizer/scheduler entries that evaluation
    does not need; they are dropped as soon as `load_state_dict` has copied
    the weights, and only `args` travels back to the caller.
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
    # The checkpoint's weights overwrite every parameter, so the model is
    # built with `init=False` straight on the target device.
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
    # Keep the training step with the args: a checkpoint can be `ckpt_best.pt`
    # (selected at a different step per run) or a mid-run save, so two
    # checkpoints sharing `max_steps` are not necessarily the same point in
    # training. Comparison tools can then refuse to attribute a delta to the
    # norm/quantization when the compared weights were not taken at the same
    # step.
    if isinstance(args, dict):
        args = dict(args)
        step = ckpt.get('step')
        args['_checkpoint_step'] = int(step) if isinstance(step, int) and not isinstance(step, bool) else None
    # Drop the heavyweight entries; only `args` travels back to the caller.
    ckpt.pop('model', None)
    ckpt.pop('optimizer', None)
    ckpt.pop('scheduler', None)
    model.to(device)
    model.eval()
    # Evaluation switches between models in-process (per-task loops, sweeps);
    # drop the per-model scratch caches between models.
    clear_ce_scratch_cache()
    clear_cast_weight_cache()
    return (model, args)

class AdaSquashLM:
    """Minimal lm-eval wrapper around the raw GPT module.

    Implements the three primitives lm-eval-harness tasks need - token
    loglikelihood (multiple-choice / completion tasks), rolling loglikelihood
    (perplexity tasks) and greedy ``generate_until`` (generation tasks) - with
    no dependence on ``transformers`` model wrappers. Scoring is one sequence
    at a time (``batch_size`` is accepted for interface compatibility).
    """

    def __init__(self, model, tokenizer, batch_size: int=1, max_gen_toks: int=256):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.max_gen_toks = int(max_gen_toks)
        self.device = next(model.parameters()).device
        cfg = getattr(model, 'cfg', None)
        self.max_length = int(getattr(cfg, 'max_seq_len', 2048))
        self.apply_chat_template = False

    def tok_encode(self, string: str, add_special_tokens: bool=False,
                   left_truncate_len: int | None=None) -> list:
        ids = self.tokenizer(string, add_special_tokens=False)['input_ids']
        if left_truncate_len is not None and len(ids) > left_truncate_len:
            ids = ids[len(ids) - left_truncate_len:]
        return list(ids)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list, list]:
        """Whitespace-safe context/continuation split.

        Tokenize the concatenation, then keep whatever the tokenizer produced
        beyond the context prefix. Tokenizing the continuation on its own can
        turn a leading space into a different token.
        """
        whole = self.tok_encode(context + continuation)
        ctx = self.tok_encode(context)
        if not ctx:
            ctx = [self.tokenizer.eos_token_id]
        cont = whole[len(ctx):]
        return (ctx, cont)

    def _forward_logprobs(self, ids: list) -> torch.Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
        return torch.log_softmax(logits[0].float(), dim=-1)

    def loglikelihood(self, requests, **kwargs):
        out = []
        for req in requests:
            context, continuation = req.args[:2]
            ctx, cont = self._encode_pair(context, continuation)
            budget = self.max_length - len(cont)
            if budget < 1:
                ctx, cont = (ctx[:1], cont[:self.max_length - 1])
                budget = 1
            if len(ctx) > budget:
                ctx = ctx[len(ctx) - budget:]
            ids = ctx + cont
            logprobs = self._forward_logprobs(ids)
            start = len(ctx) - 1
            rows = logprobs[start:start + len(cont)]
            tgt = torch.tensor(cont, device=self.device)
            lp = rows.gather(-1, tgt.unsqueeze(-1)).sum().item()
            greedy = bool((rows.argmax(-1) == tgt).all().item())
            out.append((lp, greedy))
        return out

    def loglikelihood_rolling(self, requests, **kwargs):
        out = []
        for req in requests:
            (text,) = req.args[:1]
            ids = self.tok_encode(text)
            if not ids:
                out.append(0.0)
                continue
            ids = [self.tokenizer.eos_token_id] + ids
            total = 0.0
            for start in range(0, len(ids), self.max_length):
                window = ids[start:start + self.max_length]
                if len(window) < 2:
                    continue
                logprobs = self._forward_logprobs(window)
                tgt = torch.tensor(window[1:], device=self.device)
                total += logprobs[:-1].gather(-1, tgt.unsqueeze(-1)).sum().item()
            out.append(total)
        return out

    def generate_until(self, requests, stop=None):
        stops = list(stop) if stop else []
        out = []
        for req in requests:
            context, gen_kwargs = req.args[:2]
            if not isinstance(gen_kwargs, dict):
                gen_kwargs = {}
            until = list(gen_kwargs.get('until', [])) + [
                s for s in stops if s not in gen_kwargs.get('until', [])]
            max_toks = int(gen_kwargs.get('max_gen_toks', self.max_gen_toks))
            budget = max(1, self.max_length - max_toks)
            ids = self.tok_encode(context, left_truncate_len=budget)
            generated: list = []
            with torch.no_grad():
                for _ in range(max_toks):
                    x = torch.tensor([ids], dtype=torch.long, device=self.device)
                    logits = self.model(x)[0]
                    nxt = int(logits[0, -1].argmax(-1).item())
                    ids.append(nxt)
                    generated.append(nxt)
                    text = self.tokenizer.decode(ids[len(ids) - len(generated):])
                    if any((s in text) for s in until):
                        break
            text = self.tokenizer.decode(generated)
            for s in until:
                if s and s in text:
                    text = text.split(s)[0]
            out.append(text)
        return out

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

def _checkpoint_tokenizer(ckpt_args: dict, fallback_name: str):
    """The tokenizer identity the checkpoint was trained with, else the fallback.

    Prefer the concrete commit recorded at training time over the mutable
    branch/tag the user requested; otherwise a moved branch silently
    tokenizes the evaluation differently from training.
    """
    name = ckpt_args.get('tokenizer') or fallback_name
    revision = (ckpt_args.get('tokenizer_revision_resolved')
                or ckpt_args.get('tokenizer_revision'))
    return name, revision


def needle_target_ids(tokenizer, needle: str=DEFAULT_NEEDLE) -> list:
    secret = needle.strip().rstrip('.!?').split()
    secret = secret[-1] if secret else ''
    if not secret:
        return []
    return tokenizer(' ' + secret, add_special_tokens=False)['input_ids']

def needle_decode_overhead(tokenizer, needle: str=DEFAULT_NEEDLE) -> int:
    return max(0, len(needle_target_ids(tokenizer, needle)) - 1)

def needle_in_haystack(model, tokenizer, needle: str=DEFAULT_NEEDLE, haystack_prompt: str=DEFAULT_HAYSTACK, context_len: int=4096, n_trials: int=5, device: str='cuda', seed: int=0, return_counts: bool=False, decode_batch: int=8, return_details: bool=False):
    """Exact-match accuracy of the secret over `n_trials` independent placements.

    Cost: every trial generates `len(target_ids)` steps and each step runs one
    full forward over the whole prefix, so the flops are
    ``O(n_trials * len(target_ids) * prompt_len)``. There is no KV cache (the
    SDPA baseline path is fixed, see docs/design_notes.md D1); `decode_batch`
    only batches independent trials together.

    A trial is pass/fail, so the point estimate alone hides its resolution:
    with `return_counts=True` the result is
    `{'accuracy', 'n_correct', 'n_trials', 'trial_correct', 'ci95_low',
    'ci95_high', 'failures', 'failure_causes', 'failure_confidence'}` with a
    Wilson 95% interval over the trials. `return_details=True` instead returns
    `(accuracy, per_trial)` where each record carries the needle's relative
    position (`pos_frac`, `bucket` = early/middle/late) and pass/fail, so a
    scalar cannot hide failure that is concentrated at deep positions. The
    Wilson interval describes the
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
        body_len = context_len - len(needle_ids) - len(question_ids)
        if body_len < len(prompt_ids):
            raise ValueError(f'context_len={context_len} is too short to hold the needle ({len(needle_ids)} tok) + question ({len(question_ids)} tok) + one haystack repeat ({len(prompt_ids)} tok)')
        n_repeats, remainder = divmod(body_len, len(prompt_ids))
        # Fill the final partial repeat too: flooring alone left the context
        # up to one prompt shorter than the requested length, so trials were
        # silently scored at different (and smaller) than reported contexts.
        body = prompt_ids * n_repeats + prompt_ids[:remainder]
        pos = rng.randint(0, len(body))
        full = body[:pos] + needle_ids + body[pos:] + question_ids
        contexts.append((pos, full))
    n_run = len(contexts)
    if n_run == 0:
        raise ValueError('needle test produced no trials')
    n_correct = 0
    trial_correct: list = []
    failures: list = []
    details: list = []
    last_logits = getattr(model, 'last_position_logits', None)
    decode_batch = max(1, int(decode_batch))
    with torch.inference_mode():
        for b0 in range(0, n_run, decode_batch):
            group = contexts[b0:b0 + decode_batch]
            n_rows = len(group)
            seq_len = len(group[0][1])
            buf = torch.empty((n_rows, seq_len + len(target_ids)), dtype=torch.long, device=device)
            # Host construction for the whole decode batch.
            host = torch.tensor([full for _, full in group], dtype=torch.long)
            buf[:, :seq_len].copy_(host, non_blocking=True)
            generated: list[list] = [[] for _ in range(n_rows)]
            failed = [False] * n_rows
            # Only trials that have not failed need further decode steps.
            active = list(range(n_rows))
            n = seq_len
            for want in target_ids:
                if not active:
                    break
                ctx = buf[:, :n] if len(active) == n_rows else buf[active, :n]
                if last_logits is not None:
                    step_logits = last_logits(ctx)
                else:
                    step_logits, _ = model(ctx)
                    step_logits = step_logits[:, -1]
                nxt = step_logits.argmax(-1)
                # Only rows that diverged are examined individually.
                neq = nxt != want
                if bool(neq.any().item()):
                    bad_rows = neq.nonzero(as_tuple=True)[0].tolist()
                    nxt_list = nxt.tolist()
                    for pos in bad_rows:
                        i = active[pos]
                        # Bad-case record: enough statistics to say *how* the
                        # decode failed, without decoding any token id to text
                        # (tokenizer mocks in tests have no decode method).
                        nxt_i = nxt_list[pos]
                        generated[i].append(nxt_i)
                        row_logits = step_logits[pos]
                        probs = torch.softmax(row_logits.float(), dim=-1)
                        # Rank (0-based): how many logits beat the expected
                        # token.
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
                    for pos, i in enumerate(active):
                        if not failed[i]:
                            generated[i].append(nxt_list[pos])
                else:
                    # All rows correct: keep the device scalars; they are only
                    # read back if a later step fails.
                    for pos, i in enumerate(active):
                        generated[i].append(nxt[pos])
                if len(active) == n_rows:
                    buf[:, n] = nxt
                else:
                    for pos, i in enumerate(active):
                        if not failed[i]:
                            buf[i, n] = nxt[pos]
                active = [i for i in active if not failed[i]]
                n += 1
            n_correct += sum(not f for f in failed)
            trial_correct.extend(not f for f in failed)
            decode = getattr(tok, 'decode', None)
            for i, (pos, _) in enumerate(group):
                pos_frac = pos / max(1, body_len)
                generated_text = ''
                if generated[i] and callable(decode):
                    try:
                        generated_text = decode([int(v) for v in generated[i]])
                    except Exception:
                        generated_text = ''
                details.append({
                    'trial': b0 + i,
                    'pos_frac': pos_frac,
                    'bucket': ('early' if pos_frac < 1 / 3
                               else 'middle' if pos_frac < 2 / 3 else 'late'),
                    'correct': bool(not failed[i]),
                    'generated': generated_text,
                })
    if n_run == 0:
        raise ValueError('needle test produced no trials')
    acc = n_correct / n_run
    if return_details:
        # Position-resolved coverage: a scalar accuracy can hide a model that
        # only succeeds on shallow needles. The default return type stays a
        # float (and return_counts keeps its dict), so existing callers are
        # untouched.
        return acc, details
    if return_counts:
        low, high = wilson_interval(n_correct, n_run)
        return {'accuracy': acc, 'n_correct': n_correct, 'n_trials': n_run,
                'trial_correct': trial_correct, 'seed': seed,
                'ci95_low': low, 'ci95_high': high,
                'ci_basis': 'wilson_within_single_placement_draw',
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
    p.add_argument('--tokenizer', default=None, help="tokenizer name; default: the checkpoint's recorded tokenizer, else gpt2")
    p.add_argument('--tokenizer-revision', default=None, help="tokenizer revision; default: the checkpoint's recorded revision (an explicit flag wins)")
    p.add_argument('--decode-batch', type=int, default=8, help='independent needle trials decoded per forward pass (each step still recomputes the whole prefix)')
    p.add_argument('--seed', type=int, default=0, help='seed for the needle-position and generation RNGs')
    p.add_argument('--output', default=None)
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this evaluation tests; recorded in the manifest so a pass/fail is auditable as a test of a claim rather than a fishing trip')
    args = p.parse_args()
    if not args.tasks and not args.needle_test:
        # An evaluation with neither benchmarks nor the needle test measures
        # nothing; writing a finished manifest for it would let an empty run
        # be cited as an evaluation.
        p.error('nothing to evaluate: pass --tasks and/or --needle-test')
    if args.batch_size < 1:
        p.error(f'--batch-size {args.batch_size} must be >= 1')
    if args.decode_batch < 1:
        p.error(f'--decode-batch {args.decode_batch} must be >= 1')
    if args.needle_context_len is not None and args.needle_context_len < 1:
        p.error(f'--needle-context-len {args.needle_context_len} must be >= 1')
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
        if args.tokenizer is not None:
            # An explicit tokenizer is the user's choice, not a hint: the
            # checkpoint may have been trained with a different vocabulary and
            # silently overriding the flag evaluates a different model.
            tok_name = args.tokenizer
            recorded_revision = (
                (model_args.get('tokenizer_revision_resolved')
                 or model_args.get('tokenizer_revision'))
                if model_args.get('tokenizer') == args.tokenizer else None)
        else:
            tok_name, recorded_revision = _checkpoint_tokenizer(model_args, 'gpt2')
        tok_revision = (args.tokenizer_revision if args.tokenizer_revision is not None
                        else recorded_revision)
        tok = AutoTokenizer.from_pretrained(tok_name, revision=tok_revision)
        results['tokenizer'] = tok_name
        results['tokenizer_revision'] = tok_revision
        # Pin the concrete commit the tokenizer resolved to, not only the
        # mutable request: exact-match needle accuracy depends on it.
        results['tokenizer_revision_resolved'] = (
            getattr(tok, '_commit_hash', None) or tok_revision)
        trained_seq_len = model_args.get('seq_len')
        trained_seq_len = int(trained_seq_len) if isinstance(trained_seq_len, int) else None
        needle_overhead = needle_decode_overhead(tok)
        max_scoreable = max(1, model.cfg.max_seq_len - needle_overhead)
        # The longest length the model can be *scored* at is not the same as
        # the length it was *trained* at: evaluating at max_seq_len after
        # training at a shorter context measures out-of-distribution positional
        # extrapolation, so the trained length is the honest default.
        if args.needle_context_len is not None:
            ctx_len = args.needle_context_len
            ctx_source = 'cli'
        elif trained_seq_len is not None:
            # Scoring needs `needle_overhead` extra forward positions beyond
            # the prompt, so the prompt default must leave that room inside
            # the trained context; otherwise the script's own default is
            # already an extrapolation and trips the warning below.
            ctx_len = max(1, min(trained_seq_len - needle_overhead,
                                 max_scoreable))
            ctx_source = 'trained_seq_len'
        else:
            ctx_len = max_scoreable
            ctx_source = 'max_seq_len'
        results['needle_trained_seq_len'] = trained_seq_len
        results['needle_ctx_source'] = ctx_source
        if trained_seq_len is not None and ctx_len + needle_overhead > trained_seq_len:
            results['needle_ctx_exceeds_trained_seq_len'] = True
            print(f'[eval][warn] needle context length {ctx_len} + {needle_overhead} scoring '
                  f'positions exceeds the trained seq_len={trained_seq_len}: the result '
                  f'measures positional extrapolation, not in-distribution long-context '
                  f'accuracy', file=sys.stderr)
        needle = needle_in_haystack(model, tok, context_len=ctx_len, device=device, seed=args.seed, return_counts=True, decode_batch=args.decode_batch)
        results['needle_acc'] = needle['accuracy']
        results['needle_acc_basis'] = 'single_placement_seed'
        results['needle_acc_note'] = (
            'accuracy is one placement draw; its Wilson interval covers trial noise within '
            'that draw, not placement-to-placement variability. Run several --seed values '
            '(and pair arms by seed) before claiming a placement-level difference.')
        results['needle'] = needle
        results['needle_ctx_len'] = ctx_len
        print(f"[eval] needle-in-haystack accuracy @ {ctx_len} = {needle['accuracy']:.3f} "
              f"({needle['n_correct']}/{needle['n_trials']}, 95% CI [{needle['ci95_low']:.3f}, {needle['ci95_high']:.3f}])")
        if needle['failure_causes']:
            print(f"[eval] needle failure causes: {needle['failure_causes']} "
                  f"confidence: {needle['failure_confidence']} "
                  f"(per-trial bad-case records in the results JSON)")
    results['seed'] = args.seed
    results['hypothesis'] = args.hypothesis
    results['hyperparameters'] = vars(args)
    results['run_metadata'] = collect_run_metadata(seed=args.seed)
    out = args.output or Path(args.ckpt).parent / 'eval_results.json'
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    # The manifest mirrors what the run needs to be auditable: which checkpoint
    # (hashed), with which args, and what came out. Task scores live in the
    # results file, which is fingerprinted here rather than duplicated.
    manifest = dict(results['run_metadata'])
    manifest['hyperparameters'] = vars(args)
    manifest['hypothesis'] = args.hypothesis
    manifest['checkpoint'] = file_fingerprint(args.ckpt)
    manifest['results_file'] = file_fingerprint(out)
    manifest['summary'] = {k: results[k] for k in
                           ('norm', 'tasks_ran', 'needle_acc', 'needle_acc_basis',
                            'needle_ctx_len',
                            'needle_ctx_source', 'needle_trained_seq_len',
                            'tokenizer', 'tokenizer_revision',
                            'tokenizer_revision_resolved') if k in results}
    if args.tasks and not results.get('tasks_ran', False):
        # A requested benchmark that did not run is an incomplete evaluation;
        # the manifest must not claim completion (the SystemExit below says
        # the same thing to the caller).
        manifest['status'] = 'failed'
    else:
        manifest = finish_metadata(manifest)
    save_run_manifest(Path(out).with_name('eval_manifest.json'), manifest)
    if args.hypothesis:
        print(f'[eval][hypothesis] {args.hypothesis}')
    print(f'[eval] results saved to {out}')
    if args.tasks and not results.get('tasks_ran', False):
        print(f'[eval] ERROR: tasks {args.tasks} were requested but lm-eval did not run them; '
              f'nothing about {args.tasks} was measured (see tasks_error in {out})', file=sys.stderr)
        raise SystemExit(2)
if __name__ == '__main__':
    main()
