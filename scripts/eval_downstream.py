from __future__ import annotations
import argparse
import sys
import json
import random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash import GPTConfig, GPT
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b

def _strip_compile_prefix(state):
    prefix = '_orig_mod.'
    if not any(k.startswith(prefix) for k in state):
        return state
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}

def load_ckpt(path: str, device: str='cuda'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt['args']
    model_cfg = dict(norm_name=args['norm'], qk_norm=args.get('qk_norm'), qk_alpha_init=args.get('qk_alpha_init', 1.0), use_layerscale=not args.get('no_layerscale', False), alpha_0=args.get('alpha_0', 0.9), k_init=args.get('k_init', 1.0), k_min=args.get('k_min'), shared_depth_schedule=args.get('shared_depth_schedule', False), alpha_init=args.get('alpha_init', 0.5), p_init=args.get('p_init', 2.0), learnable_p=not args.get('no_learnable_p', False), learnable_k=not args.get('no_learnable_k', False), beta=args.get('beta', True), alpha_jitter=args.get('alpha_jitter', 0.05))
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

class AdaSquashLM:
    """Minimal lm-eval wrapper around the raw GPT module.

    Implements the three primitives lm-eval-harness tasks need - token
    loglikelihood (multiple-choice / completion tasks), rolling loglikelihood
    (perplexity tasks) and greedy `generate_until` (generation tasks) - with no
    dependence on `transformers` model wrappers. Scoring is one sequence at a
    time (batch_size is accepted for interface compatibility); a 124M model
    on a few tasks is minutes, not hours.
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

    def tok_encode(self, string: str, add_special_tokens: bool=False, left_truncate_len: int | None=None) -> list:
        ids = self.tokenizer(string, add_special_tokens=False)['input_ids']
        if left_truncate_len is not None and len(ids) > left_truncate_len:
            ids = ids[len(ids) - left_truncate_len:]
        return list(ids)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list, list]:
        """Whitespace-safe context/continuation split: tokenize the whole
        string, then keep whatever the tokenizer produced beyond the context
        prefix (the canonical lm-eval trick - tokenizing the continuation on
        its own would lose a leading space into a different token)."""
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
            until = list(gen_kwargs.get('until', [])) + [s for s in stops if s not in gen_kwargs.get('until', [])]
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
    """Run lm-evaluation-harness tasks against the raw GPT module.

    Uses ``AdaSquashLM`` (loglikelihood + rolling + greedy generation), so no
    HF-compat export is needed. Returns ``{task: {metric: value}}`` with every
    numeric metric the harness reports for the task.
    """
    try:
        import lm_eval
        from lm_eval import simple_evaluate
    except ImportError as e:
        raise RuntimeError('lm-eval not installed. `pip install lm-eval`.') from e
    if tokenizer is None or isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer or 'gpt2')
    lm_obj = AdaSquashLM(model, tokenizer, batch_size=batch_size)
    report = simple_evaluate(model=lm_obj, tasks=list(tasks), batch_size=batch_size, bootstrap_iters=0)
    results = {}
    for task, metrics in (report.get('results') or {}).items():
        results[task] = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return results
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

def needle_in_haystack(model, tokenizer, needle: str=DEFAULT_NEEDLE, haystack_prompt: str=DEFAULT_HAYSTACK, context_len: int=4096, n_trials: int=5, device: str='cuda', seed: int=0, return_details: bool=False):
    """Greedy needle-recall accuracy over ``n_trials`` random insertions.

    With ``return_details=True`` also returns one record per trial: the
    needle's relative position in the context (bucketed early / middle / late
    by insertion tercile) and whether recall succeeded - position-resolved
    coverage a single scalar accuracy hides (a model can ace shallow needles
    and fail deep ones, and the average would never show it)."""
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
    details = []
    last_logits = getattr(model, 'last_position_logits', None)
    for _ in range(n_trials):
        body_len = context_len - len(needle_ids) - len(question_ids) - 2
        if body_len < len(prompt_ids):
            raise ValueError(f'context_len={context_len} is too short to hold the needle ({len(needle_ids)} tok) + question ({len(question_ids)} tok) + one haystack repeat ({len(prompt_ids)} tok)')
        n_repeats = body_len // len(prompt_ids)
        body = prompt_ids * n_repeats
        pos = rng.randint(0, max(1, len(body) - 1))
        full = body[:pos] + needle_ids + body[pos:] + question_ids
        pos_frac = pos / max(1, len(body))
        bucket = 'early' if pos_frac < 1 / 3 else ('middle' if pos_frac < 2 / 3 else 'late')
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
                    break
                buf[0, n] = nxt
                n += 1
        n_correct += int(correct)
        decode = getattr(tokenizer, 'decode', None)
        details.append({'pos_frac': pos_frac, 'bucket': bucket, 'correct': bool(correct), 'generated': decode(generated) if (generated and callable(decode)) else ''})
    if n_run == 0:
        raise ValueError('needle test produced no trials')
    if return_details:
        return (n_correct / n_run, details)
    return n_correct / n_run

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tasks', nargs='+', default=['mmlu', 'gsm8k', 'hellaswag'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--needle-test', action='store_true', help='also run needle-in-haystack')
    p.add_argument('--needle-context-len', type=int, default=None, help="needle context length (default: the checkpoint's max_seq_len minus the positions the secret's decode needs - the longest the model can actually be scored at; asking for more raises)")
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        acc = needle_in_haystack(model, tok, context_len=ctx_len, device=device)
        results['needle_acc'] = acc
        results['needle_ctx_len'] = ctx_len
        print(f'[eval] needle-in-haystack accuracy @ {ctx_len} = {acc:.3f}')
    out = args.output or Path(args.ckpt).parent / 'eval_results.json'
    Path(out).write_text(json.dumps(results, indent=2))
    print(f'[eval] results saved to {out}')
if __name__ == '__main__':
    main()
