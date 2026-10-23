from __future__ import annotations
"""Tuned simplest-baseline ladder and Bayes-limit calibration for the synthetic ablation corpus.

Why this script exists
----------------------
The ablation matrix (`scripts/ablation.py`) compares normalization variants
against `ln_baseline` / `rmsnorm_baseline`. Those are *relative* references:
if every transformer in the matrix is still far above the information-theoretic
optimum of the generative process, the comparison only ranks which model is
least bad, it does not show that any of them learned the task. This ladder
closes that gap for the repo's controlled corpus
(`adasquash.utils.markov_batch`):

1. `uniform_vocab`   - uniform over all 50304 token ids: the naive strawman.
2. `uniform_targets` - knows only the target alphabet (the `n_states` walk
   states): CE = ln(n_states); still no sequence knowledge.
3. `unigram` / `bigram` / `trigram` - count models with add-k smoothing, where
   k is *tuned on a separate held-out stream* (generator seed + 2), never on
   the stream used to compare. A baseline that only looks strong because the
   smoothing grid was badly chosen shows up as `grid_limited` instead of
   silently weakening the comparison.
4. `walk_rule` - the Bayes predictor for a memoryless observer that sees only
   the current token (outlier token -> uniform over states; walk state s ->
   uniform over {s, s+1, s+2} mod n_states). Its mean CE equals the closed-form
   memoryless floor `(1 - p_spike) * ln(max_jump) + p_spike * ln(n_states)`.
5. `hmm_bayes` - the exact Bayes predictor for a *causal observer of the whole
   history*: the corpus is an observation-hidden Markov model (the walk is
   hidden; 3% of tokens are replaced by the outlier id), so the forward filter
   `b_t(s) = P(s_t = s | x_1..x_t)` is exact and closed-form, and
   `p(y_t | x_1..x_t) = b_t * P(jump)` achieves the minimum mean CE of ANY
   causal sequence model at this task (transformers included). It is strictly
   below the memoryless floor, exactly because a run of non-spiked tokens
   identifies the hidden state even when the current token is spiked - the
   memoryless rule cannot use that, the transformer can.

The two limits matter for different rows: count models and the walk rule are
memoryless(-order) readers, so the memoryless floor is their ceiling; the
ablation's transformers read the full prefix, so `hmm_bayes` is *their*
ceiling. A row is under-trained regardless of its rank in the matrix until its
`excess_over_bayes_ceiling_nats` is near zero. The script also self-checks the
corpus assumptions instead of asserting them: the rule must match the
memoryless floor within Monte-Carlo tolerance, the HMM filter must not exceed
it, and any target outside a state's jump support is counted and reported.

Protocol (mirrors `scripts/ablation.py` so the numbers are comparable)
----------------------------------------------------------------------
* corpus: `markov_batch`, n_states=64, max_jump=3, outlier channel on (3% of
  input tokens spiked to id vocab_size - 1 = 50303), batch 8 x seq 512.
* with seed s the count models are fitted on the same generator stream the
  ablation trains on (`torch.Generator` seed s), smoothing is tuned on seed
  s + 2, and every predictor is scored on the ablation's held-out stream
  (`torch.Generator` seed s + 1). The walk rule and the HMM filter need no
  fitting, so they quantify how much of the held-out loss is irreducible.
* metric: mean cross-entropy in nats over unmasked next-token targets (the
  same quantity as the ablation's `val_loss`); perplexity is its monotone
  transform, reported for readability.

The output (`runs/ablation/simple_baselines.json` by default) is auto-detected
by `scripts/ablation.py`, which then annotates every row of
`baseline_comparison.json` with `excess_over_bayes_floor_nats` (memoryless),
`excess_over_bayes_ceiling_nats` (full-context) and
`signal_captured_fraction`.
"""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash.metrics import (holm_adjust, mean_std_stderr, paired_t_one_sided,
                               student_t_ppf)
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds
from adasquash.utils import markov_batch
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_KS = (1e-06, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
SIMPLE_PREDICTORS = ('unigram', 'bigram', 'trigram')
# Mirrors the spike probability hardcoded in `adasquash.utils.markov_batch`
# (`torch.rand(...) < 0.03`). The corpus is the data; this constant only
# enters the closed-form memoryless floor.
MARKOV_SPIKE_PROB = 0.03

def build_support(n_states: int, outlier_id):
    support = list(range(int(n_states)))
    if outlier_id is not None:
        support.append(int(outlier_id))
    return support

def wrapped_jump_probs(n_states: int, max_jump: int) -> list[float]:
    """Successor distribution of `markov_batch`'s uniform jump, after wrapping.

    `markov_batch` draws `j` uniformly from `{0..max_jump-1}` and takes
    `s_next = (s + j) mod n_states`. When `max_jump > n_states` several jumps
    land on the same successor, so the distribution is no longer uniform and
    the closed-form floor is its entropy, not `ln(max_jump)`.
    """
    n_states = int(n_states)
    max_jump = int(max_jump)
    counts = [0] * n_states
    for j in range(max_jump):
        counts[j % n_states] += 1
    return [c / max_jump for c in counts]

def memoryless_bayes_floor_nats(n_states: int, max_jump: int, spike_prob: float=0.0) -> float:
    """Closed-form CE of the best predictor that sees only the current token.

    With the outlier channel off, the next state given the current state is
    the wrapped uniform jump distribution, so the floor is its entropy
    (`ln(max_jump)` whenever all jumps stay distinct, i.e. `max_jump <=
    n_states`). With the channel on, a `spike_prob` fraction of inputs is the
    outlier id, which carries no information about the hidden state, so given
    the outlier the next token is uniform over all `n_states`. This is NOT the
    ceiling for a full-context model (`hmm_bayes_stats` computes that one).
    """
    floor = -sum(p * math.log(p) for p in wrapped_jump_probs(n_states, max_jump) if p > 0.0)
    if spike_prob > 0.0:
        floor = (1.0 - float(spike_prob)) * floor + float(spike_prob) * math.log(int(n_states))
    return floor

def build_remap(vocab_size: int, support, device='cpu') -> torch.Tensor:
    """Map corpus ids to compact indices; index `len(support)` is "unseen".

    Built on `device` so it can index the (possibly CUDA) corpus batches.
    """
    remap = torch.full((int(vocab_size) + 1,), len(support), dtype=torch.long, device=device)
    if support:
        remap[torch.tensor([int(s) for s in support], dtype=torch.long, device=device)] = torch.arange(len(support), device=device)
    return remap

def generate_batches(seed: int, n_batches: int, args, device: str):
    """The ablation runner's stream: same generator seed, corpus and batch shape."""
    if not _is_seed_id(seed):
        raise ValueError(f'stream seed must be an integer, got {seed!r}')
    seed = int(seed)
    rng = torch.Generator(device=device).manual_seed(seed)
    outlier_id = int(args.vocab_size) - 1 if args.outlier_channel else None
    for _ in range(int(n_batches)):
        yield markov_batch(rng, -100, args.batch_size, args.seq_len, device,
                           n_states=args.n_states, outlier_id=outlier_id,
                           max_jump=args.max_jump)

def flatten_split(batches, remap: torch.Tensor, unknown: int, ignore_index: int=-100) -> dict:
    """Flatten a stream into (current, previous, target) supervision tokens.

    `previous` is the raw predecessor for the trigram context; position 0 of
    each row has no predecessor, so it is mapped to the "unseen" bucket and
    the trigram falls back to the bigram there - the same handling a count
    model would give an out-of-vocabulary context.
    """
    xs, x2s, ys, x_raw, y_raw = ([], [], [], [], [])
    for x, y in batches:
        # An all-ignored batch simply contributes zero rows, and the
        # all-empty case is caught below.
        mask = y.ne(ignore_index)
        xr = remap[x]
        prev = torch.empty_like(xr)
        prev[:, 0] = unknown
        prev[:, 1:] = xr[:, :-1]
        xs.append(xr[mask])
        x2s.append(prev[mask])
        ys.append(remap[y[mask]])
        x_raw.append(x[mask])
        y_raw.append(y[mask])
    if not xs or not any(t.numel() for t in xs):
        raise ValueError('corpus stream produced no supervision tokens')
    return {'x': torch.cat(xs), 'x2': torch.cat(x2s), 'y': torch.cat(ys),
            'x_raw': torch.cat(x_raw), 'y_raw': torch.cat(y_raw)}

def collect_counts(split: dict, n_symbols: int, with_trigram: bool=True) -> dict:
    """Full-order count tables over the compact alphabet.

    The corpus uses at most `n_states + 1` distinct ids, so the dense bigram
    and trigram tables are indexed by the compact remap.
    """
    uni = torch.bincount(split['y'], minlength=n_symbols).to(torch.float64)
    bi = torch.bincount(split['x'] * n_symbols + split['y'],
                        minlength=n_symbols * n_symbols).to(torch.float64).view(n_symbols, n_symbols)
    tri = None
    if with_trigram and n_symbols ** 3 <= 1 << 27:
        # Position 0 has no real predecessor; `flatten_split` marks it with
        # the unseen bucket so the trigram must fall back to the bigram
        # there. Counting those artificial contexts would make `seen2` true
        # and defeat the documented fallback.
        real_context = split['x2'].ne(n_symbols - 1)
        idx = ((split['x2'] * n_symbols + split['x']) * n_symbols + split['y'])[real_context]
        tri = torch.bincount(idx, minlength=n_symbols ** 3).to(torch.float64).view(n_symbols, n_symbols, n_symbols)
    else:
        with_trigram = False
    return {'uni': uni, 'bi': bi, 'tri': tri, 'with_trigram': with_trigram}

def unigram_probs(counts: dict, n_symbols: int, k: float) -> torch.Tensor:
    return (counts['uni'] + k) / (counts['uni'].sum() + k * n_symbols)

def ce_unigram(counts: dict, split: dict, n_symbols: int, k: float) -> float:
    return float(-unigram_probs(counts, n_symbols, k)[split['y']].log().mean())

def ce_bigram(counts: dict, split: dict, n_symbols: int, k: float) -> float:
    """Add-k bigram P(y|x), backing off to the unigram when x was unseen."""
    p_uni = unigram_probs(counts, n_symbols, k)
    bi = counts['bi']
    ctx = bi.sum(-1, keepdim=True)
    p_cond = (bi + k) / (ctx + k * n_symbols)
    x, y = (split['x'], split['y'])
    seen = ctx.squeeze(-1)[x] > 0
    p = torch.where(seen, p_cond[x, y], p_uni[y])
    return float(-p.log().mean())

def ce_trigram(counts: dict, split: dict, n_symbols: int, k: float) -> float:
    """Add-k trigram P(y|x2,x), backing off to the bigram then the unigram."""
    p_uni = unigram_probs(counts, n_symbols, k)
    bi = counts['bi']
    ctx1 = bi.sum(-1, keepdim=True)
    p_bi = (bi + k) / (ctx1 + k * n_symbols)
    x, y = (split['x'], split['y'])
    seen1 = ctx1.squeeze(-1)[x] > 0
    p = torch.where(seen1, p_bi[x, y], p_uni[y])
    tri = counts['tri']
    if tri is None:
        return float(-p.log().mean())
    ctx2 = tri.sum(-1, keepdim=True)
    p_tri = (tri + k) / (ctx2 + k * n_symbols)
    x2 = split['x2']
    seen2 = ctx2.squeeze(-1)[x2, x] > 0
    p = torch.where(seen2, p_tri[x2, x, y], p)
    return float(-p.log().mean())

def walk_rule_stats(split: dict, n_states: int, max_jump: int, outlier_id) -> dict:
    """The memoryless Bayes predictor, scored on the held-out stream.

    Also counts rule violations (`y` not among the `max_jump` successors of a
    non-outlier `x`): a nonzero count means the stream is not the process the
    closed-form floor describes, and the calibration must not be trusted.
    """
    x_raw, y_raw = (split['x_raw'], split['y_raw'])
    if outlier_id is None:
        is_outlier = torch.zeros_like(x_raw, dtype=torch.bool)
    else:
        is_outlier = x_raw.eq(int(outlier_id))
    jump_p = torch.tensor(wrapped_jump_probs(n_states, max_jump), dtype=torch.float64,
                          device=x_raw.device)
    step = (y_raw - x_raw).remainder(int(n_states))
    p = torch.where(is_outlier,
                    torch.full_like(jump_p[step], 1.0 / int(n_states)),
                    jump_p[step])
    mismatches = int((~is_outlier & (jump_p[step] == 0)).sum())
    return {'ce_nats': float(-p.log().mean()),
            'rule_target_mismatches': mismatches,
            'outlier_input_fraction': float(is_outlier.to(torch.float64).mean())}

def hmm_bayes_stats(batches, n_states: int, max_jump: int, outlier_id,
                    ignore_index: int = -100) -> dict:
    """Exact predictive CE of the full-context Bayes observer (HMM forward filter).

    Hidden chain: `s_{t+1} = s_t + j`, `j` uniform in `{0..max_jump-1}`.
    Observation: `x_t = s_t` with probability `1 - spike_prob`, else the
    outlier id (a constant likelihood across hidden states, so the filter does
    not need the spike probability). The belief
    `b_t(s) = P(s_t = s | x_1..x_t)` is updated exactly, and
    `p(y_t | x_1..x_t) = b_t * P(jump)` is the Bayes-optimal causal predictive
    distribution, i.e. the minimum mean CE of any causal model that reads the
    observed prefix (the ablation's transformer included). Used as the
    `full_context` ceiling above the memoryless `walk_rule`.
    """
    n_states = int(n_states)
    batches = list(batches)
    if not batches:
        raise ValueError('hmm filter received no batches')
    # The filter lives on the batch device: indexing with `rows`/`x`/`y`
    # (which come from the caller's batches) must not mix CPU and CUDA tensors.
    device = batches[0][0].device
    transition = torch.zeros(n_states, n_states, dtype=torch.float64)
    for s in range(n_states):
        for j in range(int(max_jump)):
            transition[s, (s + j) % n_states] += 1.0 / int(max_jump)
    transition = transition.to(device)
    # Device-side fp64 accumulator: one .item() at the very end.
    ce_sum_t = torch.zeros((), dtype=torch.float64, device=device)
    n_tokens = 0
    for x, y in batches:
        batch = x.shape[0]
        rows = torch.arange(batch, device=x.device)
        first = x[:, 0]
        belief = torch.zeros(batch, n_states, dtype=torch.float64, device=x.device)
        if outlier_id is None:
            belief[rows, first] = 1.0
        else:
            is_oid = first.eq(int(outlier_id))
            belief.fill_(1.0 / n_states)
            keep = (~is_oid).nonzero(as_tuple=True)[0]
            if keep.numel():
                belief[keep] = 0.0
                belief[keep, first[keep]] = 1.0
        # Ping-pong buffers for the per-step posterior.
        pred = torch.empty_like(belief)
        for t in range(x.shape[1] - 1):
            torch.mm(belief, transition, out=pred)
            # Unmasked held-out targets only: an ignore-index position is not
            # a prediction target (indexing it would either crash or wrap).
            valid = y[:, t] != ignore_index
            if bool(valid.any()):
                valid_rows = rows[valid]
                targets = y[:, t][valid].to(torch.long)
                ce_sum_t = ce_sum_t - pred[valid_rows, targets].log().sum()
                n_tokens += int(valid.sum())
            obs = x[:, t + 1]
            if outlier_id is None:
                pred.zero_()
                pred[rows, obs] = 1.0
            else:
                is_oid = obs.eq(int(outlier_id))
                keep = (~is_oid).nonzero(as_tuple=True)[0]
                if keep.numel():
                    pred[keep] = 0.0
                    pred[keep, obs[keep]] = 1.0
                # The outlier rows keep the posterior themselves; only the
                # observed rows were overwritten with the delta.
            belief, pred = pred, belief
    if n_tokens == 0:
        raise ValueError('hmm filter received no tokens (empty batches or seq_len < 2)')
    return {'ce_nats': float(ce_sum_t) / n_tokens, 'n_tokens': n_tokens}

def tune_smoothing(ce_fn, ks) -> dict:
    """Select the add-k smoothing constant on a held-out tuning stream.

    Returns the tuning curve, the argmin and `grid_limited` (True when the
    best k sits on a grid endpoint, i.e. the grid may be truncating the
    optimum). The evaluation stream is never touched here.
    """
    grid = []
    for k in ks:
        ce = float(ce_fn(float(k)))
        grid.append({'k': float(k), 'tuning_ce_nats': ce if math.isfinite(ce) else None,
                     '_ce': ce})
    finite = [row for row in grid if math.isfinite(row['_ce'])]
    if not finite:
        return {'grid': [{key: val for key, val in row.items() if key != '_ce'} for row in grid],
                'best_k': None, 'best_tuning_ce_nats': None, 'grid_limited': None}
    best = min(finite, key=lambda row: row['_ce'])
    position = grid.index(best)
    return {'best_k': best['k'], 'best_tuning_ce_nats': best['_ce'],
            'grid_limited': position == 0 or position == len(grid) - 1,
            'grid': [{key: val for key, val in row.items() if key != '_ce'} for row in grid]}

def evaluate_seed(seed: int, args, device: str) -> dict:
    """Fit, tune and evaluate the whole predictor ladder for one seed."""
    if not _is_seed_id(seed):
        raise ValueError(f'seed must be an integer, got {seed!r}')
    seed = int(seed)
    outlier_id = args.vocab_size - 1 if args.outlier_channel else None
    support = build_support(args.n_states, outlier_id)
    n_symbols = len(support) + 1
    remap = build_remap(args.vocab_size, support, device)
    train = flatten_split(generate_batches(seed, args.train_batches, args, device), remap, len(support))
    tune = flatten_split(generate_batches(seed + 2, args.tune_batches, args, device), remap, len(support))
    val_batches = list(generate_batches(seed + 1, args.val_batches, args, device))
    val = flatten_split(val_batches, remap, len(support))
    counts = collect_counts(train, n_symbols, with_trigram=not args.no_trigram)
    record = {'seed': seed, 'n_train_tokens': int(train['y'].numel()),
              'n_tune_tokens': int(tune['y'].numel()), 'n_val_tokens': int(val['y'].numel()),
              'predictors': {}, 'tuning': {},
              'walk_rule': walk_rule_stats(val, args.n_states, args.max_jump, outlier_id)}
    record['predictors']['uniform_vocab'] = math.log(args.vocab_size)
    record['predictors']['uniform_targets'] = math.log(args.n_states)
    for name, ce_fn in (('unigram', ce_unigram), ('bigram', ce_bigram), ('trigram', ce_trigram)):
        if name == 'trigram' and not counts['with_trigram']:
            # Report it as skipped rather than falling back to the bigram under
            # a "trigram" label, which would overstate what was evaluated.
            record['predictors'][name] = None
            continue
        tuning = tune_smoothing(lambda k, fn=ce_fn: fn(counts, tune, n_symbols, k), args.ks)
        record['tuning'][name] = tuning
        record['predictors'][name] = (None if tuning['best_k'] is None
                                      else float(ce_fn(counts, val, n_symbols, tuning['best_k'])))
    record['predictors']['walk_rule'] = record['walk_rule']['ce_nats']
    record['predictors']['hmm_bayes'] = hmm_bayes_stats(val_batches, args.n_states,
                                                       args.max_jump, outlier_id)['ce_nats']
    return record

def _perplexity(mean_nats: float) -> float:
    """exp(CE); a finite but huge CE stays visible instead of crashing."""
    try:
        return math.exp(float(mean_nats))
    except OverflowError:
        return math.inf


def aggregate_records(records: list, names) -> dict:
    out = {}
    for name in names:
        values = [r['predictors'].get(name) for r in records
                  if isinstance(r.get('predictors'), dict)
                  and isinstance(r['predictors'].get(name), (int, float))]
        stats = mean_std_stderr(values)
        out[name] = None if stats is None else {
            'mean_ce_nats': stats['mean'], 'std_ce_nats': stats['std'],
            'stderr_ce_nats': stats['stderr'], 'n_seeds': stats['n'],
            'perplexity': _perplexity(stats['mean']),
        }
    return out

def best_count_model(aggregate: dict) -> dict | None:
    """The strongest tuned count model - the baseline a complex model must beat."""
    candidates = [(name, aggregate[name]) for name in SIMPLE_PREDICTORS if aggregate.get(name)]
    if not candidates:
        return None
    name, stats = min(candidates, key=lambda item: item[1]['mean_ce_nats'])
    return {'predictor': name, **stats}

def build_calibration(aggregate: dict, memoryless_floor: float, tuned: dict | None, records: list) -> dict:
    """Consistency checks between the observed streams and the two limits."""
    rule = aggregate.get('walk_rule')
    full = aggregate.get('hmm_bayes')
    rule_delta = None if not rule else rule['mean_ce_nats'] - memoryless_floor
    rule_tolerance = None if not rule else max(0.02, 4.0 * rule['stderr_ce_nats'])
    full_delta = None if not full else full['mean_ce_nats'] - memoryless_floor
    return {
        'memoryless_floor_nats': memoryless_floor,
        'rule_minus_memoryless_nats': rule_delta,
        'rule_memoryless_tolerance_nats': rule_tolerance,
        'rule_matches_memoryless_floor': None if rule_delta is None else abs(rule_delta) <= rule_tolerance,
        'full_context_minus_memoryless_nats': full_delta,
        'full_context_below_memoryless_floor': None if full_delta is None else full_delta <= (rule_tolerance or 0.02),
        'best_count_model_minus_full_context_nats': (None if not tuned or not full
                                                     else tuned['mean_ce_nats'] - full['mean_ce_nats']),
        'rule_target_mismatches_total': sum(r['walk_rule']['rule_target_mismatches'] for r in records),
        'outlier_input_fraction': (sum(r['walk_rule']['outlier_input_fraction'] for r in records) / len(records)
                                   if records else None),
        'reading': ('walk_rule must match the closed-form memoryless floor (corpus model check); the HMM '
                    'filter is the exact causal full-context ceiling and must sit at or below that floor '
                    '(any positive gap is Monte-Carlo noise on finite streams); transformers read the full '
                    'prefix, so hmm_bayes - not the memoryless floor - is the ceiling their excess is '
                    'measured against'),
    }

def load_ablation_rows(path) -> list:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []

def _finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _is_seed_id(value) -> bool:
    """A seed id is an int; 42.9/True must not become seed 42/1."""
    return isinstance(value, int) and not isinstance(value, bool)


def _row_seed_losses(row: dict, duplicates: set | None = None) -> dict:
    """Per-seed validation losses of one ablation row, when recorded.

    A duplicate seed is one underlying run, not two independent observations.
    Keep the first record but report the collision through `duplicates` (when
    supplied) so the caller can exclude the row instead of letting the last
    duplicate silently decide the paired test.
    """
    out = {}
    per_seed = row.get('per_seed')
    if isinstance(per_seed, list):
        for item in per_seed:
            if not isinstance(item, dict):
                continue
            seed = item.get('seed')
            # A boolean/fractional/string seed is not an independent
            # replication id; int(True)/int("42") would silently pair this
            # row with a different ablation run.
            if not _is_seed_id(seed) or not _finite_number(item.get('val_loss')):
                continue
            if seed in out:
                if duplicates is not None:
                    duplicates.add(seed)
                continue
            out[seed] = float(item['val_loss'])
        return out
    seed = row.get('seed')
    if _is_seed_id(seed) and _finite_number(row.get('val_loss')):
        out[seed] = float(row['val_loss'])
    return out


def _count_seed_ce(records: list, predictor: str,
                   duplicates: set | None = None) -> dict:
    """Per-seed CE of the tuned count model on the shared held-out stream."""
    out = {}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        seed = record.get('seed')
        if not _is_seed_id(seed):
            continue
        value = (record.get('predictors') or {}).get(predictor)
        if not _finite_number(value):
            continue
        if seed in out:
            if duplicates is not None:
                duplicates.add(seed)
            continue
        out[seed] = float(value)
    return out


def _mde_from_test(test: dict | None, alpha: float = 0.05, power: float = 0.8):
    if (test is None or not test.get('df')
            or not _finite_number(test.get('ci_low'))
            or not _finite_number(test.get('ci_high'))):
        return None
    half = (test['ci_high'] - test['ci_low']) / 2.0
    critical = student_t_ppf(1.0 - alpha / 2.0, float(test['df']))
    if critical <= 0.0:
        return None
    stderr = half / critical
    return ((student_t_ppf(1.0 - alpha, float(test['df']))
             + student_t_ppf(power, float(test['df']))) * stderr)


def _row_run_identity(row: dict, parent_of: dict) -> str:
    name = row.get('name')
    seen = set()
    while isinstance(name, str) and name not in seen:
        seen.add(name)
        parent = parent_of.get(name)
        if not isinstance(parent, str):
            return name
        name = parent
    return row.get('name')

# The synthetic corpus and validation stream have to match the count model's
# for its CE to be a paired baseline. These are the fields the ablation runner
# records in each row's `controls` block; missing/unknown values never count
# as a match.
_CALIBRATION_MODEL_VOCAB = {'gpt2-124m': 50304, 'gpt2-350m': 50304,
                            'llama-1b': 32000}
_CALIBRATION_PROTOCOL_KEYS = (
    'generator', 'n_states', 'max_jump', 'vocab_size', 'batch_size', 'seq_len',
    'outlier_id', 'val_seed_offset', 'val_batches',
)

def _calibration_protocol(args, outlier_id) -> dict:
    """The held-out corpus protocol this calibration run actually uses."""
    return {
        'generator': 'markov_batch',
        # `ablation.run_one` calls `markov_batch` with its defaults; a
        # non-default calibration corpus is a different experiment.
        'n_states': int(args.n_states),
        'max_jump': int(args.max_jump),
        'vocab_size': int(args.vocab_size),
        'batch_size': int(args.batch_size),
        'seq_len': int(args.seq_len),
        'outlier_id': outlier_id,
        'val_seed_offset': 1,
        'val_batches': int(args.val_batches),
    }

def _row_calibration_protocol(row: dict) -> dict | None:
    """The corpus protocol recorded by one ablation row, or None when unknown."""
    controls = row.get('controls')
    if not isinstance(controls, dict):
        return None
    data = controls.get('data')
    if not isinstance(data, dict):
        return None
    generator = data.get('generator')
    # `markov_batch` defaults used by the ablation runner. n_states/max_jump
    # are not separately recorded, so only a recorded markov_batch generator
    # can establish them.
    known_defaults = generator == 'markov_batch'
    return {
        'generator': generator,
        'n_states': 64 if known_defaults else None,
        'max_jump': 3 if known_defaults else None,
        'vocab_size': _CALIBRATION_MODEL_VOCAB.get(row.get('model')),
        'batch_size': data.get('batch_size'),
        'seq_len': data.get('seq_len'),
        # An absent key is unknown, while an explicitly recorded `null` means
        # the outlier channel was off. Both used to `.get()` to None, so an
        # unverified row matched a no-outlier calibration run.
        'outlier_id': (data['outlier_id'] if 'outlier_id' in data else '<missing>'),
        'val_seed_offset': data.get('val_seed_offset'),
        'val_batches': data.get('val_batches'),
    }

def _same_protocol_value(actual, expected) -> bool:
    """Type-aware comparison: ``True == 1`` and ``12.0 == 12`` are not IDs."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, int):
        return (isinstance(actual, int) and not isinstance(actual, bool)
                and actual == expected)
    if expected is None:
        return actual is None
    return actual == expected


def _calibration_protocol_mismatches(row_protocol: dict | None,
                                     expected: dict) -> list:
    if row_protocol is None:
        return list(_CALIBRATION_PROTOCOL_KEYS)
    return [key for key in _CALIBRATION_PROTOCOL_KEYS
            if not _same_protocol_value(row_protocol.get(key),
                                        expected.get(key))]

def calibrate_ablation_rows(rows: list, memoryless_floor: float, ceiling, tuned_ce,
                            log_vocab, tuned_predictor: str | None = None,
                            records: list | None = None, alpha: float = 0.05,
                            protocol: dict | None = None) -> list:
    """Join ablation val_losses with their distance to the task's limits.

    `excess_over_bayes_ceiling_nats` is the gap to the best possible full-context
    predictor (0 = Bayes optimal), `excess_over_bayes_floor_nats` the gap to the
    memoryless floor, `beats_tuned_simple_baseline` whether the transformer is
    ahead of a tuned count model, and `signal_captured_fraction` rescales the
    remaining distance against the uniform-over-vocab strawman.

    When `tuned_predictor` and the per-seed `records` are supplied, the point
    estimate is extended with the test the claim actually needs: the ablation
    seed and the count-model seed see the identical held-out stream, so their
    per-seed CEs are paired. `beats_tuned_simple_significantly` is set only by
    a one-sided paired t-test over shared seeds (>= 3) after a Holm correction
    across the rows; a point-estimate win inside the seed noise stays visible
    as `beats_tuned_simple_baseline=True, beats_tuned_simple_significantly=False`.

    When `protocol` is supplied it is the calibration corpus's held-out
    protocol (`batch_size`, `seq_len`, `outlier_id`, validation generator and
    batch count, ...). Rows whose recorded `controls` block cannot prove that
    same protocol are annotated and kept out of the Holm family: the
    summaries file deliberately accumulates rows across invocations, and a
    stale row's CE is not paired with this calibration's count-model CE.
    """
    out = []
    parent_of = {row.get('name'): row.get('reused_from') for row in rows
                 if isinstance(row, dict) and isinstance(row.get('name'), str)
                 and isinstance(row.get('reused_from'), str)}
    for row in rows:
        if not isinstance(row, dict):
            continue
        val_loss = row.get('val_loss')
        valid_loss = _finite_number(val_loss)
        identity = _row_run_identity(row, parent_of)
        entry = {'name': row.get('name'),
                 'val_loss': float(val_loss) if valid_loss else None,
                 'n_seeds': row.get('n_seeds'),
                 'run_identity': identity,
                 'excluded_from_family': None}
        if val_loss is not None and not valid_loss:
            # NaN/Inf and bools are not measurements. Keeping the raw value in
            # `val_loss` made a bool loss compute a finite excess and even
            # "beat" the tuned count model.
            entry['invalid_val_loss'] = repr(val_loss)
        if protocol is not None:
            mismatches = _calibration_protocol_mismatches(
                _row_calibration_protocol(row), protocol)
            entry['protocol_matches'] = not mismatches
            entry['protocol_mismatches'] = mismatches
            entry['recorded_protocol'] = _row_calibration_protocol(row)
            if mismatches:
                entry['excluded_from_family'] = (
                    'trained/validated under a different corpus protocol than this '
                    f"calibration: {', '.join(mismatches)}")
        if valid_loss:
            entry['excess_over_bayes_floor_nats'] = val_loss - memoryless_floor
            if ceiling is not None:
                entry['excess_over_bayes_ceiling_nats'] = val_loss - ceiling
            if tuned_ce is not None:
                entry['excess_over_tuned_simple_nats'] = val_loss - tuned_ce
                entry['beats_tuned_simple_baseline'] = bool(val_loss < tuned_ce)
            if log_vocab is not None and ceiling is not None and log_vocab > ceiling:
                entry['signal_captured_fraction'] = (log_vocab - val_loss) / (log_vocab - ceiling)
        row_duplicates: set = set()
        per_seed_val_loss = _row_seed_losses(row, row_duplicates)
        entry.update({
            'beats_tuned_simple_significantly': False,
            'per_seed_val_loss': per_seed_val_loss,
            'duplicate_seed_ids': sorted(row_duplicates),
            'n_paired_seeds': 0,
            'p_one_sided_tuned_simple': None,
            'p_one_sided_tuned_simple_holm': None,
            'delta_seed_mean_tuned_simple': None,
            'delta_ci_95_tuned_simple': None,
            'minimum_detectable_delta_tuned_simple': None,
            'simple_baseline_test_basis': None,
        })
        seed_duplicates: set = set()
        seed_ce = (_count_seed_ce(records, tuned_predictor, seed_duplicates)
                   if tuned_predictor else {})
        common = sorted(set(entry['per_seed_val_loss']) & set(seed_ce))
        entry['n_paired_seeds'] = len(common)
        if len(common) >= 2:
            row_vals = [entry['per_seed_val_loss'][seed] for seed in common]
            ref_vals = [seed_ce[seed] for seed in common]
            test = paired_t_one_sided(row_vals, ref_vals, alpha=alpha)
            entry.update({
                'p_one_sided_tuned_simple': test['p_one_sided'],
                'delta_seed_mean_tuned_simple': test['mean_delta'],
                'delta_ci_95_tuned_simple': [test['ci_low'], test['ci_high']],
                'minimum_detectable_delta_tuned_simple': _mde_from_test(test, alpha=alpha),
                'simple_baseline_test_basis': 'seed_means',
                'replicated_across_seeds': len(common) >= 3,
            })
            if len(common) < 3:
                entry['simple_baseline_test_basis'] = (
                    f'descriptive only: {len(common)} shared seeds, fewer than the '
                    'required 3')
        elif len(common) == 1:
            entry['simple_baseline_test_basis'] = (
                'descriptive only: one shared seed, so a paired count-model test '
                'cannot separate an arm effect from run-to-run noise')
        elif tuned_predictor:
            entry['simple_baseline_test_basis'] = 'no_shared_seeds'
        duplicate_ids = row_duplicates | seed_duplicates
        if duplicate_ids:
            # Repeated seed ids are one run, not extra replication; the paired
            # test above is not independent evidence and must not enter the
            # Holm family (or set the significant flag).
            entry['duplicate_seed_ids'] = sorted(duplicate_ids)
            entry['replicated_across_seeds'] = False
            entry['simple_baseline_test_basis'] = (
                'duplicate seed record(s) '
                f'{sorted(duplicate_ids)}: the same run cannot be counted twice')
            if entry.get('excluded_from_family') is None:
                entry['excluded_from_family'] = (
                    'duplicate seed record(s) '
                    f'{sorted(duplicate_ids)}: not independent evidence')
        out.append(entry)
    # Alias rows (`reused_from`) are the same underlying run under a second
    # name. Choose the canonical row by evidence quality, not by summary
    # order: an under-replicated alias that happens to appear first must not
    # shadow the replicated source and knock its significant win out of the
    # Holm family.
    by_identity: dict[str, list] = {}
    for entry in out:
        by_identity.setdefault(entry['run_identity'], []).append(entry)
    for aliases in by_identity.values():
        if len(aliases) < 2:
            continue
        canonical = min(
            aliases,
            key=lambda entry: (
                0 if _finite_number(entry.get('val_loss')) else 1,
                0 if entry.get('protocol_matches', True) else 1,
                0 if entry.get('replicated_across_seeds') else 1,
                -int(entry.get('n_seeds') or 0),
                0 if entry.get('p_one_sided_tuned_simple') is not None else 1,
                -int(entry.get('n_paired_seeds') or 0),
                str(entry.get('name')),
            ),
        )
        for entry in aliases:
            if entry is not canonical:
                entry['excluded_from_family'] = (
                    f"duplicate run of {canonical.get('name')} "
                    '(reused summary, not independent evidence)')
    family = [entry for entry in out if entry.get('replicated_across_seeds')
              and entry.get('p_one_sided_tuned_simple') is not None
              and entry.get('excluded_from_family') is None]
    adjusted = holm_adjust([entry['p_one_sided_tuned_simple'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_tuned_simple_holm'] = p_holm
        # The certifying test is paired on the shared seeds, so its direction
        # comes from that paired delta, not the all-seed point estimate.
        paired_delta = entry.get('delta_seed_mean_tuned_simple')
        entry['beats_tuned_simple_significantly'] = bool(
            paired_delta is not None and paired_delta < 0.0
            and p_holm is not None and p_holm < alpha)
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-states', type=int, default=64, help='state count of the Markov walk')
    p.add_argument('--max-jump', type=int, default=3, help='jump support {0..max_jump-1} of the walk')
    p.add_argument('--vocab-size', type=int, default=50304, help='token vocabulary of the ablation model; the outlier id is vocab_size - 1')
    p.add_argument('--no-outlier-channel', dest='outlier_channel', action='store_false', help='disable the 3%% spike to vocab_size - 1 (the ablation default keeps it on)')
    p.add_argument('--no-trigram', dest='no_trigram', action='store_true', help='skip the trigram row (only useful when the alphabet is too large for the dense count table)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--train-batches', type=int, default=200, help='batches fitted, drawn from the ablation training generator (seed s)')
    p.add_argument('--tune-batches', type=int, default=32, help='batches for the add-k selection, drawn from a separate generator (seed s + 2)')
    p.add_argument('--val-batches', type=int, default=20, help="batches scored, drawn from the ablation's held-out generator (seed s + 1)")
    p.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS), help='one independent fit/tune/eval per seed; the report is Mean +/- Std across seeds')
    p.add_argument('--ks', nargs='+', type=float, default=list(DEFAULT_KS), help='add-k smoothing grid, selected on the tuning stream')
    p.add_argument('--hypothesis', default=('The tuned count baselines approach the memoryless Bayes floor of the synthetic corpus, the exact HMM filter is strictly below it and bounds what the ablation transformers can achieve, so every ablation row can be read as a fraction of the learnable signal rather than only against the other transformer rows.'), help='pre-registered claim this calibration supports; recorded in the report manifest')
    p.add_argument('--ablation-summaries', default=None, help='ablation all_summaries.json to annotate with the limits; defaults to the sibling of --output')
    p.add_argument('--output', default='runs/ablation/simple_baselines.json')
    args = p.parse_args()
    raw_seeds = list(args.seeds)
    if not raw_seeds:
        p.error('--seeds must not be empty')
    args.seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(args.seeds) != len(raw_seeds):
        print(f'[baseline][warn] duplicate seeds removed: {raw_seeds} -> {args.seeds} '
              '(a repeated seed reruns the identical fit, not an independent sample)')
    for flag, value in (('--batch-size', args.batch_size),
                        ('--seq-len', args.seq_len),
                        ('--train-batches', args.train_batches),
                        ('--tune-batches', args.tune_batches),
                        ('--val-batches', args.val_batches)):
        if value < 1:
            p.error(f'{flag} {value} must be >= 1')
    if args.seq_len < 2:
        # A one-token sequence has no next-token target: the synthetic
        # generator labels every position ignore_index, so no baseline/HMM
        # statistic or supervised loss can be computed.
        p.error(f'--seq-len {args.seq_len} must be >= 2 for next-token '
                'supervision')
    if args.n_states < 1:
        p.error(f'--n-states {args.n_states} must be >= 1')
    if args.max_jump < 1:
        p.error(f'--max-jump {args.max_jump} must be >= 1')
    if args.vocab_size < 1:
        p.error(f'--vocab-size {args.vocab_size} must be >= 1')
    if args.n_states > args.vocab_size:
        p.error(f'--n-states {args.n_states} cannot exceed --vocab-size {args.vocab_size}')
    if args.outlier_channel and args.n_states >= args.vocab_size:
        p.error(f'--n-states {args.n_states} must be smaller than --vocab-size {args.vocab_size} '
                'when the outlier channel is enabled (the outlier id is vocab_size - 1)')
    if not args.ks:
        p.error('--ks must not be empty')
    if any((not math.isfinite(k)) or k <= 0.0 for k in args.ks):
        p.error('--ks entries must be finite and > 0')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(args.seeds[0])
    outlier_id = args.vocab_size - 1 if args.outlier_channel else None
    memoryless_floor = memoryless_bayes_floor_nats(args.n_states, args.max_jump,
                                                   MARKOV_SPIKE_PROB if args.outlier_channel else 0.0)
    print(f"[baseline] protocol: n_states={args.n_states} max_jump={args.max_jump} "
          f"outlier_id={outlier_id} vocab={args.vocab_size} "
          f"batch={args.batch_size}x{args.seq_len} seeds={args.seeds}")
    print(f"[baseline] memoryless Bayes floor = {memoryless_floor:.4f} nats "
          f"(ppl {math.exp(memoryless_floor):.3f}); ln(vocab) strawman = "
          f"{math.log(args.vocab_size):.4f} nats; hypothesis: {args.hypothesis}")
    records = []
    for seed in args.seeds:
        record = evaluate_seed(seed, args, device)
        records.append(record)
        preds = record['predictors']
        tuned = {name: record['tuning'][name]['best_k'] for name in record['tuning']}
        trigram = ('skipped' if preds.get('trigram') is None
                   else f"{preds['trigram']:.4f}")
        print(f"[baseline] seed {seed}: uniform_vocab={preds['uniform_vocab']:.4f} "
              f"uniform_targets={preds['uniform_targets']:.4f} "
              f"unigram={preds['unigram']:.4f} bigram={preds['bigram']:.4f} "
              f"trigram={trigram} walk_rule={preds['walk_rule']:.4f} "
              f"hmm_bayes={preds['hmm_bayes']:.4f} (ks: {tuned})")
    aggregate = aggregate_records(records, ('uniform_vocab', 'uniform_targets', *SIMPLE_PREDICTORS,
                                            'walk_rule', 'hmm_bayes'))
    tuned_baseline = best_count_model(aggregate)
    calibration = build_calibration(aggregate, memoryless_floor, tuned_baseline, records)
    full_context = aggregate.get('hmm_bayes')
    if tuned_baseline:
        print(f"[baseline] tuned simple baseline: {tuned_baseline['predictor']} "
              f"{tuned_baseline['mean_ce_nats']:.4f} +/- {tuned_baseline['std_ce_nats']:.4f} nats")
    if full_context:
        print(f"[baseline] full-context causal ceiling (exact HMM filter): "
              f"{full_context['mean_ce_nats']:.4f} +/- {full_context['std_ce_nats']:.4f} nats "
              f"({calibration['full_context_minus_memoryless_nats']:+.4f} vs memoryless floor)")
    print(f"[baseline] walk rule - memoryless floor = "
          f"{calibration['rule_minus_memoryless_nats']:+.4f} nats "
          f"(tol {calibration['rule_memoryless_tolerance_nats']:.4f}) -> "
          f"rule_matches_memoryless_floor={calibration['rule_matches_memoryless_floor']}")
    if not calibration['rule_matches_memoryless_floor'] or calibration['full_context_below_memoryless_floor'] is False:
        print('[baseline] WARNING: the corpus does not behave like the assumed process; do not '
              'calibrate against these limits until the discrepancy is explained')
    ceiling = None if not full_context else full_context['mean_ce_nats']
    summaries_path = Path(args.ablation_summaries) if args.ablation_summaries else Path(args.output).with_name('all_summaries.json')
    ablation_rows = calibrate_ablation_rows(
        load_ablation_rows(summaries_path), memoryless_floor, ceiling,
        None if not tuned_baseline else tuned_baseline['mean_ce_nats'],
        math.log(args.vocab_size),
        tuned_predictor=None if not tuned_baseline else tuned_baseline['predictor'],
        records=records,
        protocol=_calibration_protocol(args, outlier_id))
    if ablation_rows:
        above = sum(1 for row in ablation_rows if row.get('beats_tuned_simple_baseline'))
        certified = sum(1 for row in ablation_rows if row.get('beats_tuned_simple_significantly'))
        mismatched = sum(1 for row in ablation_rows if row.get('protocol_matches') is False)
        print(f"[baseline] ablation rows from {summaries_path}: {above}/{len(ablation_rows)} beat the tuned "
              f"count model on the point estimate, {certified}/{len(ablation_rows)} significantly "
              f"(paired t, Holm, >=3 shared seeds); {mismatched} row(s) were excluded as "
              f"trained/validated under a different corpus protocol; every row is annotated "
              f"with excess_over_bayes_floor_nats and excess_over_bayes_ceiling_nats")
    report = {
        'hypothesis': args.hypothesis,
        'protocol': {
            'corpus': 'markov_batch', 'n_states': args.n_states, 'max_jump': args.max_jump,
            'outlier_channel': bool(args.outlier_channel), 'outlier_id': outlier_id,
            'spike_prob': MARKOV_SPIKE_PROB if args.outlier_channel else 0.0,
            'vocab_size': args.vocab_size, 'batch_size': args.batch_size, 'seq_len': args.seq_len,
            'train_batches': args.train_batches, 'tune_batches': args.tune_batches,
            'val_batches': args.val_batches, 'seeds': list(args.seeds), 'ks': list(args.ks),
            'evaluation_metric': 'mean cross-entropy in nats over unmasked next-token targets',
            'fit_stream': "torch.Generator seed s - the same training stream scripts/ablation.py optimizes on",
            'tuning_stream': 'torch.Generator seed s + 2 - separate from every reported evaluation',
            'eval_stream': "torch.Generator seed s + 1 - the same held-out stream scripts/ablation.py scores on",
        },
        'bayes_limits': {
            'memoryless_nats': memoryless_floor,
            'memoryless_perplexity': math.exp(memoryless_floor),
            'memoryless_formula': ('H(wrapped jump)' if not args.outlier_channel else
                                   '(1 - spike_prob) * H(wrapped jump) + spike_prob * ln(n_states)'),
            'without_outlier_nats': memoryless_bayes_floor_nats(args.n_states, args.max_jump, 0.0),
            'full_context_nats': ceiling,
            'full_context_stderr_nats': None if not full_context else full_context['stderr_ce_nats'],
            'full_context_definition': ('exact forward filtering of the observation-hidden Markov walk; the '
                                        'minimum achievable mean causal CE for any model that reads the observed '
                                        'prefix (the ablation transformers included)'),
            'context_gain_nats': None if ceiling is None else memoryless_floor - ceiling,
        },
        'predictors': aggregate,
        'tuned_count_baseline': tuned_baseline,
        'calibration': calibration,
        'per_seed': records,
        'ablation_calibration': {'summaries': str(summaries_path), 'rows': ablation_rows},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    manifest = collect_run_metadata(seed=args.seeds[0], extra={'hypothesis': args.hypothesis,
                                                               'seeds': list(args.seeds)})
    manifest['hyperparameters'] = vars(args)
    manifest['report_file'] = str(out_path)
    manifest['bayes_limits'] = report['bayes_limits']
    manifest['tuned_count_baseline'] = tuned_baseline
    manifest = finish_metadata(manifest, artifacts={'report': file_fingerprint(out_path)})
    save_run_manifest(out_path.with_name('simple_baselines_manifest.json'), manifest)
    print(f'[done] saved to {out_path}')

if __name__ == '__main__':
    main()
