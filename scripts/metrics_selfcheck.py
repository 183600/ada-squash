"""Independent numerical self-checks for adasquash.metrics.

Run with ``python3 scripts/metrics_selfcheck.py``. The reference calculations
below are deliberately written from their definitions (pairwise concordance,
step-wise PR area, per-class confusion sums, explicit binomial tails, ranked
gains) instead of calling the helpers they check, so a shared implementation
bug cannot make both sides agree.
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import (
    average_precision,
    benjamini_hochberg_adjust,
    chi2_sf,
    f1_from_counts,
    holm_adjust,
    macro_weighted_f1,
    mcnemar_test,
    mean_ci,
    mean_prediction_interval,
    mean_std_stderr,
    max_p_combination,
    meta_analysis,
    ndcg_at_k,
    paired_t_one_sided,
    paired_trend_test,
    ratio_effect_report,
    recall_at_k,
    roc_auc,
    student_t_cdf,
    student_t_ppf,
    tost_power,
    tost_required_n,
    welch_t_one_sided,
    wilson_interval,
)

CHECKS = 0


def close(actual, expected, tol=1e-12, label='value'):
    global CHECKS
    CHECKS += 1
    if actual is None or not math.isclose(float(actual), float(expected),
                                         rel_tol=tol, abs_tol=tol):
        raise AssertionError(f'{label}: {actual!r} != {expected!r} (tol={tol})')


def equal(actual, expected, label='value'):
    global CHECKS
    CHECKS += 1
    if actual != expected:
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')


def brute_auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def brute_ap(scores, labels):
    n_pos = sum(1 for y in labels if y)
    if n_pos == 0:
        return None
    previous_recall = 0.0
    ap = 0.0
    for threshold in sorted(set(scores), reverse=True):
        selected = [(s, y) for s, y in zip(scores, labels) if s >= threshold]
        tp = sum(1 for _, y in selected if y)
        fp = len(selected) - tp
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - previous_recall)
        previous_recall = recall
    return ap


def brute_f1(pred, gold, n_classes):
    per_class = {}
    support = len(gold)
    correct = 0
    for cls in set(pred) | set(gold):
        tp = sum(1 for p, g in zip(pred, gold) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred, gold) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred, gold) if p != cls and g == cls)
        correct += tp
        denom = 2 * tp + fp + fn
        per_class[cls] = (2 * tp / denom) if denom else 0.0
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    weighted = sum(per_class[g] for g in gold) / support if support else 0.0
    return macro, weighted, correct / support if support else 0.0


def brute_ndcg(ranked, relevant, k):
    query_scores = []
    for ranking, rel in zip(ranked, relevant):
        rel_set = set(rel)
        if not rel_set:
            query_scores.append(0.0)
            continue
        seen = set()
        dcg = 0.0
        for index, item in enumerate(ranking[:k]):
            if item in rel_set and item not in seen:
                dcg += 1.0 / math.log2(index + 2)
                seen.add(item)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel_set), k)))
        query_scores.append(min(1.0, dcg / ideal) if ideal else 0.0)
    return sum(query_scores) / len(query_scores) if query_scores else None


def brute_binomial_tail(n, at_least):
    return sum(math.comb(n, k) for k in range(at_least, n + 1)) / (2 ** n)


def check_ranking():
    rng = random.Random(0)
    for trial in range(50):
        n = rng.randint(2, 30)
        scores = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, rng.random()])
                  for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]
        if 0 < sum(labels) < n:
            close(roc_auc(scores, labels), brute_auc(scores, labels),
                  label=f'roc_auc trial {trial}')
            close(average_precision(scores, labels), brute_ap(scores, labels),
                  label=f'average_precision trial {trial}')
    close(roc_auc([1, 1, 0, 0], [1, 0, 1, 0]), 0.5, label='roc_auc all ties')
    close(average_precision([1, 1, 0, 0], [1, 0, 1, 0]), 0.5,
          label='average_precision tied group')
    equal(roc_auc([0.1, 0.2], [1, 1]), None, label='roc_auc undefined')
    equal(average_precision([0.1, 0.2], [0, 0]), None,
          label='average_precision undefined')


def check_ranking_at_k():
    ranked = [[3, 1, 2], [5, 6, 7], [], [9, 9, 8]]
    relevant = [[1, 2], [5], [4], [9, 10]]
    equal(recall_at_k(ranked, relevant, 2),
          (1 / 2 + 1 / 1 + 0.0 + 1 / 2) / 4, label='recall@2')
    equal(recall_at_k([], [], 2), None, label='recall empty')
    equal(ndcg_at_k(ranked, relevant, 2), brute_ndcg(ranked, relevant, 2),
          label='ndcg@2')
    close(ndcg_at_k([[1], [2]], [[1], [2]], 5), 1.0, label='ndcg perfect')
    equal(ndcg_at_k([[1]], [[]], 5), None, label='ndcg undefined')
    repeated = ndcg_at_k([[1, 1, 1, 2]], [[1, 2]], 4)
    close(repeated, (1.0 + 1.0 / math.log2(5)) / (1.0 + 1.0 / math.log2(3)),
          label='ndcg repeated id')


def check_f1():
    pred = [0, 0, 1, 2, 2, 2, 1, 0, 0]
    gold = [0, 1, 1, 2, 2, 0, 1, 0, 2]
    macro, weighted, micro = brute_f1(pred, gold, 3)
    report = macro_weighted_f1(pred, gold, 3)
    close(report['macro_f1'], macro, label='macro_f1')
    close(report['weighted_f1'], weighted, label='weighted_f1')
    close(report['micro_f1'], micro, label='micro_f1')
    counts = Counter(gold)
    predicted = Counter(pred)
    correct = Counter(g for p, g in zip(pred, gold) if p == g)
    from_counts = f1_from_counts(
        [counts[i] for i in range(3)],
        [predicted[i] for i in range(3)],
        [correct[i] for i in range(3)],
    )
    close(from_counts['macro_f1'], macro, label='f1_from_counts macro')
    close(from_counts['weighted_f1'], weighted, label='f1_from_counts weighted')
    close(from_counts['micro_f1'], micro, label='f1_from_counts micro')
    close(from_counts['macro_recall'], (2 / 3 + 2 / 3 + 2 / 3) / 3,
          label='macro_recall')


def check_statistics():
    low, high = wilson_interval(5, 10)
    close(low, 0.236593, tol=1e-6, label='wilson 5/10 low')
    close(high, 0.763407, tol=1e-6, label='wilson 5/10 high')
    low, high = wilson_interval(0, 10)
    close(low, 0.0, tol=1e-12, label='wilson 0/10 low')
    close(high, 0.277533, tol=1e-6, label='wilson 0/10 high')

    exact = mcnemar_test(8, 2)
    close(exact['p_one_sided'], brute_binomial_tail(10, 8),
          label='McNemar one-sided')
    close(exact['p_two_sided'], min(1.0, 2 * brute_binomial_tail(10, 8)),
          label='McNemar two-sided')
    equal(mcnemar_test(0, 0)['p_one_sided'], None, label='McNemar no discordance')

    equal(holm_adjust([0.01, 0.04, 0.03, None]), [0.03, 0.06, 0.06, None],
          label='Holm adjustment')
    equal(holm_adjust([]), [], label='Holm empty')
    equal(benjamini_hochberg_adjust([0.01, 0.04, 0.03, None]),
          [0.03, 0.04, 0.04, None], label='BH q-values')
    equal(benjamini_hochberg_adjust([0.01, 1.5, '0.01', True]),
          [0.01, None, None, None], label='BH invalid p-values')

    # TOST design power from its explicit two-tail rejection probability.
    tost_se = 0.10 / math.sqrt(12)
    tost_critical = student_t_ppf(0.95, 11)
    tost_expected = (
        student_t_cdf((0.10 - 0.0) / tost_se - tost_critical, 11)
        - student_t_cdf((-0.10 - 0.0) / tost_se + tost_critical, 11))
    close(tost_power(0.10, 12, 0.10)['approximate_power'], tost_expected,
          label='TOST design power')
    tost_n = tost_required_n(0.10, 0.10, power=0.8)
    equal(tost_power(0.10, tost_n, 0.10)['approximate_power'] >= 0.8, True,
          label='TOST required n reaches target')
    equal(tost_power(0.10, tost_n - 1, 0.10)['approximate_power'] < 0.8, True,
          label='TOST required n is minimal')

    paired = paired_t_one_sided([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    close(paired['mean_delta'], 2.0, label='paired mean')
    close(paired['t'], 2.0 / (1.0 / math.sqrt(3.0)), label='paired t')
    close(paired['df'], 2.0, label='paired df')
    close(paired['ci_low'], 2.0 - student_t_ppf(0.975, 2) / math.sqrt(3.0),
          label='paired ci low')
    close(student_t_ppf(0.95, 2), 2.919986, tol=1e-5, label='t ppf df=2')
    close(student_t_ppf(0.95, 4), 2.131847, tol=1e-5, label='t ppf df=4')

    welch = welch_t_one_sided([1.0, 2.0, 3.0], [0.0, 1.0, -1.0])
    close(welch['mean_delta'], 2.0, label='welch mean')
    close(welch['df'], 4.0, label='welch df equal variance')

    # Each seed gets its own OLS slope; the trend test averages those
    # slopes, not the six pooled (seed, dose) points.
    trend = paired_trend_test([0.0, 1.0, 2.0], {
        'a': [0.0, -1.0, -2.0],
        'b': [0.0, -1.1, -2.2],
        'c': [0.0, -0.9, -1.8],
    })
    close(trend['per_seed_slopes']['a'], -1.0, label='trend slope a')
    close(trend['per_seed_slopes']['b'], -1.1, label='trend slope b')
    close(trend['mean_slope_per_dose'], -1.0, label='trend mean slope')
    equal(trend['n_seeds'], 3, label='trend seed count')
    equal(trend['certifiable'], True, label='trend certifiable')

    stats = mean_std_stderr([1.0, 2.0, 3.0])
    close(stats['mean'], 2.0, label='mean_std_stderr mean')
    close(stats['std'], 1.0, label='mean_std_stderr std')
    close(stats['stderr'], 1.0 / math.sqrt(3.0), label='mean_std_stderr stderr')
    ci = mean_ci([1.0, 2.0, 3.0])
    close(ci['ci_low'], 2.0 - student_t_ppf(0.975, 2) / math.sqrt(3.0),
          label='mean_ci low')
    close(ci['ci_high'], 2.0 + student_t_ppf(0.975, 2) / math.sqrt(3.0),
          label='mean_ci high')

    # A prediction interval must be wider than the CI for the mean by the
    # factor sqrt(1 + 1/n); the same t critical value is used.
    pi = mean_prediction_interval([1.0, 2.0, 3.0])
    half = student_t_ppf(0.975, 2) * 1.0 * math.sqrt(1.0 + 1.0 / 3.0)
    close(pi['prediction_low'], 2.0 - half, label='prediction low')
    close(pi['prediction_high'], 2.0 + half, label='prediction high')
    equal(pi['meaningful'], True, label='prediction meaningful at n=3')
    close(pi['same_direction_probability'],
          student_t_cdf_for_check(2.0 / (1.0 * math.sqrt(1.0 + 1.0 / 3.0)), 2.0),
          tol=1e-8, label='same-direction probability')

    # The intersection-union combination is the max, not a corrected min.
    close(max_p_combination([0.01, 0.04, None]), 0.04, label='max-p combine')
    close(max_p_combination([0.4, 0.3, 0.2]), 0.4, label='max-p combine three')
    close(max_p_combination([0.9, 0.8]), 0.9, label='max-p combine two')
    equal(max_p_combination([]), None, label='max-p combine empty')
    equal(max_p_combination([None, float('nan')]), None, label='max-p combine invalid')


def student_t_cdf_for_check(t, df):
    """Independent Student-t CDF for the predictive-probability check.

    The metric module has its own CDF (used by the implementation); for df=2
    this check uses the closed form of the t CDF, which is derived from the
    density by direct integration rather than from the incomplete-beta code
    path, so a sign error in the helper cannot make both sides agree.
    """
    if df == 2.0:
        return 0.5 + t / (2.0 * math.sqrt(2.0) * math.sqrt(1.0 + t * t / 2.0))
    raise ValueError('this independent check only covers df=2')


def check_meta_analysis():
    # Known chi-square critical values at alpha = 0.05 (df = 1, 2) from the
    # closed-form chi-square CDFs, not from the helper under test.
    close(chi2_sf(3.841458820694124, 1.0), 0.05, tol=1e-12,
          label='chi2 df=1 5% point')
    close(chi2_sf(5.991464547107979, 2.0), 0.05, tol=1e-12,
          label='chi2 df=2 5% point')

    # Recompute every step of the DerSimonian-Laird pooling from its
    # definition: inverse-variance weights, Q, tau^2, then random weights.
    ys = [-1.0, 0.0, 1.0]
    ses = [0.2, 0.2, 0.2]
    out = meta_analysis([
        {'label': str(i), 'effect': y, 'stderr': se}
        for i, (y, se) in enumerate(zip(ys, ses))
    ])
    w = [1.0 / se ** 2 for se in ses]
    sum_w = sum(w)
    pooled = sum(wi * y for wi, y in zip(w, ys)) / sum_w
    q = sum(wi * (y - pooled) ** 2 for wi, y in zip(w, ys))
    df = len(ys) - 1.0
    i_squared = 100.0 * max(0.0, (q - df) / q)
    tau2 = max(0.0, (q - df) / (sum_w - sum(wi ** 2 for wi in w) / sum_w))
    wr = [1.0 / (se ** 2 + tau2) for se in ses]
    random_pooled = sum(wi * y for wi, y in zip(wr, ys)) / sum(wr)

    close(out['fixed_effect']['pooled_effect'], pooled,
          label='meta fixed pooled')
    close(out['fixed_effect']['stderr'], 1.0 / math.sqrt(sum_w),
          label='meta fixed stderr')
    close(out['heterogeneity']['q'], q, label='meta Cochran Q')
    close(out['heterogeneity']['i_squared'], i_squared,
          label='meta I squared')
    close(out['heterogeneity']['tau_squared'], tau2,
          label='meta tau squared')
    close(out['random_effects']['pooled_effect'], random_pooled,
          label='meta random pooled')
    close(out['random_effects']['stderr'], 1.0 / math.sqrt(sum(wr)),
          label='meta random stderr')
    close(out['random_effects']['ci_low'],
          random_pooled - student_t_ppf(0.975, df) / math.sqrt(sum(wr)),
          label='meta random ci low')
    equal(out['n_studies'], 3, label='meta study count')

    # A CI-only study must recover the stderr it was built from, using the
    # t critical value for that CI's own df rather than 1.96.
    se = 0.35
    half = student_t_ppf(0.975, 4.0) * se
    ci_only = meta_analysis([{
        'label': 'ci', 'effect': -0.5,
        'ci_low': -0.5 - half, 'ci_high': -0.5 + half,
        'df': 4.0, 'ci_level': 0.95,
    }])
    close(ci_only['studies'][0]['stderr'], se,
          label='meta stderr from ci')


def check_ratio_effect():
    # PPL = exp(loss); the ratio of geometric-mean PPLs is exp of the mean
    # log-ratio, so the interval endpoints must be exponentiated, not the
    # arithmetic PPL difference.
    out = ratio_effect_report(-0.2, -0.3, -0.1)
    close(out['ratio'], math.exp(-0.2), label='ratio effect point')
    close(out['ratio_ci'][0], math.exp(-0.3), label='ratio effect low')
    close(out['ratio_ci'][1], math.exp(-0.1), label='ratio effect high')
    close(out['percent_change'], 100.0 * (math.exp(-0.2) - 1.0),
          label='ratio effect percent')
    close(out['percent_change_ci'][0], 100.0 * (math.exp(-0.3) - 1.0),
          label='ratio effect percent low')
    close(out['percent_change_ci'][1], 100.0 * (math.exp(-0.1) - 1.0),
          label='ratio effect percent high')


def main():
    check_ranking()
    check_ranking_at_k()
    check_f1()
    check_statistics()
    check_meta_analysis()
    check_ratio_effect()
    print(f'[metrics-selfcheck] ok: {CHECKS} independent numerical checks passed')


if __name__ == '__main__':
    main()
