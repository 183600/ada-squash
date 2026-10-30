from __future__ import annotations
"""Uncertainty-aware evaluation metrics shared by the experiment scripts.

Why this module exists
----------------------
An experiment must not decide "variant X beats the baseline" on a bare point
estimate. This module keeps the three comparisons the repo actually makes
honest:

* a held-out loss is a *sample mean*: `mean_std_stderr` reports its standard
  error so a delta smaller than the noise is visible as such;
* two arms evaluated on the *same* held-out batches are paired samples, so
  `paired_t_one_sided` (with `welch_t_one_sided` as the unpaired fallback) is
  the test used for "better by more than `margin`". The repo replicates most
  comparisons over only 2-4 seeds, so the Student-t reference distribution
  matters: at n = 3 the one-sided 5% critical value is t = 2.92, while the
  normal approximation (`paired_z_one_sided` / `welch_z_one_sided`, kept for
  large-n callers) would use z = 1.64 and declare wins from noise. The two
  t-tests also report a two-sided confidence interval for the mean difference,
  so a point estimate without a plausible range is never the whole story;
* a pass/fail rate over independent trials (needle-in-a-haystack) gets a
  Wilson interval via `wilson_interval`, which stays inside [0, 1] and is not
  degenerate at 0/1 like the normal approximation;
* token classification over a large vocabulary is evaluated with
  `macro_weighted_f1` / `f1_from_counts`, which *name* the averaging they
  use: `macro_f1` (unweighted mean over classes present), `weighted_f1`
  (support-weighted) and `micro_f1` (= accuracy). Macro and micro can differ
  wildly under class imbalance, so the caller must state which one it
  reports;
* class-imbalanced binary/multi-label decisions are scored with
  `average_precision` (PR-AUC) and `roc_auc`, and `binary_auc_report` names
  the macro/weighted averaging and reports the negatives-per-positive
  imbalance. This matters because at >= 1000 negatives per positive the ROC
  curve's false-positive rate (denominator: all negatives) grows so slowly
  that `roc_auc` stays optimistic while precision has already collapsed;
  in that regime read PR-AUC (`average_precision`), not ROC-AUC;
  `imbalance_report` flags exactly that regime for a target distribution;
* a scan that tests many rows against one reference gets `holm_adjust`, so a
  row-by-row alpha is corrected for the family of tests actually performed
  (Holm-Bonferroni), instead of reporting the best of many uncorrected draws.
  `benjamini_hochberg_adjust` supplies the complementary FDR-controlled
  q-values for exploratory screens, where the goal is a stated false-discovery
  rate rather than a family-wise guarantee;
* a claim that two arms are *the same* is an equivalence claim, not a
  non-significant superiority test: `tost_paired` runs the two one-sided
  tests and reports the matching interval. A small-seed effect is also
  checked for being carried by a single observation with
  `leave_one_out_jackknife`, and a factorial interaction is evaluated
  two-sided (`two_sided_p_from_t`) because antagonism is a finding too.
  `paired_randomization_test` adds the exact sign-flip counterpart of the
  t-test: it makes no normality assumption and reports the smallest p-value
  attainable at the achieved seed count, so a 3-seed t-test's reliance on its
  parametric model is visible instead of implicit;
* a dose-response claim ("the advantage grows with depth") is tested as a
  trend, not inferred from a row of significant per-depth deltas:
  `paired_trend_test` fits one slope per seed on the paired deltas and tests
  the mean slope one-sided, so repeated depths within a seed are not
  pseudo-replicated and a flat advantage is not mistaken for growth;
* "the effect replicates across model scales" is not a sign vote. Each scale
  estimates the same contrast with a different precision, and the scales can
  disagree by more than their own standard errors. `meta_analysis` pools the
  per-scale effects by inverse variance (fixed effect), adds a
  DerSimonian-Laird random-effects estimate, and reports Cochran's Q, I^2 and
  tau^2 together with a prediction interval for a *new* scale. A pooled
  number under large heterogeneity is an average of unlike effects, not
  evidence of one replicated mechanism.

Everything is pure `math` (no scipy/sklearn), so the training scripts can run
in a minimal environment.
"""
import functools
import math
import numbers
import random
from typing import Optional, Sequence

__all__ = [
    "average_precision",
    "benjamini_hochberg_adjust",
    "binary_auc_report",
    "chi2_sf",
    "f1_from_counts",
    "holm_adjust",
    "imbalance_report",
    "leave_one_out_jackknife",
    "macro_weighted_f1",
    "MIN_SEEDS_FOR_SIGNIFICANCE",
    "mcnemar_test",
    "mean_ci",
    "mean_prediction_interval",
    "mean_std_stderr",
    "meta_analysis",
    "max_p_combination",
    "paired_design_summary",
    "paired_randomization_test",
    "paired_required_n",
    "paired_t_one_sided",
    "paired_trend_test",
    "paired_z_one_sided",
    "roc_auc",
    "scale_replication_summary",
    "student_t_cdf",
    "student_t_ppf",
    "tost_design_summary",
    "tost_paired",
    "tost_power",
    "tost_required_n",
    "two_sided_p_from_t",
    "welch_t_one_sided",
    "welch_z_one_sided",
    "wilson_interval",
    "recall_at_k",
    "ndcg_at_k",
]


from .metrics_stats import (MIN_SEEDS_FOR_SIGNIFICANCE,
    _EXTREME_IMBALANCE,
    _RANDOMIZATION_EXACT_MAX,
    _Z_95,
    _as_real,
    _betacf,
    _binary_labels,
    _gamma_p_series,
    _gamma_q_cf,
    _is_boolean_value,
    _p_one_sided_from_z,
    _paired_mde,
    _reg_incomplete_beta,
    _strict_int,
    _validate_alpha,
    _validate_design_inputs,
    benjamini_hochberg_adjust,
    chi2_sf,
    holm_adjust,
    leave_one_out_jackknife,
    max_p_combination,
    mcnemar_test,
    mean_ci,
    mean_prediction_interval,
    mean_std_stderr,
    paired_design_summary,
    paired_required_n,
    paired_t_one_sided,
    paired_z_one_sided,
    student_t_cdf,
    student_t_ppf,
    tost_design_summary,
    tost_paired,
    tost_power,
    tost_required_n,
    two_sided_p_from_t,
    wilson_interval,
)
from .metrics_meta import (_META_EFFECT_KEYS,
    _META_STDERR_KEYS,
    _meta_influence,
    _meta_study,
    _ols_slope_and_r2,
    _pool_meta_studies,
    meta_analysis,
    paired_randomization_test,
    paired_trend_test,
    scale_replication_summary,
    welch_t_one_sided,
    welch_z_one_sided,
)
from .metrics_classification import (_average_precision_from_groups,
    _roc_auc_from_groups,
    _tie_groups_ascending,
    average_precision,
    binary_auc_report,
    f1_from_counts,
    imbalance_report,
    macro_weighted_f1,
    ndcg_at_k,
    recall_at_k,
    roc_auc,
)
