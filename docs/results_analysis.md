# 结果分析说明：复制区间与参照选择

> 中文摘要：本仓库的比较脚本回答两个结果分析问题。(1) 置信区间描述的是
> **均值**，不是下一次独立训练会落在哪里；因此新增 95% 预测区间与“同方向
> 概率”。(2) 默认参照通常是**看完 loss 后**选出的最好臂；此时只报告
> “候选 vs 最好臂”的条件比较会掩盖选择本身。新实现要求候选在
> 选择池内**击败每一个成员**（intersection-union max-p），并把合并后的
> p 值送入 Holm 校正。两者都不改变原有显著性检验的方向，只是把结论的
> 适用范围写清楚。

This note documents the results-analysis layer added to `adasquash.metrics`,
`scripts/compare_arms.py` and `scripts/ablation.py`. It is deliberately
separate from the fixed design decisions in `docs/design_notes.md`.

## 1. Confidence interval is not a replication interval

With `n` seed-level observations the comparison reports two different
intervals:

* **CI for the mean** (`delta_ci_95`, from `mean_ci` / `paired_t_one_sided`):
  `mean +/- t_{1-alpha/2, n-1} * s / sqrt(n)`. It narrows as `n` grows and
  answers "where is the population mean?".
* **Prediction interval for the next run** (`delta_prediction_interval_95`,
  from `mean_prediction_interval`):
  `mean +/- t_{1-alpha/2, n-1} * s * sqrt(1 + 1/n)`. It converges to the
  run-to-run spread `s`, not to zero, and answers "where can a new independent
  seed still land?".

At `n = 3` the prediction interval is `sqrt(4/3) ~= 1.155x` wider than the CI
in half-width, before the different reading is taken into account. A delta
whose CI excludes zero but whose PI covers zero is a mean-level result that
has not demonstrated seed-level reproducibility.

`mean_prediction_interval` also reports
`same_direction_probability = P(T_{n-1} <= |mean| / (s * sqrt(1 + 1/n)))`.
This is a **model-based predictive probability** under exchangeable,
approximately normal seed measurements. It is descriptive: it is not a
p-value, it is not corrected for multiplicity, and it does not replace the
pre-specified one-sided test. With `n < 3` the interval is returned but
marked `meaningful = False`; with `n < 2` no interval is invented.

Where it appears:

| Artifact | Fields |
|---|---|
| `scripts/compare_arms.py` row | `delta_prediction_interval_95`, `delta_same_direction_probability` |
| `scripts/compare_sealed_tests.py` seed-paired row | same two fields |
| `scripts/ablation.py` baseline row | same two fields (via `_paired_effect_summary`) |
| `paired_trend_test` (depth sweep) | `slope_prediction_interval_95`, `slope_same_direction_probability` |

## 2. A data-selected reference is a result-analysis decision

Both comparison paths choose the reference/baseline **after** seeing the
losses:

* `compare_arms` with no explicit reference takes the best replicated arm;
* `baseline_comparison` in `scripts/ablation.py` takes the lowest-loss
  replicated baseline.

This is a sensible rule for *ranking*, but it makes the reported comparison
conditional on which member happened to look best. If the claim is "the
candidate is better than the baselines", the candidate has to beat **every**
eligible baseline, not only the one that won the selection. The new logic
therefore treats the reference pool as a family of one-sided comparisons and
combines them with an intersection-union test:

```
p_selection = max(p_1, ..., p_m)
```

The IUT rejects only when every member is beaten, needs no independence
assumption, and is valid even though the pool members are correlated through
shared seeds. It is conservative by construction; that is the correct
direction for a confirmatory claim built on a data-selected comparator.
Using the *minimum* p would answer "the candidate beats at least one pool
member" and can certify a candidate that loses to the selected reference.

A worked example (synthetic, three seeds each):

| arm | mean loss |
|---|---|
| weak baseline | 2.00 |
| strong baseline | 1.90 |
| candidate | 1.95 |

If the reference were fixed to the weak baseline, the candidate would look
like a significant win. With the pool `{weak, strong}`, the candidate loses to
the strong baseline, so `p_selection = 1` and the claim is not certified.
`beats_every_pool_member` and `delta_vs_pool_worst_case` make that visible
without reading the individual contrasts.

### Wiring

`compare_arms(..., reference_selection=..., reference_pool=...)`:

* an explicit `reference` is treated as pre-specified by default;
* `reference_selection='best-of-pool'` declares a data-selected reference;
* `reference_pool` names the members it was selected from (the reference must
  be one of them). Without an explicit reference, the pool is the replicated
  arms that share the reference's controlled protocol.

`baseline_comparison` always selects the reference from its replicated
baseline pool, so it always applies the same IUT and records
`reference_selection.data_dependent = True`.

Per-row fields:

| Field | Meaning |
|---|---|
| `p_one_sided` | raw paired test against the selected reference |
| `p_one_sided_selection_adjusted` | max-p over the eligible pool (IUT) |
| `p_inferential` | the p used for the Holm family and certification |
| `selection_adjustment_complete` | every eligible pool member had a usable paired test |
| `pool_contrasts` | per-member delta, p, shared-seed count and comparability |
| `beats_every_pool_member` | point-estimate check that all deltas are `< -margin` |
| `delta_vs_pool_worst_case` | largest candidate-minus-baseline delta (smallest advantage) |

If the pool cannot be tested completely, `p_inferential` is `None` and the row
fails closed: a partial pool must not shrink the claim to the members that
happened to be convenient.

### CLI

```bash
# Reference fixed before the study (default for an explicit --reference):
python scripts/compare_arms.py runs/*/aggregate.json \
    --names ln rmsnorm dytcd --reference ln

# Reference chosen from a declared pool after seeing the results:
python scripts/compare_arms.py runs/*/aggregate.json \
    --names ln rmsnorm dytcd --reference rmsnorm \
    --reference-selection best-of-pool --reference-pool ln rmsnorm
```

The run schedule / exclusion policy / analysis-plan fingerprint remain the
place to freeze *which* reference pool was eligible. This layer only makes
the consequence of the selection explicit in the comparison artifact.

## 3. What this does not fix

The correction covers one specific researcher degree of freedom: choosing the
comparator after seeing outcomes. It does **not** cover:

* metric shopping between `final_val_loss`, `best_val_loss` and `test_loss`
  (choose the primary metric before the run; `best_val_loss` is a
  selection statistic and is already labelled as such);
* optional stopping while runs arrive (the pre-registered seed count and
  fixed-step analysis remain the guard);
* endpoint multiplicity across downstream tasks, FP8, W8A8 and MoE reports
  (the study-level/scale-level gates handle their own families);
* non-exchangeable seeds (different data order or hardware in a way that
  changes the estimand);
* equivalence claims (TOST still needs a pre-specified margin and is not
  affected by the reference-selection IUT).

Prediction intervals are a description of replication variability, not a
guarantee that the next run will be accepted by a significance test. A result
whose PI crosses zero should be reported as "mean-level evidence, not yet
seed-replicated", not as a confirmed effect.

## 4. Reporting checklist for a comparison artifact

Before writing "the candidate is better", check the artifact for all of:

1. **N and unit** - the number of shared *training seeds*, not the number of
   validation batches. Batch-level tests are descriptive under this protocol.
2. **CI and PI** - the mean-level interval (`delta_ci_95`) and the
   next-seed interval (`delta_prediction_interval_95`). If the PI covers
   zero, say "seed-level replication not yet demonstrated".
3. **Raw and inferential p** - `p_one_sided` and `p_inferential` should both
   be reported when they differ; `p_one_sided_holm` must be the value used
   for the claim.
4. **Reference provenance** - if `reference_selection.data_dependent` is
   true, report the pool and `beats_every_pool_member` /
   `delta_vs_pool_worst_case`; do not quote only the best-looking contrast.
5. **Equivalence and superiority are different** - a non-significant
   superiority test is not equivalence; report the TOST margin and p_tost
   when making a sameness claim.
6. **Protocol audit** - shape, final step, data version, ledger and training
   recipe must be identical; a descriptive-only row cannot become a
   certified claim by being the smallest p-value in the table.
7. **Multiplicity** - state the family (candidates, endpoints, depths,
   scales). A p-value outside the pre-specified family is exploratory.
