from .norms import LayerNorm, RMSNorm, DyT, DyTC, DyTD, DyTDTanh, DyTCD, DyTCDTanh, build_norm, generalized_softsign, NORM_REGISTRY
from .attention import CausalSelfAttention
from .model import GPTConfig, GPT, LayerConfig, build_gpt_124m, build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache
from .utils import ActivationStats, LossSpikeCounter, KurtosisMeter, MaxAbsMeter, measure_activation_stats, markov_batch, count_parameters
from .diagnostics import diagnose_train_val_curve, triage_nonfinite, classify_nan_stage, tensor_finite_report, NAN_TRIAGE_ORDER
from .efficiency import analytic_flops_per_token, efficiency_verdict, parameter_breakdown
from .repro import (set_all_seeds, seed_worker, collect_run_metadata, save_run_manifest,
                    environment_info, apply_deterministic_algorithms,
                    deterministic_algorithms_context)
from .design import (block_randomized_schedule, randomized_complete_block_schedule,
                     schedule_balance_report, order_drift_report,
                     default_exclusion_policy, normalize_exclusion_policy,
                     exclusion_policy_fingerprint, classify_run_exclusions,
                     run_exclusion_audit,
                     design_readiness_audit, simulate_paired_design)
__version__ = '0.1.0'
__all__ = ['LayerNorm', 'RMSNorm', 'DyT', 'DyTC', 'DyTD', 'DyTDTanh', 'DyTCD', 'DyTCDTanh', 'build_norm', 'generalized_softsign', 'NORM_REGISTRY', 'CausalSelfAttention', 'GPTConfig', 'GPT', 'LayerConfig', 'build_gpt_124m', 'build_gpt_350m', 'build_llama_tiny_1b', 'clear_ce_scratch_cache', 'ActivationStats', 'LossSpikeCounter', 'KurtosisMeter', 'MaxAbsMeter', 'measure_activation_stats', 'markov_batch', 'count_parameters', 'diagnose_train_val_curve', 'triage_nonfinite', 'classify_nan_stage', 'tensor_finite_report', 'NAN_TRIAGE_ORDER', 'analytic_flops_per_token', 'efficiency_verdict', 'parameter_breakdown', 'set_all_seeds', 'seed_worker', 'collect_run_metadata', 'save_run_manifest', 'environment_info', 'apply_deterministic_algorithms', 'deterministic_algorithms_context', 'block_randomized_schedule', 'randomized_complete_block_schedule', 'schedule_balance_report', 'order_drift_report', 'default_exclusion_policy', 'normalize_exclusion_policy', 'exclusion_policy_fingerprint', 'classify_run_exclusions', 'run_exclusion_audit', 'design_readiness_audit', 'simulate_paired_design']
