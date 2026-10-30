from .norms import (LayerNorm, RMSNorm, DyT, DyTS, DyTP, DyTC, DyTD, DyTDTanh,
                    DyTDI, DyTCD, DyTCDTanh, DyTCDI, build_norm,
                    generalized_softsign, NORM_REGISTRY)
from .attention import CausalSelfAttention
from .model import GPTConfig, GPT, LayerConfig, build_gpt_124m, build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache
from .utils import ActivationStats, LossSpikeCounter, KurtosisMeter, MaxAbsMeter, measure_activation_stats, markov_batch, count_parameters
from .diagnostics import diagnose_train_val_curve, triage_nonfinite, classify_nan_stage, tensor_finite_report, NAN_TRIAGE_ORDER
from .efficiency import analytic_flops_per_token, efficiency_verdict, parameter_breakdown
from .repro import (set_all_seeds, seed_worker, collect_run_metadata, save_run_manifest,
                    environment_info, apply_deterministic_algorithms,
                    deterministic_algorithms_context, TrainStreamFingerprint,
                    compare_train_streams, execution_identity_mismatches,
                    EXECUTION_IDENTITY_KEYS, RUN_MANIFEST_SCHEMA)
from .design import (block_randomized_schedule, randomized_complete_block_schedule,
                     schedule_balance_report, schedule_fingerprint_status,
                     order_drift_report,
                     default_exclusion_policy, normalize_exclusion_policy,
                     exclusion_policy_fingerprint, classify_run_exclusions,
                     run_exclusion_audit,
                     design_readiness_audit, prospective_power_plan,
                     tost_power_plan,
                     simulate_paired_design)
from .decontamination import (DEFAULT_NGRAM_SIZE, NgramIndex, build_ngram_index,
                              find_ngram_matches, scan_token_documents,
                              contamination_scan_bound, contamination_protocol,
                              contamination_ledger_hashes)
__version__ = '0.1.0'
__all__ = ['LayerNorm', 'RMSNorm', 'DyT', 'DyTC', 'DyTD', 'DyTDTanh', 'DyTCD', 'DyTCDTanh', 'build_norm', 'generalized_softsign', 'NORM_REGISTRY', 'CausalSelfAttention', 'GPTConfig', 'GPT', 'LayerConfig', 'build_gpt_124m', 'build_gpt_350m', 'build_llama_tiny_1b', 'clear_ce_scratch_cache', 'ActivationStats', 'LossSpikeCounter', 'KurtosisMeter', 'MaxAbsMeter', 'measure_activation_stats', 'markov_batch', 'count_parameters', 'diagnose_train_val_curve', 'triage_nonfinite', 'classify_nan_stage', 'tensor_finite_report', 'NAN_TRIAGE_ORDER', 'analytic_flops_per_token', 'efficiency_verdict', 'parameter_breakdown', 'set_all_seeds', 'seed_worker', 'collect_run_metadata', 'save_run_manifest', 'environment_info', 'apply_deterministic_algorithms', 'deterministic_algorithms_context', 'TrainStreamFingerprint', 'compare_train_streams', 'execution_identity_mismatches', 'EXECUTION_IDENTITY_KEYS', 'RUN_MANIFEST_SCHEMA', 'block_randomized_schedule', 'randomized_complete_block_schedule', 'schedule_balance_report', 'schedule_fingerprint_status', 'order_drift_report', 'default_exclusion_policy', 'normalize_exclusion_policy', 'exclusion_policy_fingerprint', 'classify_run_exclusions', 'run_exclusion_audit', 'design_readiness_audit', 'prospective_power_plan', 'tost_power_plan', 'simulate_paired_design', 'DEFAULT_NGRAM_SIZE', 'NgramIndex', 'build_ngram_index', 'find_ngram_matches', 'scan_token_documents', 'contamination_scan_bound', 'contamination_protocol', 'contamination_ledger_hashes', 'DyTS', 'DyTP', 'DyTDI', 'DyTCDI']
