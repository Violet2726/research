from __future__ import annotations

from dataclasses import replace

import pytest

from research_experiments.families.cred_v.config import (
    CRED_ACTIVE_METHODS,
    CRED_LEGACY_METHODS,
    load_experiment_config,
    load_protocol_config,
    validate_experiment_protocol_contract,
)
from research_experiments.families.registry import registered_family_names


def test_cred_v_main_config_loads() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_output_protocol == "free_text_answer_v1"
    assert experiment.cred_stage_a_output_protocol == "free_text_answer_v1"
    assert experiment.cred_verification_output_protocol == "json_object_answer_v3"
    assert experiment.cred_methods == ["cred_rfs_vote_5_anchor", "cred_rfs_repair_only_v6"]
    assert experiment.verifier_model_refs == ["xiaomimimo/mimo-v2.5-pro"]
    assert protocol.stage_a_prompt_mode == "sc5_anchor_free_text_v1"
    assert protocol.max_verifications == 1
    assert protocol.max_verification_calls == 1
    assert protocol.verification_modes == ("deterministic_repair", "tool_verified", "hetero_verified")
    assert protocol.selection_modes == (
        "deterministic_repair",
        "math_deterministic_repair",
        "hotpot_context_span_repair",
    )
    assert protocol.expansion_modes == ()
    assert protocol.disabled_selection_modes == (
        "gpqa_unanimous_pairwise_duel",
        "mmlu_pairwise_promotion",
        "strategyqa_minority_resample",
        "pairwise_2of3_promotion",
        "math_equivalence_repair_v2",
    )
    assert protocol.disabled_expansion_modes == (
        "strategyqa_dual_polarity",
        "hotpot_span_extract",
        "rfs_extra_solver",
        "mc_choice_shuffle",
        "strategyqa_minority_resample",
        "mc_blind_pairwise_duel",
        "gpqa_unanimous_pairwise_duel",
    )
    assert protocol.expansion_model_refs == ()
    assert protocol.max_expansion_calls == 0
    assert protocol.adaptive_extra_solver_calls == 0
    assert protocol.max_total_solver_calls == 5
    assert protocol.promotion_min_independent_support == 3
    assert protocol.promotion_margin_min == 1.25
    assert protocol.mc_shuffle_min_agreement == 3
    assert protocol.pairwise_allowed_datasets == ()
    assert protocol.pairwise_option_count_max == 0
    assert protocol.pairwise_duel_replicates == 0
    assert protocol.pairwise_promotion_min_wins == 0
    assert protocol.require_stage_a_challenger_support is True
    assert protocol.new_candidate_policy == "block_semantic_promotion"
    assert protocol.allow_single_verifier_promotion is False
    assert protocol.allow_strong_majority_pairwise_promotion is False
    assert protocol.allow_semantic_promotion is False
    assert protocol.false_consensus_probe is False
    assert protocol.max_trigger_rate == 0.20
    assert protocol.trigger_buckets == ("weak_split", "deterministic_repair_only")
    assert protocol.allow_same_model_promotion is False
    assert protocol.leader_lock_count == 4
    assert protocol.promotion_score_margin == 0.15
    assert protocol.stage_a_max_tokens == 0
    assert protocol.verifier_max_tokens == 0


def test_cred_cvs_v1_config_loads_certificate_contract() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_cvs_v1.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.control_methods == ["sc_5"]
    assert experiment.cred_methods == [
        "cred_rfs_vote_5_anchor",
        "cred_rfs_repair_only_v6",
        "cred_cvs_budget_matched_vote_v1",
        "cred_cvs_v1",
        "cred_isp_shadow_v1",
    ]
    assert experiment.verifier_model_refs == ["xiaomimimo/mimo-v2.5-pro", "dashscope/qwen-flash"]
    assert experiment.max_concurrent_requests == 1000
    assert experiment.requests_per_minute_limit == 1000
    assert protocol.stage_a_prompt_mode == "sc5_anchor_free_text_v1"
    assert protocol.certificate_modes == ("math_symbolic", "hotpot_context_span", "mc_option_mapping")
    assert protocol.certificate_proposer_model_refs == ("xiaomimimo/mimo-v2.5-pro", "dashscope/qwen-flash")
    assert protocol.certificate_dsl_version == "math_cert_v1_question_bound"
    assert protocol.max_certificate_calls == 2
    assert protocol.certificate_min_independent_support == 2
    assert protocol.allow_unverified_promotion is False
    assert protocol.shadow_aggregation_modes == ("isp",)
    assert protocol.max_trigger_rate == 0.30
    validate_experiment_protocol_contract(experiment, protocol)


def test_cred_cvs_contract_rejects_unconfigured_or_insufficient_proposers() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_cvs_v1.toml")
    protocol = load_protocol_config(experiment.protocol)

    with pytest.raises(ValueError, match="present in verifier_model_refs"):
        validate_experiment_protocol_contract(
            replace(experiment, verifier_model_refs=["xiaomimimo/mimo-v2.5-pro"]),
            protocol,
        )
    with pytest.raises(ValueError, match="enough distinct proposer models"):
        validate_experiment_protocol_contract(
            experiment,
            replace(protocol, certificate_proposer_model_refs=("xiaomimimo/mimo-v2.5-pro",)),
        )


def test_cred_v_rfs_v3_pairwise_ablation_config_retains_retired_selector() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v3_pairwise_ablation.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == ["cred_rfs_vote_5_anchor", "cred_rfs_safe_select_v3"]
    assert "ablation" in experiment.description
    assert protocol.stage_a_prompt_mode == "sc5_anchor_free_text_v1"
    assert protocol.selection_modes == (
        "deterministic_repair",
        "gpqa_unanimous_pairwise_duel",
        "hotpot_context_span_repair",
    )
    assert protocol.expansion_modes == ("gpqa_unanimous_pairwise_duel",)
    assert protocol.pairwise_allowed_datasets == ("gpqa_diamond",)
    assert protocol.pairwise_promotion_min_wins == 3


def test_cred_v_rfs_v7_shadow_evidence_select_config_is_shadow_only() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v7_shadow_evidence_select.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == ["cred_rfs_vote_5_anchor", "cred_rfs_repair_only_v6", "cred_rfs_shadow_evidence_select_v7"]
    assert protocol.selection_modes == (
        "deterministic_repair",
        "math_deterministic_repair",
        "hotpot_context_span_repair",
    )
    assert protocol.expansion_modes == ()
    assert protocol.shadow_selection_modes == (
        "direct_option_contrast_shadow",
        "constraint_elimination_shadow",
        "minimal_evidence_certificate_shadow",
        "strategyqa_resample_shadow",
    )
    assert "gpqa_unanimous_pairwise_duel" in protocol.disabled_selection_modes
    assert protocol.shadow_pairwise_allowed_datasets == ("gpqa_diamond", "mmlu_pro")
    assert protocol.shadow_gate_min_valid_duels == 3
    assert protocol.shadow_gate_min_wins == 3
    assert protocol.allow_semantic_promotion is False


def test_cred_v_rfs_v8_repair_bank_config_is_forward_repair_only() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v8_repair_bank.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == ["cred_rfs_vote_5_anchor", "cred_rfs_repair_only_v6", "cred_rfs_repair_bank_v8"]
    assert protocol.stage_a_prompt_mode == "sc5_anchor_free_text_v1"
    assert protocol.selection_modes == (
        "deterministic_repair",
        "math_deterministic_repair",
        "math_repair_bank_v8",
        "hotpot_context_span_repair",
        "mc_option_text_repair",
    )
    assert protocol.expansion_modes == ()
    assert protocol.shadow_selection_modes == ()
    assert "gpqa_unanimous_pairwise_duel" in protocol.disabled_selection_modes
    assert "math_equivalence_repair_v2" in protocol.disabled_selection_modes
    assert protocol.allow_semantic_promotion is False
    assert protocol.max_expansion_calls == 0


def test_cred_v_rfs_v9_certificate_shadow_config_is_shadow_only() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v9_certificate_shadow.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == [
        "cred_rfs_vote_5_anchor",
        "cred_rfs_repair_only_v6",
        "cred_rfs_repair_bank_v8",
        "cred_rfs_certificate_shadow_v9",
    ]
    assert "mc_option_text_repair" in protocol.selection_modes
    assert protocol.shadow_selection_modes == (
        "direct_option_contrast_shadow",
        "constraint_elimination_shadow",
        "minimal_evidence_certificate_shadow",
        "strategyqa_resample_shadow",
    )
    assert protocol.new_candidate_policy == "certificate_shadow_only"
    assert protocol.shadow_precision_threshold == 0.90
    assert protocol.shadow_harm_per_correction_max == 0.10


def test_cred_v_rfs_v2_ablation_config_retains_pairwise_failure_baseline() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v2_ablation.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == ["cred_rfs_vote_5_anchor", "cred_rfs_pairwise_select_v2"]
    assert protocol.stage_a_prompt_mode == "sc5_anchor_free_text_v1"
    assert protocol.selection_modes == (
        "deterministic_repair",
        "mc_blind_pairwise_duel",
        "strategyqa_minority_resample",
        "hotpot_context_span_repair",
    )
    assert protocol.pairwise_promotion_min_wins == 2
    assert protocol.false_consensus_probe is True


def test_cred_v_rfs_v4_shadow_config_loads_shadow_modes() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v4_shadow.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == ["cred_rfs_vote_5_anchor", "cred_rfs_safe_select_v3", "cred_rfs_shadow_select_v4"]
    assert protocol.selection_modes == (
        "deterministic_repair",
        "gpqa_unanimous_pairwise_duel",
        "hotpot_context_span_repair",
    )
    assert protocol.shadow_selection_modes == (
        "gpqa_2of3_retry_shadow",
        "mmlu_unanimous_pairwise_shadow",
        "strategyqa_resample_shadow",
    )
    assert protocol.shadow_pairwise_allowed_datasets == ("gpqa_diamond", "mmlu_pro")
    assert protocol.shadow_pairwise_retry_replicates == 2
    assert protocol.shadow_gate_min_valid_duels == 5
    assert protocol.shadow_gate_min_wins == 4
    assert protocol.shadow_precision_threshold == 0.85
    assert protocol.shadow_harm_per_correction_max == 0.20


def test_cred_v_rfs_v5_evidence_repair_config_loads_without_negative_shadow_branches() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v5_evidence_repair.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == ["cred_rfs_vote_5_anchor", "cred_rfs_safe_select_v3", "cred_rfs_evidence_repair_v5"]
    assert "negative ablation" in experiment.description
    assert protocol.stage_a_prompt_mode == "sc5_anchor_free_text_v1"
    assert protocol.selection_modes == (
        "deterministic_repair",
        "hotpot_context_span_repair_v2",
        "gpqa_unanimous_pairwise_duel",
    )
    assert protocol.expansion_modes == ("gpqa_unanimous_pairwise_duel",)
    assert protocol.shadow_selection_modes == ()
    assert "math_equivalence_repair_v2" in protocol.disabled_selection_modes
    assert "mmlu_unanimous_pairwise_shadow" in protocol.disabled_selection_modes
    assert "gpqa_2of3_retry_shadow" in protocol.disabled_selection_modes
    assert "strategyqa_minority_resample" in protocol.disabled_selection_modes
    assert protocol.pairwise_allowed_datasets == ("gpqa_diamond",)
    assert protocol.pairwise_promotion_min_wins == 3
    assert protocol.require_stage_a_challenger_support is True
    assert protocol.allow_strong_majority_pairwise_promotion is False


def test_cred_v_rfs_v1_ablation_config_retains_role_guided_stage_a() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v1_ablation.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_methods == ["cred_rfs_vote_5", "cred_rfs_adaptive_sc_v1"]
    assert protocol.stage_a_prompt_mode == "reasoning_first_roles_v1"
    assert protocol.expansion_modes == ("mc_choice_shuffle",)


def test_cred_v_legacy_config_retains_old_baselines() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_legacy.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_stage_a_output_protocol == "json_object_answer_v3"
    assert experiment.cred_methods == ["cred_v_vote_5", "cred_v_task_verify_v3", "cred_verify_safe_v1", "cred_acs_v1"]
    assert protocol.expansion_modes == ("math_symbolic_repair", "hotpot_span_extract", "mc_choice_shuffle", "strategyqa_dual_polarity")


def test_cred_v_family_is_registered() -> None:
    assert "cred_v" in registered_family_names()


def test_active_registry_contains_only_v6_and_cvs_forward_methods() -> None:
    assert {
        "cred_rfs_vote_5_anchor",
        "cred_rfs_repair_only_v6",
        "cred_cvs_budget_matched_vote_v1",
        "cred_cvs_v1",
        "cred_isp_shadow_v1",
    } == CRED_ACTIVE_METHODS
    assert "cred_rfs_safe_select_v3" in CRED_LEGACY_METHODS
    assert "cred_rfs_certificate_shadow_v9" in CRED_LEGACY_METHODS


def test_legacy_experiment_is_explicitly_marked() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_rfs_v9_certificate_shadow.toml")

    assert experiment.legacy_experiment is True
