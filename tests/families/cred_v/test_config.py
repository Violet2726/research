from __future__ import annotations

from research_experiments.families.cred_v.config import load_experiment_config, load_protocol_config
from research_experiments.families.registry import registered_family_names


def test_cred_v_main_config_loads() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_output_protocol == "free_text_answer_v1"
    assert experiment.cred_stage_a_output_protocol == "free_text_answer_v1"
    assert experiment.cred_verification_output_protocol == "json_object_answer_v3"
    assert experiment.cred_methods == ["cred_rfs_vote_5", "cred_rfs_adaptive_sc_v1"]
    assert experiment.verifier_model_refs == ["xiaomimimo/mimo-v2.5-pro"]
    assert protocol.stage_a_prompt_mode == "reasoning_first_roles_v1"
    assert protocol.max_verifications == 1
    assert protocol.max_verification_calls == 1
    assert protocol.verification_modes == ("deterministic_repair", "tool_verified", "hetero_verified")
    assert protocol.expansion_modes == ("mc_choice_shuffle",)
    assert protocol.disabled_expansion_modes == ("strategyqa_dual_polarity", "hotpot_span_extract")
    assert protocol.expansion_model_refs == ("xiaomimimo/mimo-v2.5-pro",)
    assert protocol.max_expansion_calls == 3
    assert protocol.adaptive_extra_solver_calls == 2
    assert protocol.max_total_solver_calls == 7
    assert protocol.promotion_min_independent_support == 2
    assert protocol.promotion_margin_min == 1.25
    assert protocol.mc_shuffle_min_agreement == 2
    assert protocol.require_stage_a_challenger_support is True
    assert protocol.allow_single_verifier_promotion is False
    assert protocol.false_consensus_probe is False
    assert protocol.max_trigger_rate == 0.30
    assert protocol.trigger_buckets == ("weak_split", "format_risk", "deterministic_repair_only")
    assert protocol.allow_same_model_promotion is False
    assert protocol.leader_lock_count == 4
    assert protocol.promotion_score_margin == 0.15
    assert protocol.stage_a_max_tokens == 0
    assert protocol.verifier_max_tokens == 1024


def test_cred_v_legacy_config_retains_old_baselines() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_legacy.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_stage_a_output_protocol == "json_object_answer_v3"
    assert experiment.cred_methods == ["cred_v_vote_5", "cred_v_task_verify_v3", "cred_verify_safe_v1", "cred_acs_v1"]
    assert protocol.expansion_modes == ("math_symbolic_repair", "hotpot_span_extract", "mc_choice_shuffle", "strategyqa_dual_polarity")


def test_cred_v_family_is_registered() -> None:
    assert "cred_v" in registered_family_names()
