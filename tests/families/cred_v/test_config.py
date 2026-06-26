from __future__ import annotations

from research_experiments.families.cred_v.config import load_experiment_config, load_protocol_config
from research_experiments.families.registry import registered_family_names


def test_cred_v_main_config_loads() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_output_protocol == "json_object_answer_v3"
    assert experiment.cred_stage_a_output_protocol == "free_text_answer_v1"
    assert experiment.cred_verification_output_protocol == "json_object_answer_v3"
    assert experiment.cred_methods == ["cred_v_vote_5", "cred_v_task_verify_v3", "cred_verify_safe_v1"]
    assert experiment.verifier_model_refs == ["xiaomimimo/mimo-v2.5-pro"]
    assert protocol.max_verifications == 1
    assert protocol.max_verification_calls == 1
    assert protocol.verification_modes == ("deterministic_repair", "tool_verified", "hetero_verified")
    assert protocol.allow_same_model_promotion is False
    assert protocol.promotion_score_margin == 0.15
    assert protocol.stage_a_max_tokens == 0
    assert protocol.verifier_max_tokens == 1024


def test_cred_v_family_is_registered() -> None:
    assert "cred_v" in registered_family_names()
