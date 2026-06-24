from __future__ import annotations

from research_experiments.families.cred_v.config import load_experiment_config, load_protocol_config
from research_experiments.families.registry import registered_family_names


def test_cred_v_main_config_loads() -> None:
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_output_protocol == "json_object_answer_v3"
    assert experiment.cred_stage_a_output_protocol == "free_text_answer_v1"
    assert experiment.cred_debate_output_protocol == "json_object_answer_v3"
    assert experiment.cred_methods == ["cred_v_vote_5", "cred_v_selective_verify_v1"]
    assert protocol.max_refutations == 1
    assert protocol.locked_override_margin == 1.25
    assert protocol.stage_a_max_tokens == 0
    assert protocol.judge_max_tokens == 1024


def test_cred_v_family_is_registered() -> None:
    assert "cred_v" in registered_family_names()
