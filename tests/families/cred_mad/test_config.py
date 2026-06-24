from __future__ import annotations

from research_experiments.families.cred_mad.config import load_experiment_config, load_protocol_config


def test_cred_mad_main_config_loads() -> None:
    experiment = load_experiment_config("configs/families/cred_mad/experiments/cred_mad_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.cred_output_protocol == "json_object_answer_v3"
    assert experiment.cred_stage_a_output_protocol == "free_text_answer_v1"
    assert experiment.cred_debate_output_protocol == "json_object_answer_v3"
    assert experiment.cred_methods == ["cred_vote_5", "cred_refute_queue_v1_lock"]
    assert protocol.stage_a_agent_count == 5
    assert protocol.risk_trigger_count == 3
    assert protocol.max_refutations == 1
    assert protocol.locked_override_margin == 1.25
    assert protocol.stage_a_max_tokens == 0
    assert protocol.judge_max_tokens == 1024
