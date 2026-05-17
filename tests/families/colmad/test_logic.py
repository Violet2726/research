from __future__ import annotations

from research_experiments.families.colmad.config import load_experiment_config, load_protocol_config
from research_experiments.families.colmad.prompts import validate_debater_output
from research_experiments.families.colmad.run.sample import _build_metrics


def test_load_colmad_experiment_and_protocol() -> None:
    experiment = load_experiment_config("configs/families/colmad/experiments/colmad_realmistake_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.name == "colmad_realmistake_main"
    assert [method.name for method in experiment.methods] == [
        "single_agent_detector",
        "copmad_competitive",
        "colmad_collaborative",
    ]
    assert protocol.max_debate_rounds == 1
    assert protocol.max_failure_modes == 3


def test_validate_debater_output_enforces_protocol_specific_fields() -> None:
    competitive = validate_debater_output(
        '{"verdict":"contains_error","rationale":"bad claim","attack_points":["peer overstates"],"supportive_critique":"ignore","complemented_peer_points":["x"],"observed_failure_modes":["overconfident_claim"]}',
        "",
        debate_protocol="competitive",
    )
    collaborative = validate_debater_output(
        '{"verdict":"contains_no_error","rationale":"good answer","supportive_critique":"peer missed the date constraint","complemented_peer_points":["date constraint"],"attack_points":["ignore"],"observed_failure_modes":[]}',
        "",
        debate_protocol="collaborative",
    )

    assert competitive["supportive_critique"] == ""
    assert competitive["attack_points"] == ["peer overstates"]
    assert collaborative["supportive_critique"] == "peer missed the date constraint"
    assert collaborative["complemented_peer_points"] == ["date constraint"]
    assert collaborative["attack_points"] == []


def test_build_metrics_reports_shift_and_protocol_fields() -> None:
    metrics = _build_metrics(
        [
            {
                "dataset": "realmistake_math_problem_generation",
                "method_name": "single_agent_detector",
                "score": 0.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 30.0,
                "total_tokens_per_question": 130.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 5.0,
                "calls_per_question": 1,
                "changed_after_debate": False,
                "judge_flip_after_debate": False,
                "correct_to_wrong_shift_flag": False,
                "wrong_to_correct_shift_flag": False,
                "correction_flag": False,
                "degradation_flag": False,
                "competitive_hacking_flag": False,
                "supportive_critique_observed": False,
                "evidence_complementarity_observed": False,
                "judge_confidence": 0.4,
            },
            {
                "dataset": "realmistake_math_problem_generation",
                "method_name": "copmad_competitive",
                "score": 0.0,
                "prompt_tokens_per_question": 300.0,
                "completion_tokens_per_question": 80.0,
                "total_tokens_per_question": 380.0,
                "communication_tokens_per_question": 250.0,
                "latency_ms_per_question": 20.0,
                "calls_per_question": 5,
                "changed_after_debate": True,
                "judge_flip_after_debate": True,
                "correct_to_wrong_shift_flag": True,
                "wrong_to_correct_shift_flag": False,
                "correction_flag": False,
                "degradation_flag": True,
                "competitive_hacking_flag": True,
                "supportive_critique_observed": False,
                "evidence_complementarity_observed": False,
                "judge_confidence": 0.7,
            },
            {
                "dataset": "realmistake_math_problem_generation",
                "method_name": "colmad_collaborative",
                "score": 1.0,
                "prompt_tokens_per_question": 320.0,
                "completion_tokens_per_question": 70.0,
                "total_tokens_per_question": 390.0,
                "communication_tokens_per_question": 255.0,
                "latency_ms_per_question": 21.0,
                "calls_per_question": 5,
                "changed_after_debate": True,
                "judge_flip_after_debate": True,
                "correct_to_wrong_shift_flag": False,
                "wrong_to_correct_shift_flag": True,
                "correction_flag": True,
                "degradation_flag": False,
                "competitive_hacking_flag": False,
                "supportive_critique_observed": True,
                "evidence_complementarity_observed": True,
                "judge_confidence": 0.8,
            },
        ],
        load_experiment_config("configs/families/colmad/experiments/colmad_realmistake_main.toml").methods,
        model_name="xiaomimimo/mimo-v2.5",
    )

    collaborative_row = next(
        row for row in metrics["summary"] if row["dataset"] == "realmistake_math_problem_generation" and row["method_name"] == "colmad_collaborative"
    )
    overall_competitive = next(row for row in metrics["summary"] if row["dataset"] == "overall" and row["method_name"] == "copmad_competitive")

    assert collaborative_row["wrong_to_correct_shift_rate"] == 1.0
    assert collaborative_row["supportive_critique_rate"] == 1.0
    assert collaborative_row["gain_over_competitive"] == 1.0
    assert overall_competitive["competitive_hacking_rate"] == 1.0
