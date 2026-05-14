from __future__ import annotations

from research_experiments.families.imad.config import DebateMethodSpec, load_experiment_config, load_protocol_config
from research_experiments.families.imad.run.sample import _build_metrics, assess_stability_gate


def test_load_imad_experiment_config_reads_methods_and_protocol() -> None:
    experiment = load_experiment_config("configs/families/imad/experiments/imad_same_context_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.name == "imad_same_context_main"
    assert [method.name for method in experiment.methods] == [
        "mad_fixed_r1",
        "mad_fixed_r2",
        "mad_fixed_r3",
        "imad_adaptive",
    ]
    assert protocol.agent_count == 3
    assert protocol.max_rounds == 3
    assert experiment.methods[0].matched_controls == ["mv_6"]
    assert experiment.methods[-1].matched_controls == []


def test_assess_stability_gate_requires_same_top_answer_and_low_ks() -> None:
    ks_statistic, passed = assess_stability_gate(
        previous_top_answer="42",
        current_top_answer="42",
        previous_posterior_samples=[0.60, 0.61, 0.62, 0.63],
        current_posterior_samples=[0.61, 0.62, 0.63, 0.64],
        posterior_mean=0.70,
        ks_threshold=0.5,
        stable_posterior_mean_threshold=0.60,
    )

    assert ks_statistic is not None
    assert passed is True

    _, changed_top_passed = assess_stability_gate(
        previous_top_answer="41",
        current_top_answer="42",
        previous_posterior_samples=[0.60, 0.61, 0.62, 0.63],
        current_posterior_samples=[0.61, 0.62, 0.63, 0.64],
        posterior_mean=0.70,
        ks_threshold=0.5,
        stable_posterior_mean_threshold=0.60,
    )
    assert changed_top_passed is False


def test_build_metrics_reports_round_and_early_stop_fields() -> None:
    methods = [
        DebateMethodSpec(name="imad_adaptive", mode="adaptive", round_limit=3, matched_controls=[]),
        DebateMethodSpec(name="mad_fixed_r1", mode="fixed", round_limit=1, matched_controls=["mv_6"]),
    ]
    metrics = _build_metrics(
        [
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "imad_adaptive",
                "method_type": "mad",
                "method_mode": "adaptive",
                "configured_round_limit": 3,
                "score": 1.0,
                "prompt_tokens_per_question": 10.0,
                "completion_tokens_per_question": 5.0,
                "total_tokens_per_question": 15.0,
                "debate_total_tokens_per_question": 6.0,
                "latency_ms_per_question": 20.0,
                "calls_per_question": 9,
                "debate_rounds": 2,
                "executed_round_count": 2,
                "stopped_early": True,
                "stop_reason": "stability_gate",
                "ks_statistic_last": 0.02,
                "posterior_mean_last": 0.72,
            },
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "mad_fixed_r1",
                "method_type": "mad",
                "method_mode": "fixed",
                "configured_round_limit": 1,
                "score": 1.0,
                "prompt_tokens_per_question": 8.0,
                "completion_tokens_per_question": 4.0,
                "total_tokens_per_question": 12.0,
                "debate_total_tokens_per_question": 3.0,
                "latency_ms_per_question": 15.0,
                "calls_per_question": 6,
                "debate_rounds": 1,
                "executed_round_count": 1,
                "stopped_early": False,
                "stop_reason": "fixed_round_limit_r1",
                "ks_statistic_last": None,
                "posterior_mean_last": 0.68,
            },
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "mv_6",
                "method_type": "control",
                "method_mode": "control",
                "configured_round_limit": 0,
                "score": 0.5,
                "prompt_tokens_per_question": 8.0,
                "completion_tokens_per_question": 4.0,
                "total_tokens_per_question": 12.0,
                "debate_total_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 6,
                "debate_rounds": 0,
                "executed_round_count": 0,
                "stopped_early": False,
                "stop_reason": "control_no_debate",
                "ks_statistic_last": None,
                "posterior_mean_last": None,
            },
        ],
        methods,
    )

    adaptive_row = next(row for row in metrics["summary"] if row["method_name"] == "imad_adaptive")
    fixed_row = next(row for row in metrics["summary"] if row["method_name"] == "mad_fixed_r1")
    assert adaptive_row["executed_round_count_mean"] == 2.0
    assert adaptive_row["stopped_early_rate"] == 1.0
    assert adaptive_row["stability_stop_rate"] == 1.0
    assert adaptive_row["debate_gain_over_vote"] is None
    assert fixed_row["debate_gain_over_vote"] == 0.5
