from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json, write_registered_family_manifest

from research_experiments.families.baseline_compare.config import load_experiment_config
from research_experiments.families.baseline_compare.run.report import summarize_run
from research_experiments.families.baseline_compare.run.sample import _build_metrics


def test_load_experiment_config_reads_control_methods_and_setups() -> None:
    experiment = load_experiment_config("configs/families/baseline_compare/experiments/core_six_method_baseline.toml")

    assert experiment.name == "core_six_method_baseline"
    assert experiment.answer_contract == "json_answer_core"
    assert experiment.control_methods == ["cot_1", "sc_3", "sc_5"]
    assert experiment.method_order == [
        "cot_1",
        "sc_3",
        "sc_5",
        "mad_3a_r1",
        "mad_3a_r2",
        "mad_5a_r1",
    ]
    assert [setup.name for setup in experiment.setups] == ["mad_3a_r1", "mad_3a_r2", "mad_5a_r1"]
    assert experiment.raw["phases"]["count300"]["split_overrides"]["gpqa_diamond"] == "full198_seed42"


def test_load_paper_mad_experiment_config_uses_faithful_prompt_inventory() -> None:
    experiment = load_experiment_config(
        "configs/families/baseline_compare/experiments/core_six_method_baseline_paper_mad.toml"
    )

    assert experiment.name == "core_six_method_baseline_paper_mad"
    assert experiment.prompt_version == "multi_agent_paper_text"
    assert experiment.answer_contract == "paper_transcript_hardened"
    assert experiment.control_methods == ["cot_1", "sc_3", "sc_5"]
    assert experiment.method_order == [
        "cot_1",
        "sc_3",
        "sc_5",
        "mad_paper_3a_r1",
        "mad_paper_3a_r2",
        "mad_paper_5a_r1",
    ]
    assert [setup.name for setup in experiment.setups] == [
        "mad_paper_3a_r1",
        "mad_paper_3a_r2",
        "mad_paper_5a_r1",
    ]


def test_build_metrics_adds_macro_micro_and_relative_fields() -> None:
    metrics = _build_metrics(
        [
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "cot_1",
                "method_type": "control",
                "score": 0.0,
                "initial_vote_score": 0.0,
                "prompt_tokens_per_question": 10.0,
                "completion_tokens_per_question": 5.0,
                "total_tokens_per_question": 15.0,
                "debate_total_tokens_per_question": 0.0,
                "latency_ms_per_question": 20.0,
                "calls_per_question": 1,
                "debate_rounds": 0,
                "agent_count": 1,
                "corrected_by_debate": False,
                "harmed_by_debate": False,
                "vote_flipped": False,
                "initial_consensus": True,
                "final_consensus": True,
            },
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "sc_5",
                "method_type": "control",
                "score": 1.0,
                "initial_vote_score": 1.0,
                "prompt_tokens_per_question": 25.0,
                "completion_tokens_per_question": 10.0,
                "total_tokens_per_question": 35.0,
                "debate_total_tokens_per_question": 0.0,
                "latency_ms_per_question": 40.0,
                "calls_per_question": 5,
                "debate_rounds": 0,
                "agent_count": 5,
                "corrected_by_debate": False,
                "harmed_by_debate": False,
                "vote_flipped": False,
                "initial_consensus": False,
                "final_consensus": False,
            },
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "mad_3a_r1",
                "method_type": "mad",
                "score": 1.0,
                "initial_vote_score": 0.0,
                "prompt_tokens_per_question": 24.0,
                "completion_tokens_per_question": 12.0,
                "total_tokens_per_question": 36.0,
                "debate_total_tokens_per_question": 12.0,
                "latency_ms_per_question": 45.0,
                "calls_per_question": 6,
                "debate_rounds": 1,
                "agent_count": 3,
                "corrected_by_debate": True,
                "harmed_by_debate": False,
                "vote_flipped": True,
                "initial_consensus": False,
                "final_consensus": True,
            },
            {
                "dataset": "math500",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "cot_1",
                "method_type": "control",
                "score": 1.0,
                "initial_vote_score": 1.0,
                "prompt_tokens_per_question": 12.0,
                "completion_tokens_per_question": 6.0,
                "total_tokens_per_question": 18.0,
                "debate_total_tokens_per_question": 0.0,
                "latency_ms_per_question": 25.0,
                "calls_per_question": 1,
                "debate_rounds": 0,
                "agent_count": 1,
                "corrected_by_debate": False,
                "harmed_by_debate": False,
                "vote_flipped": False,
                "initial_consensus": True,
                "final_consensus": True,
            },
            {
                "dataset": "math500",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "sc_5",
                "method_type": "control",
                "score": 0.0,
                "initial_vote_score": 0.0,
                "prompt_tokens_per_question": 28.0,
                "completion_tokens_per_question": 12.0,
                "total_tokens_per_question": 40.0,
                "debate_total_tokens_per_question": 0.0,
                "latency_ms_per_question": 42.0,
                "calls_per_question": 5,
                "debate_rounds": 0,
                "agent_count": 5,
                "corrected_by_debate": False,
                "harmed_by_debate": False,
                "vote_flipped": False,
                "initial_consensus": False,
                "final_consensus": False,
            },
            {
                "dataset": "math500",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "mad_3a_r1",
                "method_type": "mad",
                "score": 0.0,
                "initial_vote_score": 0.0,
                "prompt_tokens_per_question": 30.0,
                "completion_tokens_per_question": 12.0,
                "total_tokens_per_question": 42.0,
                "debate_total_tokens_per_question": 14.0,
                "latency_ms_per_question": 50.0,
                "calls_per_question": 6,
                "debate_rounds": 1,
                "agent_count": 3,
                "corrected_by_debate": False,
                "harmed_by_debate": False,
                "vote_flipped": False,
                "initial_consensus": True,
                "final_consensus": True,
            },
        ],
        dataset_order=["gsm8k", "math500"],
        method_order=["cot_1", "sc_3", "sc_5", "mad_3a_r1"],
        control_names=["cot_1", "sc_3", "sc_5"],
    )

    summary = metrics["summary"]
    overall_methods = [row["method_name"] for row in summary if row["dataset"] == "overall"]
    assert overall_methods == ["cot_1", "sc_5", "mad_3a_r1"]
    assert any(row["dataset"] == "overall_micro" for row in summary)

    overall_mad = next(row for row in summary if row["dataset"] == "overall" and row["method_name"] == "mad_3a_r1")
    assert overall_mad["best_no_comm_method"] == "cot_1"
    assert overall_mad["accuracy_delta_vs_cot_1"] == 0.0
    assert overall_mad["accuracy_delta_vs_best_no_comm"] == 0.0
    assert overall_mad["debate_gain_over_initial_vote"] == 0.5
    assert overall_mad["communication_tokens_mean"] == 13.0


def test_summarize_run_groups_rows_by_dataset(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="baseline_compare")
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {"dataset": "gsm8k", "method_name": "cot_1", "accuracy_mean": 0.7},
                {"dataset": "overall", "method_name": "cot_1", "accuracy_mean": 0.7},
            ]
        },
    )

    payload = summarize_run(tmp_path)

    assert payload["row_count"] == 2
    assert payload["datasets"] == ["gsm8k", "overall"]
    assert "gsm8k" in payload["summary_by_dataset"]
