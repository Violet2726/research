from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.table_critic.config import TableCriticMethodSpec, load_experiment_config, load_protocol_config
from research_experiments.families.table_critic.prompts import build_chain_of_table_messages, build_refiner_messages
from research_experiments.families.table_critic.run.sample import (
    _build_metrics,
    _seed_template_tree,
    _stabilize_answer_format_answer,
    _update_template_tree,
)


def test_load_table_critic_experiment_config_reads_methods_and_protocol() -> None:
    experiment = load_experiment_config("configs/families/table_critic/experiments/table_critic_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.name == "table_critic_main"
    assert [method.name for method in experiment.methods] == [
        "end_to_end_qa",
        "few_shot_qa",
        "chain_of_table",
        "critic_cot",
        "table_critic_paper",
    ]
    assert protocol.max_refine_rounds == 2
    assert protocol.use_curator is True
    assert "ROOT/Sub-table Error/Row Error" in protocol.seed_template_paths


def test_template_tree_update_reuses_existing_summary() -> None:
    protocol = load_protocol_config("configs/families/table_critic/protocols/paper_main.toml")
    tree = _seed_template_tree(protocol)

    first_id = _update_template_tree(
        tree,
        ["ROOT", "Final Query Error", "Calculation Error"],
        {
            "pattern_summary": "Arithmetic aggregation used the wrong subtotal.",
            "reuse_hint": "Recompute the subtotal before the final claim.",
            "template_title": "Subtotal mismatch",
        },
        improved=True,
        max_summaries_per_node=protocol.template_tree_max_summaries_per_node,
    )
    second_id = _update_template_tree(
        tree,
        ["ROOT", "Final Query Error", "Calculation Error"],
        {
            "pattern_summary": "Arithmetic aggregation used the wrong subtotal.",
            "reuse_hint": "Recompute the subtotal before the final claim.",
            "template_title": "Subtotal mismatch",
        },
        improved=False,
        max_summaries_per_node=protocol.template_tree_max_summaries_per_node,
    )

    node = tree["children"]["Final Query Error"]["children"]["Calculation Error"]

    assert first_id == second_id
    assert node["summaries"][0]["usage_count"] == 2
    assert node["summaries"][0]["success_count"] == 1


def test_build_metrics_reports_correction_and_degradation_fields() -> None:
    methods = [
        TableCriticMethodSpec(name="chain_of_table", mode="chain_of_table"),
        TableCriticMethodSpec(name="table_critic_paper", mode="table_critic_paper", matched_controls=["chain_of_table"]),
    ]
    metrics = _build_metrics(
        [
            {
                "dataset": "wikitq",
                "method_name": "chain_of_table",
                "score": 0.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 20.0,
                "total_tokens_per_question": 120.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 1,
                "refinement_round_count": 0,
                "judge_error_detected": False,
                "correction_flag": False,
                "degradation_flag": False,
                "template_reused": False,
                "answer_changed": False,
            },
            {
                "dataset": "wikitq",
                "method_name": "table_critic_paper",
                "score": 1.0,
                "prompt_tokens_per_question": 200.0,
                "completion_tokens_per_question": 50.0,
                "total_tokens_per_question": 250.0,
                "latency_ms_per_question": 20.0,
                "calls_per_question": 5,
                "refinement_round_count": 1,
                "judge_error_detected": True,
                "correction_flag": True,
                "degradation_flag": False,
                "template_reused": True,
                "answer_changed": True,
            },
        ],
        methods,
        model_name="xiaomimimo/mimo-v2.5",
    )

    paper_row = next(row for row in metrics["summary"] if row["dataset"] == "wikitq" and row["method_name"] == "table_critic_paper")
    overall_chain = next(row for row in metrics["summary"] if row["dataset"] == "overall" and row["method_name"] == "chain_of_table")

    assert paper_row["correction_rate"] == 1.0
    assert paper_row["degradation_rate"] == 0.0
    assert paper_row["template_reuse_rate"] == 1.0
    assert paper_row["gain_over_chain_of_table"] == 1.0
    assert overall_chain["accuracy_mean"] == 0.0


def test_build_chain_of_table_messages_adds_compact_guard_for_large_tables() -> None:
    sample = DatasetSample(
        dataset="wikitq",
        sample_id="demo",
        question="which comet has the longest orbital period?",
        reference_answer="halley",
        prompt_context="row\n" * 4000,
        metadata={"question_type": "lookup"},
    )

    content = build_chain_of_table_messages(sample)[1]["content"]

    assert "This table is large. Keep the reasoning compact." in content
    assert "Do not enumerate every row" in content


def test_build_refiner_messages_preserves_answer_under_answer_format_error() -> None:
    sample = DatasetSample(
        dataset="wikitq",
        sample_id="demo",
        question="what language is listed?",
        reference_answer="hindi",
        prompt_context="/* demo table */",
        metadata={"question_type": "lookup"},
    )

    messages = build_refiner_messages(
        sample,
        current_reasoning="The table lists hindi.",
        current_answer="hindi",
        judge_payload={"error_step": "Answer Format Error", "rationale": "format only"},
        critic_payload={"critic_feedback": "Keep the answer but normalize the output.", "conflicting_evidence": [], "repair_hint": "Do not change the answer."},
        template_hints=[],
    )

    assert "Preserve the original answer content unless it is empty." in messages[1]["content"]


def test_stabilize_answer_format_answer_keeps_previous_semantics() -> None:
    stabilized = _stabilize_answer_format_answer(
        previous_answer="hindi",
        refined_answer="cannot be determined from the table",
        judge_payload={"error_step": "Answer Format Error"},
    )

    assert stabilized == "hindi"
