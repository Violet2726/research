"""覆盖 `consensagent` family 的基本摘要链路。"""

from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json, write_registered_family_manifest

from research_experiments.families.consensagent.run.report import render_report, summarize_run


def test_summarize_run_counts_methods_and_questions(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="consensagent")
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {"method_name": "consensagent_3a", "prediction_rows": 5},
                {"method_name": "mad_3a_r1", "prediction_rows": 5},
            ]
        },
    )

    payload = summarize_run(tmp_path)

    assert payload["run_dir"] == str(tmp_path)
    assert payload["total_questions"] == 10
    assert payload["method_count"] == 2


def test_render_report_abstract_reflects_mixed_consensagent_results(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="consensagent",
        payload={
            "created_at": "2026-06-04T12:00:00+00:00",
            "experiment_name": "consensagent_main",
            "phase_name": "count100",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "consensagent_paper_v1",
        },
    )
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "gsm8k",
                    "method_name": "consensagent_3a",
                    "method_type": "consensagent",
                    "prediction_rows": 100,
                    "accuracy_mean": 0.61,
                    "total_tokens_mean": 3122.84,
                    "calls_per_question_mean": 6.27,
                    "actual_debate_rounds_mean": 1.09,
                    "trigger_rate": 0.35,
                    "sycophancy_rate_mean": 0.335,
                    "accuracy_per_1k_tokens": 0.19,
                    "matched_vote_control": "mv_6",
                },
                {
                    "dataset": "gsm8k",
                    "method_name": "mv_6",
                    "method_type": "control",
                    "prediction_rows": 100,
                    "accuracy_mean": 0.97,
                    "total_tokens_mean": 2073.4,
                    "calls_per_question_mean": 6.0,
                    "actual_debate_rounds_mean": 0.0,
                    "trigger_rate": 0.0,
                    "sycophancy_rate_mean": 0.0,
                    "accuracy_per_1k_tokens": 0.47,
                },
                {
                    "dataset": "hotpotqa",
                    "method_name": "consensagent_3a",
                    "method_type": "consensagent",
                    "prediction_rows": 100,
                    "accuracy_mean": 0.75,
                    "total_tokens_mean": 10850.91,
                    "calls_per_question_mean": 6.12,
                    "actual_debate_rounds_mean": 1.04,
                    "trigger_rate": 0.06,
                    "sycophancy_rate_mean": 0.055,
                    "accuracy_per_1k_tokens": 0.06,
                    "matched_vote_control": "mv_6",
                },
                {
                    "dataset": "hotpotqa",
                    "method_name": "mv_6",
                    "method_type": "control",
                    "prediction_rows": 100,
                    "accuracy_mean": 0.72,
                    "total_tokens_mean": 10144.86,
                    "calls_per_question_mean": 6.0,
                    "actual_debate_rounds_mean": 0.0,
                    "trigger_rate": 0.0,
                    "sycophancy_rate_mean": 0.0,
                    "accuracy_per_1k_tokens": 0.07,
                },
            ]
        },
    )
    write_json(
        tmp_path / "diagnostics" / "debate_diagnostics.json",
        {
            "rows": [
                {
                    "dataset": "gsm8k",
                    "method_name": "consensagent_3a",
                    "initial_disagreement_rate": 0.37,
                    "post_debate_consensus_rate": 0.61,
                    "vote_flip_rate": 0.06,
                    "wrong_consensus_rate": 0.12,
                    "sycophancy_rate_mean": 0.335,
                    "trigger_rate": 0.35,
                    "avg_debate_rounds": 1.09,
                },
                {
                    "dataset": "hotpotqa",
                    "method_name": "consensagent_3a",
                    "initial_disagreement_rate": 0.08,
                    "post_debate_consensus_rate": 0.91,
                    "vote_flip_rate": 0.03,
                    "wrong_consensus_rate": 0.26,
                    "sycophancy_rate_mean": 0.055,
                    "trigger_rate": 0.06,
                    "avg_debate_rounds": 1.04,
                },
            ]
        },
    )
    write_json(
        tmp_path / "diagnostics" / "cost_breakdown.json",
        {
            "rows": [
                {
                    "dataset": "gsm8k",
                    "method_name": "consensagent_3a",
                    "method_type": "consensagent",
                    "prompt_tokens": 100.0,
                    "completion_tokens": 20.0,
                    "total_tokens": 120.0,
                    "latency_ms": 10.0,
                    "turn_count": 6.0,
                    "initial_tokens": 30.0,
                    "debate_tokens": 90.0,
                    "control_tokens": 0.0,
                }
            ]
        },
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "均优于" not in local_report
    assert "1 个更优，1 个更差，0 个持平" in local_report
