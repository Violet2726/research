from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json

from research_experiments.families.selective_comm.run.report import render_report as render_selective_report
from research_experiments.families.single_agent.run.report import render_report as render_single_agent_report


def test_single_agent_render_report_outputs_scientific_markdown(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-09T12:00:00+00:00",
            "experiment": "same_context_core_benchmarks",
            "phase": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "single_agent_v1",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "cot_1",
                    "accuracy_mean": 0.72,
                    "accuracy_std": 0.01,
                    "total_tokens_mean": 120.0,
                    "calls_per_question_mean": 1.0,
                    "acc_per_1k_tokens": 6.0,
                    "prompt_tokens_mean": 80.0,
                    "completion_tokens_mean": 40.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "sc_5",
                    "accuracy_mean": 0.78,
                    "accuracy_std": 0.02,
                    "total_tokens_mean": 300.0,
                    "calls_per_question_mean": 5.0,
                    "acc_per_1k_tokens": 2.6,
                    "prompt_tokens_mean": 150.0,
                    "completion_tokens_mean": 150.0,
                },
                {
                    "dataset": "gsm8k",
                    "method_name": "cot_1",
                    "accuracy_mean": 0.70,
                    "accuracy_std": 0.01,
                    "total_tokens_mean": 120.0,
                    "calls_per_question_mean": 1.0,
                    "acc_per_1k_tokens": 5.8,
                    "prompt_tokens_mean": 80.0,
                    "completion_tokens_mean": 40.0,
                },
                {
                    "dataset": "gsm8k",
                    "method_name": "sc_5",
                    "accuracy_mean": 0.80,
                    "accuracy_std": 0.02,
                    "total_tokens_mean": 300.0,
                    "calls_per_question_mean": 5.0,
                    "acc_per_1k_tokens": 2.7,
                    "prompt_tokens_mean": 150.0,
                    "completion_tokens_mean": 150.0,
                },
            ]
        },
    )

    payload = render_single_agent_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")
    published_report = Path(payload["published_report"]).read_text(encoding="utf-8")

    assert "## 摘要" in local_report
    assert "## 研究问题与判读口径" in local_report
    assert "## 图表资产" in local_report
    assert "figures/frontier_overall.svg" in local_report
    assert "../figures/frontier_overall.svg" in published_report


def test_selective_comm_render_report_outputs_scientific_markdown(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-09T12:00:00+00:00",
            "experiment": "trigger_early_exit_main",
            "phase": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "selective_comm_trigger_json",
        },
    )
    write_json(
        tmp_path / "policy_metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "always_communicate",
                    "display_name": "always",
                    "accuracy_mean": 0.80,
                    "communication_tokens_mean": 300.0,
                    "total_tokens_mean": 500.0,
                    "acc_per_1k_tokens": 1.60,
                },
                {
                    "dataset": "overall",
                    "method_name": "hybrid_trigger",
                    "display_name": "hybrid",
                    "accuracy_mean": 0.82,
                    "communication_tokens_mean": 120.0,
                    "total_tokens_mean": 320.0,
                    "acc_per_1k_tokens": 2.56,
                },
                {
                    "dataset": "gsm8k",
                    "method_name": "always_communicate",
                    "display_name": "always",
                    "accuracy_mean": 0.78,
                    "communication_tokens_mean": 300.0,
                    "total_tokens_mean": 500.0,
                    "acc_per_1k_tokens": 1.56,
                },
                {
                    "dataset": "gsm8k",
                    "method_name": "hybrid_trigger",
                    "display_name": "hybrid",
                    "accuracy_mean": 0.81,
                    "communication_tokens_mean": 120.0,
                    "total_tokens_mean": 320.0,
                    "acc_per_1k_tokens": 2.53,
                },
            ]
        },
    )
    write_json(
        tmp_path / "policy_diagnostics.json",
        {
            "policy_rows": [
                {
                    "dataset": "overall",
                    "policy_name": "always_communicate",
                    "display_name": "always",
                    "accuracy_mean": 0.80,
                    "trigger_rate": 1.0,
                    "early_exit_rate": 0.0,
                    "precision": 0.55,
                    "recall": 1.0,
                    "false_trigger_rate": 0.10,
                    "missed_beneficial_comm_rate": 0.00,
                    "communication_tokens_mean": 300.0,
                },
                {
                    "dataset": "overall",
                    "policy_name": "hybrid_trigger",
                    "display_name": "hybrid",
                    "accuracy_mean": 0.82,
                    "trigger_rate": 0.40,
                    "early_exit_rate": 0.35,
                    "precision": 0.72,
                    "recall": 0.68,
                    "false_trigger_rate": 0.08,
                    "missed_beneficial_comm_rate": 0.12,
                    "communication_tokens_mean": 120.0,
                },
            ],
            "voc_policy_rows": [],
            "shared_prefix_rows": [
                {
                    "dataset": "overall",
                    "shared_actual_tokens": 1000.0,
                    "naive_independent_tokens": 2000.0,
                    "shared_prefix_savings_ratio": 0.50,
                }
            ],
            "recommended_next_default_policy": {"selected_policy": "hybrid_trigger"},
        },
    )
    write_json(tmp_path / "oracle_trigger_eval.json", {"sample_rows": []})
    (tmp_path / "policy_predictions.jsonl").write_text("", encoding="utf-8")

    payload = render_selective_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "## 摘要" in local_report
    assert "## 研究问题与实验设计" in local_report
    assert "## 共享前缀节省情况" in local_report
    assert "## 图表资产" in local_report
    assert Path(payload["frontier_report"]).exists()
    assert Path(payload["trigger_diagnostic_report"]).exists()
