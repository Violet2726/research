from __future__ import annotations

import json
from pathlib import Path

from selective_comm.reporting import render_report as render_selective_report
from single_agent.reporting import render_report as render_single_agent_report
from sparc.reporting import render_report as render_sparc_report


def test_single_agent_render_report_outputs_scientific_markdown(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-09T12:00:00+00:00",
            "experiment": "same_context_core_benchmarks",
            "phase": "smoke20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "single_agent_v1",
        },
    )
    _write_json(
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
    _write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-09T12:00:00+00:00",
            "experiment": "trigger_early_exit_main",
            "phase": "smoke20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "selective_comm_trigger_json",
        },
    )
    _write_json(
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
    _write_json(
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
    _write_json(tmp_path / "oracle_trigger_eval.json", {"sample_rows": []})
    (tmp_path / "policy_predictions.jsonl").write_text("", encoding="utf-8")

    payload = render_selective_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "## 摘要" in local_report
    assert "## 研究问题与实验设计" in local_report
    assert "## 共享前缀节省情况" in local_report
    assert "## 图表资产" in local_report
    assert Path(payload["frontier_report"]).exists()
    assert Path(payload["trigger_diagnostic_report"]).exists()


def test_sparc_render_report_outputs_scientific_markdown(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-09T12:00:00+00:00",
            "experiment": "content_ablation",
            "experiment_kind": "content_ablation",
            "phase": "smoke20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
        },
    )
    _write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "full_cot",
                    "display_name": "full_cot",
                    "accuracy_mean": 0.74,
                    "communication_tokens_mean": 3500.0,
                    "audit_tokens_mean": 0.0,
                    "total_tokens_mean": 6100.0,
                    "calls_per_question_mean": 6.0,
                    "acc_per_1k_tokens": 0.12,
                    "compression_ratio_vs_full_cot": 1.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "answer_only",
                    "display_name": "answer_only",
                    "accuracy_mean": 0.61,
                    "communication_tokens_mean": 2600.0,
                    "audit_tokens_mean": 0.0,
                    "total_tokens_mean": 5200.0,
                    "calls_per_question_mean": 6.0,
                    "acc_per_1k_tokens": 0.11,
                    "compression_ratio_vs_full_cot": 0.23,
                },
                {
                    "dataset": "gsm8k",
                    "method_name": "full_cot",
                    "display_name": "full_cot",
                    "accuracy_mean": 0.75,
                    "communication_tokens_mean": 3400.0,
                    "audit_tokens_mean": 0.0,
                    "total_tokens_mean": 6000.0,
                    "calls_per_question_mean": 6.0,
                    "acc_per_1k_tokens": 0.12,
                    "compression_ratio_vs_full_cot": 1.0,
                },
            ]
        },
    )
    _write_json(
        tmp_path / "diagnostics.json",
        {
            "experiment_kind": "content_ablation",
            "recommended_next_default": {
                "method_name": "answer_only",
                "accuracy_mean": 0.61,
                "total_tokens_mean": 5200.0,
                "compression_ratio_vs_full_cot": 0.23,
            },
        },
    )
    (tmp_path / "final_predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "gsm8k",
                        "sample_id": "gsm8k-1",
                        "model_name": "xiaomimimo/mimo-v2.5",
                        "method_name": "full_cot",
                        "display_name": "full_cot",
                        "score": 1.0,
                        "communication_tokens_per_question": 3400.0,
                        "total_tokens_per_question": 6000.0,
                        "question_preview": "question 1",
                        "gold": "42",
                        "prediction": "42",
                        "initial_disagreement": True,
                        "oracle_positive": True,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "gsm8k",
                        "sample_id": "gsm8k-1",
                        "model_name": "xiaomimimo/mimo-v2.5",
                        "method_name": "answer_only",
                        "display_name": "answer_only",
                        "score": 0.0,
                        "communication_tokens_per_question": 2600.0,
                        "total_tokens_per_question": 5200.0,
                        "question_preview": "question 1",
                        "gold": "42",
                        "prediction": "41",
                        "initial_disagreement": True,
                        "oracle_positive": True,
                        "note": "compression hurts this case",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "gsm8k",
                        "sample_id": "gsm8k-2",
                        "model_name": "xiaomimimo/mimo-v2.5",
                        "method_name": "full_cot",
                        "display_name": "full_cot",
                        "score": 1.0,
                        "communication_tokens_per_question": 3400.0,
                        "total_tokens_per_question": 6000.0,
                        "question_preview": "question 2",
                        "gold": "84",
                        "prediction": "84",
                        "initial_disagreement": False,
                        "oracle_positive": False,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "gsm8k",
                        "sample_id": "gsm8k-2",
                        "model_name": "xiaomimimo/mimo-v2.5",
                        "method_name": "answer_only",
                        "display_name": "answer_only",
                        "score": 1.0,
                        "communication_tokens_per_question": 2600.0,
                        "total_tokens_per_question": 5200.0,
                        "question_preview": "question 2",
                        "gold": "84",
                        "prediction": "84",
                        "initial_disagreement": False,
                        "oracle_positive": False,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = render_sparc_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "# SPARC 内容消融科研报告" in local_report
    assert "## 研究问题与实验设计" in local_report
    assert "## 局限性" in local_report
    assert "## 图表资产" in local_report
    assert Path(payload["figure_manifest"]).exists()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
