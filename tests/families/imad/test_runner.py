from __future__ import annotations

from pathlib import Path

from research_experiments.families.imad.run.report import render_report
from research_experiments.families.imad.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_imad_render_report_outputs_markdown_and_figures(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-14T12:00:00+00:00",
            "experiment": "imad_same_context_main",
            "phase": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "imad_controlled_json",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "mv_6",
                    "accuracy_mean": 0.72,
                    "total_tokens_mean": 1200.0,
                    "communication_tokens_mean": 0.0,
                    "calls_per_question_mean": 6.0,
                    "accuracy_per_1k_tokens": 0.60,
                    "executed_round_count_mean": 0.0,
                    "stopped_early_rate": 0.0,
                    "stability_stop_rate": 0.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "mad_fixed_r1",
                    "accuracy_mean": 0.79,
                    "total_tokens_mean": 1500.0,
                    "communication_tokens_mean": 400.0,
                    "calls_per_question_mean": 6.0,
                    "accuracy_per_1k_tokens": 0.526,
                    "executed_round_count_mean": 1.0,
                    "stopped_early_rate": 0.0,
                    "stability_stop_rate": 0.0,
                    "matched_vote_control": "mv_6",
                    "debate_gain_over_vote": 0.07,
                },
                {
                    "dataset": "overall",
                    "method_name": "mad_fixed_r3",
                    "accuracy_mean": 0.80,
                    "total_tokens_mean": 2200.0,
                    "communication_tokens_mean": 900.0,
                    "calls_per_question_mean": 12.0,
                    "accuracy_per_1k_tokens": 0.36,
                    "executed_round_count_mean": 3.0,
                    "stopped_early_rate": 0.0,
                    "stability_stop_rate": 0.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "imad_adaptive",
                    "accuracy_mean": 0.79,
                    "total_tokens_mean": 1800.0,
                    "communication_tokens_mean": 500.0,
                    "calls_per_question_mean": 9.0,
                    "accuracy_per_1k_tokens": 0.438,
                    "executed_round_count_mean": 2.0,
                    "stopped_early_rate": 0.6,
                    "stability_stop_rate": 0.6,
                    "matched_vote_control": None,
                    "debate_gain_over_vote": None,
                },
            ]
        },
    )
    write_json(
        tmp_path / "stability_diagnostics.json",
        {
            "summary_rows": [
                {
                    "dataset": "overall",
                    "method_name": "imad_adaptive",
                    "executed_round_count_mean": 2.0,
                    "stopped_early_rate": 0.6,
                    "stability_stop_rate": 0.6,
                    "max_round_reached_rate": 0.4,
                    "ks_statistic_last_mean": 0.04,
                    "posterior_mean_last_mean": 0.73,
                }
            ],
            "round_rows": [
                {
                    "dataset": "overall",
                    "method_name": "imad_adaptive",
                    "round_index": 2,
                    "mean_support_rate": 0.7,
                    "mean_posterior_mean": 0.68,
                    "same_top_rate": 0.8,
                    "stability_gate_pass_rate": 0.6,
                    "mean_ks_statistic": 0.04,
                }
            ],
        },
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "## 摘要" in local_report
    assert "## 研究问题与实验设计" in local_report
    assert "mad_fixed_r1" in local_report
    assert "mv_6` 只对应 `mad_fixed_r1`" in local_report
    assert "## 稳定性诊断" in local_report
    assert Path(payload["figure_manifest"]).exists()


def test_imad_validate_run_accepts_complete_artifacts(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "demo", "experiment": "imad_same_context_main"})
    write_jsonl(
        tmp_path / "agent_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "imad_adaptive",
                "output_status": "ok",
            }
        ],
    )
    write_jsonl(tmp_path / "debate_messages.jsonl", [])
    write_jsonl(
        tmp_path / "round_diagnostics.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "imad_adaptive",
                "round_index": 1,
            }
        ],
    )
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "imad_adaptive",
                "method_type": "mad",
                "executed_round_count": 2,
                "configured_round_limit": 3,
                "stopped_early": True,
                "stop_reason": "stability_gate",
                "round_1_score": 0.0,
                "round_2_score": 1.0,
                "round_3_score": None,
                "ks_statistic_last": 0.03,
                "posterior_mean_last": 0.71,
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(tmp_path / "stability_diagnostics.json", {"summary_rows": [{"dataset": "gsm8k"}]})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True
