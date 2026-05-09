"""覆盖 faithful analysis 构建与聚合行为的测试。"""

from __future__ import annotations

import json
from pathlib import Path

from experiment_core.faithful_analysis import build_faithful_analysis, render_faithful_analysis


def test_build_faithful_analysis_computes_reference_deltas_and_stage_ceiling(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "budget_comm" / "dala_lite_same_context_main" / "smoke20" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "dataset": "overall",
                        "method_name": "mv_3",
                        "accuracy_mean": 0.60,
                        "total_tokens_mean": 1000.0,
                        "calls_per_question_mean": 3.0,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "all_to_all_full",
                        "accuracy_mean": 0.72,
                        "total_tokens_mean": 2000.0,
                        "communication_tokens_mean": 400.0,
                        "calls_per_question_mean": 6.0,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "budget_confidence",
                        "accuracy_mean": 0.73,
                        "total_tokens_mean": 1700.0,
                        "communication_tokens_mean": 100.0,
                        "calls_per_question_mean": 6.0,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "dala_lite",
                        "accuracy_mean": 0.70,
                        "total_tokens_mean": 1600.0,
                        "communication_tokens_mean": 90.0,
                        "calls_per_question_mean": 6.0,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "final_predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "overall",
                        "method_name": "dala_lite",
                        "score": 0.0,
                        "stage_a_score": 1.0,
                        "stage_b_score": 0.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "overall",
                        "method_name": "dala_lite",
                        "score": 1.0,
                        "stage_a_score": 1.0,
                        "stage_b_score": 1.0,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state_payload = {
        "semantic_entries": [
            {
                "family": "budget_comm",
                "config_path": "configs/budget_comm/experiments/dala_lite_same_context_main.toml",
                "experiment_name": "dala_lite_same_context_main",
                "run_dir": run_dir.as_posix(),
                "status": "completed",
            }
        ]
    }

    analysis = build_faithful_analysis(state_payload)
    assert len(analysis["rows"]) == 1
    row = analysis["rows"][0]
    assert row["best_no_comm_control"] == "mv_3"
    assert row["evidence_tier"] == "headline"
    assert row["full_comm_reference"] == "all_to_all_full"
    assert row["family_envelope"] == "budget_confidence"
    assert row["delta_vs_best_no_comm"] == 0.1
    assert row["delta_vs_full_comm"] == -0.02
    assert row["delta_vs_family_envelope"] == -0.03
    assert row["stage_ceiling"] == 1.0
    assert row["stage_ceiling_gap"] == 0.3


def test_render_faithful_analysis_writes_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "cue" / "cue_black_box_utility_main" / "smoke20" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "policy_metrics.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "dataset": "overall",
                        "method_name": "mv_3",
                        "accuracy_mean": 0.6,
                        "total_tokens_mean": 1000.0,
                        "calls_per_question_mean": 3.0,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "always_communicate",
                        "accuracy_mean": 0.7,
                        "total_tokens_mean": 2500.0,
                        "communication_tokens_mean": 1500.0,
                        "calls_per_question_mean": 6.0,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "cue_v1",
                        "accuracy_mean": 0.68,
                        "total_tokens_mean": 1500.0,
                        "communication_tokens_mean": 400.0,
                        "calls_per_question_mean": 3.5,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "disagreement_triggered",
                        "accuracy_mean": 0.69,
                        "total_tokens_mean": 1800.0,
                        "communication_tokens_mean": 700.0,
                        "calls_per_question_mean": 4.0,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "policy_predictions.jsonl").write_text(
        json.dumps({"dataset": "overall", "method_name": "cue_v1", "score": 1.0, "stage_a_score": 1.0}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    state_dir = tmp_path / "matrix"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "semantic_entries": [
                    {
                        "family": "cue",
                        "config_path": "configs/cue/experiments/cue_black_box_utility_main.toml",
                        "experiment_name": "cue_black_box_utility_main",
                        "run_dir": run_dir.as_posix(),
                        "status": "completed",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = render_faithful_analysis(
        state_dir,
        published_path=tmp_path / "published-faithful.md",
    )

    assert Path(outputs["json_path"]).exists()
    assert Path(outputs["markdown_path"]).exists()


def test_build_faithful_analysis_synthesizes_overall_when_metrics_lack_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "multi_agent" / "same_context_controlled_debate" / "smoke20" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "dataset": "gsm8k",
                        "method_name": "mad_3a_r1",
                        "accuracy_mean": 0.9,
                        "total_tokens_mean": 2400.0,
                        "calls_per_question_mean": 6.0,
                    },
                    {
                        "dataset": "gsm8k",
                        "method_name": "mv_6",
                        "accuracy_mean": 0.7,
                        "total_tokens_mean": 1800.0,
                        "calls_per_question_mean": 6.0,
                    },
                    {
                        "dataset": "hotpotqa",
                        "method_name": "mad_3a_r1",
                        "accuracy_mean": 0.8,
                        "total_tokens_mean": 10400.0,
                        "calls_per_question_mean": 6.0,
                    },
                    {
                        "dataset": "hotpotqa",
                        "method_name": "mv_6",
                        "accuracy_mean": 0.75,
                        "total_tokens_mean": 9800.0,
                        "calls_per_question_mean": 6.0,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "final_predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"dataset": "gsm8k", "method_name": "mad_3a_r1", "score": 1.0}, ensure_ascii=False),
                json.dumps({"dataset": "hotpotqa", "method_name": "mad_3a_r1", "score": 0.0}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state_payload = {
        "semantic_entries": [
            {
                "family": "multi_agent",
                "config_path": "configs/multi_agent/experiments/same_context_controlled_debate.toml",
                "experiment_name": "same_context_controlled_debate",
                "run_dir": run_dir.as_posix(),
                "status": "completed",
            }
        ]
    }

    analysis = build_faithful_analysis(state_payload)
    overall_row = next(
        row for row in analysis["rows"] if row["dataset"] == "overall" and row["experiment_name"] == "same_context_controlled_debate"
    )
    assert overall_row["faithful_score"] == 0.85
    assert overall_row["best_no_comm_control"] == "mv_6"
    assert overall_row["best_no_comm_score"] == 0.725
