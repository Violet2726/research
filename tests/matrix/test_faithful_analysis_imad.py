from __future__ import annotations

from pathlib import Path
import json

from research_experiments.matrix.faithful_analysis import build_faithful_analysis


def test_build_faithful_analysis_uses_round_scores_for_stage_ceiling(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "imad" / "imad_same_context_main" / "count20" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "dataset": "overall",
                        "method_name": "mv_6",
                        "accuracy_mean": 0.60,
                        "total_tokens_mean": 1000.0,
                        "calls_per_question_mean": 6.0,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "mad_fixed_r3",
                        "accuracy_mean": 0.76,
                        "total_tokens_mean": 2200.0,
                        "calls_per_question_mean": 12.0,
                    },
                    {
                        "dataset": "overall",
                        "method_name": "imad_adaptive",
                        "accuracy_mean": 0.72,
                        "total_tokens_mean": 1800.0,
                        "calls_per_question_mean": 9.0,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "final_predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "overall",
                        "method_name": "imad_adaptive",
                        "score": 0.0,
                        "round_1_score": 0.0,
                        "round_2_score": 1.0,
                        "round_3_score": 1.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "overall",
                        "method_name": "imad_adaptive",
                        "score": 1.0,
                        "round_1_score": 1.0,
                        "round_2_score": 1.0,
                        "round_3_score": None,
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
                "family": "imad",
                "config_path": "configs/families/imad/experiments/imad_same_context_main.toml",
                "experiment_name": "imad_same_context_main",
                "run_dir": run_dir.as_posix(),
                "status": "completed",
            }
        ]
    }

    analysis = build_faithful_analysis(state_payload)
    row = analysis["rows"][0]
    assert row["stage_ceiling"] == 1.0
    assert row["stage_ceiling_gap"] == 0.28
