from __future__ import annotations

import json
from pathlib import Path

from experiment_core.reporting.paper_statistics import build_paper_statistics, render_paper_statistics


def test_build_paper_statistics_pairs_binary_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "trigger"
    run_dir.mkdir()
    rows = [
        {"dataset": "gsm8k", "sample_id": "1", "method_name": "hybrid_trigger", "score": 1.0},
        {"dataset": "gsm8k", "sample_id": "1", "method_name": "always_communicate", "score": 0.0},
        {"dataset": "gsm8k", "sample_id": "2", "method_name": "hybrid_trigger", "score": 0.0},
        {"dataset": "gsm8k", "sample_id": "2", "method_name": "always_communicate", "score": 0.0},
        {"dataset": "gsm8k", "sample_id": "3", "method_name": "hybrid_trigger", "score": 1.0},
        {"dataset": "gsm8k", "sample_id": "3", "method_name": "always_communicate", "score": 1.0},
    ]
    (run_dir / "policy_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    state_payload = {
        "semantic_entries": [
            {
                "status": "completed",
                "experiment_name": "trigger_early_exit_main",
                "run_dir": run_dir.as_posix(),
            }
        ]
    }

    stats = build_paper_statistics(state_payload, bootstrap_samples=20)

    comparison = stats["bootstrap_ci"]["hybrid_trigger_vs_always_communicate"]
    assert comparison["paired_n"] == 3
    assert comparison["mean_delta"] == 0.333333
    assert stats["paired_win_loss"]["hybrid_trigger_vs_always_communicate"]["wins"] == 1
    assert stats["mcnemar_tests"]["hybrid_trigger_vs_always_communicate"]["b"] == 1


def test_render_paper_statistics_writes_fixed_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "dala"
    run_dir.mkdir()
    rows = [
        {"dataset": "gsm8k", "sample_id": "1", "method_name": "dala_lite", "score": 1.0},
        {"dataset": "gsm8k", "sample_id": "1", "method_name": "all_to_all_full", "score": 1.0},
    ]
    (run_dir / "final_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    state_dir = tmp_path / "matrix"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "semantic_entries": [
                    {
                        "status": "completed",
                        "experiment_name": "dala_lite_same_context_main",
                        "run_dir": run_dir.as_posix(),
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    paths = render_paper_statistics(state_dir, bootstrap_samples=5)

    assert Path(paths["paper_statistics"]).exists()
    assert Path(paths["bootstrap_ci"]).exists()
    assert Path(paths["paired_win_loss"]).exists()
    assert Path(paths["mcnemar_tests"]).exists()
    assert Path(paths["dataset_breakdown"]).exists()

