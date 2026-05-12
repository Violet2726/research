from __future__ import annotations

import json
from pathlib import Path

from research_experiments.reporting.paper_package import build_paper_package_payload, render_paper_package


def test_build_paper_package_payload_splits_evidence_tiers() -> None:
    analysis = {
        "combined_overall": [
            {
                "family": "selective_comm",
                "experiment_name": "trigger_early_exit_main",
                "evaluation_track": "same_context",
                "evidence_tier": "headline",
                "primary_method_name": "hybrid_trigger",
                "faithful_score": 0.84,
                "delta_vs_best_no_comm": -0.01,
                "total_tokens_mean": 1000.0,
                "calls_per_question_mean": 3.0,
            },
            {
                "family": "cue",
                "experiment_name": "cue_black_box_utility_main",
                "evaluation_track": "same_context",
                "evidence_tier": "diagnostic",
                "primary_method_name": "cue_v1",
                "faithful_score": 0.58,
                "total_tokens_mean": 900.0,
                "calls_per_question_mean": 3.0,
            },
        ]
    }
    package = build_paper_package_payload(
        {"semantic_entries": [], "overrides": {"phase_name": "count100", "model_ref": "xiaomimimo/mimo-v2.5"}},
        analysis,
        {"comparisons": []},
    )

    assert package["sections"]["same_context_main_table"][0]["experiment_name"] == "trigger_early_exit_main"
    assert package["sections"]["diagnostic_evidence_table"][0]["experiment_name"] == "cue_black_box_utility_main"


def test_render_paper_package_writes_markdown_and_figures(tmp_path: Path) -> None:
    state_dir = tmp_path / "matrix"
    state_dir.mkdir()
    run_dir = tmp_path / "trigger_run"
    run_dir.mkdir()
    external_figures = tmp_path / "external_figures"

    (run_dir / "policy_predictions.jsonl").write_text(
        json.dumps(
            {
                "dataset": "gsm8k",
                "sample_id": "1",
                "method_name": "hybrid_trigger",
                "score": 1.0,
                "stage_a_score": 0.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    state_payload = {
        "overrides": {"phase_name": "count100", "model_ref": "xiaomimimo/mimo-v2.5"},
        "counts": {"completed": 1},
        "semantic_entries": [
            {
                "family": "selective_comm",
                "config_path": "configs/families/selective_comm/experiments/trigger_early_exit_main.toml",
                "experiment_name": "trigger_early_exit_main",
                "status": "completed",
                "run_dir": run_dir.as_posix(),
            }
        ],
    }
    (state_dir / "state.json").write_text(json.dumps(state_payload, ensure_ascii=False), encoding="utf-8")
    (state_dir / "faithful_analysis.json").write_text(
        json.dumps(
            {
                "combined_overall": [
                    {
                        "family": "selective_comm",
                        "experiment_name": "trigger_early_exit_main",
                        "evaluation_track": "same_context",
                        "evidence_tier": "headline",
                        "primary_method_name": "hybrid_trigger",
                        "faithful_score": 0.84,
                        "delta_vs_best_no_comm": -0.01,
                        "delta_vs_full_comm": 0.0,
                        "token_ratio_vs_full_comm": 0.55,
                        "total_tokens_mean": 1000.0,
                        "calls_per_question_mean": 3.0,
                        "stage_ceiling_gap": 0.0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "paper_statistics.json").write_text(
        json.dumps(
            {"comparisons": [], "bootstrap_ci": {}, "paired_win_loss": {}, "mcnemar_tests": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    paths = render_paper_package(
        state_dir,
        published_path=tmp_path / "published.md",
        figures_root=external_figures,
    )

    assert Path(paths["package_markdown"]).exists()
    assert Path(paths["published_path"]).exists()
    assert Path(paths["figures_root"]) == external_figures
    assert (external_figures / "budget_frontier_same_context.svg").exists()
    assert (state_dir / "figure_manifest.json").exists()
    svg_text = (external_figures / "budget_frontier_same_context.svg").read_text(encoding="utf-8")
    package_markdown = (state_dir / "paper_package.md").read_text(encoding="utf-8")
    assert "![预算前沿：同上下文 headline 方法]" in package_markdown
    assert "Faithful score" in svg_text
