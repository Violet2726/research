from __future__ import annotations

from pathlib import Path

from research_experiments.reporting.family_landscape import build_family_landscape_payload, render_family_landscape
from testsupport.filesystem import write_json


def test_build_family_landscape_payload_creates_global_track_and_rollup_views() -> None:
    payload = build_family_landscape_payload(
        {
            "overrides": {"phase_name": "count300", "model_ref": "xiaomimimo/mimo-v2.5"},
            "counts": {"completed": 4, "semantic_unique_targets": 4},
        },
        {
            "combined_overall": [
                {
                    "family": "selective_comm",
                    "experiment_name": "trigger_early_exit_main",
                    "evaluation_track": "same_context",
                    "evidence_tier": "headline",
                    "primary_method_name": "hybrid_trigger",
                    "faithful_score": 0.835947,
                    "delta_vs_best_no_comm": -0.010856,
                    "delta_vs_full_comm": 0.001206,
                    "total_tokens_mean": 1200.0,
                    "communication_tokens_mean": 200.0,
                    "calls_per_question_mean": 3.4,
                    "stage_ceiling_gap": 0.006031,
                },
                {
                    "family": "selective_comm",
                    "experiment_name": "voc_trigger_main",
                    "evaluation_track": "same_context",
                    "evidence_tier": "headline",
                    "primary_method_name": "voc_trigger_v2",
                    "faithful_score": 0.727382,
                    "delta_vs_best_no_comm": 0.088058,
                    "delta_vs_full_comm": -0.007238,
                    "total_tokens_mean": 1400.0,
                    "communication_tokens_mean": 260.0,
                    "calls_per_question_mean": 4.4,
                    "stage_ceiling_gap": 0.021713,
                },
                {
                    "family": "budget_comm",
                    "experiment_name": "dala_lite_split_context_main",
                    "evaluation_track": "split_context",
                    "evidence_tier": "headline",
                    "primary_method_name": "dala_lite",
                    "faithful_score": 0.648393,
                    "delta_vs_best_no_comm": 0.037807,
                    "delta_vs_full_comm": -0.066163,
                    "delta_vs_full_context": None,
                    "total_tokens_mean": 1800.0,
                    "communication_tokens_mean": 500.0,
                    "calls_per_question_mean": 6.0,
                    "stage_ceiling_gap": 0.04915,
                },
                {
                    "family": "cue",
                    "experiment_name": "cue_black_box_utility_main",
                    "evaluation_track": "same_context",
                    "evidence_tier": "diagnostic",
                    "primary_method_name": "cue_v1",
                    "faithful_score": 0.572428,
                    "delta_vs_best_no_comm": 0.047585,
                    "delta_vs_full_comm": -0.055983,
                    "total_tokens_mean": 1100.0,
                    "communication_tokens_mean": 180.0,
                    "calls_per_question_mean": 3.8,
                    "stage_ceiling_gap": 0.004899,
                },
            ]
        },
        package_payload={
            "helpful_harmful_communication": [
                {"experiment_name": "trigger_early_exit_main", "method_name": "hybrid_trigger", "helpful_rate": 0.008444, "harmful_rate": 0.006031},
                {"experiment_name": "voc_trigger_main", "method_name": "voc_trigger_v2", "helpful_rate": 0.101327, "harmful_rate": 0.013269},
            ]
        },
        acceptance_payload={
            "all_rows": [
                {
                    "family": "selective_comm",
                    "experiment_name": "trigger_early_exit_main",
                    "primary_method_name": "hybrid_trigger",
                    "status": "accepted_same_context",
                    "notes": "passed_same_context_gate",
                },
                {
                    "family": "budget_comm",
                    "experiment_name": "dala_lite_split_context_main",
                    "primary_method_name": "dala_lite",
                    "status": "accepted_split_context",
                    "notes": "passed_split_context_gate",
                },
            ]
        },
    )

    global_board = payload["global_total_board"]
    assert [row["experiment_name"] for row in global_board] == [
        "trigger_early_exit_main",
        "voc_trigger_main",
        "dala_lite_split_context_main",
        "cue_black_box_utility_main",
    ]
    assert global_board[0]["global_rank_by_score"] == 1
    assert global_board[1]["track_rank_by_score"] == 2
    assert global_board[3]["tier_track_rank_by_score"] == 1
    assert global_board[0]["helpful_rate"] == 0.008444
    assert global_board[2]["acceptance_status"] == "accepted_split_context"

    same_context_headline = payload["tier_track_boards"]["same_context"]["headline"]
    assert [row["experiment_name"] for row in same_context_headline] == [
        "trigger_early_exit_main",
        "voc_trigger_main",
    ]

    selective_rollup = next(row for row in payload["family_rollup"] if row["family"] == "selective_comm")
    assert selective_rollup["row_count"] == 2
    assert "trigger_early_exit_main" in selective_rollup["experiments"]
    assert "voc_trigger_main" in selective_rollup["experiments"]
    assert "family_score" not in selective_rollup


def test_render_family_landscape_writes_json_and_markdown(tmp_path: Path) -> None:
    state_dir = tmp_path / "matrix"
    write_json(
        state_dir / "state.json",
        {
            "overrides": {"phase_name": "count100", "model_ref": "xiaomimimo/mimo-v2.5"},
            "counts": {"completed": 1, "semantic_unique_targets": 1},
        },
    )
    write_json(
        state_dir / "faithful_analysis.json",
        {
            "combined_overall": [
                {
                    "family": "sparc",
                    "experiment_name": "end_to_end_main",
                    "evaluation_track": "same_context",
                    "evidence_tier": "headline",
                    "primary_method_name": "sparc_v1",
                    "faithful_score": 0.664656,
                    "delta_vs_best_no_comm": 0.05187,
                    "delta_vs_full_comm": -0.004825,
                    "total_tokens_mean": 1500.0,
                    "communication_tokens_mean": 300.0,
                    "calls_per_question_mean": 4.3,
                    "stage_ceiling_gap": 0.016888,
                }
            ]
        },
    )

    outputs = render_family_landscape(
        state_dir,
        published_path=tmp_path / "published-family-landscape.md",
    )

    markdown = Path(outputs["markdown_path"]).read_text(encoding="utf-8")
    assert Path(outputs["json_path"]).exists()
    assert Path(outputs["published_path"]).exists()
    assert "## Global Total Board" in markdown
    assert "end_to_end_main" in markdown
