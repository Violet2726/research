from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl, write_registered_family_manifest

from research_experiments.families.adaptive_sparse_mad.run.validate import validate_run


def test_validate_run_rejects_legacy_stage_b_judge_rows(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="adaptive_sparse_mad")
    write_json(
        tmp_path / "progress.json",
        {
            "rate_limit_429_count": 0,
        },
    )
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "hetero_vote_3",
                    "accuracy_mean": 0.85,
                    "trigger_rate": 0.2,
                    "early_exit_rate": 0.8,
                    "changed_answer_rate": 0.05,
                    "corrected_rate": 0.03,
                    "harmed_rate": 0.01,
                    "judge_fallback_rate": 0.0,
                    "total_tokens_mean": 1800.0,
                }
            ]
        },
    )
    write_json(tmp_path / "diagnostics" / "router_eval.json", {"summary_rows": [], "sample_rows": []})
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {"policy_rows": [], "recommended_next_default_policy": {"selected_policy": "hetero_vote_3"}},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_resolver_breakdown.json",
        {"summary_rows": [], "sample_rows": [], "example_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_error_buckets.json",
        {"summary": {"error_count": 0}, "dataset_rows": [], "sample_rows": [], "example_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_solver_contributions.json",
        {"summary_rows": [], "sample_pattern_rows": []},
    )
    touch_figure_contract(tmp_path)
    write_jsonl(
        tmp_path / "turns" / "stage_a_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "solver_cot",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:00+00:00",
            }
        ],
    )
    write_jsonl(
        tmp_path / "turns" / "stage_b_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "legacy_stage_b",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:01+00:00",
            }
        ],
    )
    write_jsonl(
        tmp_path / "turns" / "judge_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "legacy_judge",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:02+00:00",
            }
        ],
    )
    write_jsonl(
        tmp_path / "turns" / "control_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "cot_1",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:03+00:00",
            }
        ],
    )
    write_jsonl(
        tmp_path / "turns" / "router_decisions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "policy_name": "legacy_policy",
                "triggered": True,
            },
        ],
    )
    write_jsonl(
        tmp_path / "views" / "predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "triggered": False,
                "early_exit": False,
                "communication_tokens_per_question": 0.0,
            },
        ],
    )

    result = validate_run(tmp_path)

    assert result["passed"] is False
    assert result["checks"]["router_empty_check"]["passed"] is False
    assert result["checks"]["stage_b_judge_empty_check"]["passed"] is False


def test_validate_run_accepts_fast_stage_a_only_contract(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="adaptive_sparse_mad")
    write_json(tmp_path / "progress.json", {"rate_limit_429_count": 0})
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "hetero_vote_3",
                    "accuracy_mean": 0.86,
                    "trigger_rate": 0.0,
                    "early_exit_rate": 1.0,
                    "changed_answer_rate": 0.0,
                    "corrected_rate": 0.0,
                    "harmed_rate": 0.0,
                    "judge_fallback_rate": 0.0,
                    "total_tokens_mean": 1200.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "cot_1",
                    "accuracy_mean": 0.84,
                    "trigger_rate": 0.0,
                    "early_exit_rate": 0.0,
                    "changed_answer_rate": 0.0,
                    "corrected_rate": 0.0,
                    "harmed_rate": 0.0,
                    "judge_fallback_rate": 0.0,
                    "total_tokens_mean": 300.0,
                },
            ]
        },
    )
    write_json(tmp_path / "diagnostics" / "router_eval.json", {"summary_rows": [], "sample_rows": []})
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {"policy_rows": [], "recommended_next_default_policy": {"selected_policy": "hetero_vote_3"}},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_resolver_breakdown.json",
        {"summary_rows": [], "sample_rows": [], "example_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_error_buckets.json",
        {"summary": {"error_count": 0}, "dataset_rows": [], "sample_rows": [], "example_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_solver_contributions.json",
        {"summary_rows": [], "sample_pattern_rows": []},
    )
    touch_figure_contract(tmp_path)
    write_jsonl(
        tmp_path / "turns" / "stage_a_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "solver_cot",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:00+00:00",
            }
        ],
    )
    write_jsonl(tmp_path / "turns" / "stage_b_turns.jsonl", [])
    write_jsonl(tmp_path / "turns" / "judge_turns.jsonl", [])
    write_jsonl(
        tmp_path / "turns" / "control_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "cot_1",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:01+00:00",
            }
        ],
    )
    write_jsonl(tmp_path / "turns" / "router_decisions.jsonl", [])
    write_jsonl(
        tmp_path / "views" / "predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "triggered": False,
                "early_exit": True,
                "communication_tokens_per_question": 0.0,
            },
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "cot_1",
                "method_kind": "control",
                "triggered": False,
                "early_exit": False,
                "communication_tokens_per_question": 0.0,
            },
        ],
    )

    result = validate_run(tmp_path)

    assert result["passed"] is True
    assert result["checks"]["router_empty_check"]["row_count"] == 0
    assert result["checks"]["stage_b_judge_empty_check"]["row_count"] == 0


def test_validate_run_accepts_adaptive_gate_router_rows(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="adaptive_sparse_mad",
        payload={
            "aggregate_methods": ["hetero_vote_3", "adaptive_gate_v4"],
        },
    )
    write_json(tmp_path / "progress.json", {"rate_limit_429_count": 0})
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "adaptive_gate_v4",
                    "accuracy_mean": 0.9,
                    "trigger_rate": 0.5,
                    "early_exit_rate": 0.5,
                    "changed_answer_rate": 0.1,
                    "corrected_rate": 0.1,
                    "harmed_rate": 0.0,
                    "judge_fallback_rate": 0.0,
                    "total_tokens_mean": 1400.0,
                }
            ]
        },
    )
    write_json(
        tmp_path / "diagnostics" / "router_eval.json",
        {"summary_rows": [{"dataset": "overall", "trigger_rate": 0.5}], "sample_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {
            "policy_rows": [{"dataset": "overall", "method_name": "adaptive_gate_v4"}],
            "recommended_next_default_policy": {"selected_policy": "adaptive_gate_v4"},
        },
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_resolver_breakdown.json",
        {"summary_rows": [], "sample_rows": [], "example_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_error_buckets.json",
        {"summary": {"error_count": 0}, "dataset_rows": [], "sample_rows": [], "example_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "stage_a_solver_contributions.json",
        {"summary_rows": [], "sample_pattern_rows": []},
    )
    touch_figure_contract(tmp_path)
    write_jsonl(
        tmp_path / "turns" / "stage_a_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "solver_cot",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:00+00:00",
            }
        ],
    )
    write_jsonl(tmp_path / "turns" / "stage_b_turns.jsonl", [])
    write_jsonl(tmp_path / "turns" / "judge_turns.jsonl", [])
    write_jsonl(
        tmp_path / "turns" / "control_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "cot_1",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:01+00:00",
            }
        ],
    )
    write_jsonl(
        tmp_path / "turns" / "router_decisions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "policy_name": "adaptive_gate_v4",
                "triggered": True,
                "selected_addon_solver": "solver_verify",
            }
        ],
    )
    write_jsonl(
        tmp_path / "views" / "predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "adaptive_gate_v4",
                "method_kind": "aggregate",
                "triggered": True,
                "early_exit": False,
                "communication_tokens_per_question": 0.0,
            },
        ],
    )

    result = validate_run(tmp_path)

    assert result["passed"] is True
    assert result["checks"]["router_empty_check"]["row_count"] == 1
