from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl, write_registered_family_manifest

from research_experiments.families.adaptive_sparse_mad.run.validate import validate_run


def _adaptive_router_summary_row(policy_name: str) -> dict[str, object]:
    return {
        "dataset": "overall",
        "policy_name": policy_name,
        "trigger_rate": 0.5,
        "false_consensus_risk_rate": 0.25,
        "changed_answer_rate": 0.1,
        "debate_trigger_rate": 0.0,
        "debate_rounds_mean": 0.0,
        "corrected_count": 1,
        "harmed_count": 0,
        "probe_accepted_count": 1,
        "debate_after_probe_triggered_count": 0,
        "avg_support_gap": 0.2,
        "avg_avg_confidence": 0.8,
        "addon_solver_counts": {"solver_verify": 1},
        "stage_a_accuracy": 0.75,
        "pre_route_accuracy": 0.8,
        "stage_a_oracle_accuracy": 0.9,
        "oracle_gap_vs_hetero": 0.15,
        "oracle_gap_capture_by_preroute": 0.333333,
        "high_value_trigger_precision": 0.8,
        "high_value_trigger_recall": 0.5,
        "all_three_wrong_trigger_rate": 0.25,
        "correct_to_wrong_rate_on_stage_a_correct": 0.0,
        "stage_a_oracle_3core": 0.9,
        "stage_a_oracle_5expert": 0.95,
        "all_three_wrong_before_expansion_rate": 0.1,
        "all_three_wrong_after_expansion_rate": 0.05,
        "specialist_pair_override_precision": 0.8,
        "arbiter_precision": 0.7,
    }


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
    assert result["checks"]["stage_b_judge_empty_check"]["files_present_count"] == 0


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
        {"summary_rows": [_adaptive_router_summary_row("adaptive_gate_v4")], "sample_rows": [], "bucket_rows": []},
    )
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {
            "policy_rows": [{"dataset": "overall", "method_name": "adaptive_gate_v4"}],
            "router_summary_rows": [_adaptive_router_summary_row("adaptive_gate_v4")],
            "router_bucket_rows": [],
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


def test_validate_run_accepts_v6_probe_router_rows(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="adaptive_sparse_mad",
        payload={
            "aggregate_methods": ["hetero_vote_3", "adaptive_sparse_rescue_probe_v1"],
        },
    )
    write_json(tmp_path / "progress.json", {"rate_limit_429_count": 0})
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "adaptive_sparse_rescue_probe_v1",
                    "accuracy_mean": 0.9,
                    "trigger_rate": 0.5,
                    "early_exit_rate": 0.5,
                    "changed_answer_rate": 0.1,
                    "corrected_rate": 0.1,
                    "harmed_rate": 0.0,
                    "total_tokens_mean": 1400.0,
                }
            ]
        },
    )
    write_json(
        tmp_path / "diagnostics" / "router_eval.json",
        {
            "summary_rows": [_adaptive_router_summary_row("adaptive_sparse_rescue_probe_v1")],
            "sample_rows": [],
            "bucket_rows": [],
        },
    )
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {
            "policy_rows": [{"dataset": "overall", "method_name": "adaptive_sparse_rescue_probe_v1"}],
            "router_summary_rows": [_adaptive_router_summary_row("adaptive_sparse_rescue_probe_v1")],
            "router_bucket_rows": [],
            "recommended_next_default_policy": {"selected_policy": "adaptive_sparse_rescue_probe_v1"},
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
                "policy_name": "adaptive_sparse_rescue_probe_v1",
                "triggered": True,
                "selected_addon_solver": "solver_verify",
                "false_consensus_risk": True,
                "answer_family_count": 1,
                "slot_mismatch_risk": False,
                "probe_accepted": True,
                "debate_after_probe_triggered": False,
            }
        ],
    )
    write_jsonl(tmp_path / "turns" / "debate_messages.jsonl", [])
    write_jsonl(
        tmp_path / "views" / "predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "adaptive_sparse_rescue_probe_v1",
                "method_kind": "aggregate",
                "triggered": True,
                "debate_triggered": False,
                "early_exit": False,
                "communication_tokens_per_question": 0.0,
            },
        ],
    )

    result = validate_run(tmp_path)

    assert result["passed"] is True
    assert result["checks"]["router_empty_check"]["row_count"] == 1


def test_validate_run_requires_debate_messages_for_sparse_debate_manifest(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="adaptive_sparse_mad",
        payload={
            "aggregate_methods": ["hetero_vote_3", "adaptive_sparse_debate_v1"],
        },
    )
    write_json(tmp_path / "progress.json", {"rate_limit_429_count": 0})
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "adaptive_sparse_debate_v1",
                    "accuracy_mean": 0.9,
                    "trigger_rate": 0.0,
                    "early_exit_rate": 1.0,
                    "changed_answer_rate": 0.0,
                    "corrected_rate": 0.0,
                    "harmed_rate": 0.0,
                    "total_tokens_mean": 1200.0,
                }
            ]
        },
    )
    write_json(
        tmp_path / "diagnostics" / "router_eval.json",
        {
            "summary_rows": [_adaptive_router_summary_row("adaptive_sparse_debate_v1")],
            "sample_rows": [],
            "bucket_rows": [],
        },
    )
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {
            "policy_rows": [{"dataset": "overall", "method_name": "adaptive_sparse_debate_v1"}],
            "router_summary_rows": [_adaptive_router_summary_row("adaptive_sparse_debate_v1")],
            "router_bucket_rows": [],
            "recommended_next_default_policy": {"selected_policy": "adaptive_sparse_debate_v1"},
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
                "method_name": "adaptive_sparse_debate_v1",
                "method_kind": "aggregate",
                "triggered": False,
                "debate_triggered": False,
                "early_exit": True,
                "communication_tokens_per_question": 0.0,
            },
        ],
    )

    missing_result = validate_run(tmp_path)

    assert missing_result["passed"] is False
    assert "turns/debate_messages.jsonl" in missing_result["missing_files"]

    write_jsonl(tmp_path / "turns" / "debate_messages.jsonl", [])
    present_result = validate_run(tmp_path)

    assert present_result["passed"] is True
    assert present_result["checks"]["debate_messages_check"]["required"] is True


def test_validate_run_accepts_v7_meta_route_router_rows(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="adaptive_sparse_mad",
        payload={
            "aggregate_methods": ["hetero_vote_3", "adaptive_sparse_meta_route_v7"],
        },
    )
    write_json(tmp_path / "progress.json", {"rate_limit_429_count": 0})
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "adaptive_sparse_meta_route_v7",
                    "accuracy_mean": 0.82,
                    "trigger_rate": 0.3,
                    "early_exit_rate": 0.7,
                    "changed_answer_rate": 0.08,
                    "corrected_rate": 0.08,
                    "harmed_rate": 0.01,
                    "total_tokens_mean": 2100.0,
                }
            ]
        },
    )
    router_summary = _adaptive_router_summary_row("adaptive_sparse_meta_route_v7")
    write_json(
        tmp_path / "diagnostics" / "router_eval.json",
        {
            "summary_rows": [router_summary],
            "sample_rows": [],
            "bucket_rows": [
                {
                    "dataset": "overall",
                    "policy_name": "adaptive_sparse_meta_route_v7",
                    "stage_a_error_bucket": "clean_pseudo_majority",
                    "question_count": 10,
                    "trigger_rate": 0.8,
                    "changed_answer_rate": 0.5,
                    "corrected_count": 4,
                    "harmed_count": 0,
                    "override_accepted_rate": 0.4,
                }
            ],
        },
    )
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {
            "policy_rows": [{"dataset": "overall", "method_name": "adaptive_sparse_meta_route_v7"}],
            "router_summary_rows": [router_summary],
            "router_bucket_rows": [
                {
                    "dataset": "overall",
                    "policy_name": "adaptive_sparse_meta_route_v7",
                    "stage_a_error_bucket": "clean_pseudo_majority",
                    "trigger_rate": 0.8,
                }
            ],
            "recommended_next_default_policy": {"selected_policy": "adaptive_sparse_meta_route_v7"},
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
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "method_name": "solver_cot",
                "output_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:00+00:00",
            }
        ],
    )
    write_jsonl(
        tmp_path / "turns" / "control_turns.jsonl",
        [
            {
                "dataset": "hotpotqa",
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
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "policy_name": "adaptive_sparse_meta_route_v7",
                "triggered": True,
                "selected_addon_solver": "solver_evidence",
            }
        ],
    )
    write_jsonl(
        tmp_path / "views" / "predictions.jsonl",
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "method_name": "adaptive_sparse_meta_route_v7",
                "method_kind": "aggregate",
                "triggered": True,
                "early_exit": False,
                "communication_tokens_per_question": 0.0,
            },
        ],
    )

    result = validate_run(tmp_path)

    assert result["passed"] is True
    assert result["checks"]["router_diagnostics_check"]["passed"] is True
