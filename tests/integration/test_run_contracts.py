"""覆盖运行产物字段契约与报告契约的测试。"""

from __future__ import annotations

from pathlib import Path
import json

from research_experiments.families.budget_comm.run.report import summarize_run as summarize_budget
from research_experiments.families.budget_comm.run.validate import validate_run as validate_budget
from research_experiments.families.comm_necessary.run.report import summarize_run as summarize_comm_necessary
from research_experiments.families.comm_necessary.run.validate import validate_run as validate_comm_necessary
from research_experiments.families.cue.run.report import summarize_run as summarize_cue
from research_experiments.families.cue.run.validate import validate_run as validate_cue
from research_experiments.families.free_mad_lite.run.report import summarize_run as summarize_free_mad
from research_experiments.families.free_mad_lite.run.validate import validate_run as validate_free_mad
from research_experiments.families.multi_agent.run.report import summarize_run as summarize_multi_agent
from research_experiments.families.multi_agent.run.validate import validate_run as validate_multi_agent
from research_experiments.families.sparc.run.report import summarize_run as summarize_sparc
from research_experiments.families.sparc.run.validate import validate_run as validate_sparc
from research_experiments.families.selective_comm.config import load_control_catalog, load_policies, load_protocol_config
from research_experiments.families.selective_comm.run.report import summarize_run as summarize_selective
from research_experiments.families.selective_comm.run.sample import _load_resume_seed_state
from research_experiments.families.selective_comm.run.validate import validate_run as validate_selective
from research_experiments.families.sid_lite.run.report import summarize_run as summarize_sid
from research_experiments.families.sid_lite.run.validate import validate_run as validate_sid
from research_experiments.families.single_agent.run.report import summarize_run as summarize_single_agent
from research_experiments.families.single_agent.run.validate import validate_run as validate_single_agent
from research_experiments.core.structured_outputs import (
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
    validate_or_recover_structured_output,
)
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_single_agent_reporting_and_validation_use_method_name(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {})
    write_jsonl(
        tmp_path / "raw_responses.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "sc_5",
                "rerun_index": 0,
                "prompt_hash": "hash-1",
                "output_status": "ok",
            },
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "cot_1",
                "rerun_index": 0,
                "prompt_hash": "hash-1",
                "output_status": "ok",
            },
        ],
    )
    write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {"dataset": "gsm8k", "method_name": "sc_5", "rerun_index": 0},
            {"dataset": "gsm8k", "method_name": "cot_1", "rerun_index": 0},
        ],
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "dataset": "gsm8k",
                        "model_name": "m",
                        "method_name": "sc_5",
                        "total_tokens_mean": 10.0,
                    },
                    {
                        "dataset": "gsm8k",
                        "model_name": "m",
                        "method_name": "cot_1",
                        "total_tokens_mean": 10.0,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    touch_figure_contract(tmp_path)

    summary = summarize_single_agent(tmp_path)
    validation = validate_single_agent(tmp_path)
    assert summary["dataset_count"] == 1
    assert validation["passed"] is True


def test_multi_agent_validation_contract(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {})
    write_jsonl(tmp_path / "agent_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(tmp_path / "debate_messages.jsonl", [{"x": 1}])
    write_jsonl(tmp_path / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(tmp_path / "cost_breakdown.json", {"rows": []})
    write_json(tmp_path / "debate_diagnostics.json", {"rows": []})
    touch_figure_contract(tmp_path)
    assert summarize_multi_agent(tmp_path)["row_count"] == 1
    assert validate_multi_agent(tmp_path)["passed"] is True


def test_selective_comm_validation_contract(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {})
    write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(tmp_path / "stage_b_turns.jsonl", [])
    write_jsonl(tmp_path / "control_turns.jsonl", [])
    write_jsonl(
        tmp_path / "trigger_decisions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "policy_name": "always_communicate",
                "triggered": True,
                "initial_disagreement": True,
                "any_invalid_confidence": False,
            },
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "policy_name": "disagreement_triggered",
                "triggered": True,
                "initial_disagreement": True,
                "any_invalid_confidence": False,
            },
        ],
    )
    write_jsonl(
        tmp_path / "policy_predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "always_communicate",
                "method_kind": "policy",
                "triggered": True,
                "early_exit": False,
                "communication_tokens_per_question": 1.0,
                "stage_a_trace_hash": "a",
                "stage_b_trace_hash_used": "b",
            },
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "disagreement_triggered",
                "method_kind": "policy",
                "triggered": True,
                "early_exit": False,
                "communication_tokens_per_question": 1.0,
                "stage_a_trace_hash": "a",
                "stage_b_trace_hash_used": "b",
            },
        ],
    )
    write_json(tmp_path / "policy_metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(
        tmp_path / "policy_diagnostics.json",
        {
            "voc_policy_rows": [],
            "recommended_next_default_policy": {"selected_policy": "voc_trigger_v2"},
        },
    )
    write_json(tmp_path / "oracle_trigger_eval.json", {"summary_rows": []})
    write_json(tmp_path / "policy_reference_summary.json", {"run_id": "seed", "overall_policies": {}})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    touch_figure_contract(tmp_path)

    assert summarize_selective(tmp_path)["row_count"] == 1
    assert validate_selective(tmp_path)["passed"] is True


def test_cue_validation_contract(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {})
    write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(tmp_path / "communication_turns.jsonl", [])
    write_jsonl(tmp_path / "audit_turns.jsonl", [])
    write_jsonl(
        tmp_path / "policy_predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "always_communicate",
                "stage_a_trace_hash": "stage-a",
            },
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "cue_v1",
                "stage_a_trace_hash": "stage-a",
            },
        ],
    )
    write_json(tmp_path / "policy_metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(tmp_path / "policy_diagnostics.json", {"policy_rows": []})
    write_json(tmp_path / "oracle_trigger_eval.json", {"summary_rows": []})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    touch_figure_contract(tmp_path)

    assert summarize_cue(tmp_path)["row_count"] == 1
    assert validate_cue(tmp_path)["passed"] is True


def test_selective_comm_resume_seed_state_keeps_only_complete_samples(tmp_path: Path) -> None:
    protocol = load_protocol_config("configs/families/selective_comm/protocols/shared_3a_r1.toml")
    policies = load_policies(
        [
            "configs/families/selective_comm/policies/always_communicate.toml",
            "configs/families/selective_comm/policies/disagreement_triggered.toml",
            "configs/families/selective_comm/policies/confidence_triggered.toml",
            "configs/families/selective_comm/policies/hybrid_trigger.toml",
        ]
    )
    controls = load_control_catalog("configs/families/selective_comm/controls/trigger_equal_budget.toml")
    write_json(tmp_path / "manifest.json", {"run_id": "seed-run"})

    stage_a_rows = [
        {"dataset": "gsm8k", "sample_id": "ok", "output_status": "ok"},
        {"dataset": "gsm8k", "sample_id": "ok", "output_status": "ok"},
        {"dataset": "gsm8k", "sample_id": "ok", "output_status": "ok"},
        {"dataset": "gsm8k", "sample_id": "bad", "output_status": "request_fail"},
        {"dataset": "gsm8k", "sample_id": "bad", "output_status": "ok"},
        {"dataset": "gsm8k", "sample_id": "bad", "output_status": "ok"},
    ]
    stage_b_rows = [
        {"dataset": "gsm8k", "sample_id": sample_id, "output_status": "ok"}
        for sample_id in ["ok", "ok", "ok", "bad", "bad", "bad"]
    ]
    control_rows = []
    for sample_id in ["ok", "bad"]:
        for method_name, budget_calls in [("mv_6", 6), ("sc_6", 6)]:
            control_rows.extend(
                {
                    "dataset": "gsm8k",
                    "sample_id": sample_id,
                    "method_name": method_name,
                    "output_status": "ok",
                }
                for _ in range(budget_calls)
            )
    trigger_rows = []
    prediction_rows = []
    for sample_id in ["ok", "bad"]:
        for policy_name in [
            "always_communicate",
            "disagreement_triggered",
            "confidence_triggered",
            "hybrid_trigger",
        ]:
            trigger_rows.append({"dataset": "gsm8k", "sample_id": sample_id, "policy_name": policy_name})
            prediction_rows.append(
                {
                    "dataset": "gsm8k",
                    "sample_id": sample_id,
                    "model_name": "deepseek/deepseek-v4-flash",
                    "method_name": policy_name,
                    "method_kind": "policy",
                }
            )
        trigger_rows.append({"dataset": "gsm8k", "sample_id": sample_id, "policy_name": "voc_trigger_v2"})
        prediction_rows.append(
            {
                "dataset": "gsm8k",
                "sample_id": sample_id,
                "model_name": "deepseek/deepseek-v4-flash",
                "method_name": "voc_trigger_v2",
                "method_kind": "policy",
            }
        )
        for control_name in ["mv_3", "mv_6", "sc_6"]:
            prediction_rows.append(
                {
                    "dataset": "gsm8k",
                    "sample_id": sample_id,
                    "model_name": "deepseek/deepseek-v4-flash",
                    "method_name": control_name,
                    "method_kind": "control",
                }
            )
        prediction_rows.append(
            {
                "dataset": "gsm8k",
                "sample_id": sample_id,
                "model_name": "deepseek/deepseek-v4-flash",
                "method_name": "bonus_control",
                "method_kind": "control",
            }
        )
    control_rows.append(
        {
            "dataset": "gsm8k",
            "sample_id": "ok",
            "method_name": "bonus_control",
            "output_status": "ok",
        }
    )

    write_jsonl(tmp_path / "stage_a_turns.jsonl", stage_a_rows)
    write_jsonl(tmp_path / "stage_b_turns.jsonl", stage_b_rows)
    write_jsonl(tmp_path / "control_turns.jsonl", control_rows)
    write_jsonl(tmp_path / "trigger_decisions.jsonl", trigger_rows)
    write_jsonl(tmp_path / "policy_predictions.jsonl", prediction_rows)

    state = _load_resume_seed_state(tmp_path, protocol, policies, controls)
    assert state.source_run_id == "seed-run"
    assert state.completed_sample_keys == {("gsm8k", "ok")}
    assert len(state.stage_a_turns) == 3
    assert len(state.stage_b_turns) == 3
    assert len(state.control_turns) == 12
    assert len(state.trigger_rows) == 4
    assert len(state.prediction_rows) == 7
    assert state.initial_completed_calls == 18
    assert state.initial_completed_predictions == 7


def test_selective_comm_reasoning_fallback_recovers_math_answer() -> None:
    payload = validate_or_recover_structured_output(
        "",
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
        dataset="gsm8k",
        provider_reasoning_text="The steps are straightforward. Final answer should be 68. Confidence 0.99.",
    )
    assert payload["final_answer"] == "68"
    assert payload["claim_span"] == "68"


def test_selective_comm_reasoning_fallback_recovers_strategy_answer() -> None:
    payload = validate_or_recover_structured_output(
        "",
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
        dataset="strategyqa",
        provider_reasoning_text="Avengers are Marvel, not DC. So the answer should be yes.",
    )
    assert payload["final_answer"] == "yes"


def test_selective_comm_reasoning_fallback_recovers_hotpot_answer() -> None:
    payload = validate_or_recover_structured_output(
        "",
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
        dataset="hotpotqa",
        provider_reasoning_text="Tivolis Koncertsal is at Tivoli Gardens, and the answer should be 15 August 1843.",
    )
    assert payload["final_answer"] == "15 August 1843"


def test_sparc_validation_contract(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "variant_name": "content_ablation",
        },
    )
    write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok", "cache_hit": False, "dataset": "gsm8k", "method_name": "shared_stage_a", "sample_id": "gsm8k-00001"}])
    write_jsonl(tmp_path / "message_packets.jsonl", [{"x": 1}])
    write_jsonl(tmp_path / "belief_updates.jsonl", [{"output_status": "ok", "cache_hit": False, "dataset": "gsm8k", "method_name": "shared_stage_b::full_cot", "sample_id": "gsm8k-00001"}])
    write_jsonl(tmp_path / "audit_turns.jsonl", [])
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "full_cot",
                "stage_a_trace_hash": "a",
                "audit_status": "not_applicable",
                "audit_tokens_per_question": 0.0,
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(tmp_path / "diagnostics.json", {"recommended_next_default": {"method_name": "full_cot"}})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    (tmp_path / "paper_summary.csv").write_text("dataset,model_name,method_name,accuracy_mean,communication_tokens_mean,total_tokens_mean,calls_per_question_mean,acc_per_1k_tokens\n", encoding="utf-8")
    touch_figure_contract(tmp_path)

    assert summarize_sparc(tmp_path)["row_count"] == 1
    assert validate_sparc(tmp_path)["passed"] is True


def test_sparc_auditing_ablation_paired_design_contract(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "variant_name": "auditing_ablation",
            "phase_metadata": {"split_suffix": "count20_seed42"},
            "aggregation_methods": ["majority_vote", "single_judge", "final_round_vote", "local_auditing"],
            "benchmarks": [
                {"slug": "gsm8k", "smoke_size": 1},
                {"slug": "strategyqa", "smoke_size": 1},
            ],
        },
    )
    write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(tmp_path / "message_packets.jsonl", [{"x": 1}])
    write_jsonl(tmp_path / "belief_updates.jsonl", [{"output_status": "ok"}])
    write_jsonl(
        tmp_path / "audit_turns.jsonl",
        [
            {"output_status": "ok", "method_name": "single_judge", "input_includes_full_debate": False},
            {"output_status": "ok", "method_name": "local_auditing", "input_includes_full_debate": False},
        ],
    )
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        _sparc_auditing_prediction_rows("gsm8k", "gsm8k-00001")
        + _sparc_auditing_prediction_rows("strategyqa", "strategyqa-00001"),
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "diagnostics.json", {"recommended_next_default": {"method_name": "local_auditing"}})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    (tmp_path / "paper_summary.csv").write_text("dataset,model_name,method_name,accuracy_mean,communication_tokens_mean,total_tokens_mean,calls_per_question_mean,acc_per_1k_tokens\n", encoding="utf-8")
    touch_figure_contract(tmp_path)

    validation = validate_sparc(tmp_path)
    assert validation["passed"] is True
    paired_check = validation["checks"]["auditing_ablation_paired_design_check"]
    assert paired_check["expected_count_per_method"] == 2
    assert paired_check["observed_count_per_method"] == {
        "majority_vote": 2,
        "single_judge": 2,
        "final_round_vote": 2,
        "local_auditing": 2,
    }


def test_budget_comm_validation_contract(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "context_view": {"track_name": "split_context"},
        },
    )
    write_jsonl(
        tmp_path / "sample_views.jsonl",
        [
            {
                "dataset": "strategyqa",
                "sample_id": "s1",
                "agent_id": 1,
                "includes_full_context": False,
                "full_context_hash": "full",
                "view_context_hash": "a",
                "coverage_items": ["fact_a"],
                "required_coverage_items": ["fact_a", "fact_b"],
            },
            {
                "dataset": "strategyqa",
                "sample_id": "s1",
                "agent_id": 2,
                "includes_full_context": False,
                "full_context_hash": "full",
                "view_context_hash": "b",
                "coverage_items": ["fact_b"],
                "required_coverage_items": ["fact_a", "fact_b"],
            },
            {
                "dataset": "strategyqa",
                "sample_id": "s1",
                "agent_id": 3,
                "includes_full_context": False,
                "full_context_hash": "full",
                "view_context_hash": "c",
                "coverage_items": [],
                "required_coverage_items": ["fact_a", "fact_b"],
            },
        ],
    )
    write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok", "stage_name": "stage_a", "method_name": "shared_stage_a", "dataset": "strategyqa", "sample_id": "s1"}])
    write_jsonl(
        tmp_path / "candidate_packets.jsonl",
        [
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "dala_lite", "agent_id": 1, "density_score": 0.1, "dala_assigned_mode": "keywords", "selected_mode": "keywords", "selected_packet_tokens": 1, "candidate_cost": 1},
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "dala_lite", "agent_id": 2, "density_score": 0.2, "dala_assigned_mode": "summary", "selected_mode": "summary", "selected_packet_tokens": 1, "candidate_cost": 1},
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "dala_lite", "agent_id": 3, "density_score": 0.3, "dala_assigned_mode": "full", "selected_mode": "full", "selected_packet_tokens": 1, "candidate_cost": 1},
        ],
    )
    write_jsonl(
        tmp_path / "auction_decisions.jsonl",
        [
            {
                "dataset": "strategyqa",
                "sample_id": "s1",
                "method_name": "budget_random",
                "selection_rule": "knapsack_random_full",
                "round_budget_tokens": 3,
                "winner_agent_ids": [1, 2, 3],
                "candidate_scores": {"1": 0.1, "2": 0.2, "3": 0.3},
                "candidate_costs": {"1": 1, "2": 1, "3": 1},
                "total_cost": 3,
            },
            {
                "dataset": "strategyqa",
                "sample_id": "s1",
                "method_name": "budget_confidence",
                "selection_rule": "knapsack_confidence_full",
                "round_budget_tokens": 3,
                "winner_agent_ids": [1, 2, 3],
                "candidate_scores": {"1": 0.1, "2": 0.2, "3": 0.3},
                "candidate_costs": {"1": 1, "2": 1, "3": 1},
                "total_cost": 3,
            },
            {
                "dataset": "strategyqa",
                "sample_id": "s1",
                "method_name": "dala_lite",
                "selection_rule": "knapsack_density_tiered",
                "round_budget_tokens": 3,
                "winner_agent_ids": [1, 2, 3],
                "candidate_scores": {"1": 0.1, "2": 0.2, "3": 0.3},
                "candidate_costs": {"1": 1, "2": 1, "3": 1},
                "total_cost": 3,
            },
        ],
    )
    write_jsonl(tmp_path / "belief_updates.jsonl", [{"output_status": "ok", "stage_name": "stage_b", "method_name": "all_to_all_full", "dataset": "strategyqa", "sample_id": "s1"}])
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "mv_3"},
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "all_to_all_full"},
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "budget_random"},
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "budget_confidence"},
            {"dataset": "strategyqa", "sample_id": "s1", "method_name": "dala_lite"},
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "budget_diagnostics.json", {"full_dala_gate": {"ready_for_full_dala": False}})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    touch_figure_contract(tmp_path)
    (tmp_path / "paper_summary.csv").write_text("dataset,track_name,model_name,method_name,accuracy_mean,communication_tokens_mean,total_tokens_mean,calls_per_question_mean,acc_per_1k_tokens\n", encoding="utf-8")

    assert summarize_budget(tmp_path)["row_count"] == 1
    assert validate_budget(tmp_path)["passed"] is True


def test_sid_lite_validation_contract(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {"methods": ["mv_3", "always_full", "compression_only", "sid_lite"]},
    )
    write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(
        tmp_path / "message_packets.jsonl",
        [{"dataset": "gsm8k", "sample_id": "s1", "agent_id": 1, "approx_packet_tokens": 5, "token_cap": 10}],
    )
    write_jsonl(tmp_path / "belief_updates.jsonl", [{"output_status": "ok"}])
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": method,
                "stage_a_trace_hash": "stage-a",
                "early_exit": method in {"mv_3", "sid_lite"},
                "communication_tokens_per_question": 0.0 if method in {"mv_3", "sid_lite"} else 1.0,
                "any_invalid_confidence": False,
            }
            for method in ["mv_3", "always_full", "compression_only", "sid_lite"]
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "diagnostics.json", {"sid_early_exit_rate": 1.0})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    touch_figure_contract(tmp_path)

    assert summarize_sid(tmp_path)["row_count"] == 1
    assert validate_sid(tmp_path)["passed"] is True


def test_free_mad_lite_validation_contract(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "methods": [
                "mv_3_initial",
                "vanilla_mad_r1_final_vote",
                "anti_conformity_final_vote",
                "free_mad_lite_llm_trajectory",
            ],
            "protocol": {"debate_rounds": 1},
            "anti_conformity_prompt_hash": "hash",
        },
    )
    write_jsonl(tmp_path / "agent_turns.jsonl", [{"output_status": "ok", "role": "initial"}])
    write_jsonl(tmp_path / "debate_messages.jsonl", [{"x": 1}])
    write_jsonl(
        tmp_path / "trajectory_scores.jsonl",
        [{"dataset": "gsm8k", "sample_id": "s1", "output_status": "ok", "judge_fallback_used": False}],
    )
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": method,
                "stage_a_trace_hash": "stage-a",
            }
            for method in [
                "mv_3_initial",
                "vanilla_mad_r1_final_vote",
                "anti_conformity_final_vote",
                "free_mad_lite_llm_trajectory",
            ]
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "diagnostics.json", {"judge_fallback_rate": 0.0})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    touch_figure_contract(tmp_path)

    assert summarize_free_mad(tmp_path)["row_count"] == 1
    assert validate_free_mad(tmp_path)["passed"] is True


def test_comm_necessary_validation_contract(tmp_path: Path) -> None:
    methods = [
        "full_context_single",
        "split_no_comm_mv3",
        "answer_only_exchange",
        "evidence_exchange",
        "full_packet_exchange",
    ]
    write_json(
        tmp_path / "manifest.json",
        {
            "methods": methods,
            "requests_per_minute_limit": 60,
            "tokens_per_minute_limit": 2000000,
        },
    )
    write_jsonl(
        tmp_path / "sample_views.jsonl",
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "agent_id": 1,
                "view_kind": "supporting_shard_a",
                "includes_full_context": False,
                "full_context_hash": "full",
                "view_context_hash": "a",
                "coverage_titles": ["A"],
                "required_titles": ["A", "B"],
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "agent_id": 2,
                "view_kind": "supporting_shard_b",
                "includes_full_context": False,
                "full_context_hash": "full",
                "view_context_hash": "b",
                "coverage_titles": ["B"],
                "required_titles": ["A", "B"],
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "agent_id": 3,
                "view_kind": "distractor_titles_only",
                "includes_full_context": False,
                "full_context_hash": "full",
                "view_context_hash": "c",
                "coverage_titles": [],
                "required_titles": ["A", "B"],
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "agent_id": 0,
                "view_kind": "full_context",
                "includes_full_context": True,
                "full_context_hash": "full",
                "view_context_hash": "full",
                "coverage_titles": ["A", "B"],
                "required_titles": ["A", "B"],
            },
        ],
    )
    write_jsonl(
        tmp_path / "stage_a_turns.jsonl",
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": "shared_split_stage_a",
                "stage_name": "stage_a",
                "agent_id": 1,
                "output_status": "ok",
                "cache_hit": False,
                "request_started_at": "2026-04-24T00:00:00+00:00",
                "estimated_request_tokens": 100,
            }
        ],
    )
    write_jsonl(
        tmp_path / "message_packets.jsonl",
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": "evidence_exchange",
                "agent_id": 1,
                "approx_packet_tokens": 10,
                "token_cap": 128,
            }
        ],
    )
    write_jsonl(
        tmp_path / "stage_b_turns.jsonl",
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": "evidence_exchange",
                "agent_id": 1,
                "output_status": "ok",
                "cache_hit": False,
                "request_started_at": "2026-04-24T00:00:01+00:00",
                "estimated_request_tokens": 100,
            }
        ],
    )
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": method,
                "prediction": "alpha",
                "supporting_facts": [["A", 0], ["B", 0]],
            }
            for method in methods
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "diagnostics.json", {"key_deltas": []})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    touch_figure_contract(tmp_path)
    (tmp_path / "paper_summary.csv").write_text("dataset,model_name,method_name,answer_em_mean\n", encoding="utf-8")
    hotpot_dir = tmp_path / "hotpot_predictions"
    hotpot_dir.mkdir()
    for method in methods:
        write_json(hotpot_dir / f"{method}.json", {"answer": {"h1": "alpha"}, "sp": {"h1": [["A", 0], ["B", 0]]}})

    assert summarize_comm_necessary(tmp_path)["row_count"] == 1
    assert validate_comm_necessary(tmp_path)["passed"] is True


def _sparc_auditing_prediction_rows(dataset: str, sample_id: str) -> list[dict[str, object]]:
    common = {
        "dataset": dataset,
        "sample_id": sample_id,
        "stage_a_trace_hash": f"stage-a-{dataset}-{sample_id}",
        "audit_tokens_per_question": 0.0,
    }
    return [
        {
            **common,
            "method_name": "majority_vote",
            "audit_status": "not_applicable",
            "stage_b_trace_hash_used": None,
        },
        {
            **common,
            "method_name": "single_judge",
            "audit_status": "judge",
            "audit_tokens_per_question": 1.0,
            "stage_b_trace_hash_used": f"stage-b-{dataset}-{sample_id}",
        },
        {
            **common,
            "method_name": "final_round_vote",
            "audit_status": "not_applicable",
            "stage_b_trace_hash_used": f"stage-b-{dataset}-{sample_id}",
        },
        {
            **common,
            "method_name": "local_auditing",
            "audit_status": "resolved",
            "audit_tokens_per_question": 1.0,
            "stage_b_trace_hash_used": f"stage-b-{dataset}-{sample_id}",
        },
    ]



