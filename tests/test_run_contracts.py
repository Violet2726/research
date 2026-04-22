from __future__ import annotations

from pathlib import Path
import json

from multi_agent_baselines.reporting import summarize_run as summarize_multi_agent
from multi_agent_baselines.validation import validate_run as validate_multi_agent
from sparc.reporting import summarize_run as summarize_sparc
from sparc.validation import validate_run as validate_sparc
from selective_comm.reporting import summarize_run as summarize_selective
from selective_comm.validation import validate_run as validate_selective
from single_agent_baselines.reporting import budget_fairness_check, summarize_run as summarize_single_agent
from single_agent_baselines.validation import validate_run as validate_single_agent


def test_single_agent_reporting_and_validation_use_method_name(tmp_path: Path) -> None:
    _write_jsonl(
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
                "method_name": "mv_5",
                "rerun_index": 0,
                "prompt_hash": "hash-1",
                "output_status": "ok",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {"dataset": "gsm8k", "method_name": "sc_5", "rerun_index": 0},
            {"dataset": "gsm8k", "method_name": "mv_5", "rerun_index": 0},
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
                        "method_name": "mv_5",
                        "total_tokens_mean": 10.0,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = summarize_single_agent(tmp_path)
    fairness = budget_fairness_check(tmp_path)
    validation = validate_single_agent(tmp_path)
    assert summary["dataset_count"] == 1
    assert fairness[0]["within_threshold"] is True
    assert validation["passed"] is True


def test_multi_agent_validation_contract(tmp_path: Path) -> None:
    _touch_json(tmp_path / "manifest.json", {})
    _write_jsonl(tmp_path / "agent_turns.jsonl", [{"output_status": "ok"}])
    _write_jsonl(tmp_path / "debate_messages.jsonl", [{"x": 1}])
    _write_jsonl(tmp_path / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    _touch_json(tmp_path / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(tmp_path / "cost_breakdown.json", {"rows": []})
    _touch_json(tmp_path / "debate_diagnostics.json", {"rows": []})
    assert summarize_multi_agent(tmp_path)["row_count"] == 1
    assert validate_multi_agent(tmp_path)["passed"] is True


def test_selective_comm_validation_contract(tmp_path: Path) -> None:
    _touch_json(tmp_path / "manifest.json", {})
    _write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok"}])
    _write_jsonl(tmp_path / "stage_b_turns.jsonl", [])
    _write_jsonl(tmp_path / "control_turns.jsonl", [])
    _write_jsonl(
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
    _write_jsonl(
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
    _touch_json(tmp_path / "policy_metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(
        tmp_path / "policy_diagnostics.json",
        {"recommended_next_default_policy": {"selected_policy": "hybrid_trigger"}},
    )
    _touch_json(tmp_path / "oracle_trigger_eval.json", {"summary_rows": []})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    (tmp_path / "trigger_report.md").write_text("# report\n", encoding="utf-8")

    assert summarize_selective(tmp_path)["row_count"] == 1
    assert validate_selective(tmp_path)["passed"] is True


def test_sparc_validation_contract(tmp_path: Path) -> None:
    _touch_json(
        tmp_path / "manifest.json",
        {
            "experiment_kind": "content_ablation",
        },
    )
    _write_jsonl(tmp_path / "stage_a_turns.jsonl", [{"output_status": "ok", "cache_hit": False, "dataset": "gsm8k", "method_name": "shared_stage_a", "sample_id": "gsm8k-00001"}])
    _write_jsonl(tmp_path / "message_packets.jsonl", [{"x": 1}])
    _write_jsonl(tmp_path / "belief_updates.jsonl", [{"output_status": "ok", "cache_hit": False, "dataset": "gsm8k", "method_name": "shared_stage_b::full_cot", "sample_id": "gsm8k-00001"}])
    _write_jsonl(tmp_path / "audit_turns.jsonl", [])
    _write_jsonl(
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
    _touch_json(tmp_path / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(tmp_path / "diagnostics.json", {"recommended_next_default": {"method_name": "full_cot"}})
    (tmp_path / "progress.json").write_text("{}", encoding="utf-8")
    (tmp_path / "paper_summary.csv").write_text("dataset,model_name,method_name,accuracy_mean,communication_tokens_mean,total_tokens_mean,calls_per_question_mean,acc_per_1k_tokens\n", encoding="utf-8")
    (tmp_path / "content_ablation_report.md").write_text("# report\n", encoding="utf-8")

    assert summarize_sparc(tmp_path)["row_count"] == 1
    assert validate_sparc(tmp_path)["passed"] is True


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _touch_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
