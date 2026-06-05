"""覆盖 `selective_comm` family 的基本摘要链路。"""

from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json, write_registered_family_manifest

from research_experiments.families.selective_comm.run.report import summarize_run
from research_experiments.families.selective_comm.run.sample import (
    _apply_debate_turn_fallback,
    _select_next_default_policy,
)


def test_summarize_run_reads_policy_metrics(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="selective_comm")
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {"dataset": "gsm8k", "method_name": "hybrid_trigger", "accuracy_mean": 0.8},
                {"dataset": "overall", "method_name": "hybrid_trigger", "accuracy_mean": 0.8},
            ]
        },
    )

    payload = summarize_run(tmp_path)

    assert payload["row_count"] == 2
    assert payload["datasets"] == ["gsm8k", "overall"]


def test_apply_debate_turn_fallback_keeps_previous_answer_after_reasoning_only_output() -> None:
    row = {
        "stage_name": "stage_b",
        "output_status": "schema_fail",
        "request_error": None,
        "assistant_text": '{"reasoning":"Peers agree with my earlier answer."}',
        "prediction": "",
        "normalized_answer": "",
        "reasoning": "",
        "confidence_raw": None,
        "confidence_raw_display": "",
        "confidence_value": None,
        "confidence_valid": False,
        "confidence_source": "missing",
        "claim_span": None,
        "uncertainty_type": None,
        "key_evidence": None,
        "uncertain_point": None,
        "validated_output": {},
    }
    previous_row = {
        "normalized_answer": "yes",
        "reasoning": "Nickel's boiling point is below the outer-core temperature.",
        "confidence_raw": 0.95,
        "confidence_raw_display": 0.95,
        "confidence_value": 0.95,
        "confidence_valid": True,
        "confidence_source": "unit_interval",
        "claim_span": "yes",
        "uncertainty_type": "commonsense_gap",
        "key_evidence": "Outer-core temperature exceeds nickel boiling point.",
        "uncertain_point": None,
        "validated_output": {
            "final_answer": "yes",
            "confidence_raw": 0.95,
            "reasoning": "Nickel would boil in the outer core.",
            "claim_span": "yes",
            "uncertainty_type": "commonsense_gap",
            "key_evidence": "Outer-core temperature exceeds nickel boiling point.",
            "uncertain_point": None,
        },
    }

    _apply_debate_turn_fallback(row, previous_row)

    assert row["output_status"] == "ok"
    assert row["prediction"] == "yes"
    assert row["normalized_answer"] == "yes"
    assert row["debate_fallback"] == "kept_previous_answer_after_reasoning_only_output"
    validated_output = row["validated_output"]
    assert isinstance(validated_output, dict)
    assert validated_output["final_answer"] == "yes"


def test_select_next_default_policy_avoids_dominated_voc_policy() -> None:
    metric_lookup = {
        ("overall", "always_communicate"): {"accuracy_mean": 0.766667, "total_tokens_mean": 3899.566667},
        ("overall", "disagreement_triggered"): {"accuracy_mean": 0.766667, "total_tokens_mean": 3008.016667},
        ("overall", "hybrid_trigger"): {"accuracy_mean": 0.766667, "total_tokens_mean": 3008.016667},
        ("overall", "voc_trigger_v2"): {"accuracy_mean": 0.766667, "total_tokens_mean": 3252.016667},
        ("overall", "confidence_triggered"): {"accuracy_mean": 0.7, "total_tokens_mean": 2449.466667},
    }

    decision = _select_next_default_policy(metric_lookup)

    assert decision["selected_policy"] == "hybrid_trigger"
    assert decision["rule_passed"] is True
