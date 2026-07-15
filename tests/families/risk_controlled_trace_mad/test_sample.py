import json
from types import SimpleNamespace

import pytest

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.risk_controlled_trace_mad.config import load_experiment_config, load_protocol_config
from research_experiments.families.risk_controlled_trace_mad.run import hsgsa_sample
from research_experiments.families.risk_controlled_trace_mad.run.hsgsa_sample import parse_blind_reviewer_output
from research_experiments.families.risk_controlled_trace_mad.run.sample import _parse_audit, _parse_selector


def test_selector_rejects_anchor_and_novel_labels() -> None:
    mapping = {"A": "1", "B": "2"}
    assert (
        _parse_selector(json.dumps({"challenger_label": "B", "decisive_difference": "check"}), mapping, "1")[
            "challenger_answer"
        ]
        == "2"
    )
    with pytest.raises(ValueError):
        _parse_selector(json.dumps({"challenger_label": "A", "decisive_difference": "check"}), mapping, "1")
    with pytest.raises(ValueError):
        _parse_selector(json.dumps({"challenger_label": "C", "decisive_difference": "check"}), mapping, "1")


def test_audit_maps_random_labels_back_to_existing_answers() -> None:
    raw = json.dumps(
        {
            "preferred_label": "B",
            "decisive_claim": "2 + 3 = 5",
            "evidence": [
                {
                    "target_label": "B",
                    "claim_kind": "support",
                    "test_type": "arithmetic",
                    "payload": {"left": "2+3", "right": "5", "relation": "eq"},
                }
            ],
        }
    )
    result = _parse_audit(raw, {"A": "4", "B": "5"}, "Compute 2 + 3.")
    assert result["preferred_answer"] == "5"
    assert result["evidence_results"][0]["status"] == "pass"


def test_blind_pick_drives_selection_without_requiring_explanation() -> None:
    result = parse_blind_reviewer_output(
        "PICK: B\nFINAL_ANSWER: (b)",
        label_to_key={"A": "a", "B": "b"},
        label_to_answer={"A": "a", "B": "b"},
        dataset="bbeh",
    )
    assert result["picked_answer_class_key"] == "b"
    with pytest.raises(ValueError):
        parse_blind_reviewer_output(
            "PICK: C\nFINAL_ANSWER: c",
            label_to_key={"A": "a"},
            label_to_answer={"A": "a"},
            dataset="bbeh",
        )


def test_hsgsa_uses_one_shared_eleven_call_graph_and_never_promotes_new_answer(monkeypatch) -> None:
    experiment = load_experiment_config(
        "configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml"
    )
    protocol = load_protocol_config(experiment.protocol)
    stage_answers = ["a", "a", "a", "b", "b"]
    resample_answers = ["b", "b", "a"]

    def fake_solve(**kwargs):
        source = stage_answers if kwargs["method_name"] == "hsgsa_stage_a_shared" else resample_answers
        answer = source[kwargs["agent_id"] - 1]
        return {
            "run_id": kwargs["run_id"], "dataset": kwargs["dataset"], "split": kwargs["split_name"],
            "sample_id": kwargs["sample"].sample_id, "method_name": kwargs["method_name"],
            "method_type": kwargs["method_type"], "round_index": kwargs["round_index"],
            "agent_id": kwargs["agent_id"], "role": kwargs["role"], "model_lineage": "mimo",
            "model_name": "mimo-v2.5", "prompt_hash": f"p-{kwargs['method_name']}-{kwargs['agent_id']}",
            "normalized_answer": answer, "prediction": answer, "assistant_text": f"reason {answer}",
            "validated_output": {"reasoning": f"reason {answer}", "final_answer": answer},
            "output_status": "ok", "protocol_parse_status": "ok", "provider_abstention": False,
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "latency_ms": 1,
            "network_attempt_count": 0,
        }

    def fake_review(**kwargs):
        picked_label = next(label for label, key in kwargs["label_to_key"].items() if key == "b")
        return {
            "run_id": kwargs["run_id"], "dataset": kwargs["dataset"], "split": kwargs["split_name"],
            "sample_id": kwargs["sample"].sample_id, "method_name": "hsgsa_blind_reviewer_shared",
            "method_type": "support_blind_review", "round_index": 1, "agent_id": kwargs["reviewer_index"],
            "role": "blind_reviewer", "model_lineage": "mimo", "model_name": "mimo-v2.5",
            "prompt_hash": f"review-{kwargs['reviewer_index']}", "normalized_answer": "b",
            "assistant_text": f"PICK: {picked_label}\nFINAL_ANSWER: invented-shadow",
            "validated_output": {"pick": picked_label, "picked_answer_class_key": "b", "picked_answer": "b",
                                 "generated_final_answer": "invented-shadow"},
            "output_status": "ok", "protocol_parse_status": "ok", "provider_abstention": False,
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "latency_ms": 1,
            "network_attempt_count": 0,
        }

    monkeypatch.setattr(hsgsa_sample, "_execute_free_text_turn", fake_solve)
    monkeypatch.setattr(hsgsa_sample, "_execute_reviewer_turn", fake_review)
    sample = DatasetSample("bbeh", "s", "Choose.", "b", "", {"task": "task"})
    turns, _, _, predictions = hsgsa_sample.run_hsgsa_sample(
        sample,
        run_id="r",
        dataset="bbeh",
        split_name="count1_seed42",
        experiment=experiment,
        protocol=protocol,
        active_methods=experiment.methods,
        endpoint=SimpleNamespace(backbone=SimpleNamespace(name="mimo-v2.5")),
        network_budget=hsgsa_sample.NetworkAttemptBudget(50_000),
    )
    by_method = {row["method_name"]: row for row in predictions}
    assert len(turns) == 11
    assert by_method["hsgsa_unanimous_3"]["logical_calls_per_question"] == 8
    assert by_method["adaptive_sc_8"]["logical_calls_per_question"] == 8
    assert by_method["hsgsa_unanimous_3"]["prediction"] == "b"
    assert by_method["hsgsa_unanimous_3"]["novel_answer"] is False
    assert "invented-shadow" in by_method["hsgsa_unanimous_3"]["reviewer_generated_answers_shadow"]
