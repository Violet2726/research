from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.cred_v.config import load_experiment_config, load_protocol_config
from research_experiments.families.cred_v.run import sample as sample_runner


def test_certificate_proposals_use_two_configured_models_and_attach_checker_results(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_certificate_turn)
    sample = DatasetSample("math500", "m1", "Evaluate 1/2 + 1/3.", "5/6", "", {})
    protocol = SimpleNamespace(
        certificate_modes=("math_symbolic",),
        certificate_proposer_model_refs=("pro", "qwen"),
        certificate_dsl_version="math_cert_v1",
        max_certificate_calls=2,
        verifier_temperature=0.0,
        top_p=1.0,
        verifier_max_tokens=256,
    )

    rows = sample_runner._run_cvs_certificate_proposals(
        run_id="r",
        benchmark_slug="math500",
        split_name="s",
        sample=sample,
        experiment=SimpleNamespace(global_seed=42, cred_verification_output_protocol="json_object_answer_v3"),
        protocol=protocol,
        stage_rows=[],
        leader_answer="1/2",
        verifier_runtimes=[_runtime("pro"), _runtime("qwen")],
    )

    assert [row["certificate_model_ref"] for row in rows] == ["pro", "qwen"]
    assert all(row["certificate_validation"]["valid"] for row in rows)
    assert all(row["certificate_validation"]["leader_pass"] is False for row in rows)


def test_hotpot_certificate_proposals_use_context_bound_checker(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_hotpot_certificate_turn)
    raw_context = {
        "title": ["Expedition"],
        "sentences": [["The expedition was led by Captain John Underhill."]],
    }
    sample = DatasetSample(
        "hotpotqa",
        "h1",
        "Who led the expedition?",
        "Captain John Underhill",
        "[Expedition] The expedition was led by Captain John Underhill.",
        {"raw_context": raw_context},
    )
    protocol = SimpleNamespace(
        certificate_modes=("hotpot_context_span",),
        certificate_proposer_model_refs=("pro", "qwen"),
        certificate_dsl_version="math_cert_v1",
        max_certificate_calls=2,
        verifier_temperature=0.0,
        top_p=1.0,
        verifier_max_tokens=256,
    )

    rows = sample_runner._run_cvs_certificate_proposals(
        run_id="r",
        benchmark_slug="hotpotqa",
        split_name="s",
        sample=sample,
        experiment=SimpleNamespace(global_seed=42, cred_verification_output_protocol="json_object_answer_v3"),
        protocol=protocol,
        stage_rows=[],
        leader_answer="John Underhill",
        verifier_runtimes=[_runtime("pro"), _runtime("qwen")],
    )

    assert len(rows) == 2
    assert all(row["certificate_mode"] == "hotpot_context_span" for row in rows)
    assert all(row["certificate_validation"]["valid"] for row in rows)


def test_isp_shadow_rows_are_short_json_calls_and_keep_agent_answers(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_isp_turn)
    sample = DatasetSample("strategyqa", "s1", "Can X happen?", "yes", "", {})
    stage_rows = [
        {"normalized_answer": "yes", "prediction": "yes"},
        {"normalized_answer": "no", "prediction": "no"},
    ]
    protocol = SimpleNamespace(
        max_shadow_calls=2,
        verifier_temperature=0.0,
        top_p=1.0,
        verifier_max_tokens=128,
    )

    rows = sample_runner._run_isp_shadow_rows(
        run_id="r",
        benchmark_slug="strategyqa",
        split_name="s",
        sample=sample,
        experiment=SimpleNamespace(global_seed=42, cred_verification_output_protocol="json_object_answer_v3"),
        protocol=protocol,
        stage_rows=stage_rows,
        backbone=SimpleNamespace(name="base"),
        provider=None,
        cache=None,
        throttle=None,
    )

    assert len(rows) == 2
    assert [row["isp_own_answer"] for row in rows] == ["yes", "no"]
    assert all(row["isp_validation_pass"] for row in rows)


def test_cvs_methods_share_stage_a_and_attribute_only_their_own_calls(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_full_cvs_turn)
    experiment = load_experiment_config("configs/families/cred_v/experiments/cred_v_cvs_v1.toml")
    protocol = replace(load_protocol_config(experiment.protocol), max_trigger_rate=1.0)
    sample = DatasetSample("math500", "m1", "Evaluate 1/2 + 1/3.", "5/6", "", {})

    turns, _, _, predictions = sample_runner._run_cred_sample(
        sample,
        run_id="r",
        benchmark_slug="math500",
        split_name="s",
        experiment=experiment,
        protocol=protocol,
        backbone=SimpleNamespace(name="base"),
        provider=None,
        cache=None,
        throttle=None,
        verifier_runtimes=[_runtime("xiaomimimo/mimo-v2.5-pro"), _runtime("dashscope/qwen-flash")],
    )
    by_method = {row["method_name"]: row for row in predictions}

    assert sum(1 for row in turns if row["method_name"] == "cred_stage_a") == 5
    assert by_method["cred_rfs_vote_5_anchor"]["prediction"] == "1/2"
    assert by_method["cred_rfs_repair_only_v6"]["prediction"] == "1/2"
    assert by_method["cred_cvs_budget_matched_vote_v1"]["prediction"] == "5/6"
    assert by_method["cred_cvs_v1"]["prediction"] == "5/6"
    assert by_method["cred_isp_shadow_v1"]["prediction"] == "1/2"
    assert by_method["cred_rfs_vote_5_anchor"]["calls_per_question"] == 5
    assert by_method["cred_rfs_repair_only_v6"]["calls_per_question"] == 5
    assert by_method["cred_cvs_v1"]["calls_per_question"] == 7
    assert by_method["cred_cvs_v1"]["cross_model_agreement"] == 2
    assert by_method["cred_cvs_v1"]["semantic_entropy"] > 0.0
    assert by_method["cred_isp_shadow_v1"]["calls_per_question"] == 5


def _runtime(model_ref: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_ref=model_ref,
        backbone=SimpleNamespace(name=model_ref),
        provider=None,
        cache=None,
        throttle=None,
    )


def _fake_certificate_turn(**kwargs):
    payload = {
        "answer": "5/6",
        "final_answer": "5/6",
        "certificate_type": "expression_evaluation",
        "problem_expression": "1/2 + 1/3",
        "problem_constants": ["1", "2", "1", "3"],
        "problem_variables": [],
        "unit": "",
    }
    return {
        "dataset": kwargs["dataset"],
        "method_name": kwargs["method_name"],
        "prediction": "5/6",
        "normalized_answer": "5/6",
        "request_status": "ok",
        "output_status": "ok",
        "protocol_parse_status": "ok",
        "validated_output": payload,
        "total_tokens": 10.0,
        "prompt_tokens": 5.0,
        "completion_tokens": 5.0,
        "latency_ms": 1.0,
    }


def _fake_isp_turn(**kwargs):
    own = "yes" if int(kwargs["agent_id"]) == 1201 else "no"
    return {
        "dataset": kwargs["dataset"],
        "method_name": kwargs["method_name"],
        "prediction": own,
        "normalized_answer": own,
        "request_status": "ok",
        "output_status": "ok",
        "protocol_parse_status": "ok",
        "validated_output": {"answer": own, "peer_distribution": {"yes": 0.6, "no": 0.4}},
        "total_tokens": 10.0,
        "prompt_tokens": 5.0,
        "completion_tokens": 5.0,
        "latency_ms": 1.0,
    }


def _fake_hotpot_certificate_turn(**kwargs):
    payload = {
        "answer": "Captain John Underhill",
        "final_answer": "Captain John Underhill",
        "certificate_type": "context_span_completion",
        "source_title": "Expedition",
        "source_sentence_index": 0,
        "evidence_span": "Captain John Underhill",
        "missing_tokens": ["Captain"],
    }
    return {
        "dataset": kwargs["dataset"],
        "method_name": kwargs["method_name"],
        "prediction": payload["answer"],
        "normalized_answer": payload["answer"],
        "request_status": "ok",
        "output_status": "ok",
        "protocol_parse_status": "ok",
        "validated_output": payload,
        "total_tokens": 10.0,
        "prompt_tokens": 5.0,
        "completion_tokens": 5.0,
        "latency_ms": 1.0,
    }


def _fake_full_cvs_turn(**kwargs):
    method_name = kwargs["method_name"]
    if method_name == "cred_stage_a":
        answer = "1/2" if int(kwargs["agent_id"]) <= 3 else "5/6"
        payload = {"answer": answer, "final_answer": answer, "reasoning": "solve"}
    else:
        answer = "5/6"
        payload = {
            "answer": answer,
            "final_answer": answer,
            "reasoning": "derive target",
            "certificate_type": "expression_evaluation",
            "problem_expression": "1/2 + 1/3",
            "problem_constants": ["1", "2", "1", "3"],
            "problem_variables": [],
            "unit": "",
        }
    return {
        "dataset": kwargs["dataset"],
        "method_name": method_name,
        "prediction": answer,
        "normalized_answer": answer,
        "request_status": "ok",
        "output_status": "ok",
        "protocol_parse_status": "ok",
        "validated_output": payload,
        "total_tokens": 10.0,
        "prompt_tokens": 5.0,
        "completion_tokens": 5.0,
        "latency_ms": 1.0,
        "reason_present": True,
        "output_protocol": kwargs["output_protocol"],
        "protocol_recovery": "",
        "raw_finish_reason": "stop",
    }
