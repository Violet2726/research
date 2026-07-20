from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.artifact_replay import (
    audit_v3_artifact_recomputation,
)
from research_experiments.families.contrastive_active_testing.config import CatchProtocolConfig
from research_experiments.families.contrastive_active_testing.run import sample as sample_runner


def test_network_attempt_budget_is_a_soft_warning_counter() -> None:
    budget = sample_runner.NetworkAttemptBudget(10)
    first = budget.reserve()
    second = budget.reserve()
    third = budget.reserve()
    assert (first, second, third) == (5, 5, 5)
    budget.settle(first, 3)
    budget.settle(second, 5)
    budget.settle(third, 4)
    assert budget.actual == 12
    assert budget.snapshot()["limit_exceeded"] is True
    assert budget.overage == 2


def test_network_attempt_budget_never_blocks_competing_workers() -> None:
    budget = sample_runner.NetworkAttemptBudget(25)

    def reserve_once(_index):
        return budget.reserve()

    with ThreadPoolExecutor(max_workers=20) as executor:
        reservations = list(executor.map(reserve_once, range(100)))
    assert sum(reservations) == 500
    for reservation in reservations:
        if reservation:
            budget.settle(reservation, reservation)
    assert budget.actual == 500
    assert budget.overage == 475


def test_all_failed_stage_requests_produce_unavailable_predictions_without_raising() -> None:
    sample = DatasetSample(
        dataset="bbeh",
        sample_id="all-failed",
        question="Return yes.",
        reference_answer="yes",
        prompt_context="",
        metadata={"task": "fixture", "answer_contract": {"kind": "free_text"}},
    )
    stage_rows = tuple(
        {
            "dataset": "bbeh",
            "sample_id": sample.sample_id,
            "role": "stage_a_solver",
            "agent_id": index,
            "answer_class_key": "",
            "prediction": "",
            "normalized_answer": "",
            "request_error": "timeout after retries",
            "network_attempt_count": 5,
        }
        for index in range(1, 6)
    )
    turns, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="run",
        split_name="dev",
        experiment=SimpleNamespace(global_seed=42),
        protocol=SimpleNamespace(protocol_version="catch_v3", stage_candidates=5),
        endpoint=None,
        network_budget=sample_runner.NetworkAttemptBudget(1),
        phase_name="development",
        run_direct_judge=False,
        precomputed_stage_rows=stage_rows,
    )
    assert len(turns) == 5
    assert router["triggered"] is False
    assert {row["method_name"] for row in predictions} == {"sc_5", "adaptive_sc_8", "catch"}
    assert all(not row["prediction"] for row in predictions)


def test_development_grid_keeps_each_catch_variant_at_five_plus_three_calls(monkeypatch) -> None:
    sample = DatasetSample(
        dataset="bbeh",
        sample_id="catch-unit",
        question="The quantity is either even or odd.\nOptions:\n(A) even\n(B) odd",
        reference_answer="B",
        prompt_context="",
        metadata={
            "task": "unit",
            "options": [{"label": "A", "text": "even"}, {"label": "B", "text": "odd"}],
        },
    )
    protocol = CatchProtocolConfig(5, 3, 2, 3, 6, 4, 0.7, 1.0, 16_384, 4_096, (2, 3, 4), (1, 2), 62_000)
    stage_answers = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B"}
    calls: list[str] = []

    def answer_turn(_sample, *, role, agent_id, **_kwargs):
        answer = stage_answers[agent_id] if role == "stage_a_solver" else "B"
        calls.append(role)
        reasoning = "even fact first; parity check second; final relation third" if answer == "A" else "odd fact first; parity check second; final relation third"
        return {
            "sample_id": sample.sample_id,
            "role": role,
            "answer_class_key": answer,
            "normalized_answer": answer,
            "prediction": answer,
            "validated_output": {"reasoning": reasoning, "final_answer": answer},
            "total_tokens": 1,
            "actual_total_tokens": 1,
            "network_attempt_count": 1,
        }

    def json_turn(_sample, *, role, messages, **_kwargs):
        calls.append(role)
        if role == "test_designer":
            hypotheses = json.loads(
                re.search(r"Anonymous hypotheses:\n(.+?)\n\nCompile", messages[-1]["content"], re.S).group(1)
            )
            tests = []
            quotes = (
                {"A": "even fact", "B": "odd fact"},
                {"A": "parity check", "B": "parity check"},
                {"A": "second", "B": "second"},
                {"A": "final relation", "B": "final relation"},
            )
            for index, per_answer_quote in enumerate(quotes):
                per_test = {
                    hypothesis["id"]: {
                        "outcome_id": "O0" if hypothesis["answer"] == "A" else "O1",
                        "evidence_quote": per_answer_quote[hypothesis["answer"]],
                    }
                    for hypothesis in hypotheses
                }
                tests.append(
                    {
                        "test_id": f"T{index}",
                        "question": f"Is parity diagnostic fact {index} even?",
                        "outcomes": [{"id": "O0", "text": "even"}, {"id": "O1", "text": "odd"}],
                        "commitments": per_test,
                    }
                )
            return {"total_tokens": 1, "actual_total_tokens": 1, "network_attempt_count": 1}, {"tests": tests}
        if role == "blinded_witness":
            rendered = json.loads(re.search(r"Diagnostic tests:\n(.+?)\n\nReturn", messages[-1]["content"], re.S).group(1))
            answers = []
            for test in rendered:
                odd = next(outcome for outcome in test["outcomes"] if outcome["text"] == "odd")
                answers.append({"test_id": test["test_id"], "outcome_id": odd["id"], "check": "odd"})
            return {"total_tokens": 1, "actual_total_tokens": 1, "network_attempt_count": 1}, {"answers": answers}
        hypothesis_id = re.search(r'"id": "(H\d+)"', messages[-1]["content"]).group(1)
        return {"total_tokens": 1, "actual_total_tokens": 1, "network_attempt_count": 1}, {"selected_id": hypothesis_id, "check": "ok"}

    monkeypatch.setattr(sample_runner, "_answer_turn", answer_turn)
    monkeypatch.setattr(sample_runner, "_json_turn", json_turn)
    turns, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="run",
        split_name="dev",
        experiment=SimpleNamespace(global_seed=42),
        protocol=protocol,
        endpoint=SimpleNamespace(cache_namespace="catch-dev-v1"),
        network_budget=sample_runner.NetworkAttemptBudget(62_000),
        phase_name="development",
        frozen_decoding=None,
        run_direct_judge=True,
    )

    variants = [row for row in predictions if row["method_name"].startswith("catch_d")]
    assert len(turns) == 18
    assert len(variants) == 6
    assert all(row["logical_calls_per_question"] == 8 for row in variants)
    assert all(row["actual_intervention_calls_per_question"] == 3 for row in variants)
    assert router["candidate_oracle_correct"] is True
    assert any(row["override_accepted"] and row["prediction"] == "B" for row in variants)


def test_empty_code_packet_abstains_without_witness_calls(monkeypatch) -> None:
    sample = DatasetSample(
        "bbeh",
        "empty-packet",
        "Question\nOptions:\n(A) yes\n(B) no",
        "A",
        "",
        {"task": "unit", "options": [{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}]},
    )
    protocol = CatchProtocolConfig(5, 3, 2, 3, 6, 4, 0.7, 1.0, 16_384, 4_096, (2, 3, 4), (1, 2), 62_000)
    calls = []

    def answer_turn(_sample, *, role, agent_id, **_kwargs):
        answer = "A" if role != "stage_a_solver" or agent_id < 4 else "B"
        return {
            "answer_class_key": answer,
            "normalized_answer": answer,
            "prediction": answer,
            "validated_output": {"reasoning": f"reason {answer}", "final_answer": answer},
            "actual_total_tokens": 1,
            "network_attempt_count": 1,
        }

    def json_turn(_sample, *, role, **_kwargs):
        calls.append(role)
        if role == "blinded_witness":
            raise AssertionError("empty packets must never schedule a witness")
        if role == "test_designer":
            return {"actual_total_tokens": 1, "network_attempt_count": 1}, {"tests": []}
        return {"actual_total_tokens": 1, "network_attempt_count": 1}, {"selected_id": "H0"}

    monkeypatch.setattr(sample_runner, "_answer_turn", answer_turn)
    monkeypatch.setattr(sample_runner, "_json_turn", json_turn)
    _, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="run",
        split_name="dev",
        experiment=SimpleNamespace(global_seed=42),
        protocol=protocol,
        endpoint=SimpleNamespace(cache_namespace="catch-dev-v2"),
        network_budget=sample_runner.NetworkAttemptBudget(62_000),
        phase_name="development",
        frozen_decoding=None,
        run_direct_judge=True,
    )
    assert "blinded_witness" not in calls
    assert all(
        row["resolver"] == "insufficient_code_distance"
        and row["logical_calls_per_question"] == 6
        for row in predictions
        if row["method_name"].startswith("catch_d")
    )
    assert all(not variant["code_eligible"] for variant in router["catch_variants"])


def test_v3_icv_runs_fixed_five_plus_three_and_pair_judge(monkeypatch) -> None:
    sample = DatasetSample(
        "bbeh",
        "icv-integration",
        "The beta facts are supported by the source.\nOptions:\n(A) alpha\n(B) beta",
        "B",
        "",
        {"task": "unit", "options": [{"label": "A", "text": "alpha"}, {"label": "B", "text": "beta"}]},
    )
    protocol = CatchProtocolConfig(
        5, 3, 2, 3, 6, 4, 0.7, 1.0, 16_384, 4_096, (2, 3, 4), (1, 2), 62_000,
        protocol_version="catch_v3",
        preflight_sample_count=20,
        preflight_code_coverage_threshold=0.60,
        preflight_coordinate_validity_threshold=0.95,
        preflight_usable_pair_threshold=0.90,
        coordinates_per_pair=3,
        max_selected_contrasts=6,
        pair_judge_count=3,
        preflight_decisive_threshold=0.80,
        preflight_panel_agreement_threshold=0.70,
    )
    stage_answers = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B"}
    calls: list[str] = []

    def answer_turn(_sample, *, role, agent_id, **_kwargs):
        answer = stage_answers[agent_id] if role == "stage_a_solver" else "B"
        calls.append(role)
        name = "alpha" if answer == "A" else "beta"
        reasoning = (
            f"The {name} premise is established from source. "
            f"The {name} implication follows from context. "
            f"The {name} boundary condition remains satisfied."
        )
        return {
            "sample_id": sample.sample_id,
            "role": role,
            "answer_class_key": answer,
            "normalized_answer": answer,
            "prediction": answer,
            "validated_output": {"reasoning": reasoning, "final_answer": answer},
            "actual_total_tokens": 1,
            "network_attempt_count": 1,
        }

    def json_turn(_sample, *, role, messages, **_kwargs):
        calls.append(role)
        content = messages[-1]["content"]
        turn = {
            "sample_id": sample.sample_id,
            "role": role,
            "actual_total_tokens": 1,
            "network_attempt_count": 1,
        }
        if role == "icv_selector":
            return turn, {
                "contrasts": [
                    {
                        "pair_id": "P0",
                        "contrast_id": f"C{index}",
                        "left_unit_ids": [f"L:E{index}"],
                        "right_unit_ids": [f"R:E{index}"],
                    }
                    for index in range(3)
                ]
            }
        if role == "icv_witness":
            contrasts = json.loads(re.search(r"Anonymous local contrasts:\n(.+?)\n\nFor every", content, re.S).group(1))
            answers = []
            for item in contrasts:
                verdict = "LEFT_ONLY" if "beta" in item["statement_left"] else "RIGHT_ONLY"
                answers.append({"contrast_id": item["contrast_id"], "verdict": verdict})
            return turn, {"answers": answers}
        selected = re.search(r'"id": "([HJ]\d+)", "answer": "B"', content).group(1)
        return turn, {"selected_id": selected}

    monkeypatch.setattr(sample_runner, "_answer_turn", answer_turn)
    monkeypatch.setattr(sample_runner, "_json_turn", json_turn)
    turns, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="run",
        split_name="dev",
        experiment=SimpleNamespace(global_seed=42),
        protocol=protocol,
        endpoint=SimpleNamespace(cache_namespace="catch-dev-v3"),
        network_budget=sample_runner.NetworkAttemptBudget(62_000),
        phase_name="development",
        frozen_decoding=None,
        run_direct_judge=True,
    )

    by_method = {row["method_name"]: row for row in predictions}
    assert len(turns) == 17
    assert set(by_method) == {"sc_5", "adaptive_sc_8", "catch", "direct_judge_3", "pair_judge_3"}
    assert by_method["catch"]["prediction"] == "B"
    assert by_method["catch"]["logical_calls_per_question"] == 8
    assert by_method["catch"]["actual_intervention_calls_per_question"] == 3
    assert router["eligible_challengers"] == ["B"]
    assert router["target_oracle_correct"] is True
    assert calls.count("pair_judge") == 3
    audit = audit_v3_artifact_recomputation(
        turns=turns,
        routers=[router],
        predictions=predictions,
    )
    assert audit["passed"]
    tampered = [dict(row) for row in predictions]
    next(row for row in tampered if row["method_name"] == "catch")["prediction"] = "A"
    assert not audit_v3_artifact_recomputation(
        turns=turns,
        routers=[router],
        predictions=tampered,
    )["passed"]


def test_cert_v2_uses_answer_nodes_all_candidates_and_gold_free_seq_adapter(monkeypatch) -> None:
    complete = '["start: A1","move_to: A2","rescue: Alice"]'
    incomplete = '["start: A1","move_to: A2"]'
    sample = DatasetSample(
        "seqbench",
        "cert-v2-integration",
        "Room A1 and A2 are connected by an open door. Bob is in room A1. Alice is in room A2.",
        complete,
        "",
        {"seqbench_instance_metadata": {"agent_name": "Bob", "target_name": "Alice"}},
    )
    protocol = CatchProtocolConfig(
        5,
        3,
        2,
        3,
        6,
        6,
        0.7,
        1.0,
        16_384,
        4_096,
        (2, 3, 4),
        (1, 2),
        62_000,
        protocol_version="catch_cert_v2",
        pair_judge_count=3,
    )
    stage_answers = {1: incomplete, 2: incomplete, 3: incomplete, 4: complete, 5: complete}
    calls: list[str] = []

    def answer_turn(_sample, *, role, agent_id, **_kwargs):
        answer = stage_answers[agent_id] if role == "stage_a_solver" else complete
        calls.append(role)
        return {
            "sample_id": sample.sample_id,
            "role": role,
            "answer_class_key": answer,
            "normalized_answer": answer,
            "prediction": answer,
            "validated_output": {"reasoning": f"Plan is {answer}", "final_answer": answer},
            "actual_total_tokens": 1,
            "network_attempt_count": 1,
        }

    def json_turn(_sample, *, role, messages, **_kwargs):
        calls.append(role)
        if role == "certificate_verifier_v2":
            raise AssertionError("a fully executable seq certificate must not call a model verifier")
        assert role == "certificate_designer_v2"
        content = messages[-1]["content"]
        nodes = json.loads(
            re.search(r"Anonymous answer nodes \(these are candidate meanings, not correctness labels\):\n(.+?)\n\nShort", content, re.S).group(1)
        )
        pairs = json.loads(re.search(r"Anonymous pairs:\n(.+?)\n\nExact", content, re.S).group(1))
        contract = json.loads(
            re.search(r"Question contract and mandatory obligations:\n(.+?)\n\nAnonymous answer", content, re.S).group(1)
        )
        source = json.loads(re.search(r"Indexed source spans:\n(.+?)\n\nQuestion contract", content, re.S).group(1))
        complete_public = next(key for key, node in nodes.items() if "rescue" in node["rendered_content"])
        pair = next(item for item in pairs if complete_public in {item["left_candidate"], item["right_candidate"]})
        other = pair["right_candidate"] if pair["left_candidate"] == complete_public else pair["left_candidate"]
        payload = {
            "tests": [
                {
                    "test_id": "T0",
                    "pair_id": pair["pair_id"],
                    "obligation_ids": [item["obligation_id"] for item in contract["mandatory_obligations"]],
                    "operation_kind": "seq_plan",
                    "question_or_operation": "Which candidate is a complete executable rescue plan?",
                    "finite_outcomes": [
                        {"outcome_id": "O0", "text": "incomplete"},
                        {"outcome_id": "O1", "text": "complete"},
                    ],
                    "expected_outcome_by_candidate": {other: "O0", complete_public: "O1"},
                    "source_span_ids": [source["spans"][0]["span_id"]],
                    "deterministic_payload": {},
                }
            ],
            "certificates": [
                {
                    "candidate_key_anon": complete_public,
                    "answer_hash": nodes[complete_public]["answer_hash"],
                    "required_test_ids": ["T0"],
                }
            ],
        }
        return {
            "sample_id": sample.sample_id,
            "role": role,
            "actual_total_tokens": 1,
            "network_attempt_count": 1,
        }, payload

    monkeypatch.setattr(sample_runner, "_answer_turn", answer_turn)
    monkeypatch.setattr(sample_runner, "_json_turn", json_turn)
    turns, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="run",
        split_name="dev",
        experiment=SimpleNamespace(global_seed=42),
        protocol=protocol,
        endpoint=SimpleNamespace(cache_namespace="catch-dev-cert_v2"),
        network_budget=sample_runner.NetworkAttemptBudget(62_000),
        phase_name="development",
        run_direct_judge=False,
    )

    by_method = {row["method_name"]: row for row in predictions}
    assert set(by_method) == {"sc_5", "adaptive_sc_8", "catch_cert_v2"}
    assert by_method["catch_cert_v2"]["prediction"] == complete
    assert by_method["catch_cert_v2"]["logical_calls_per_question"] == 6
    assert by_method["catch_cert_v2"]["adapter_executed_test_count"] == 1
    assert router["target_oracle_correct"] is True
    assert len(router["public_pairs"]) == len(router["candidate_answer_nodes"]) - 1
    assert "certificate_verifier_v2" not in calls
    assert len(turns) == 9
