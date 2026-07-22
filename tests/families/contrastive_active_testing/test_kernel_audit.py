from __future__ import annotations

import hashlib
import json

import pytest

from research_experiments.families.contrastive_active_testing.causal_ledger import (
    build_causal_ledger,
    summarize_counterfactual_ledger,
)
from research_experiments.families.contrastive_active_testing.comparison_replay import (
    merge_comparison_predictions,
)
from research_experiments.families.contrastive_active_testing.kernel_mechanism import (
    STUDY_ARMS,
    ingest_kernel_mechanism_results,
    summarize_kernel_mechanism,
    write_kernel_mechanism_template,
    write_kernel_run_mechanism_results,
)


def _jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_causal_ledger_keeps_recoverable_harm_and_safe_cases(tmp_path) -> None:
    run = tmp_path / "run"
    predictions = []
    routers = []
    cases = [
        ("recover", 0.0, 1.0, True, True, "kernel_verified_override"),
        ("harm", 1.0, 0.0, True, True, "kernel_verified_override"),
        ("safe", 1.0, 1.0, True, False, "proof_incomplete"),
        ("missing", 0.0, 0.0, False, False, "no_certificate"),
    ]
    for sample_id, sc_score, final_score, oracle, override, resolver in cases:
        common = {"run_id": "run", "dataset": "bbeh", "sample_id": sample_id, "task": "unit"}
        predictions.extend(
            [
                {**common, "method_name": "sc_5", "score": sc_score},
                {
                    **common,
                    "method_name": "catch_kernel",
                    "score": final_score,
                    "candidate_oracle_correct": oracle,
                    "target_oracle_correct": oracle,
                    "override_accepted": override,
                    "resolver": resolver,
                    "protocol_version": "catch_kernel_v1",
                    "answer_link_coverage": 1.0,
                    "obligation_coverage": 1.0,
                    "verifier_jurisdiction_coverage": 1.0,
                    "proof_completeness": 1.0 if final_score else 0.0,
                },
            ]
        )
        routers.append(
            {
                **common,
                "protocol_version": "catch_kernel_v1",
                "certificate_protocol_error": None,
                "adapter_results": {},
            }
        )
    _jsonl(run / "views" / "predictions.jsonl", predictions)
    _jsonl(run / "turns" / "router_decisions.jsonl", routers)
    summary = build_causal_ledger([run], tmp_path / "audit", matched_safe_count=1)
    assert summary["recoverable_wrong_count"] == 1
    assert summary["harm_count"] == 1
    assert summary["intensive_audit_count"] == 3
    assert summary["transition_counts"]["wrong→correct"] == 1
    assert summary["first_failure_bookkeeping"]["status"] == "heuristic_first_failure_bookkeeping_not_causal"
    assert summary["counterfactual_decomposition"]["status"] == "pending_counterfactual_adjudication"
    assert summary["safe_match_diagnostics"]["selected"] == 1

    ledger_path = tmp_path / "audit" / "causal_ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    recover = next(item for item in rows if item["sample_id"] == "recover")
    recover.update(
        {
            "in_intensive_audit": True,
            "counterfactual_contract_correct": True,
            "counterfactual_verifier_correct": False,
            "counterfactual_certificate_correct": True,
            "counterfactual_complete_proof": True,
            "adjudicated_first_failure_layer": "compiler",
            "counterfactual_annotation_status": "complete",
        }
    )
    _jsonl(ledger_path, rows)
    counterfactual = summarize_counterfactual_ledger(ledger_path)
    assert counterfactual["decomposition"]["status"] == "complete"
    assert counterfactual["decomposition"]["additive_layer_counts"] == {"compiler": 1}


def test_kernel_mechanism_matrix_is_non_blocking_and_summarizes_partial_rows(tmp_path) -> None:
    audit = {
        "cases": [
            {
                "run_id": "r",
                "sample_id": "s",
                "dataset": "bbeh",
                "task": "unit",
                "case_class": "harm",
                "sc_correct": True,
            }
        ]
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix = write_kernel_mechanism_template(audit_path, matrix_path)
    assert matrix["row_count"] == sum(len(value) for value in STUDY_ARMS.values())
    assert matrix["non_blocking"] is True
    for row in matrix["rows"]:
        if row["arm"] in {
            "model_obligation__model_verifier",
            "human_obligation__model_verifier",
            "model_obligation__human_or_deterministic_verifier",
            "human_obligation__human_or_deterministic_verifier",
            "adapter_then_model_fallback",
            "capability_routed_kernel",
        }:
            row.update(
                {
                    "completed": True,
                    "final_correct": row["arm"]
                    in {
                        "human_obligation__model_verifier",
                        "human_obligation__human_or_deterministic_verifier",
                        "capability_routed_kernel",
                    },
                }
            )
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    summary = summarize_kernel_mechanism(matrix_path)
    assert summary["completed_rows"] == 6
    assert summary["matrix_complete"] is False
    assert summary["attribution"] is None
    assert summary["interpretation"] == "incomplete_mechanism_matrix"


def test_mechanism_result_ingest_enforces_frozen_candidate_identity(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "run_id": "r",
                        "sample_id": "s",
                        "dataset": "bbeh",
                        "task": "unit",
                        "case_class": "harm",
                        "sc_correct": True,
                        "candidate_set_hash": "candidate-hash",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.json"
    write_kernel_mechanism_template(audit_path, matrix_path)
    result_path = tmp_path / "result.jsonl"
    _jsonl(
        result_path,
        [
            {
                "run_id": "r",
                "sample_id": "s",
                "dataset": "bbeh",
                "study": "jurisdiction",
                "arm": "capability_routed_kernel",
                "candidate_set_hash": "candidate-hash",
                "gold_blind": True,
                "final_correct": True,
                "proof_complete": True,
            }
        ],
    )
    merged = ingest_kernel_mechanism_results(matrix_path, [result_path], tmp_path / "merged.json")
    assert merged["imported_result_count"] == 1
    completed = [item for item in merged["rows"] if item["completed"]]
    assert len(completed) == 1
    assert completed[0]["execution_mode"] == "kernel_run_or_replay"


def test_comparator_merge_requires_identical_stage_a_candidates(tmp_path) -> None:
    primary = tmp_path / "primary"
    comparator = tmp_path / "comparator"
    for root, method in ((primary, "catch_kernel"), (comparator, "catch")):
        _jsonl(
            root / "views" / "predictions.jsonl",
            [
                {
                    "run_id": root.name,
                    "dataset": "bbeh",
                    "sample_id": "s1",
                    "method_name": method,
                    "prediction": "A",
                }
            ],
        )
        _jsonl(
            root / "turns" / "agent_turns.jsonl",
            [
                {
                    "dataset": "bbeh",
                    "sample_id": "s1",
                    "role": "stage_a_solver",
                    "agent_id": index,
                    "answer_class_key": "A" if index < 4 else "B",
                    "prediction": "A" if index < 4 else "B",
                }
                for index in range(1, 6)
            ],
        )
    merged = merge_comparison_predictions(primary, [comparator], tmp_path / "comparison.jsonl")
    assert merged["primary_sample_count"] == 1
    assert merged["imported_methods"] == {"catch": 1}

    turns_path = comparator / "turns" / "agent_turns.jsonl"
    rows = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["prediction"] = "C"
    _jsonl(turns_path, rows)
    with pytest.raises(ValueError, match="signature mismatch"):
        merge_comparison_predictions(primary, [comparator], tmp_path / "bad.jsonl")


def test_kernel_run_materializes_all_offline_proof_arms_from_one_proof_set(tmp_path) -> None:
    candidate_payload = {
        "candidate_answer_nodes": {"H0": {"answer_hash": "a"}, "H1": {"answer_hash": "b"}},
        "candidate_public_to_answer_class_key": {"H0": "A", "H1": "B"},
        "public_pairs": [{"pair_id": "P0"}],
    }
    candidate_hash = hashlib.sha256(
        json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    arms = {
        "capability_routed_kernel": "jurisdiction",
        "support_only": "proof_completeness",
        "support_and_refutation": "proof_completeness",
        "support_refutation_and_complete_obligations": "proof_completeness",
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "run_id": "old-run",
                        "dataset": "bbeh",
                        "sample_id": "s1",
                        "study": study,
                        "arm": arm,
                        "candidate_set_hash": candidate_hash,
                    }
                    for arm, study in arms.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    run = tmp_path / "kernel-run"
    proof = {
        "candidate_key_anon": "H1",
        "status": "PASS",
        "provenance_valid": True,
        "entailment_valid": True,
        "obligation_valid": True,
        "sufficiency_valid": True,
        "obligation_id": "Q0",
    }
    router = {
        "run_id": "kernel-run",
        "dataset": "bbeh",
        "sample_id": "s1",
        "protocol_version": "catch_kernel_v1",
        **candidate_payload,
        "anchor_key": "A",
        "gold_candidate_keys": ["B"],
        "task_semantics": {"mandatory_obligation_templates": [{"obligation_id": "Q0", "required": True}]},
        "verifier_bindings": {"T0": {"binding_status": "BOUND"}},
        "proof_results": [
            {**proof, "test_id": "T0_support"},
            {**proof, "test_id": "T0_refute_anchor"},
        ],
        "kernel_decision": {
            "decision": "OVERRIDE",
            "challenger_id": "H1",
            "failure_layer": "none",
        },
    }
    _jsonl(run / "turns" / "router_decisions.jsonl", [router])
    _jsonl(
        run / "views" / "predictions.jsonl",
        [
            {
                "dataset": "bbeh",
                "sample_id": "s1",
                "method_name": "catch_kernel",
                "score": 1,
                "syntax_validity": 1,
                "schema_validity": 1,
                "total_tokens_per_question": 100,
            }
        ],
    )
    output = tmp_path / "results.jsonl"
    result = write_kernel_run_mechanism_results(matrix_path, [run], output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["result_count"] == 4
    assert {row["arm"] for row in rows} == set(arms)
    assert all(row["final_correct"] is True for row in rows)
    assert all(row["override_accepted"] is True for row in rows)
