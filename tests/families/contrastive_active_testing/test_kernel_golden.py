from __future__ import annotations

import pytest

from research_experiments.families.contrastive_active_testing.kernel import VERIFIER_CAPABILITIES
from research_experiments.families.contrastive_active_testing.kernel_adapters import validate_typed_payload

EXECUTABLE_GOLDEN_PAYLOADS = {
    "seq_plan": {"executor": "official_seqbench_v1"},
    "stack_trace": {"executor": "dyck_stack_trace_v1"},
    "grid_path": {"executor": "ordered_grid_path_v1"},
    "custom_sort_order": {"words": ["b", "a"], "alphabet": list("abcdefghijklmnopqrstuvwxyz")},
    "permutation": {
        "initial_order": ["Alice", "Bob"],
        "swaps": [["Alice", "Bob"]],
        "query_item": "Alice",
    },
    "arithmetic_dsl": {
        "checks_by_candidate": {
            "H0": {"left_expression": "1+1", "right_expression": "2"},
            "H1": {"left_expression": "1+2", "right_expression": "3"},
        }
    },
    "constraint_witness": {
        "assignments_by_candidate": {"H0": {"x": 1}, "H1": {"x": 2}},
        "constraints": [{"left": "x", "operator": "!=", "right": 0}],
    },
}


@pytest.mark.parametrize(
    "operation_kind",
    sorted(
        {
            operation
            for capability in VERIFIER_CAPABILITIES
            if capability.guarantee_level == "executable"
            for operation in capability.supported_operation_kinds
        }
    ),
)
def test_every_executable_jurisdiction_has_pass_fail_and_unsupported_payload_examples(operation_kind: str) -> None:
    payload = EXECUTABLE_GOLDEN_PAYLOADS[operation_kind]
    valid, _ = validate_typed_payload(
        operation_kind,
        payload,
        candidate_ids={"H0", "H1"},
        payload_source="local"
        if operation_kind in {"seq_plan", "stack_trace", "grid_path", "custom_sort_order"}
        else "model_slots",
    )
    invalid, _ = validate_typed_payload(
        operation_kind,
        {"unexpected": True},
        candidate_ids={"H0", "H1"},
        payload_source="local"
        if operation_kind in {"seq_plan", "stack_trace", "grid_path", "custom_sort_order"}
        else "model_slots",
    )
    assert valid is True
    assert invalid is False


@pytest.mark.parametrize(
    "operation_kind",
    sorted(
        {
            operation
            for capability in VERIFIER_CAPABILITIES
            if capability.guarantee_level == "bounded_semantic"
            for operation in capability.supported_operation_kinds
        }
    ),
)
def test_every_bounded_semantic_jurisdiction_forbids_hidden_payloads(operation_kind: str) -> None:
    valid, _ = validate_typed_payload(
        operation_kind,
        {},
        candidate_ids={"H0", "H1"},
        payload_source="none",
    )
    invalid, _ = validate_typed_payload(
        operation_kind,
        {"model_invented_fact": "x"},
        candidate_ids={"H0", "H1"},
        payload_source="none",
    )
    assert valid is True
    assert invalid is False


def test_capability_manifests_declare_failure_semantics_and_finite_results() -> None:
    for capability in VERIFIER_CAPABILITIES:
        assert capability.failure_semantics
        assert "PASS" in capability.possible_results
        assert "FAIL" in capability.possible_results
        assert capability.capability_version


def test_executable_jurisdiction_excludes_model_authored_source_semantics() -> None:
    executable = {
        operation
        for capability in VERIFIER_CAPABILITIES
        if capability.guarantee_level == "executable"
        for operation in capability.supported_operation_kinds
    }
    assert executable == {"seq_plan", "stack_trace", "grid_path", "custom_sort_order"}
    assert not executable & {"permutation", "arithmetic_dsl", "constraint_witness"}
