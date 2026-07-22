"""Kernel-only typed payload compilers and deterministic adapters.

This module deliberately sits outside ``certificates_v2.py`` so frozen
CATCH-Cert v2 reruns keep their historical adapter semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    AdapterResult,
    CandidateAnswerNode,
    CandidatePairV2,
    CertificateTestV2,
    TaskContractV2,
    run_deterministic_adapters_v2,
)


def compile_local_typed_payload(sample: DatasetSample, operation_kind: str) -> tuple[dict[str, Any] | None, str | None]:
    """Compile source structure, never a gold label or hidden candidate answer."""

    source = question_without_answer_contract(sample)
    if operation_kind == "seq_plan":
        return {"executor": "official_seqbench_v1"}, None
    if operation_kind == "stack_trace":
        return {"executor": "dyck_stack_trace_v1"}, None
    if operation_kind == "grid_path":
        circular, error = _compile_circular_path(source)
        if circular is not None:
            return circular, None
        if "circular path" in source.casefold():
            return None, error
        return {"executor": "ordered_grid_path_v1"}, None
    if operation_kind == "custom_sort_order":
        return _compile_custom_sort(source)
    if operation_kind == "sort_trace_earliest":
        return {"executor": "unsupported_sort_trace_v1"}, None
    return {}, None


def validate_typed_payload(
    operation_kind: str,
    payload: Any,
    *,
    candidate_ids: set[str],
    payload_source: str,
) -> tuple[bool, str]:
    """Validate the operation-specific payload union used by the kernel."""

    if not isinstance(payload, dict):
        return False, "typed_payload_not_object"
    if payload_source == "none":
        return (not payload, "ok" if not payload else "semantic_payload_must_be_empty")
    if operation_kind == "seq_plan":
        return _exact(payload, {"executor": "official_seqbench_v1"}, "seq_plan_payload_invalid")
    if operation_kind == "stack_trace":
        return _exact(payload, {"executor": "dyck_stack_trace_v1"}, "stack_trace_payload_invalid")
    if operation_kind == "sort_trace_earliest":
        return _exact(payload, {"executor": "unsupported_sort_trace_v1"}, "sort_trace_payload_invalid")
    if operation_kind == "grid_path":
        if payload == {"executor": "ordered_grid_path_v1"}:
            return True, "ok"
        fields = {"path_kind", "entities_clockwise", "start_entity", "signed_moves"}
        valid = (
            set(payload) == fields
            and payload.get("path_kind") == "circular"
            and _string_list(payload.get("entities_clockwise"), minimum=2)
            and isinstance(payload.get("start_entity"), str)
            and bool(payload.get("start_entity"))
            and isinstance(payload.get("signed_moves"), list)
            and bool(payload.get("signed_moves"))
            and all(isinstance(item, int) for item in payload["signed_moves"])
        )
        return valid, "ok" if valid else "grid_path_payload_invalid"
    if operation_kind == "custom_sort_order":
        valid = (
            set(payload) == {"words", "alphabet"}
            and _string_list(payload.get("words"), minimum=1)
            and _string_list(payload.get("alphabet"), minimum=1)
            and len(set(payload["alphabet"])) == len(payload["alphabet"])
        )
        return valid, "ok" if valid else "custom_sort_payload_invalid"
    if operation_kind == "permutation":
        required = {"initial_order", "swaps"}
        query_fields = {"query_item", "query_position"} & set(payload)
        valid = (
            required.issubset(payload)
            and set(payload).issubset(required | {"query_item", "query_position"})
            and len(query_fields) == 1
            and _string_list(payload.get("initial_order"), minimum=1)
            and isinstance(payload.get("swaps"), list)
            and all(
                isinstance(item, list) and len(item) == 2 and all(isinstance(value, str) and value for value in item)
                for item in payload["swaps"]
            )
            and ("query_item" not in payload or isinstance(payload["query_item"], str))
            and ("query_position" not in payload or isinstance(payload["query_position"], int))
        )
        return valid, "ok" if valid else "permutation_payload_invalid"
    if operation_kind == "arithmetic_dsl":
        checks = payload.get("checks_by_candidate")
        valid = set(payload) == {"checks_by_candidate"} and isinstance(checks, dict) and set(checks) == candidate_ids
        if valid:
            valid = all(
                isinstance(item, dict)
                and set(item) == {"left_expression", "right_expression"}
                and all(isinstance(item[key], str) and item[key].strip() for key in item)
                for item in checks.values()
            )
        return valid, "ok" if valid else "arithmetic_payload_invalid"
    if operation_kind == "constraint_witness":
        assignments = payload.get("assignments_by_candidate")
        constraints = payload.get("constraints")
        valid = (
            set(payload) == {"assignments_by_candidate", "constraints"}
            and isinstance(assignments, dict)
            and set(assignments) == candidate_ids
            and all(isinstance(item, dict) for item in assignments.values())
            and isinstance(constraints, list)
            and bool(constraints)
            and all(
                isinstance(item, dict)
                and set(item) == {"left", "operator", "right"}
                and item.get("operator") in {"==", "!="}
                for item in constraints
            )
        )
        return valid, "ok" if valid else "constraint_payload_invalid"
    return False, f"typed_payload_schema_unregistered:{operation_kind}"


def typed_payload_prompt_schema(operation_kind: str, candidate_ids: set[str]) -> dict[str, Any]:
    """Return the exact compiler-owned slot shape shown to the model."""

    ordered = sorted(candidate_ids)
    if operation_kind == "permutation":
        return {
            "initial_order": ["item"],
            "swaps": [["left_item", "right_item"]],
            "query_item": "item (use query_position instead when appropriate)",
        }
    if operation_kind == "arithmetic_dsl":
        return {
            "checks_by_candidate": {
                candidate: {"left_expression": "source-grounded expression", "right_expression": "candidate value"}
                for candidate in ordered
            }
        }
    if operation_kind == "constraint_witness":
        return {
            "assignments_by_candidate": {candidate: {"variable": "value"} for candidate in ordered},
            "constraints": [{"left": "variable_or_value", "operator": "== or !=", "right": "variable_or_value"}],
        }
    return {}


def run_kernel_adapters(
    sample: DatasetSample,
    *,
    contract: TaskContractV2,
    tests: tuple[CertificateTestV2, ...],
    answer_nodes: dict[str, CandidateAnswerNode],
    pairs: tuple[CandidatePairV2, ...],
) -> dict[str, AdapterResult]:
    """Run frozen v2 adapters, overriding only versioned Kernel operations."""

    results = run_deterministic_adapters_v2(
        sample,
        contract=contract,
        tests=tests,
        answer_nodes=answer_nodes,
        pairs=pairs,
    )
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    for test in tests:
        if test.operation_kind not in {"custom_sort_order", "grid_path"}:
            continue
        if test.operation_kind == "grid_path" and test.deterministic_payload.get("path_kind") != "circular":
            continue
        pair = pair_by_id[test.pair_id]
        checks: dict[str, tuple[bool | None, str]] = {}
        for candidate in (pair.left_candidate, pair.right_candidate):
            node = answer_nodes[candidate]
            if test.operation_kind == "custom_sort_order":
                checks[candidate] = _check_custom_sort(node, test.deterministic_payload)
            else:
                checks[candidate] = _check_circular_path(node, test.deterministic_payload)
        valid_candidates = [candidate for candidate, (valid, _) in checks.items() if valid is True]
        status = "EXECUTED"
        observed = None
        if any(valid is None for valid, _ in checks.values()):
            status = "UNSUPPORTED"
        elif len(valid_candidates) != 1:
            status = "CONFLICT"
        else:
            observed = test.expected_outcome_by_candidate[valid_candidates[0]]
        trace = {
            "kernel_adapter_version": "catch_kernel_typed_adapters_v1",
            "test_id": test.test_id,
            "operation_kind": test.operation_kind,
            "payload": test.deterministic_payload,
            "checks": checks,
            "status": status,
            "observed": observed,
        }
        results[test.test_id] = AdapterResult(
            test.test_id,
            observed,
            status,  # type: ignore[arg-type]
            "; ".join(f"{key}:{detail}" for key, (_, detail) in checks.items()),
            _sha256(trace),
        )
    return results


def _compile_custom_sort(source: str) -> tuple[dict[str, Any] | None, str | None]:
    match = re.search(r"Sort the following words.*?:\s*([^\r\n?]+)", source, re.IGNORECASE)
    if match is None:
        return None, "source_word_list_missing"
    words = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if not words:
        return None, "source_word_list_empty"
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    exception = re.search(
        r"except that\s+(.+?)\s+are the last(?:\s+\w+)?\s+letters",
        source,
        re.IGNORECASE,
    )
    if exception is not None:
        tail = re.findall(r"\b([a-z])\b", exception.group(1).casefold())
        if not tail:
            return None, "custom_alphabet_tail_invalid"
        alphabet = [item for item in alphabet if item not in tail] + tail
    return {"words": words, "alphabet": alphabet}, None


def _compile_circular_path(source: str) -> tuple[dict[str, Any] | None, str | None]:
    if "circular path" not in source.casefold():
        return None, "not_a_circular_path"
    top = re.search(r"top of the path,\s*where you find (?:an?|the)\s+([^.,]+)", source, re.IGNORECASE)
    clockwise = re.search(
        r"moving in a clockwise direction from .*?,\s*the elements on the path are\s+(.+?)\.\s*Starting from",
        source,
        re.IGNORECASE | re.DOTALL,
    )
    start = re.search(r"Starting from (?:an?|the)?\s*([^,]+),\s*you move", source, re.IGNORECASE)
    if top is None or clockwise is None or start is None:
        return None, "circular_path_schema_missing"
    tail = re.sub(r",?\s+and\s+", ",", clockwise.group(1).strip(), flags=re.IGNORECASE)
    entities = [_clean_entity(top.group(1)), *(_clean_entity(item) for item in tail.split(",") if item.strip())]
    start_entity = _clean_entity(start.group(1))
    normalized = {_normalize(item) for item in entities}
    if _normalize(start_entity) not in normalized:
        return None, "circular_start_entity_missing"
    raw_moves = re.findall(
        r"by\s+(\d+)\s+steps?\s+in\s+a\s+(counter-clockwise|clockwise)\s+direction",
        source[start.end() :],
        re.IGNORECASE,
    )
    if not raw_moves:
        return None, "circular_moves_missing"
    moves = [int(count) * (-1 if direction.casefold() == "counter-clockwise" else 1) for count, direction in raw_moves]
    return {
        "path_kind": "circular",
        "entities_clockwise": entities,
        "start_entity": start_entity,
        "signed_moves": moves,
    }, None


def _check_custom_sort(node: CandidateAnswerNode, payload: dict[str, Any]) -> tuple[bool | None, str]:
    valid, reason = validate_typed_payload(
        "custom_sort_order",
        payload,
        candidate_ids={node.candidate_key_anon},
        payload_source="local",
    )
    if not valid:
        return None, reason
    rank = {character: index for index, character in enumerate(payload["alphabet"])}

    def key(value: str) -> tuple[int, ...]:
        return tuple(rank.get(character.casefold(), len(rank) + ord(character)) for character in value)

    expected = tuple(sorted(payload["words"], key=key))
    actual = tuple(item.strip() for item in node.rendered_content.split(",") if item.strip())
    return actual == expected, f"expected_order={','.join(expected)}"


def _check_circular_path(node: CandidateAnswerNode, payload: dict[str, Any]) -> tuple[bool | None, str]:
    valid, reason = validate_typed_payload(
        "grid_path",
        payload,
        candidate_ids={node.candidate_key_anon},
        payload_source="local",
    )
    if not valid:
        return None, reason
    entities = payload["entities_clockwise"]
    normalized = [_normalize(item) for item in entities]
    position = normalized.index(_normalize(payload["start_entity"]))
    for move in payload["signed_moves"]:
        position = (position + move) % len(entities)
    expected = entities[position]
    return _normalize(node.rendered_content) == _normalize(expected), f"final_entity={expected}"


def _exact(payload: dict[str, Any], expected: dict[str, Any], reason: str) -> tuple[bool, str]:
    return (payload == expected, "ok" if payload == expected else reason)


def _string_list(value: Any, *, minimum: int) -> bool:
    return isinstance(value, list) and len(value) >= minimum and all(isinstance(item, str) and item for item in value)


def _clean_entity(value: str) -> str:
    return re.sub(r"^(?:an?|the)\s+", "", value.strip(), flags=re.IGNORECASE)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
