"""Kernel-only typed payload compilers and deterministic adapters.

This module deliberately sits outside ``certificates_v2.py`` so frozen
CATCH-Cert v2 reruns keep their historical adapter semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from typing import Any, Literal

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    AdapterResult,
    CandidateAnswerNode,
    CandidatePairV2,
    CertificateTestV2,
    TaskContractV2,
    run_deterministic_adapters_v2,
)

CandidateAdapterStatus = Literal["VALID", "INVALID", "UNSUPPORTED"]


@dataclass(frozen=True)
class CandidateAdapterResult:
    """A candidate-local verdict; unlike pair tests it has no global conflict state."""

    candidate_key_anon: str
    operation_kind: str
    status: CandidateAdapterStatus
    detail: str
    execution_trace_hash: str


def candidate_adapter_result_to_dict(result: CandidateAdapterResult) -> dict[str, Any]:
    return asdict(result)


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
        ordered, ordered_error = _compile_ordered_grid_path(source)
        return (ordered if ordered is not None else {"executor": "ordered_grid_path_v1"}), None
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
        if set(payload) == {"path_kind", "expected_entity"}:
            valid = payload.get("path_kind") == "ordered" and isinstance(payload.get("expected_entity"), str)
            return valid, "ok" if valid else "grid_path_payload_invalid"
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
        if test.operation_kind == "grid_path" and test.deterministic_payload == {
            "executor": "ordered_grid_path_v1"
        }:
            # Historical D1 payload: it declared jurisdiction without compiling
            # the route.  Preserve the frozen adapter result and let D2's unary
            # compiler return UNSUPPORTED; never reinterpret it as circular.
            continue
        pair = pair_by_id[test.pair_id]
        checks: dict[str, tuple[bool | None, str]] = {}
        for candidate in (pair.left_candidate, pair.right_candidate):
            node = answer_nodes[candidate]
            if test.operation_kind == "custom_sort_order":
                checks[candidate] = _check_custom_sort(node, test.deterministic_payload)
            else:
                checks[candidate] = _check_grid_path(node, test.deterministic_payload)
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


def run_kernel_unary_adapters(
    sample: DatasetSample,
    *,
    tests: tuple[CertificateTestV2, ...],
    answer_nodes: dict[str, CandidateAnswerNode],
) -> dict[str, CandidateAdapterResult]:
    """Verify every answer class independently under the local D2 compiler.

    A result is shared across duplicate obligation tests.  This avoids the D1
    failure mode in which an invalid-vs-invalid pair became a global veto even
    when one different candidate had an exact, deterministic certificate.
    """

    if not tests:
        return {}
    kinds = {test.operation_kind for test in tests}
    payloads = {_sha256(test.deterministic_payload): test.deterministic_payload for test in tests}
    if len(kinds) != 1 or len(payloads) != 1:
        detail = "inconsistent_unary_test_contract"
        return {
            key: _candidate_result(key, "mixed", "UNSUPPORTED", detail, {}) for key in answer_nodes
        }
    operation_kind = next(iter(kinds))
    payload = next(iter(payloads.values()))
    results: dict[str, CandidateAdapterResult] = {}
    for candidate, node in answer_nodes.items():
        valid, detail = _check_candidate(sample, node, operation_kind, payload)
        status: CandidateAdapterStatus = "UNSUPPORTED" if valid is None else "VALID" if valid else "INVALID"
        results[candidate] = _candidate_result(candidate, operation_kind, status, detail, payload)
    return results


def _candidate_result(
    candidate: str,
    operation_kind: str,
    status: CandidateAdapterStatus,
    detail: str,
    payload: dict[str, Any],
) -> CandidateAdapterResult:
    trace = {
        "kernel_adapter_version": "catch_kernel_unary_exact_v1",
        "candidate": candidate,
        "operation_kind": operation_kind,
        "payload": payload,
        "status": status,
        "detail": detail,
    }
    return CandidateAdapterResult(candidate, operation_kind, status, detail, _sha256(trace))


def _check_candidate(
    sample: DatasetSample,
    node: CandidateAnswerNode,
    operation_kind: str,
    payload: dict[str, Any],
) -> tuple[bool | None, str]:
    if operation_kind == "seq_plan":
        expected, error = _canonical_seqbench_plan(sample)
        if expected is None:
            return None, error or "seqbench_compile_failed"
        try:
            actual = tuple(str(item) for item in json.loads(node.canonical_value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False, "invalid_sequence_syntax"
        return actual == expected, "canonical_plan_match" if actual == expected else "canonical_plan_mismatch"
    if operation_kind == "stack_trace":
        expected, reason = _strict_dyck_earliest_error(question_without_answer_contract(sample))
        if expected is None:
            return None, reason
        actual = _normalize(node.rendered_content)
        return actual == expected, f"expected_earliest={expected};reason={reason}"
    if operation_kind == "custom_sort_order":
        compiled, error = _compile_custom_sort(question_without_answer_contract(sample), strict=True)
        if compiled is None:
            return None, error or "custom_alphabet_rule_unrecognized"
        return _check_custom_sort(node, compiled)
    if operation_kind == "grid_path":
        compiled, error = compile_local_typed_payload(sample, "grid_path")
        if compiled is None or compiled.get("executor") == "ordered_grid_path_v1":
            compiled, error = _compile_ordered_grid_path(question_without_answer_contract(sample))
        if compiled is None:
            circular, circular_error = _compile_circular_path(question_without_answer_contract(sample))
            compiled, error = circular, circular_error
        if compiled is None:
            return None, error or "grid_path_compile_failed"
        return _check_grid_path(node, compiled)
    return None, f"unary_adapter_unregistered:{operation_kind}"


def _compile_custom_sort(source: str, *, strict: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    match = re.search(r"Sort the following words.*?:\s*([^\r\n?]+)", source, re.IGNORECASE)
    if match is None:
        return None, "source_word_list_missing"
    words = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if not words:
        return None, "source_word_list_empty"
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    explicit = re.search(r"new alphabet order\s*\[([^\]]+)\]", source, re.IGNORECASE)
    first = re.search(r"except that\s+(.+?)\s+(?:is the first letter|are the first two letters)", source, re.IGNORECASE)
    last = re.search(r"except that\s+(.+?)\s+(?:is the last letter|are the last two letters)", source, re.IGNORECASE)
    swapped = re.search(r"except that\s+([a-z])\s+and\s+([a-z])\s+are swapped", source, re.IGNORECASE)
    if explicit is not None:
        alphabet = [item.casefold() for item in re.findall(r"\b([a-z])\b", explicit.group(1), re.IGNORECASE)]
    elif first is not None:
        head = re.findall(r"\b([a-z])\b", first.group(1).casefold())
        alphabet = head + [item for item in alphabet if item not in head]
    elif last is not None:
        tail = re.findall(r"\b([a-z])\b", last.group(1).casefold())
        alphabet = [item for item in alphabet if item not in tail] + tail
    elif swapped is not None:
        left, right = (item.casefold() for item in swapped.groups())
        left_index, right_index = alphabet.index(left), alphabet.index(right)
        alphabet[left_index], alphabet[right_index] = alphabet[right_index], alphabet[left_index]
    else:
        if strict:
            return None, "custom_alphabet_rule_unrecognized"
    if len(alphabet) != 26 or set(alphabet) != set("abcdefghijklmnopqrstuvwxyz"):
        return None, "custom_alphabet_not_a_permutation"
    return {"words": words, "alphabet": alphabet}, None


def _compile_ordered_grid_path(source: str) -> tuple[dict[str, Any] | None, str | None]:
    folded = source.casefold()
    if "jump to a random vertex" in folded or "jump back to" in folded:
        return None, "grid_random_jump_unsupported"
    if "hexagonal tile map" in folded:
        vectors = {
            "up": (0, 2), "down": (0, -2), "up-left": (-1, 1),
            "up-right": (1, 1), "down-left": (-1, -1), "down-right": (1, -1),
        }
    elif "triangular tile map" in folded:
        vectors = {
            "left": (-2, 0), "right": (2, 0), "up-left": (-1, 1),
            "up-right": (1, 1), "down-left": (-1, -1), "down-right": (1, -1),
        }
    else:
        return None, "ordered_grid_family_unsupported"
    initial = re.search(
        r"(?:Initially, you are positioned|You are initially).*?where you (?:find|see)\s+(?:an?\s+|the\s+)?([^.,]+)",
        source,
        re.IGNORECASE,
    )
    if initial is None:
        return None, "ordered_grid_initial_observation_missing"
    position = (0, 0)
    observations: dict[tuple[int, int], str] = {position: _clean_entity(initial.group(1))}
    moves = re.finditer(
        r"(?:Then\s+)?you move\s+(up-left|up-right|down-left|down-right|up|down|left|right)\s+"
        r"(?:by|for)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+steps?"
        r"(?:,?\s+where you (?:find|see)|\s+and see)?\s*(?:an?\s+|the\s+)?([^.]*)\.",
        source,
        re.IGNORECASE,
    )
    move_count = 0
    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    for move in moves:
        direction = move.group(1).casefold()
        if direction not in vectors:
            return None, f"ordered_grid_direction_unsupported:{direction}"
        raw_count = move.group(2).casefold()
        count = number_words.get(raw_count, int(raw_count) if raw_count.isdigit() else 0)
        dx, dy = vectors[direction]
        position = (position[0] + dx * count, position[1] + dy * count)
        observation = move.group(3).strip(" ,")
        if observation:
            entity = _clean_entity(observation)
            previous = observations.get(position)
            if previous is not None and _normalize(previous) != _normalize(entity):
                return None, "ordered_grid_inconsistent_observation"
            observations[position] = entity
        move_count += 1
    if move_count == 0:
        return None, "ordered_grid_moves_missing"
    return {"path_kind": "ordered", "expected_entity": observations.get(position, "unknown")}, None


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


def _check_grid_path(node: CandidateAnswerNode, payload: dict[str, Any]) -> tuple[bool | None, str]:
    valid, reason = validate_typed_payload(
        "grid_path",
        payload,
        candidate_ids={node.candidate_key_anon},
        payload_source="local",
    )
    if not valid:
        return None, reason
    if payload.get("path_kind") == "ordered":
        expected = payload["expected_entity"]
        return _normalize(node.rendered_content) == _normalize(expected), f"final_entity={expected}"
    entities = payload["entities_clockwise"]
    normalized = [_normalize(item) for item in entities]
    position = normalized.index(_normalize(payload["start_entity"]))
    for move in payload["signed_moves"]:
        position = (position + move) % len(entities)
    expected = entities[position]
    return _normalize(node.rendered_content) == _normalize(expected), f"final_entity={expected}"


def _canonical_seqbench_plan(sample: DatasetSample) -> tuple[tuple[str, ...] | None, str | None]:
    """Compile the exact deterministic shortest plan used by seqBench Pass@1."""

    context = sample.question
    room = r"([A-Z]+[0-9]+)"
    open_edges: set[frozenset[str]] = {
        frozenset((left.upper(), right.upper()))
        for left, right in re.findall(
            rf"Room\s+{room}\s+and\s+{room}\s+are connected by an open door",
            context,
            flags=re.IGNORECASE,
        )
    }
    locked: dict[frozenset[str], str] = {}
    for left, right in re.findall(
        rf"Room\s+{room}\s+and\s+{room}\s+are connected by a closed and locked door",
        context,
        flags=re.IGNORECASE,
    ):
        locked[frozenset((left.upper(), right.upper()))] = ""
    for left, right, key in re.findall(
        rf"(?:The )?locked door between\s+{room}\s+and\s+{room}\s+requires key\s+([0-9]+)",
        context,
        flags=re.IGNORECASE,
    ):
        locked[frozenset((left.upper(), right.upper()))] = key
    key_locations = {
        key: location.upper()
        for key, location in re.findall(rf"Key\s+([0-9]+)\s+is in room\s+{room}", context, flags=re.IGNORECASE)
    }
    metadata = sample.metadata.get("seqbench_instance_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    agent = str(metadata.get("agent_name") or "Bob")
    target = str(metadata.get("target_name") or "Alice")
    positions = {
        name: location.upper()
        for name, location in re.findall(
            rf"([A-Za-z][A-Za-z0-9_-]*)\s+is in room\s+{room}", context
        )
    }
    start, goal = positions.get(agent), positions.get(target)
    if start is None or goal is None:
        return None, "seqbench_agent_or_target_room_missing"
    if any(not key for key in locked.values()):
        return None, "seqbench_locked_door_key_missing"
    keys = sorted(key_locations, key=lambda value: (int(value), value))
    key_bit = {key: 1 << index for index, key in enumerate(keys)}
    locked_edges = sorted(locked, key=lambda edge: tuple(sorted(edge)))
    edge_bit = {edge: 1 << index for index, edge in enumerate(locked_edges)}
    adjacency: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for edge in open_edges | set(locked):
        if len(edge) != 2:
            continue
        left, right = sorted(edge)
        adjacency.setdefault(left, []).append((right, edge))
        adjacency.setdefault(right, []).append((left, edge))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: item[0])

    # Heap ordering by the complete action tuple makes shortest-path ties
    # deterministic and reproduces the benchmark's canonical reference path.
    initial_state = (start, 0, 0)
    initial_actions = (f"start: {start}",)
    heap: list[tuple[int, tuple[str, ...], tuple[str, int, int]]] = [(1, initial_actions, initial_state)]
    best: dict[tuple[str, int, int], tuple[int, tuple[str, ...]]] = {initial_state: (1, initial_actions)}
    while heap:
        cost, actions, state = heappop(heap)
        if best.get(state) != (cost, actions):
            continue
        current, held_mask, unlocked_mask = state
        if current == goal:
            return (*actions, f"rescue: {target}"), None
        for key in keys:
            bit = key_bit[key]
            if key_locations[key] == current and not held_mask & bit:
                next_state = (current, held_mask | bit, unlocked_mask)
                next_actions = (*actions, f"pick_up_key: {key}")
                _push_seq_state(heap, best, cost + 1, next_actions, next_state)
        for neighbor, edge in adjacency.get(current, []):
            if edge in open_edges or unlocked_mask & edge_bit.get(edge, 0):
                next_state = (neighbor, held_mask, unlocked_mask)
                next_actions = (*actions, f"move_to: {neighbor}")
                _push_seq_state(heap, best, cost + 1, next_actions, next_state)
                continue
            key = locked.get(edge)
            if key is None or not held_mask & key_bit.get(key, 0):
                continue
            next_state = (neighbor, held_mask, unlocked_mask | edge_bit[edge])
            next_actions = (
                *actions,
                f"use_key: {key}",
                f"unlock_and_open_door_to: {neighbor}",
                f"move_to: {neighbor}",
            )
            _push_seq_state(heap, best, cost + 3, next_actions, next_state)
    return None, "seqbench_goal_unreachable"


def _push_seq_state(
    heap: list[tuple[int, tuple[str, ...], tuple[str, int, int]]],
    best: dict[tuple[str, int, int], tuple[int, tuple[str, ...]]],
    cost: int,
    actions: tuple[str, ...],
    state: tuple[str, int, int],
) -> None:
    score = (cost, actions)
    if state not in best or score < best[state]:
        best[state] = score
        heappush(heap, (cost, actions, state))


def _strict_dyck_earliest_error(source: str) -> tuple[str | None, str]:
    input_match = re.search(r"\bInput:\s*(.*?)\s*Thought\s+1:", source, re.IGNORECASE | re.DOTALL)
    if input_match is None:
        return None, "dyck_input_missing"
    tokens = re.findall(r"[\[\](){}<>]", input_match.group(1))
    if not tokens:
        return None, "dyck_input_empty"
    thoughts = {
        int(number): body.strip()
        for number, body in re.findall(r"^Thought\s+(\d+):\s*(.*)$", source, re.IGNORECASE | re.MULTILINE)
    }
    if _normalize(thoughts.get(2, "")) != "stack: empty":
        return "2", "initial_stack_mismatch"
    stack: list[str] = []
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    for index, token in enumerate(tokens):
        thought_number = index + 3
        body = thoughts.get(thought_number)
        if body is None:
            return str(thought_number), "missing_token_thought"
        parsed = re.fullmatch(r"\s*([\[\](){}<>])\s*;\s*stack:\s*(.*?)\s*", body)
        if parsed is None:
            return str(thought_number), "token_thought_schema_mismatch"
        if parsed.group(1) != token:
            return str(thought_number), "input_token_mismatch"
        if token in "([{<":
            stack.append(token)
        elif not stack or stack[-1] != closing[token]:
            return str(thought_number), "invalid_bracket_transition"
        else:
            stack.pop()
        expected_stack = " ".join(stack) if stack else "empty"
        if parsed.group(2).strip() != expected_stack:
            return str(thought_number), "reported_stack_lexical_mismatch"
    return "no", "no_error"


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
