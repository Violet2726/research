"""CATCH-Kernel 语义义务使用的短式、仅填槽提示词。"""

from __future__ import annotations

import json

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    CandidateAnswerNode,
    CertificateBankValidationV2,
    CertificateVerifierPacketV2,
    SourceSpanGraph,
    candidate_answer_node_to_dict,
)
from research_experiments.families.contrastive_active_testing.kernel import (
    KERNEL_SCHEMA_VERSION,
    TaskSemantics,
    task_semantics_to_dict,
)
from research_experiments.families.contrastive_active_testing.kernel_adapters import (
    typed_payload_prompt_schema,
)
from research_experiments.families.contrastive_active_testing.kernel_d3 import D3_IR_SCHEMA, D3_IR_VERSION
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    D4_PROMPT_VERSION,
    D4RouteDecision,
)

KERNEL_PROMPT_VERSION = "catch_kernel_atomic_truth_v3"
D3_PROMPT_VERSION = "catch_kernel_d3_source_blind_v1"


def build_d4_source_compiler_messages(
    sample: DatasetSample,
    *,
    source_spans: list[dict[str, str]],
    answer_contract: dict[str, object],
    decision: D4RouteDecision,
) -> list[dict[str, str]]:
    """Build a D4 compiler prompt from source-only inputs.

    The caller cannot pass Stage-A state.  ``canonical_ir_hash`` is emitted as
    an empty placeholder and is computed by trusted local code after parsing.
    """

    schema = {
        "capability_id": decision.capability_id,
        "query_operator": decision.query_operator,
        "entities": [{"entity_id": "source entity", "kind": "typed kind"}],
        "facts": [{"fact_id": "F0", "kind": "typed fact", "source_span_ids": ["S0"]}],
        "events": [{"event_id": "E0", "kind": "typed event", "source_span_ids": ["S0"]}],
        "constraints": [
            {
                "constraint_id": "C0",
                "kind": "numeric_expression",
                "expression": "closed expression using source literals only",
                "source_span_ids": ["S0"],
            }
        ],
        "query": {"kind": decision.query_operator, "source_span_ids": ["S0"]},
        "answer_contract": answer_contract,
        "source_span_map": source_spans,
        "mandatory_spans": ["S0"],
        "uncovered_spans": [],
        "canonical_ir_hash": "",
    }
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object and no other text. You are a candidate-blind source compiler. "
                "You never receive Stage-A answers, candidates, an anchor, vote counts, gold, or a candidate "
                "oracle. Compile only source-grounded entities, facts, ordered events, constraints, and query. "
                "Do not infer open-world facts. Copy source_span_map and answer_contract exactly. Leave "
                "canonical_ir_hash empty for trusted local computation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Compiler protocol: {D4_PROMPT_VERSION}\n"
                f"Kernel: {decision.kernel_id}\n"
                f"Capability: {decision.capability_id}\n"
                f"Query operator: {decision.query_operator}\n\n"
                f"Source task (answer key removed):\n{question_without_answer_contract(sample)}\n\n"
                f"Allowed answer contract (schema only, never gold):\n"
                f"{json.dumps(answer_contract, ensure_ascii=False)}\n\n"
                f"Exact indexed source spans:\n{json.dumps(source_spans, ensure_ascii=False)}\n\n"
                "Every decisive fact, event, constraint, and query field must cite source_span_ids. "
                "Put genuinely irrelevant spans in uncovered_spans; all query-critical spans must be mandatory. "
                "For evaluate_numeric_expression, use one closed numeric_expression and no unmentioned constants, "
                "formulas, conversions, chemical structures, or retrieved knowledge. If the source cannot be "
                "compiled into the declared operator, preserve the source span map but return empty typed lists and "
                "a query whose status is unsupported. Return exactly these keys and no others:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_d3_source_compiler_messages(
    sample: DatasetSample,
    *,
    source_spans: list[dict[str, str]],
    answer_schema: list[dict[str, str]],
    operation_kind: str,
) -> list[dict[str, str]]:
    """Prompt a source-only compiler; Stage-A candidates are intentionally absent."""

    schema = {
        "schema": D3_IR_SCHEMA,
        "ir_version": D3_IR_VERSION,
        "query": {
            "kind": "evaluate_numeric_expression",
            "source_span_ids": ["S0"],
            "constraint_ids": ["C0"],
        },
        "constraints": [
            {
                "constraint_id": "C0",
                "kind": "numeric_expression",
                "expression": "closed numeric expression only",
                "source_span_ids": ["S0"],
            }
        ],
        "covered_span_ids": ["S0"],
        "uncovered_span_ids": [],
    }
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You are a candidate-blind source compiler. "
                "You do not see Stage-A answers, candidate labels, votes, an anchor, or gold. "
                "Compile only the source into a closed typed IR. Do not invent facts, variables, "
                "constraints, or answer choices. Every source span must be in exactly one of "
                "covered_span_ids or uncovered_span_ids."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Compiler protocol: {D3_PROMPT_VERSION}\n"
                f"Operation kind: {operation_kind}\n"
                f"Source task (answer region removed):\n{question_without_answer_contract(sample)}\n\n"
                f"Answer schema (allowed options only; no gold):\n{json.dumps(answer_schema, ensure_ascii=False)}\n\n"
                f"Indexed source spans:\n{json.dumps(source_spans, ensure_ascii=False)}\n\n"
                "For safe_numeric_expression, bind the query and the single C0 constraint to exact source spans. "
                "Every numeric literal, constant, and function in expression must occur in those bound spans; "
                "do not import conversion factors, formulas, or world knowledge. The expression may contain only "
                "numeric literals, arithmetic, parentheses, and constants/functions permitted by the local checker. "
                "Return exactly this object shape and no extra fields:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_kernel_designer_messages(
    sample: DatasetSample,
    *,
    semantics: TaskSemantics,
    answer_nodes: dict[str, CandidateAnswerNode],
    source_graph: SourceSpanGraph,
    skeleton: CertificateBankValidationV2,
    reasoning_claims: dict[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    """The model fills values; operation names and meanings are compiler-owned."""

    templates = {item.obligation_id: item for item in semantics.mandatory_obligation_templates}
    slots = [
        {
            "test_id": test.test_id,
            "operation_kind": test.operation_kind,
            "obligation_id": test.obligation_ids[0],
            "candidate_ids": sorted(test.expected_outcome_by_candidate),
            "typed_payload_schema": typed_payload_prompt_schema(
                test.operation_kind,
                set(test.expected_outcome_by_candidate),
            ),
        }
        for test in skeleton.tests
        if templates[test.obligation_ids[0]].payload_source == "model_slots"
    ]
    schema = {
        "payloads": [{"test_id": item["test_id"], "typed_payload": item["typed_payload_schema"]} for item in slots]
    }
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You are an untrusted slot filler for a locally defined verification "
                "specification. Never invent tests, operation kinds, obligations, finite outcomes, candidate "
                "commitments, candidate IDs, answer hashes, or a new answer. Fill only the explicitly declared typed "
                "payload slots."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Protocol schema version: {KERNEL_SCHEMA_VERSION}\n\n"
                f"Source task without answer key:\n{question_without_answer_contract(sample)}\n\n"
                f"Task semantics chosen by the local registry:\n"
                f"{json.dumps(task_semantics_to_dict(semantics), ensure_ascii=False)}\n\n"
                f"Exact source spans:\n"
                f"{json.dumps([{'span_id': item.span_id, 'text': item.text} for item in source_graph.spans], ensure_ascii=False)}\n\n"
                f"Anonymous answer nodes:\n"
                f"{json.dumps({key: candidate_answer_node_to_dict(value) for key, value in answer_nodes.items()}, ensure_ascii=False)}\n\n"
                f"Bounded answer-connected claims:\n{json.dumps(reasoning_claims, ensure_ascii=False)}\n\n"
                f"Compiler-owned payload slots:\n{json.dumps(slots, ensure_ascii=False)}\n\n"
                "Return one payload for every listed test_id and no other test. Copy each test_id exactly. Every "
                "typed_payload must match its operation-specific shape and use only explicit values grounded in the "
                "indexed source. Do not output certificates, outcome text, commitments, or source-span IDs. Return "
                "exactly this structure with no extra fields:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_kernel_verifier_messages(
    sample: DatasetSample,
    *,
    semantics: TaskSemantics,
    packet: CertificateVerifierPacketV2,
) -> list[dict[str, str]]:
    role = (
        "Resolve each finite atomic proposition from the indexed source."
        if packet.role == "support_auditor"
        else "Attempt to falsify each outcome independently before resolving it."
    )
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You are a blinded atomic verifier operating inside a declared semantic "
                f"jurisdiction. {role} You cannot select an answer candidate and you never see candidate commitments."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source task:\n{question_without_answer_contract(sample)}\n\n"
                f"Registry semantics:\n{json.dumps(task_semantics_to_dict(semantics), ensure_ascii=False)}\n\n"
                f"Indexed source spans:\n{json.dumps(list(packet.source_spans), ensure_ascii=False)}\n\n"
                f"Typed finite tests:\n{json.dumps(list(packet.tests), ensure_ascii=False)}\n\n"
                "For every test return one listed outcome. ENTAILED requires exact supporting span IDs; "
                "CONTRADICTED requires exact incompatible evidence; otherwise return UNDERDETERMINED. Return exactly "
                '{"results":[{"test_id":"Q0","observed_outcome":"R0","support_status":"ENTAILED",'
                '"source_span_ids":["S0"],"ruled_out_outcomes":["R1"]}]}.'
            ),
        },
    ]
