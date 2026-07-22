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

KERNEL_PROMPT_VERSION = "catch_kernel_atomic_truth_v3"


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
