"""CATCH-Cert v2 答案连接与全局义务证书的提示词。"""

from __future__ import annotations

import json
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    CandidateAnswerNode,
    CandidatePairV2,
    CertificateVerifierPacketV2,
    SourceSpanGraph,
    TaskContractV2,
    candidate_answer_node_to_dict,
    pair_v2_to_dict,
    source_span_graph_to_dict,
    task_contract_v2_to_dict,
)

CERT_V2_PROMPT_VERSION = "catch_cert_answer_linked_obligations_v2"
CERT_V2_SCHEMA_VERSION = "catch_cert_v2"


def build_certificate_designer_messages_v2(
    sample: DatasetSample,
    *,
    contract: TaskContractV2,
    answer_nodes: dict[str, CandidateAnswerNode],
    source_graph: SourceSpanGraph,
    pairs: tuple[CandidatePairV2, ...],
    reasoning_claims: dict[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    """Render pair-specific identifiers instead of a misleading fixed example."""

    public_pairs = [pair_v2_to_dict(pair) for pair in pairs]
    pair_candidates = {pair.pair_id: [pair.left_candidate, pair.right_candidate] for pair in pairs}
    first_pair = pairs[0] if pairs else None
    example_test: dict[str, Any] = {
        "test_id": "T0",
        "pair_id": first_pair.pair_id if first_pair else "P0",
        "obligation_ids": [contract.mandatory_obligations[0].obligation_id],
        "operation_kind": contract.adapter_kind,
        "question_or_operation": "one finite answer-critical operation",
        "finite_outcomes": [
            {"outcome_id": "O0", "text": "first finite outcome"},
            {"outcome_id": "O1", "text": "incompatible finite outcome"},
        ],
        "expected_outcome_by_candidate": (
            {first_pair.left_candidate: "O0", first_pair.right_candidate: "O1"}
            if first_pair
            else {"H0": "O0", "H1": "O1"}
        ),
        "source_span_ids": [source_graph.spans[0].span_id] if source_graph.spans else [],
        "deterministic_payload": {},
    }
    first_candidate = first_pair.left_candidate if first_pair else next(iter(answer_nodes), "H0")
    example_certificate = {
        "candidate_key_anon": first_candidate,
        "answer_hash": answer_nodes[first_candidate].answer_hash if first_candidate in answer_nodes else "64-char-hash",
        "required_test_ids": ["T0"],
    }
    schema = {"certificates": [example_certificate], "tests": [example_test]}
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You compile answer-linked correctness certificates for anonymous "
                "existing answers. Do not choose a winner, infer vote counts, or invent a new answer. A local fact "
                "is not sufficient unless it covers every mandatory question obligation and deterministically "
                "connects a finite outcome to the candidate answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source task without its answer key:\n{question_without_answer_contract(sample)}\n\n"
                f"Indexed source spans:\n{json.dumps(source_span_graph_to_dict(source_graph), ensure_ascii=False)}\n\n"
                f"Question contract and mandatory obligations:\n{json.dumps(task_contract_v2_to_dict(contract), ensure_ascii=False)}\n\n"
                f"Anonymous answer nodes (these are candidate meanings, not correctness labels):\n"
                f"{json.dumps({key: candidate_answer_node_to_dict(value) for key, value in answer_nodes.items()}, ensure_ascii=False)}\n\n"
                f"Short answer-connected reasoning claims:\n{json.dumps(reasoning_claims, ensure_ascii=False)}\n\n"
                f"Anonymous pairs:\n{json.dumps(public_pairs, ensure_ascii=False)}\n\n"
                f"Exact candidates allowed for each pair:\n{json.dumps(pair_candidates, ensure_ascii=False)}\n\n"
                "Create one to six finite tests total. Every expected_outcome_by_candidate object must contain "
                "exactly the two candidate IDs listed for its pair and must assign different outcomes. Every "
                "candidate certificate must copy its exact answer_hash and cover all mandatory obligation IDs. "
                "For earliest questions, include both the candidate-step error and validity of the complete earlier "
                "prefix. For final-state, argmax, exact-set, and exact-sequence questions, explicitly cover the "
                "corresponding global obligation. source_span_ids must be selected from the indexed source. "
                "deterministic_payload may contain only explicit inputs needed to execute the stated operation; use "
                "an empty object for semantic checks. Do not emit a refutation_condition or predicted_outcome; "
                "those are compiled locally. Return exactly this field structure with no extra fields:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_certificate_verifier_messages_v2(
    sample: DatasetSample,
    *,
    contract: TaskContractV2,
    packet: CertificateVerifierPacketV2,
) -> list[dict[str, str]]:
    role_guidance = (
        "Establish a finite outcome only when the indexed source entails it."
        if packet.role == "support_auditor"
        else "Independently try to rule out each finite outcome before returning the source-entailed result."
    )
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You are a blinded finite-outcome verifier, not an answer judge. "
                f"{role_guidance} You never see candidate IDs, candidate answers, votes, or commitments."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source task:\n{question_without_answer_contract(sample)}\n\n"
                f"Question contract:\n{json.dumps(task_contract_v2_to_dict(contract), ensure_ascii=False)}\n\n"
                f"Exact indexed source spans:\n{json.dumps(list(packet.source_spans), ensure_ascii=False)}\n\n"
                f"Finite tests:\n{json.dumps(list(packet.tests), ensure_ascii=False)}\n\n"
                "For every test, choose exactly one listed public outcome. Use ENTAILED only when the source "
                "supports that outcome, CONTRADICTED when it establishes an incompatible result, and "
                "UNDERDETERMINED otherwise. A decisive result must cite one or more exact span IDs; an "
                "UNDERDETERMINED result may cite none. ruled_out_outcomes contains listed public outcome IDs that "
                "the source excludes. Return exactly "
                '{"results":[{"test_id":"Q0","observed_outcome":"R0","support_status":"ENTAILED",'
                '"source_span_ids":["S0"],"ruled_out_outcomes":["R1"]}]} with no extra fields.'
            ),
        },
    ]
