"""独立 CATCH-Cert 协议的证书设计与盲验证提示词。"""

from __future__ import annotations

import json

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.families.contrastive_active_testing.certificates import (
    CertificateVerifierPacket,
    ClaimGraph,
    TaskContract,
    claim_graph_to_dict,
    task_contract_to_dict,
)

CERT_PROMPT_VERSION = "catch_cert_question_conditioned_v1"
CERT_SCHEMA_VERSION = "catch_cert_v1"


def build_certificate_designer_messages(
    sample: DatasetSample,
    *,
    contract: TaskContract,
    graphs: dict[str, ClaimGraph],
    public_pairs: tuple[dict[str, str], ...],
) -> list[dict[str, str]]:
    public_graphs = {key: claim_graph_to_dict(value) for key, value in graphs.items()}
    schema = {
        "certificates": [
            {
                "candidate_key_anon": "H0",
                "required_conditions": ["T0"],
                "predicted_outcome": "short candidate outcome",
                "refutation_condition": "T0",
                "evidence_refs": ["H0:N0"],
                "dependency_refs": [],
            }
        ],
        "tests": [
            {
                "test_id": "T0",
                "pair_id": "P0",
                "question_or_operation": "one finite question-conditioned check",
                "finite_outcomes": [
                    {"outcome_id": "O0", "text": "finite outcome"},
                    {"outcome_id": "O1", "text": "incompatible finite outcome"},
                ],
                "expected_outcome_by_candidate": {"H0": "O0", "H1": "O1"},
                "provenance_refs": ["H0:N0", "H1:N0"],
                "task_family": contract.family,
            }
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You design question-conditioned correctness certificates for "
                "anonymous existing candidates. Do not select a final answer, use vote counts, or invent a new "
                "candidate. Tests must be finite, falsifiable, source-decidable, and backed by listed claim IDs."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source task with hidden answer contract:\n{question_without_answer_contract(sample)}\n\n"
                f"Task contract:\n{json.dumps(task_contract_to_dict(contract), ensure_ascii=False)}\n\n"
                f"Anonymous candidate pairs:\n{json.dumps(public_pairs, ensure_ascii=False)}\n\n"
                f"Anonymous provenance graphs:\n{json.dumps(public_graphs, ensure_ascii=False)}\n\n"
                "For every candidate pair, identify the smallest answer-critical conditions whose finite outcomes "
                "differ between the two candidates. A condition must test sufficiency for the source question, not "
                "mere local textual support. Include a refutation condition for each certificate. You may reference "
                "non-contiguous claim IDs, but never copy candidate IDs into question or outcome text. Use two to "
                "six tests total and at most four required conditions per certificate. For equation tasks, an "
                "optional deterministic arithmetic check may use the exact form 'CHECK: numeric_expression == "
                "numeric_expression'. Return exactly this schema with no extra fields:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_certificate_verifier_messages(
    sample: DatasetSample,
    *,
    contract: TaskContract,
    packet: CertificateVerifierPacket,
) -> list[dict[str, str]]:
    role_guidance = (
        "Establish positive support only when the source task entails the observed finite outcome."
        if packet.role == "support_auditor"
        else "Actively seek a source-grounded counterexample before marking an observed outcome entailed."
    )
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You are a blinded certificate verifier, not a final-answer judge. "
                f"{role_guidance} You never see candidate identities, answers, votes, or commitments."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source task with hidden answer contract:\n{question_without_answer_contract(sample)}\n\n"
                f"Task contract:\n{json.dumps(task_contract_to_dict(contract), ensure_ascii=False)}\n\n"
                f"Finite certificate tests:\n{json.dumps(list(packet.tests), ensure_ascii=False)}\n\n"
                "For each test choose exactly one listed public outcome. Set support_status to ENTAILED only if "
                "that outcome is sufficient under the source question and task contract, CONTRADICTED when the "
                "source establishes an incompatible result, or UNDERDETERMINED otherwise. counterexample must be "
                "a short source-grounded refutation when one exists, else an empty string. source_refs must contain "
                "short exact source snippets and must not contain candidate IDs. Return exactly "
                '{"results":[{"test_id":"Q0","observed_outcome":"R0","support_status":"ENTAILED",'
                '"counterexample":"","source_refs":["exact source snippet"]}]} with no extra fields.'
            ),
        },
    ]
