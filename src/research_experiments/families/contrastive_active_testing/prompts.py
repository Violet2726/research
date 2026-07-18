"""CATCH-v2 证据编译器与盲化测量提示。"""

from __future__ import annotations

import json

from research_experiments.core.data.datasets import DatasetSample, question_without_bbeh_options
from research_experiments.families.contrastive_active_testing.algorithms import StageDecision, WitnessPacket

CATCH_PROMPT_VERSION = "catch_v2_evidence_compiler"
CATCH_SCHEMA_VERSION = "catch_contrast_atoms_v2"


def build_designer_messages(
    sample: DatasetSample,
    *,
    stage: StageDecision,
    hypothesis_to_key: dict[str, str],
) -> list[dict[str, str]]:
    candidates = {candidate.key: candidate for candidate in stage.candidates}
    hypotheses = [
        {
            "id": hypothesis,
            "answer": candidates[key].answer,
            "reasoning": candidates[key].representative_reasoning,
        }
        for hypothesis, key in hypothesis_to_key.items()
    ]
    schema = {
        "tests": [
            {
                "test_id": "T0",
                "question": "one atomic diagnostic question",
                "outcomes": [
                    {"id": "O0", "text": "finite outcome"},
                    {"id": "O1", "text": "different finite outcome"},
                ],
                "commitments": {
                    hypothesis: {
                        "outcome_id": "O0",
                        "evidence_quote": "an exact, uniquely occurring quote from this hypothesis reasoning",
                    }
                    for hypothesis in hypothesis_to_key
                },
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. Design finite, contrastive measurements that distinguish anonymous "
                "answer hypotheses. Do not judge by style, confidence, verbosity, or consensus."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{sample.question}\n\n"
                f"Anonymous hypotheses:\n{json.dumps(hypotheses, ensure_ascii=False)}\n\n"
                "Compile at most six pair-targeted contrast atoms. Each atom must be answerable from the original "
                "stem/context, "
                "must not ask the original final-answer question, and must have 2-4 finite outcomes. Do not mention "
                "hypothesis IDs, candidate metadata, vote counts, or option labels inside question/outcome text. "
                "For every non-null commitment, copy one exact, uniquely occurring evidence_quote from that "
                "hypothesis's reasoning. Never count or emit character offsets. Use null when a trace makes no "
                "commitment. Every atom must assign different non-null outcomes to at least one pair of hypotheses; "
                "when possible, provide two independent atoms per distinguishable pair using non-overlapping quotes. "
                "Do not quote or cite FINAL_ANSWER.\n\n"
                f"Schema example:\n{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_witness_messages(sample: DatasetSample, *, packet: WitnessPacket) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You are a blinded measurement witness. Answer each finite diagnostic "
                "test from the supplied source material. You are not selecting a final candidate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source material (final answer options removed):\n{question_without_bbeh_options(sample)}\n\n"
                f"Diagnostic tests:\n{json.dumps(list(packet.tests), ensure_ascii=False)}\n\n"
                "Return {\"answers\":[{\"test_id\":\"Q0\",\"outcome_id\":\"R0\"}]}. "
                "An optional check string may explain a local derivation, but it is not scored. Use only listed IDs. "
                "Include one row for every test you can determine. Do not infer or produce the original final answer."
            ),
        },
    ]


def build_direct_judge_messages(
    sample: DatasetSample,
    *,
    stage: StageDecision,
    hypothesis_to_key: dict[str, str],
) -> list[dict[str, str]]:
    candidates = {candidate.key: candidate for candidate in stage.candidates}
    hypotheses = [
        {
            "id": hypothesis,
            "answer": candidates[key].answer,
            "reasoning": candidates[key].representative_reasoning,
        }
        for hypothesis, key in hypothesis_to_key.items()
    ]
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. Select the best existing anonymous hypothesis by correctness. "
                "Do not invent a new answer and do not infer support counts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\n"
                f"Anonymous hypotheses:\n{json.dumps(hypotheses, ensure_ascii=False)}\n\n"
                "Return exactly {\"selected_id\":\"H0\",\"check\":\"brief correctness check\"}. "
                "selected_id must be one listed hypothesis ID."
            ),
        },
    ]


def build_vote_aware_designer_messages(
    sample: DatasetSample,
    *,
    stage: StageDecision,
    hypothesis_to_key: dict[str, str],
) -> list[dict[str, str]]:
    """Dev-only leakage ablation; never used by the registered primary runner."""

    messages = build_designer_messages(sample, stage=stage, hypothesis_to_key=hypothesis_to_key)
    counts = {hypothesis: stage.vote_counts[key] for hypothesis, key in hypothesis_to_key.items()}
    messages[-1]["content"] += f"\n\nAblation-only support counts: {json.dumps(counts, sort_keys=True)}"
    return messages


def build_unblinded_witness_messages(
    sample: DatasetSample,
    *,
    packet: WitnessPacket,
    stage: StageDecision,
    hypothesis_to_key: dict[str, str],
) -> list[dict[str, str]]:
    """Dev-only blindness ablation; never used by the registered primary runner."""

    messages = build_witness_messages(sample, packet=packet)
    candidates = {candidate.key: candidate for candidate in stage.candidates}
    disclosed = [
        {"id": hypothesis, "answer": candidates[key].answer}
        for hypothesis, key in hypothesis_to_key.items()
    ]
    messages[-1]["content"] += (
        "\n\nAblation-only unblinded candidate mapping: "
        + json.dumps(disclosed, ensure_ascii=False)
    )
    return messages
