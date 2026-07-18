"""CATCH-v3 索引 selector、双盲 witness 与匹配 judge 提示。"""

from __future__ import annotations

import json

from research_experiments.core.data.datasets import (
    DatasetSample,
    question_without_answer_contract,
    question_without_bbeh_options,
)
from research_experiments.families.contrastive_active_testing.algorithms import StageDecision, WitnessPacket
from research_experiments.families.contrastive_active_testing.icv import (
    EvidenceUnit,
    IcvWitnessPacket,
    TargetPair,
    selector_public_payload,
)

CATCH_PROMPT_VERSION = "catch_v3_indexed_contrast_verification"
CATCH_SCHEMA_VERSION = "catch_icv_v3"


def build_icv_selector_messages(
    sample: DatasetSample,
    *,
    pairs: tuple[TargetPair, ...],
    evidence: dict[str, tuple[EvidenceUnit, ...]],
) -> list[dict[str, str]]:
    rendered_pairs = selector_public_payload(pairs, evidence)
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. Select existing indexed statements for blinded contrast checks. "
                "Never write, paraphrase, quote, explain, or invent any statement."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source task with its final answer region removed:\n{question_without_answer_contract(sample)}\n\n"
                f"Independent anonymous pairs with indexed reasoning statements:\n"
                f"{json.dumps(rendered_pairs, ensure_ascii=False)}\n\n"
                "For each pair, select exactly three distinct upstream contrasts when possible. Each contrast must "
                "place mutually incompatible claims on its two sides, be decidable from the source task, and avoid "
                "mere conclusions, answer choices, style, confidence, or verbosity. A side may cite one to three "
                "consecutive IDs from that side of the same pair. IDs may not be reused. Return no text fields. "
                "Return exactly {\"contrasts\":[{\"pair_id\":\"P0\",\"contrast_id\":\"C0\","
                "\"left_unit_ids\":[\"L:E0\"],\"right_unit_ids\":[\"R:E2\"]}]} or an empty contrasts list."
            ),
        },
    ]


def build_icv_witness_messages(
    sample: DatasetSample,
    *,
    packet: IcvWitnessPacket,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. You are a blinded local verification witness, not a final-answer "
                "judge. Compare only the two anonymous statements against the source task."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source task with final answer region removed:\n{question_without_answer_contract(sample)}\n\n"
                f"Anonymous local contrasts:\n{json.dumps(list(packet.contrasts), ensure_ascii=False)}\n\n"
                "For every contrast return LEFT_ONLY if only the left statement is supported, RIGHT_ONLY if only "
                "the right statement is supported, BOTH if both are supported, or NEITHER if neither is supported. "
                "Do not infer candidates, votes, or the original final answer. Return exactly "
                "{\"answers\":[{\"contrast_id\":\"X0\",\"verdict\":\"LEFT_ONLY\"}]} using only listed IDs."
            ),
        },
    ]


def build_pair_judge_messages(
    sample: DatasetSample,
    *,
    stage: StageDecision,
    public_to_key: dict[str, str],
) -> list[dict[str, str]]:
    candidates = {candidate.key: candidate for candidate in stage.candidates}
    hypotheses = [
        {
            "id": public_id,
            "answer": candidates[key].answer,
            "reasoning": candidates[key].representative_reasoning,
        }
        for public_id, key in public_to_key.items()
    ]
    return [
        {
            "role": "system",
            "content": (
                "Return one JSON object only. Select the best existing anonymous target candidate by correctness. "
                "Do not invent an answer and do not infer vote counts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{sample.question}\n\n"
                f"Anonymous target candidates:\n{json.dumps(hypotheses, ensure_ascii=False)}\n\n"
                "Return exactly {\"selected_id\":\"J0\"}. selected_id must be listed."
            ),
        },
    ]


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
