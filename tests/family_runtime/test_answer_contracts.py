from __future__ import annotations

from pathlib import Path

from research_experiments.core.config import resolve_model_ref
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.structured_outputs import SCHEMA_ANSWER_CORE
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.family_runtime.answer_contracts import (
    JSON_ANSWER_CORE_CONTRACT,
    PAPER_TRANSCRIPT_HARDENED_CONTRACT,
    execute_answer_contract_turn,
    extract_paper_transcript_answer,
    refresh_answer_contract_turn,
    validate_prompt_answer_contract,
)


def _sample(*, dataset: str, question: str, answer: str) -> DatasetSample:
    return DatasetSample(
        dataset=dataset,
        sample_id=f"{dataset}-1",
        question=question,
        reference_answer=answer,
        prompt_context="",
        metadata={},
    )


def test_validate_prompt_answer_contract_requires_explicit_match() -> None:
    assert (
        validate_prompt_answer_contract("multi_agent_paper_text", PAPER_TRANSCRIPT_HARDENED_CONTRACT)
        == PAPER_TRANSCRIPT_HARDENED_CONTRACT
    )
    assert (
        validate_prompt_answer_contract("multi_agent_controlled_json", JSON_ANSWER_CORE_CONTRACT)
        == JSON_ANSWER_CORE_CONTRACT
    )


def test_extract_paper_transcript_answer_recovers_nested_boxed_fraction() -> None:
    payload = extract_paper_transcript_answer(
        "After simplification, the final answer is \\boxed{\\frac{1}{16}}.",
        dataset="competition_math",
    )
    assert payload.status == "ok"
    assert payload.source == "explicit_boxed_answer"
    assert payload.validated_output["final_answer"] == "1/16"


def test_extract_paper_transcript_answer_does_not_guess_last_number_for_math() -> None:
    payload = extract_paper_transcript_answer(
        "The denominator is 1152 and 39916800 / 1152 = 34650, so we continue from there",
        dataset="competition_math",
    )
    assert payload.status == "failed"


def test_extract_paper_transcript_answer_does_not_take_middle_entity_for_hotpot() -> None:
    payload = extract_paper_transcript_answer(
        "Tommy's Honour stars Jack Lowden, who later found success in a BBC miniseries.",
        dataset="hotpotqa",
    )
    assert payload.status == "failed"


def test_execute_answer_contract_turn_repairs_incomplete_paper_transcript(tmp_path: Path) -> None:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    cache = RequestCache(tmp_path / "requests.sqlite")
    sample = _sample(
        dataset="gsm8k",
        question="Ben flips a fair nickel four times. What is the probability of HTHT in order?",
        answer="1/16",
    )

    def request_executor(payload, provider, throttle):
        del payload, provider, throttle
        return {
            "assistant_text": "The probability is (1/2)^4. Final answer is still being computed...",
            "provider_reasoning_text": "",
            "finish_reason": None,
            "usage_reported": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "usage_estimated": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "usage_source": "reported",
            "latency_ms": 11.0,
            "provider_request_id": "raw_req",
            "response_id": "raw_resp",
            "request_started_at": "2026-06-11T00:00:00+00:00",
            "request_error": None,
        }

    def repair_request_executor(payload, provider, throttle):
        del payload, provider, throttle
        return {
            "assistant_text": '{"final_answer":"1/16","reasoning":"repair"}',
            "provider_reasoning_text": "",
            "finish_reason": "stop",
            "usage_reported": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
            "usage_estimated": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
            "usage_source": "reported",
            "latency_ms": 7.0,
            "provider_request_id": "repair_req",
            "response_id": "repair_resp",
            "request_started_at": "2026-06-11T00:00:01+00:00",
            "request_error": None,
        }

    result = execute_answer_contract_turn(
        backbone=model,
        provider=object(),
        cache=cache,
        throttle=None,
        sample=sample,
        messages=[{"role": "user", "content": "demo"}],
        temperature=0.7,
        top_p=1.0,
        seed=42,
        dataset="gsm8k",
        answer_contract=PAPER_TRANSCRIPT_HARDENED_CONTRACT,
        use_response_format=False,
        allow_network_repair=True,
        request_executor=request_executor,
        repair_request_executor=repair_request_executor,
    )
    cache.close()

    assert result.output_status == "ok"
    assert result.answer_extraction_source == "repair_json_answer_core"
    assert result.repair_call_used is True
    assert result.request_count == 2
    assert result.validated_output["final_answer"] == "1/16"


def test_refresh_answer_contract_turn_reextracts_from_saved_transcript() -> None:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    sample = _sample(
        dataset="competition_math",
        question="Determine the probability.",
        answer="1/16",
    )
    row = {
        "dataset": "competition_math",
        "sample_id": sample.sample_id,
        "assistant_text": "Therefore the final answer is \\boxed{\\frac{1}{16}}.",
        "provider_reasoning_text": "",
        "payload": {"seed": 42},
        "prompt_hash": "hash",
        "request_error": None,
        "cache_hit": False,
        "request_started_at": "2026-06-11T00:00:00+00:00",
        "prompt_tokens": 10.0,
        "completion_tokens": 20.0,
        "total_tokens": 30.0,
        "latency_ms": 9.0,
        "output_status": "schema_fail",
    }

    refreshed = refresh_answer_contract_turn(
        row=row,
        sample=sample,
        backbone=model,
        provider=None,
        cache=None,
        throttle=None,
        answer_contract=PAPER_TRANSCRIPT_HARDENED_CONTRACT,
        allow_network_repair=False,
    )

    assert refreshed.output_status == "ok"
    assert refreshed.validated_output["final_answer"] == "1/16"
