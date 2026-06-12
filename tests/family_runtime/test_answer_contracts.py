from __future__ import annotations

from pathlib import Path

from research_experiments.core.config import resolve_model_ref
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.execution.cache import RequestCache
from research_experiments.family_runtime.answer_contracts import (
    JSON_ANSWER_ANCHOR_V2_CONTRACT,
    MULTI_AGENT_CONSISTENT_JSON_V2_PROMPT,
    execute_answer_contract_turn,
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
        validate_prompt_answer_contract(MULTI_AGENT_CONSISTENT_JSON_V2_PROMPT, JSON_ANSWER_ANCHOR_V2_CONTRACT)
        == JSON_ANSWER_ANCHOR_V2_CONTRACT
    )


def test_execute_answer_contract_turn_accepts_strict_json(tmp_path: Path) -> None:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    cache = RequestCache(tmp_path / "strict.sqlite")

    def request_executor(payload, provider, throttle):
        del payload, provider, throttle
        return {
            "assistant_text": '{"reasoning":"The spectrum best matches option B.","final_answer":"B"}',
            "provider_reasoning_text": "",
            "finish_reason": "stop",
            "usage_reported": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            "usage_estimated": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            "usage_source": "reported",
            "latency_ms": 11.0,
            "provider_request_id": "req",
            "response_id": "resp",
            "request_started_at": "2026-06-12T00:00:00+00:00",
            "request_error": None,
        }

    result = execute_answer_contract_turn(
        backbone=model,
        provider=object(),
        cache=cache,
        throttle=None,
        sample=_sample(dataset="gpqa_diamond", question="Choose.", answer="B"),
        messages=[{"role": "user", "content": "demo"}],
        temperature=0.7,
        top_p=1.0,
        seed=42,
        dataset="gpqa_diamond",
        answer_contract=JSON_ANSWER_ANCHOR_V2_CONTRACT,
        use_response_format=True,
        request_executor=request_executor,
    )
    cache.close()

    assert result.output_status == "ok"
    assert result.answer_contract_source == "full_json"
    assert result.json_parse_mode == "strict_json"
    assert result.validated_output["final_answer"] == "B"


def test_execute_answer_contract_turn_fails_for_truncated_json_without_recovery(tmp_path: Path) -> None:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    cache = RequestCache(tmp_path / "fallback.sqlite")

    def request_executor(payload, provider, throttle):
        del payload, provider, throttle
        return {
            "assistant_text": (
                '{"reasoning":"Total needed is 400. '
                'The fifth worker makes 18 toys per hour'
            ),
            "provider_reasoning_text": "",
            "finish_reason": "length",
            "usage_reported": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "usage_estimated": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "usage_source": "reported",
            "latency_ms": 11.0,
            "provider_request_id": "raw_req",
            "response_id": "raw_resp",
            "request_started_at": "2026-06-12T00:00:00+00:00",
            "request_error": None,
        }

    result = execute_answer_contract_turn(
        backbone=model,
        provider=object(),
        cache=cache,
        throttle=None,
        sample=_sample(dataset="gsm8k", question="How many?", answer="18"),
        messages=[{"role": "user", "content": "demo"}],
        temperature=0.7,
        top_p=1.0,
        seed=42,
        dataset="gsm8k",
        answer_contract=JSON_ANSWER_ANCHOR_V2_CONTRACT,
        use_response_format=True,
        request_executor=request_executor,
    )
    cache.close()

    assert result.output_status == "answer_contract_fail"
    assert result.answer_contract_source is None
    assert result.validated_output == {}


def test_execute_answer_contract_turn_fails_when_answer_fields_disagree(tmp_path: Path) -> None:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    cache = RequestCache(tmp_path / "mismatch.sqlite")

    def request_executor(payload, provider, throttle):
        del payload, provider, throttle
        return {
            "assistant_text": '{"reasoning":"Correct area is 49π.","final_answer":"98\\\\pi","unexpected":"49\\\\pi"}',
            "provider_reasoning_text": "",
            "finish_reason": "stop",
            "usage_reported": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            "usage_estimated": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            "usage_source": "reported",
            "latency_ms": 11.0,
            "provider_request_id": "req",
            "response_id": "resp",
            "request_started_at": "2026-06-12T00:00:00+00:00",
            "request_error": None,
        }

    result = execute_answer_contract_turn(
        backbone=model,
        provider=object(),
        cache=cache,
        throttle=None,
        sample=_sample(dataset="competition_math", question="Area?", answer="49\\pi"),
        messages=[{"role": "user", "content": "demo"}],
        temperature=0.7,
        top_p=1.0,
        seed=42,
        dataset="competition_math",
        answer_contract=JSON_ANSWER_ANCHOR_V2_CONTRACT,
        use_response_format=True,
        request_executor=request_executor,
    )
    cache.close()

    assert result.output_status == "answer_contract_fail"


def test_refresh_answer_contract_turn_reparses_saved_row() -> None:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    sample = _sample(dataset="mmlu_pro", question="Pick.", answer="A")
    row = {
        "dataset": "mmlu_pro",
        "sample_id": sample.sample_id,
        "assistant_text": '{"reasoning":"Option A best matches the evidence.","final_answer":"A"}',
        "provider_reasoning_text": "",
        "payload": {"seed": 42},
        "prompt_hash": "hash",
        "request_error": None,
        "cache_hit": False,
        "request_started_at": "2026-06-12T00:00:00+00:00",
        "prompt_tokens": 10.0,
        "completion_tokens": 20.0,
        "total_tokens": 30.0,
        "latency_ms": 9.0,
        "output_status": "answer_contract_fail",
    }

    refreshed = refresh_answer_contract_turn(
        row=row,
        sample=sample,
        backbone=model,
        provider=None,
        cache=None,
        throttle=None,
        answer_contract=JSON_ANSWER_ANCHOR_V2_CONTRACT,
    )

    assert refreshed.output_status == "ok"
    assert refreshed.validated_output["final_answer"] == "A"
