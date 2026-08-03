"""共享答案契约与强化 JSON 主线解析辅助。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runner_common import (
    CachedRequestResult,
    TurnRequestExecutor,
    execute_cached_request,
)
from research_experiments.core.structured_outputs import SCHEMA_ANSWER_ANCHOR_V2, validate_structured_output

JSON_ANSWER_ANCHOR_V2_CONTRACT = "json_answer_anchor_v2"
MULTI_AGENT_CONSISTENT_JSON_V2_PROMPT = "multi_agent_consistent_json_v2"

AnswerContract = Literal["json_answer_anchor_v2"]
AnswerContractStatus = Literal["ok", "failed", "not_attempted", "not_applicable"]
AnswerContractSource = Literal["full_json"] | None
JsonParseMode = Literal["strict_json"] | None


@dataclass(frozen=True)
class ParsedAnswerContract:
    status: AnswerContractStatus
    validated_output: dict[str, Any]
    source: AnswerContractSource
    parse_mode: JsonParseMode
    error: str | None
    answer_field_consistent: bool
    reasoning_present: bool


@dataclass(frozen=True)
class AnswerContractTurnResult:
    payload: dict[str, Any]
    prompt_hash: str
    cache_key: str
    cache_hit: bool
    response_payload: dict[str, Any]
    request_error: str | None
    request_status: str
    output_status: str
    validated_output: dict[str, Any]
    usage: dict[str, Any]
    answer_contract_status: AnswerContractStatus
    answer_contract_source: AnswerContractSource
    answer_contract_error: str | None
    answer_field_consistent: bool
    reasoning_present: bool
    json_parse_mode: JsonParseMode
    request_count: int
    cache_request_count: int
    network_request_count: int
    raw_finish_reason: str | None


def answer_contract_for_prompt_version(prompt_version: str) -> AnswerContract:
    if prompt_version != MULTI_AGENT_CONSISTENT_JSON_V2_PROMPT:
        raise ValueError(f"Unsupported vanilla MAD prompt_version: {prompt_version}")
    return JSON_ANSWER_ANCHOR_V2_CONTRACT


def validate_answer_contract(answer_contract: str) -> AnswerContract:
    normalized = str(answer_contract or "").strip()
    if normalized != JSON_ANSWER_ANCHOR_V2_CONTRACT:
        raise ValueError(
            f"Unsupported answer_contract {answer_contract!r}. "
            f"Expected {JSON_ANSWER_ANCHOR_V2_CONTRACT!r}."
        )
    return JSON_ANSWER_ANCHOR_V2_CONTRACT


def validate_prompt_answer_contract(prompt_version: str, answer_contract: str) -> AnswerContract:
    expected = answer_contract_for_prompt_version(prompt_version)
    normalized = validate_answer_contract(answer_contract)
    if normalized != expected:
        raise ValueError(
            f"prompt_version={prompt_version!r} requires answer_contract={expected!r}, got {normalized!r}."
        )
    return normalized


def execute_answer_contract_turn(
    *,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle | None,
    sample: DatasetSample | None,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int | None,
    dataset: str,
    answer_contract: AnswerContract,
    use_response_format: bool,
    request_executor: TurnRequestExecutor | None = None,
) -> AnswerContractTurnResult:
    del sample
    if answer_contract != JSON_ANSWER_ANCHOR_V2_CONTRACT:
        raise ValueError(f"Unsupported answer contract: {answer_contract}")

    request = execute_cached_request(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_response_format=use_response_format,
        request_executor=request_executor,
        response_validator=lambda response: _admit_answer_contract_response(
            response,
            dataset=dataset,
        ),
    )
    return _finalize_turn_result(request, dataset=dataset)


def refresh_answer_contract_turn(
    *,
    row: dict[str, Any],
    sample: DatasetSample | None,
    backbone,
    provider: OpenAICompatibleProvider | None,
    cache: RequestCache | None,
    throttle: RequestThrottle | None,
    answer_contract: AnswerContract,
) -> AnswerContractTurnResult:
    del sample, backbone, provider, cache, throttle
    if answer_contract != JSON_ANSWER_ANCHOR_V2_CONTRACT:
        raise ValueError(f"Unsupported answer contract: {answer_contract}")

    response_payload = {
        "assistant_text": str(row.get("assistant_text") or ""),
        "provider_reasoning_text": str(row.get("provider_reasoning_text") or ""),
        "finish_reason": row.get("raw_finish_reason"),
        "latency_ms": float(row.get("raw_latency_ms") or row.get("latency_ms") or 0.0),
        "request_started_at": row.get("request_started_at"),
    }
    request = CachedRequestResult(
        payload=dict(row.get("payload") or {}),
        prompt_hash=str(row.get("prompt_hash") or ""),
        cache_key=str(row.get("cache_key") or ""),
        cache_hit=bool(row.get("cache_hit")),
        response_payload=response_payload,
        request_error=str(row.get("request_error") or "") or None,
        usage={
            "prompt_tokens": float(row.get("raw_prompt_tokens") or row.get("prompt_tokens") or 0.0),
            "completion_tokens": float(row.get("raw_completion_tokens") or row.get("completion_tokens") or 0.0),
            "total_tokens": float(row.get("raw_total_tokens") or row.get("total_tokens") or 0.0),
        },
    )
    refreshed = _finalize_turn_result(request, dataset=str(row.get("dataset") or ""))
    return AnswerContractTurnResult(
        payload=request.payload,
        prompt_hash=refreshed.prompt_hash,
        cache_key=refreshed.cache_key,
        cache_hit=refreshed.cache_hit,
        response_payload=refreshed.response_payload,
        request_error=refreshed.request_error,
        request_status=refreshed.request_status,
        output_status=refreshed.output_status,
        validated_output=refreshed.validated_output,
        usage=refreshed.usage,
        answer_contract_status=refreshed.answer_contract_status,
        answer_contract_source=refreshed.answer_contract_source,
        answer_contract_error=refreshed.answer_contract_error,
        answer_field_consistent=refreshed.answer_field_consistent,
        reasoning_present=refreshed.reasoning_present,
        json_parse_mode=refreshed.json_parse_mode,
        request_count=max(1, int(row.get("request_count") or 1)),
        cache_request_count=max(0, int(row.get("cache_request_count") or (1 if row.get("cache_hit") else 0))),
        network_request_count=max(
            0,
            int(
                row.get("network_request_count")
                or max(1, int(row.get("request_count") or 1))
                - max(0, int(row.get("cache_request_count") or (1 if row.get("cache_hit") else 0)))
            ),
        ),
        raw_finish_reason=refreshed.raw_finish_reason,
    )


def _finalize_turn_result(
    request: CachedRequestResult,
    *,
    dataset: str,
) -> AnswerContractTurnResult:
    request_status = "request_fail" if request.request_error else "ok"
    if request.request_error:
        return AnswerContractTurnResult(
            payload=request.payload,
            prompt_hash=request.prompt_hash,
            cache_key=request.cache_key,
            cache_hit=request.cache_hit,
            response_payload=request.response_payload,
            request_error=request.request_error,
            request_status=request_status,
            output_status="request_fail",
            validated_output={},
            usage=dict(request.usage),
            answer_contract_status="not_attempted",
            answer_contract_source=None,
            answer_contract_error=None,
            answer_field_consistent=False,
            reasoning_present=False,
            json_parse_mode=None,
            request_count=1,
            cache_request_count=1 if request.cache_hit else 0,
            network_request_count=0 if request.cache_hit else 1,
            raw_finish_reason=_raw_finish_reason(request.response_payload),
        )

    parsed = _parse_answer_contract_response(
        str(request.response_payload.get("assistant_text") or ""),
        dataset=dataset,
    )
    output_status = "ok" if parsed.status == "ok" else "answer_contract_fail"
    return AnswerContractTurnResult(
        payload=request.payload,
        prompt_hash=request.prompt_hash,
        cache_key=request.cache_key,
        cache_hit=request.cache_hit,
        response_payload=request.response_payload,
        request_error=request.request_error,
        request_status=request_status,
        output_status=output_status,
        validated_output=parsed.validated_output,
        usage=dict(request.usage),
        answer_contract_status=parsed.status,
        answer_contract_source=parsed.source,
        answer_contract_error=parsed.error if parsed.status != "ok" else None,
        answer_field_consistent=parsed.answer_field_consistent,
        reasoning_present=parsed.reasoning_present,
        json_parse_mode=parsed.parse_mode,
        request_count=1,
        cache_request_count=1 if request.cache_hit else 0,
        network_request_count=0 if request.cache_hit else 1,
        raw_finish_reason=_raw_finish_reason(request.response_payload),
    )


def _parse_answer_contract_response(raw_text: str, *, dataset: str) -> ParsedAnswerContract:
    cleaned = str(raw_text or "").strip()
    if not cleaned:
        return ParsedAnswerContract(
            status="failed",
            validated_output={},
            source=None,
            parse_mode=None,
            error="Assistant output is empty.",
            answer_field_consistent=False,
            reasoning_present=False,
        )
    try:
        validated = validate_structured_output(cleaned, SCHEMA_ANSWER_ANCHOR_V2, dataset=dataset)
        return ParsedAnswerContract(
            status="ok",
            validated_output=validated,
            source="full_json",
            parse_mode="strict_json",
            error=None,
            answer_field_consistent=True,
            reasoning_present=bool(str(validated.get("reasoning") or "").strip()),
        )
    except Exception as exc:
        return ParsedAnswerContract(
            status="failed",
            validated_output={},
            source=None,
            parse_mode="strict_json",
            error=str(exc),
            answer_field_consistent=False,
            reasoning_present=False,
        )


def _admit_answer_contract_response(response_payload: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    parsed = _parse_answer_contract_response(
        str(response_payload.get("assistant_text") or ""),
        dataset=dataset,
    )
    if parsed.status != "ok":
        raise ValueError(parsed.error or "Answer contract validation failed.")
    return parsed.validated_output


def _raw_finish_reason(response_payload: dict[str, Any]) -> str | None:
    value = response_payload.get("finish_reason")
    if value is None:
        return None
    return str(value)
