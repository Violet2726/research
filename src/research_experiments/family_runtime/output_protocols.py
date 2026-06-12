"""主线实验共用的标签化自由文本输出协议。"""

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
from research_experiments.family_runtime.free_text_protocol import (
    FREE_TEXT_ANSWER_PROTOCOL_V1,
    FREE_TEXT_DEBATE_UPDATE_PROTOCOL_V1,
    parse_free_text_answer_output,
)

OutputProtocol = Literal["free_text_answer_v1", "free_text_debate_update_v1"]
ProtocolParseStatus = Literal["ok", "failed", "not_attempted"]


@dataclass(frozen=True)
class ParsedOutputProtocol:
    status: ProtocolParseStatus
    validated_output: dict[str, Any]
    output_protocol: OutputProtocol
    error: str | None
    reason_present: bool
    decision: str | None
    changed_answer: bool


@dataclass(frozen=True)
class OutputProtocolTurnResult:
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
    output_protocol: OutputProtocol
    protocol_parse_status: ProtocolParseStatus
    protocol_parse_error: str | None
    reason_present: bool
    decision: str | None
    changed_answer: bool
    request_count: int
    cache_request_count: int
    network_request_count: int
    raw_finish_reason: str | None


def execute_output_protocol_turn(
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
    role: str,
    output_protocol: OutputProtocol,
    request_executor: TurnRequestExecutor | None = None,
) -> OutputProtocolTurnResult:
    request = execute_cached_request(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_response_format=False,
        request_executor=request_executor,
    )
    return _finalize_turn_result(request, dataset=dataset, output_protocol=output_protocol)


def refresh_output_protocol_turn(
    *,
    row: dict[str, Any],
    sample: DatasetSample | None,
    backbone,
    provider: OpenAICompatibleProvider | None,
    cache: RequestCache | None,
    throttle: RequestThrottle | None,
    output_protocol: OutputProtocol,
) -> OutputProtocolTurnResult:
    del sample, backbone, provider, cache, throttle
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
    refreshed = _finalize_turn_result(request, dataset=str(row.get("dataset") or ""), output_protocol=output_protocol)
    return OutputProtocolTurnResult(
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
        output_protocol=refreshed.output_protocol,
        protocol_parse_status=refreshed.protocol_parse_status,
        protocol_parse_error=refreshed.protocol_parse_error,
        reason_present=refreshed.reason_present,
        decision=refreshed.decision,
        changed_answer=refreshed.changed_answer,
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


def validate_output_protocol(value: str) -> OutputProtocol:
    normalized = str(value or "").strip()
    if normalized not in {FREE_TEXT_ANSWER_PROTOCOL_V1, FREE_TEXT_DEBATE_UPDATE_PROTOCOL_V1}:
        raise ValueError(
            f"Unsupported output_protocol {value!r}. "
            f"Expected one of {FREE_TEXT_ANSWER_PROTOCOL_V1!r}, {FREE_TEXT_DEBATE_UPDATE_PROTOCOL_V1!r}."
        )
    return normalized


def build_shared_output_protocol_diagnostics(
    turn_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
) -> dict[str, Any]:
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in turn_rows:
        grouped.setdefault((str(row.get("dataset") or ""), str(row.get("method_name") or "")), []).append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, method_name), items in grouped.items():
        rows.append(_output_protocol_diagnostic_row(dataset, method_name, items))

    overall_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in turn_rows:
        overall_grouped.setdefault(str(row.get("method_name") or ""), []).append(row)
    for method_name, items in overall_grouped.items():
        rows.append(_output_protocol_diagnostic_row("overall", method_name, items))

    def _sort_key(row: dict[str, Any]) -> tuple[int, int]:
        dataset_idx = dataset_rank.get(str(row["dataset"]), len(dataset_order))
        if row["dataset"] == "overall":
            dataset_idx = len(dataset_order) + 1
        return dataset_idx, method_rank.get(str(row["method_name"]), 999)

    rows.sort(key=_sort_key)
    return {"rows": rows}


def _output_protocol_diagnostic_row(
    dataset: str,
    method_name: str,
    turn_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    turn_count = len(turn_rows)
    protocol_failures = sum(1 for row in turn_rows if row.get("protocol_parse_status") == "failed")
    reason_missing = sum(1 for row in turn_rows if not row.get("reason_present"))
    return {
        "dataset": dataset,
        "method_name": method_name,
        "turn_count": turn_count,
        "request_failure_count": sum(1 for row in turn_rows if row.get("request_status") == "request_fail"),
        "protocol_failure_count": protocol_failures,
        "protocol_failure_rate": _ratio_count(protocol_failures, turn_count),
        "reason_missing_count": reason_missing,
        "reason_missing_rate": _ratio_count(reason_missing, turn_count),
    }


def _finalize_turn_result(
    request: CachedRequestResult,
    *,
    dataset: str,
    output_protocol: OutputProtocol,
) -> OutputProtocolTurnResult:
    request_status = "request_fail" if request.request_error else "ok"
    if request.request_error:
        return OutputProtocolTurnResult(
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
            output_protocol=output_protocol,
            protocol_parse_status="not_attempted",
            protocol_parse_error=None,
            reason_present=False,
            decision=None,
            changed_answer=False,
            request_count=1,
            cache_request_count=1 if request.cache_hit else 0,
            network_request_count=0 if request.cache_hit else 1,
            raw_finish_reason=_raw_finish_reason(request.response_payload),
        )

    parsed = _parse_output_protocol_response(
        str(request.response_payload.get("assistant_text") or ""),
        dataset=dataset,
        output_protocol=output_protocol,
    )
    output_status = "ok" if parsed.status == "ok" else "protocol_fail"
    return OutputProtocolTurnResult(
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
        output_protocol=parsed.output_protocol,
        protocol_parse_status=parsed.status,
        protocol_parse_error=parsed.error if parsed.status != "ok" else None,
        reason_present=parsed.reason_present,
        decision=parsed.decision,
        changed_answer=parsed.changed_answer,
        request_count=1,
        cache_request_count=1 if request.cache_hit else 0,
        network_request_count=0 if request.cache_hit else 1,
        raw_finish_reason=_raw_finish_reason(request.response_payload),
    )


def _parse_output_protocol_response(
    raw_text: str,
    *,
    dataset: str,
    output_protocol: OutputProtocol,
) -> ParsedOutputProtocol:
    cleaned = str(raw_text or "").strip()
    if not cleaned:
        return ParsedOutputProtocol(
            status="failed",
            validated_output={},
            output_protocol=output_protocol,
            error="Assistant output is empty.",
            reason_present=False,
            decision=None,
            changed_answer=False,
        )

    try:
        validated = parse_free_text_answer_output(
            cleaned,
            dataset=dataset,
            require_decision=output_protocol == FREE_TEXT_DEBATE_UPDATE_PROTOCOL_V1,
        )
        return ParsedOutputProtocol(
            status="ok",
            validated_output=validated,
            output_protocol=output_protocol,
            error=None,
            reason_present=bool(str(validated.get("reasoning") or "").strip()),
            decision=str(validated.get("decision") or "") or None,
            changed_answer=bool(validated.get("changed_answer")),
        )
    except Exception as exc:
        return ParsedOutputProtocol(
            status="failed",
            validated_output={},
            output_protocol=output_protocol,
            error=str(exc),
            reason_present=False,
            decision=None,
            changed_answer=False,
        )


def _raw_finish_reason(response_payload: dict[str, Any]) -> str | None:
    value = response_payload.get("finish_reason")
    if value is None:
        return None
    return str(value)


def _ratio_count(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
