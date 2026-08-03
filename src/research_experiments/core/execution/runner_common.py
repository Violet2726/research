from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypeVar

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.core.execution.cache import (
    CACHE_KEY_POLICY_VERSION,
    RequestCache,
    build_request_cache_key,
    cache_successful_response,
)
from research_experiments.core.execution.providers import (
    OpenAICompatibleProvider,
    build_payload,
    execute_completion_request,
)
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.structured_outputs import SchemaId, validate_or_recover_structured_output

T = TypeVar("T")
R = TypeVar("R")

TurnValidator = Callable[[str, str], dict[str, Any]]
TurnRequestExecutor = Callable[
    [dict[str, Any], OpenAICompatibleProvider, RequestThrottle | None],
    dict[str, Any],
]
TurnResponseHook = Callable[[dict[str, Any], dict[str, Any]], None]
ResponseAdmission = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class CachedTurnResult:
    """Normalized result for one validated cached-or-live model turn."""

    payload: dict[str, Any]
    prompt_hash: str
    cache_key: str
    cache_hit: bool
    response_payload: dict[str, Any]
    request_error: str | None
    validated_output: dict[str, Any]
    output_status: str
    usage: dict[str, Any]


@dataclass(frozen=True)
class CachedRequestResult:
    """Normalized result for one cached-or-live raw provider request."""

    payload: dict[str, Any]
    prompt_hash: str
    cache_key: str
    cache_hit: bool
    response_payload: dict[str, Any]
    request_error: str | None
    usage: dict[str, Any]


def prepare_run_root(
    run_root: str | Path,
    experiment_name: str,
    phase_name: str,
    run_id: str,
) -> Path:
    """Create and return the normalized run root directory."""

    root = Path(run_root) / experiment_name / phase_name / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def prompt_hash(messages: list[dict[str, Any]]) -> str:
    """Build a stable hash for one message list."""

    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def iter_indexed_batch[T, R](
    items: Iterable[T],
    *,
    worker: Callable[[T], R],
    max_concurrent_requests: int,
) -> Iterable[tuple[int, R]]:
    """Yield indexed worker results in completion order."""

    indexed_items = list(enumerate(items))
    max_workers = max(1, min(max_concurrent_requests, len(indexed_items) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, item): index
            for index, item in indexed_items
        }
        for future in as_completed(future_to_index):
            yield (future_to_index[future], future.result())


def execute_cached_request(
    *,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle | None,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int | None,
    use_response_format: bool = True,
    max_tokens: int | None = None,
    request_executor: TurnRequestExecutor | None = None,
    response_hook: TurnResponseHook | None = None,
    response_validator: ResponseAdmission | None = None,
    cache_mode: Literal["shared", "live"] = "shared",
) -> CachedRequestResult:
    """Execute one request, caching only after an explicit validator passes."""

    payload = build_payload(
        config=backbone,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_response_format=use_response_format,
        max_tokens=max_tokens,
    )
    hashed_prompt = prompt_hash(messages)
    cache_key = build_request_cache_key(
        provider=backbone.provider,
        request_model=backbone.model_id,
        payload=payload,
    )
    cache_lookup_started_at = datetime.now(UTC).isoformat()
    cache_lookup_started = time.monotonic()
    cached = None if cache_mode == "live" else cache.get(cache_key)
    if cached is not None and not _cached_response_fits_cap(cached.completion_tokens, payload):
        cached = None
    cache_lookup_ms = max(0.0, (time.monotonic() - cache_lookup_started) * 1000)
    cache_lookup_finished_at = datetime.now(UTC).isoformat()
    if cached is None:
        response_payload = _execute_request_with_retries(
            payload=payload,
            provider=provider,
            throttle=throttle,
            request_executor=request_executor,
        )
        response_payload = dict(response_payload)
        if response_hook is not None:
            response_hook(payload, response_payload)
        cache_hit = False
        if response_payload.get("request_error") is None and cache_mode != "live" and response_validator is not None:
            try:
                validated_output = response_validator(response_payload)
                cache_successful_response(
                    cache,
                    cache_key=cache_key,
                    payload=payload,
                    response_payload=response_payload,
                    validated_output=validated_output,
                )
            except Exception:
                response_payload = dict(response_payload)
                response_payload["cache_admission_error"] = True
    else:
        response_payload = dict(json.loads(cached.response_json))
        if cached.completion_tokens is not None:
            response_payload["usage_reported"] = {
                "completion_tokens": cached.completion_tokens,
            }
        response_payload["cache_origin_completion_cap"] = None
        response_payload["cache_origin_key_policy"] = CACHE_KEY_POLICY_VERSION
        cache_hit = True

    response_payload = dict(response_payload)
    if not cache_hit:
        response_payload["cache_origin_completion_cap"] = None
        response_payload["cache_origin_key_policy"] = CACHE_KEY_POLICY_VERSION
    response_payload["cache_lookup_timeline"] = {
        "started_at": cache_lookup_started_at,
        "finished_at": cache_lookup_finished_at,
        "duration_ms": cache_lookup_ms,
        "hit": cache_hit,
    }

    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    request_error = response_payload.get("request_error")
    return CachedRequestResult(
        payload=payload,
        prompt_hash=hashed_prompt,
        cache_key=cache_key,
        cache_hit=cache_hit,
        response_payload=response_payload,
        request_error=str(request_error) if request_error else None,
        usage=usage,
    )


def _completion_cap(payload: dict[str, Any]) -> int | None:
    """Read the original provider cap without treating it as cache identity."""

    for field in ("max_completion_tokens", "max_tokens"):
        value = payload.get(field)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _cached_response_fits_cap(completion_tokens: int | None, payload: dict[str, Any]) -> bool:
    """Reject a replay that would exceed the current transport ceiling."""

    cap = _completion_cap(payload)
    if cap is None:
        return True
    return completion_tokens is not None and completion_tokens <= cap


def _execute_request_with_retries(
    *,
    payload: dict[str, Any],
    provider: OpenAICompatibleProvider,
    throttle: RequestThrottle | None,
    request_executor: TurnRequestExecutor | None,
    max_attempts: int = 5,
) -> dict[str, Any]:
    """Retry transport/429/5xx failures without changing the logical request."""

    executor = request_executor or (
        lambda value, target, limiter: execute_completion_request(target, value, throttle=limiter)
    )
    last: dict[str, Any] = {}
    request_started_at_events: list[str] = []
    attempt_timeline: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        last = dict(executor(payload, provider, throttle))
        started_at = last.get("request_started_at")
        if started_at:
            request_started_at_events.append(str(started_at))
        attempt_timeline.append(
            {
                "attempt_index": attempt,
                "queued_at": last.get("network_queued_at"),
                "rate_admitted_at": last.get("rate_admitted_at"),
                "network_started_at": last.get("network_started_at") or started_at,
                "network_finished_at": last.get("network_finished_at"),
                "latency_ms": last.get("latency_ms"),
                "throttle_wait_ms": last.get("throttle_wait_ms"),
                "http_status": last.get("http_status"),
                "error": last.get("request_error"),
                "provider_request_id": last.get("provider_request_id"),
                "response_id": last.get("response_id"),
                "retry_delay_seconds": None,
            }
        )
        status = last.get("http_status")
        retryable = bool(last.get("request_error")) and (
            status is None or int(status) == 429 or 500 <= int(status) < 600
        )
        if not retryable or attempt == max_attempts:
            last["network_attempt_count"] = attempt
            last["request_started_at_events"] = request_started_at_events
            last["attempt_timeline"] = attempt_timeline
            return last
        suggested = last.get("retry_after_seconds")
        delay = float(suggested) if suggested is not None else min(float(2 ** (attempt - 1)), 30.0)
        attempt_timeline[-1]["retry_delay_seconds"] = max(0.0, delay)
        time.sleep(max(0.0, delay))
    return last


def execute_cached_turn(
    *,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle | None,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int | None,
    validator: TurnValidator | None = None,
    schema_id: SchemaId | None = None,
    dataset: str | None = None,
    use_response_format: bool = True,
    max_tokens: int | None = None,
    request_executor: TurnRequestExecutor | None = None,
    response_hook: TurnResponseHook | None = None,
    cache_mode: Literal["shared", "live"] = "shared",
) -> CachedTurnResult:
    """Execute one cached model turn and validate its structured output."""

    def _validate_response(response_payload: dict[str, Any]) -> dict[str, Any]:
        if validator is not None:
            return validator(
                str(response_payload.get("assistant_text") or ""),
                str(response_payload.get("provider_reasoning_text") or ""),
            )
        if schema_id is None:
            raise ValueError("schema_id is required when validator is not provided.")
        return validate_or_recover_structured_output(
            str(response_payload.get("assistant_text") or ""),
            schema_id,
            dataset=dataset,
            provider_reasoning_text=str(response_payload.get("provider_reasoning_text") or ""),
        )

    request_result = execute_cached_request(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_response_format=use_response_format,
        max_tokens=max_tokens,
        request_executor=request_executor,
        response_hook=response_hook,
        response_validator=_validate_response,
        cache_mode=cache_mode,
    )
    response_payload = request_result.response_payload
    request_error = request_result.request_error
    validated_output: dict[str, Any] = {}
    output_status = "request_fail" if request_error else "schema_fail"
    if not request_error:
        try:
            validated_output = _validate_response(response_payload)
            output_status = "ok"
        except Exception:
            validated_output = {}
            output_status = "schema_fail"

    if request_result.cache_hit and output_status != "ok":
        cache.delete(request_result.cache_key)

    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    return CachedTurnResult(
        payload=request_result.payload,
        prompt_hash=request_result.prompt_hash,
        cache_key=request_result.cache_key,
        cache_hit=request_result.cache_hit,
        response_payload=response_payload,
        request_error=str(request_error) if request_error else None,
        validated_output=validated_output,
        output_status=output_status,
        usage=usage,
    )
