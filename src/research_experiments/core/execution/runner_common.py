from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.core.execution.cache import (
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
) -> CachedRequestResult:
    """Execute one cached raw provider request and persist successful raw responses."""

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
    cached = cache.get(cache_key)
    if cached is None:
        response_payload = (
            request_executor(payload, provider, throttle)
            if request_executor is not None
            else execute_completion_request(provider, payload, throttle=throttle)
        )
        response_payload = dict(response_payload)
        if response_hook is not None:
            response_hook(payload, response_payload)
        cache_hit = False
        if response_payload.get("request_error") is None:
            try:
                cache_successful_response(
                    cache,
                    cache_key=cache_key,
                    payload=payload,
                    response_payload=response_payload,
                )
            except Exception:
                response_payload = dict(response_payload)
                response_payload["cache_write_error"] = True
    else:
        response_payload = json.loads(cached.response_json)
        cache_hit = True

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
) -> CachedTurnResult:
    """Execute one cached model turn and validate its structured output."""

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
    )
    response_payload = request_result.response_payload
    request_error = request_result.request_error
    validated_output: dict[str, Any] = {}
    output_status = "request_fail" if request_error else "schema_fail"
    if not request_error:
        try:
            if validator is not None:
                validated_output = validator(
                    str(response_payload.get("assistant_text") or ""),
                    str(response_payload.get("provider_reasoning_text") or ""),
                )
            else:
                if schema_id is None:
                    raise ValueError("schema_id is required when validator is not provided.")
                validated_output = validate_or_recover_structured_output(
                    str(response_payload.get("assistant_text") or ""),
                    schema_id,
                    dataset=dataset,
                    provider_reasoning_text=str(response_payload.get("provider_reasoning_text") or ""),
                )
            output_status = "ok"
        except Exception:
            validated_output = {}
            output_status = "schema_fail"

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
