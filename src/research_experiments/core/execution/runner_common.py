"""跨实验家族共享的 runner 基础原语。

这里统一承接请求生命周期里的低层公共逻辑，包括：
- 规范 run 根目录；
- 为消息构造稳定哈希；
- 统一并发批处理顺序；
- 统一“查缓存 -> 发请求 -> 校验结构化输出 -> 选择性写缓存”的执行链路。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
from typing import Any, Callable, Iterable, TypeVar

from research_experiments.core.execution.cache import (
    RequestCache,
    build_request_cache_key,
    cache_successful_response,
)
from research_experiments.core.config import ResolvedModelConfig
from research_experiments.core.execution.providers import (
    OpenAICompatibleProvider,
    build_payload,
    execute_completion_request,
)
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.structured_outputs import SchemaId, validate_or_recover_structured_output


T = TypeVar("T")
R = TypeVar("R")

TurnValidator = Callable[[str, str], dict[str, Any]]
TurnRequestExecutor = Callable[
    [dict[str, Any], OpenAICompatibleProvider, SlidingWindowRateLimiter | None],
    dict[str, Any],
]
TurnResponseHook = Callable[[dict[str, Any], dict[str, Any]], None]


@dataclass(frozen=True)
class CachedTurnResult:
    """表示一次“命中缓存或真实请求”的标准化结果。"""

    payload: dict[str, Any]
    prompt_hash: str
    cache_key: str
    cache_hit: bool
    response_payload: dict[str, Any]
    request_error: str | None
    validated_output: dict[str, Any]
    output_status: str
    usage: dict[str, Any]


def prepare_run_root(
    run_root: str | Path,
    experiment_name: str,
    phase_name: str,
    run_id: str,
) -> Path:
    """创建并返回规范的 run 根目录路径。"""

    root = Path(run_root) / experiment_name / phase_name / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def prompt_hash(messages: list[dict[str, Any]]) -> str:
    """为一组消息构造稳定哈希，供追溯与去重使用。"""

    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def run_indexed_batch(
    items: Iterable[T],
    *,
    worker: Callable[[T], R],
    max_concurrent_requests: int,
) -> list[tuple[int, R]]:
    """并发执行带索引批次，并按输入顺序返回结果。"""

    indexed_items = list(enumerate(items))
    max_workers = max(1, min(max_concurrent_requests, len(indexed_items) or 1))
    completed: list[tuple[int, R]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, item): index
            for index, item in indexed_items
        }
        for future in as_completed(future_to_index):
            completed.append((future_to_index[future], future.result()))
    completed.sort(key=lambda item: item[0])
    return completed


def execute_cached_turn(
    *,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter | None,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int | None,
    validator: TurnValidator | None = None,
    schema_id: SchemaId | None = None,
    dataset: str | None = None,
    use_response_format: bool = True,
    request_executor: TurnRequestExecutor | None = None,
    response_hook: TurnResponseHook | None = None,
) -> CachedTurnResult:
    """执行一次带缓存的调用，并统一请求生命周期输出。

    这个入口会显式区分“请求失败”“结构化失败”和“结构化成功”，
    避免后续报告层把工程失败误判为方法行为。
    """

    payload = build_payload(
        config=backbone,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
        use_response_format=use_response_format,
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
            request_executor(payload, provider, limiter)
            if request_executor is not None
            else execute_completion_request(provider, payload, limiter=limiter)
        )
        response_payload = dict(response_payload)
        if response_hook is not None:
            response_hook(payload, response_payload)
        cache_hit = False
    else:
        response_payload = json.loads(cached.response_json)
        cache_hit = True

    request_error = response_payload.get("request_error")
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
            # 只有结构化成功的响应才进入缓存，避免把坏输出固化成后续缓存命中。
            if not cache_hit:
                cache_successful_response(
                    cache,
                    cache_key=cache_key,
                    payload=payload,
                    response_payload=response_payload,
                )
        except Exception:
            validated_output = {}
            output_status = "schema_fail"

    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    return CachedTurnResult(
        payload=payload,
        prompt_hash=hashed_prompt,
        cache_key=cache_key,
        cache_hit=cache_hit,
        response_payload=response_payload,
        request_error=str(request_error) if request_error else None,
        validated_output=validated_output,
        output_status=output_status,
        usage=usage,
    )


