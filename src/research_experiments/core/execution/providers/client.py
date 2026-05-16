"""provider 客户端与请求执行入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from threading import Lock
import time
from typing import Any

import httpx

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.core.execution.providers.normalization import (
    ProviderRequestError,
    ProviderResponse,
    estimate_usage,
    extract_finish_reason,
    extract_message_channels,
    looks_like_provider_soft_rejection,
    retry_delay_seconds,
    sanitize_payload_messages,
)
from research_experiments.core.execution.providers.payloads import (
    estimate_request_tokens,
    realized_total_tokens,
)
from research_experiments.core.execution.rate_limits import RateLimitReservation, SlidingWindowRateLimiter


@dataclass
class _SharedClientHandle:
    """按 provider/base_url 共享的长生命周期 HTTP client。"""

    client: httpx.Client
    refcount: int = 0
    retired_clients: list[httpx.Client] = field(default_factory=list)


class OpenAICompatibleProvider:
    """最小可用的 OpenAI-compatible provider 包装器。"""

    _shared_clients: dict[str, _SharedClientHandle] = {}
    _shared_clients_lock = Lock()

    def __init__(self, config: ResolvedModelConfig) -> None:
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {config.api_key_env}. "
                "Create `.env.local` or export it in the current shell."
            )
        self.api_key = api_key
        self._client_key = f"{config.provider}|{config.base_url.rstrip('/')}"
        self._closed = False
        self._client_handle = self._acquire_shared_client_handle()

    def close(self) -> None:
        """释放当前 provider 持有的共享传输层引用。"""

        if self._closed:
            return
        with self._shared_clients_lock:
            handle = self._shared_clients.get(self._client_key)
            if handle is not None:
                handle.refcount -= 1
                if handle.refcount <= 0:
                    try:
                        handle.client.close()
                    except Exception:
                        pass
                    finally:
                        for retired in handle.retired_clients:
                            try:
                                retired.close()
                            except Exception:
                                pass
                        self._shared_clients.pop(self._client_key, None)
        self._closed = True

    def chat_completion(self, payload: dict[str, Any]) -> ProviderResponse:
        """执行一次带有限重试的 chat completion 请求。"""

        if self._closed:
            raise RuntimeError("Provider client has already been closed.")
        url = self.config.base_url.rstrip("/") + self.config.chat_path
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(self.config.timeout_seconds)
        last_error: Exception | None = None
        active_payload = payload
        sanitized_retry_used = False

        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            client = self._client_handle.client
            try:
                response = client.post(url, headers=headers, json=active_payload, timeout=timeout)
                latency_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                body = response.json()
                usage_reported = body.get("usage")
                assistant_text, provider_reasoning_text = extract_message_channels(body)
                if looks_like_provider_soft_rejection(assistant_text) and not sanitized_retry_used:
                    active_payload = sanitize_payload_messages(active_payload)
                    sanitized_retry_used = True
                    time.sleep(retry_delay_seconds(None, attempt))
                    continue
                response_id = body.get("id")
                provider_request_id = (
                    response.headers.get("x-request-id")
                    or response.headers.get("x-b3-traceid")
                    or response_id
                )
                return ProviderResponse(
                    http_status=response.status_code,
                    raw_payload=body,
                    assistant_text=assistant_text,
                    provider_reasoning_text=provider_reasoning_text,
                    finish_reason=extract_finish_reason(body),
                    usage_reported=usage_reported,
                    usage_estimated=estimate_usage(active_payload, assistant_text),
                    usage_source="reported" if usage_reported else "estimated",
                    latency_ms=latency_ms,
                    provider_request_id=provider_request_id,
                    response_id=response_id,
                )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in {408, 409, 429, 500, 502, 503, 504} or attempt == self.config.max_retries:
                    response_text = exc.response.text
                    provider_request_id = (
                        exc.response.headers.get("x-request-id")
                        or exc.response.headers.get("x-b3-traceid")
                    )
                    raise ProviderRequestError(
                        message=f"Provider returned HTTP {exc.response.status_code}: {response_text}",
                        http_status=exc.response.status_code,
                        response_text=response_text,
                        provider_request_id=provider_request_id,
                    ) from exc
                time.sleep(retry_delay_seconds(exc.response, attempt))
            except httpx.TransportError as exc:
                last_error = exc
                self._reset_shared_client(client)
                if attempt == self.config.max_retries:
                    raise ProviderRequestError(
                        message=f"Provider connection error after retries: {exc}",
                        http_status=None,
                        response_text=None,
                        provider_request_id=None,
                    ) from exc
                time.sleep(min(2**attempt, 8))

        raise ProviderRequestError(
            message=f"API request failed after retries: {last_error}",
            http_status=None,
            response_text=None,
            provider_request_id=None,
        )

    def _acquire_shared_client_handle(self) -> _SharedClientHandle:
        with self._shared_clients_lock:
            handle = self._shared_clients.get(self._client_key)
            if handle is None:
                handle = _SharedClientHandle(client=_build_http_client())
                self._shared_clients[self._client_key] = handle
            handle.refcount += 1
            return handle

    def _reset_shared_client(self, failed_client: httpx.Client) -> None:
        """在协议级断连后轮换共享 client，避免复用坏连接池。

        这里不能立刻关闭旧 client。
        高并发下，其他线程可能仍在使用它；若此时直接 close，会把仍在运行的请求一起打断。
        因此我们只把旧实例挂到 `retired_clients`，等最后一个 provider 释放引用时统一回收。
        """

        with self._shared_clients_lock:
            handle = self._shared_clients.get(self._client_key)
            if handle is None or handle is not self._client_handle or handle.client is not failed_client:
                return
            handle.retired_clients.append(failed_client)
            handle.client = _build_http_client()


def execute_completion_request(
    provider: OpenAICompatibleProvider,
    payload: dict[str, Any],
    *,
    limiter: SlidingWindowRateLimiter | None = None,
) -> dict[str, Any]:
    """统一执行一次 provider 请求，并在限流器上做预留与对账。"""

    reservation: RateLimitReservation | None = None
    response_payload: dict[str, Any] | None = None
    if limiter is not None:
        reservation = limiter.acquire(estimate_request_tokens(payload))
    try:
        response = provider.chat_completion(payload)
        response_payload = {
            "http_status": response.http_status,
            "raw_payload": response.raw_payload,
            "assistant_text": response.assistant_text,
            "provider_reasoning_text": response.provider_reasoning_text,
            "finish_reason": response.finish_reason,
            "usage_reported": response.usage_reported,
            "usage_estimated": response.usage_estimated,
            "usage_source": response.usage_source,
            "latency_ms": response.latency_ms,
            "provider_request_id": response.provider_request_id,
            "response_id": response.response_id,
            "request_error": None,
        }
        return response_payload
    except ProviderRequestError as exc:
        response_payload = {
            "http_status": exc.http_status,
            "raw_payload": {"error": exc.message},
            "assistant_text": "",
            "provider_reasoning_text": "",
            "finish_reason": None,
            "usage_reported": None,
            "usage_estimated": None,
            "usage_source": "missing",
            "latency_ms": 0.0,
            "provider_request_id": exc.provider_request_id,
            "response_id": None,
            "request_error": exc.message,
        }
        return response_payload
    finally:
        if limiter is not None and reservation is not None and response_payload is not None:
            limiter.settle(reservation, realized_total_tokens(response_payload))


def _build_http_client() -> httpx.Client:
    """构造统一的长生命周期 HTTP client。"""

    return httpx.Client(
        limits=httpx.Limits(
            max_connections=128,
            max_keepalive_connections=32,
            keepalive_expiry=30.0,
        ),
    )
