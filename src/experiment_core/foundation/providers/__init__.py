"""共享的 OpenAI-compatible provider 客户端。

本模块负责：
- 统一构造请求载荷；
- 复用长生命周期 HTTP 连接；
- 规范化 provider 响应；
- 为限流器提供 token 预留与 usage 对账辅助。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from threading import Lock
import time
from typing import Any

from h2.exceptions import ProtocolError as H2ProtocolError
import httpx

from experiment_core.foundation.config import ResolvedModelConfig
from experiment_core.foundation.rate_limits import RateLimitReservation, SlidingWindowRateLimiter


@dataclass(frozen=True)
class ProviderResponse:
    """表示统一归一化后的 provider 响应。"""

    http_status: int
    raw_payload: dict[str, Any]
    assistant_text: str
    provider_reasoning_text: str
    finish_reason: str | None
    usage_reported: dict[str, Any] | None
    usage_estimated: dict[str, Any]
    usage_source: str
    latency_ms: float
    provider_request_id: str | None
    response_id: str | None


@dataclass(frozen=True)
class ProviderRequestError(RuntimeError):
    """表示向 runner 暴露的 provider 请求失败。"""

    message: str
    http_status: int | None
    response_text: str | None
    provider_request_id: str | None

    def __str__(self) -> str:
        return self.message


@dataclass
class _SharedClientHandle:
    """表示一个按 provider/base_url 共享的长生命周期 HTTP client。"""

    client: httpx.Client
    refcount: int = 0


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
                        # HTTP/2 连接在远端已关闭时，close 阶段不应反过来污染实验结果。
                        pass
                    finally:
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
                assistant_text, provider_reasoning_text = _extract_message_channels(body)
                if _looks_like_provider_soft_rejection(assistant_text) and not sanitized_retry_used:
                    active_payload = _sanitize_payload_messages(active_payload)
                    sanitized_retry_used = True
                    time.sleep(_retry_delay_seconds(None, attempt))
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
                    finish_reason=_extract_finish_reason(body),
                    usage_reported=usage_reported,
                    usage_estimated=_estimate_usage(active_payload, assistant_text),
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
                time.sleep(_retry_delay_seconds(exc.response, attempt))
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.WriteError,
                httpx.CloseError,
                httpx.RemoteProtocolError,
                H2ProtocolError,
            ) as exc:
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
        """在协议级断连后轮换共享 client，避免后续请求继续复用坏连接池。"""
        with self._shared_clients_lock:
            handle = self._shared_clients.get(self._client_key)
            if handle is None or handle is not self._client_handle or handle.client is not failed_client:
                return
            try:
                failed_client.close()
            except Exception:
                pass
            handle.client = _build_http_client()


def _build_http_client() -> httpx.Client:
    """构造统一的长生命周期 HTTP/2 client。"""
    return httpx.Client(
        http2=True,
        limits=httpx.Limits(
            max_connections=128,
            max_keepalive_connections=32,
            keepalive_expiry=30.0,
        ),
    )


def build_payload(
    config: ResolvedModelConfig,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int | None,
    *,
    use_response_format: bool = True,
) -> dict[str, Any]:
    """把内部请求参数映射成 provider 所需载荷。"""
    payload: dict[str, Any] = {
        "model": config.model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_output_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    _apply_thinking_control(config, payload)
    if use_response_format and config.supports_response_format and config.response_format:
        payload["response_format"] = {"type": config.response_format}
    return payload


def estimate_request_tokens(payload: dict[str, Any]) -> int:
    """估算一次请求需要预留的 token 配额。"""
    prompt_tokens = estimate_prompt_tokens(payload)
    completion_reserve = estimate_completion_reservation(payload, prompt_tokens=prompt_tokens)
    return prompt_tokens + completion_reserve


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    """估算 prompt 部分的 token 数。"""
    prompt_chars = len(json.dumps(payload.get("messages", []), ensure_ascii=False))
    return max(1, prompt_chars // 4)


def estimate_completion_reservation(payload: dict[str, Any], *, prompt_tokens: int | None = None) -> int:
    """为 completion 预留一个比 `max_tokens` 更保守、但不至于过满的上界。"""
    prompt_estimate = prompt_tokens if prompt_tokens is not None else estimate_prompt_tokens(payload)
    max_completion_tokens = max(0, int(payload.get("max_tokens") or 0))
    if max_completion_tokens <= 0:
        return 0
    heuristic_completion = max(128, min(512, prompt_estimate))
    return min(max_completion_tokens, heuristic_completion)


def realized_total_tokens(response_payload: dict[str, Any]) -> int:
    """从响应载荷中提取真实或回退估算的总 token 数。"""
    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        try:
            return max(0, int(float(total_tokens)))
        except (TypeError, ValueError):
            pass
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    try:
        return max(0, int(float(prompt_tokens or 0)) + int(float(completion_tokens or 0)))
    except (TypeError, ValueError):
        return 0


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


def _apply_thinking_control(config: ResolvedModelConfig, payload: dict[str, Any]) -> None:
    if config.reasoning_effort is None:
        return
    if config.provider == "local_ollama":
        payload["reasoning_effort"] = config.reasoning_effort
        return
    if config.provider == "dashscope":
        payload["enable_thinking"] = config.reasoning_effort != "none"
        return
    if config.provider == "siliconflow":
        payload["enable_thinking"] = config.reasoning_effort != "none"
        return
    if config.provider == "deepseek":
        payload["thinking"] = {"type": "disabled" if config.reasoning_effort == "none" else "enabled"}
        return
    if config.provider == "xiaomimimo":
        if config.reasoning_effort == "none":
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["reasoning_effort"] = config.reasoning_effort
        return
    payload["reasoning_effort"] = config.reasoning_effort


def _extract_message_channels(body: dict[str, Any]) -> tuple[str, str]:
    choices = body.get("choices") or []
    if not choices:
        return "", ""
    message = choices[0].get("message") or {}
    assistant_text = _extract_text_field(message.get("content", ""))
    provider_reasoning_text = _extract_text_field(
        message.get("reasoning", message.get("reasoning_content", ""))
    )
    return assistant_text, provider_reasoning_text


def _looks_like_provider_soft_rejection(text: str) -> bool:
    normalized = " ".join((text or "").split()).lower()
    if not normalized:
        return False
    return normalized in {
        "the request was rejected because it was considered high risk",
        "request rejected because it was considered high risk",
    }


def _sanitize_payload_messages(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    sanitized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", ""))
        ascii_content = content.encode("ascii", "ignore").decode("ascii")
        ascii_content = " ".join(ascii_content.split())
        sanitized_messages.append(
            {
                "role": str(message.get("role", "user")),
                "content": ascii_content,
            }
        )
    return {**payload, "messages": sanitized_messages}


def _retry_delay_seconds(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
    return min(2**attempt, 8)


def _extract_text_field(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
                continue
            if "text" in item:
                parts.append(str(item.get("text", "")))
                continue
            if "content" in item:
                parts.append(str(item.get("content", "")))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _extract_finish_reason(body: dict[str, Any]) -> str | None:
    choices = body.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def _estimate_usage(payload: dict[str, Any], assistant_text: str) -> dict[str, int]:
    prompt_chars = len(json.dumps(payload, ensure_ascii=False))
    completion_chars = len(assistant_text)
    return {
        "prompt_tokens": max(1, prompt_chars // 4),
        "completion_tokens": max(1, completion_chars // 4),
        "total_tokens": max(2, (prompt_chars + completion_chars) // 4),
    }
