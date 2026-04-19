"""共享 provider 客户端实现。

当前项目统一把远端模型视作 OpenAI-compatible chat completion 接口。
本模块负责请求发送、重试、错误包装与 usage 估算，不夹带实验特定逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

import httpx

from experiment_core.config import ResolvedModelConfig


@dataclass(frozen=True)
class ProviderResponse:
    """一次成功 provider 调用的规范化响应。"""

    http_status: int
    raw_payload: dict[str, Any]
    raw_text: str
    finish_reason: str | None
    usage_reported: dict[str, Any] | None
    usage_estimated: dict[str, Any]
    usage_source: str
    latency_ms: float
    provider_request_id: str | None
    response_id: str | None


@dataclass(frozen=True)
class ProviderRequestError(RuntimeError):
    """对外暴露的 provider 请求异常。"""

    message: str
    http_status: int | None
    response_text: str | None
    provider_request_id: str | None

    def __str__(self) -> str:
        return self.message


class OpenAICompatibleProvider:
    """最小化的 OpenAI-compatible provider 封装。"""

    def __init__(self, config: ResolvedModelConfig) -> None:
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {config.api_key_env}. "
                "Create `.env.local` or export it in the current shell."
            )
        self.api_key = api_key

    def chat_completion(self, payload: dict[str, Any]) -> ProviderResponse:
        """发起一次 chat completion 请求，并在可恢复错误时重试。"""
        url = self.config.base_url.rstrip("/") + self.config.chat_path
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(self.config.timeout_seconds)
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                latency_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                body = response.json()
                usage_reported = body.get("usage")
                content = _extract_text(body)
                response_id = body.get("id")
                provider_request_id = (
                    response.headers.get("x-request-id")
                    or response.headers.get("x-b3-traceid")
                    or response_id
                )
                return ProviderResponse(
                    http_status=response.status_code,
                    raw_payload=body,
                    raw_text=content,
                    finish_reason=_extract_finish_reason(body),
                    usage_reported=usage_reported,
                    usage_estimated=_estimate_usage(payload, content),
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
                time.sleep(min(2**attempt, 8))
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
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


def build_payload(
    config: ResolvedModelConfig,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    """构造统一的请求 payload。"""
    payload: dict[str, Any] = {
        "model": config.model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_output_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if config.supports_response_format and config.response_format:
        payload["response_format"] = {"type": config.response_format}
    return payload


def estimate_request_tokens(payload: dict[str, Any]) -> int:
    """用字符数对请求 token 做近似估算，供限流器占位。"""
    prompt_chars = len(json.dumps(payload.get("messages", []), ensure_ascii=False))
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = int(payload.get("max_tokens") or 0)
    return prompt_tokens + completion_tokens


def _extract_text(body: dict[str, Any]) -> str:
    """从 provider 响应中提取首个文本内容。"""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_finish_reason(body: dict[str, Any]) -> str | None:
    """提取首个 choice 的 finish reason。"""
    choices = body.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def _estimate_usage(payload: dict[str, Any], raw_text: str) -> dict[str, int]:
    """当 provider 未回 usage 时，用字符数提供保守估计。"""
    prompt_chars = len(json.dumps(payload, ensure_ascii=False))
    completion_chars = len(raw_text)
    return {
        "prompt_tokens": max(1, prompt_chars // 4),
        "completion_tokens": max(1, completion_chars // 4),
        "total_tokens": max(2, (prompt_chars + completion_chars) // 4),
    }
