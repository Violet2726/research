"""Provider 抽象层。

当前项目统一走 OpenAI-compatible chat completions 协议，
这里负责 API key 读取、请求重试、响应抽取和用量估算。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

import httpx

from api_baselines.config import ResolvedModelConfig


@dataclass(frozen=True)
class ProviderResponse:
    """一次 provider 成功响应的标准化结果。"""

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
    """对网络异常和 HTTP 错误做统一包装。"""

    message: str
    http_status: int | None
    response_text: str | None
    provider_request_id: str | None

    def __str__(self) -> str:
        return self.message


class OpenAICompatibleProvider:
    """兼容 OpenAI chat completions 协议的 provider 客户端。"""

    def __init__(self, config: ResolvedModelConfig) -> None:
        """读取模型解析后的连接配置，并校验 API key 是否可用。"""
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {config.api_key_env}. "
                "Create `.env.local` or export it in the current shell."
            )
        self.api_key = api_key

    def chat_completion(self, payload: dict[str, Any]) -> ProviderResponse:
        """发送一次 chat completion 请求，并按配置执行指数退避重试。"""
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
    """把方法参数与模型默认配置合并成最终请求体。"""
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
    """在真正发送请求前做保守 token 估算，用于限流。"""
    prompt_chars = len(json.dumps(payload.get("messages", []), ensure_ascii=False))
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = int(payload.get("max_tokens") or 0)
    return prompt_tokens + completion_tokens


def _extract_text(body: dict[str, Any]) -> str:
    """从 provider 返回体中抽取首个候选回答文本。"""
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
    """提取 completion finish_reason，供排障与统计使用。"""
    choices = body.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def _estimate_usage(payload: dict[str, Any], raw_text: str) -> dict[str, int]:
    """当 provider 没有上报 usage 时，用字符数估一个可比较的近似值。"""
    prompt_chars = len(json.dumps(payload, ensure_ascii=False))
    completion_chars = len(raw_text)
    return {
        "prompt_tokens": max(1, prompt_chars // 4),
        "completion_tokens": max(1, completion_chars // 4),
        "total_tokens": max(2, (prompt_chars + completion_chars) // 4),
    }
