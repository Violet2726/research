"""Shared OpenAI-compatible provider client."""

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
    """Normalized provider response."""

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
    """Provider request failure surfaced to runners."""

    message: str
    http_status: int | None
    response_text: str | None
    provider_request_id: str | None

    def __str__(self) -> str:
        return self.message


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible provider wrapper."""

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
        """Execute one chat-completion request with bounded retries."""
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
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=active_payload)
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
    *,
    use_response_format: bool = True,
) -> dict[str, Any]:
    """Map internal request parameters to provider payloads."""
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
    """Estimate request tokens with a lightweight heuristic."""
    prompt_chars = len(json.dumps(payload.get("messages", []), ensure_ascii=False))
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = int(payload.get("max_tokens") or 0)
    return prompt_tokens + completion_tokens


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
