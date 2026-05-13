"""provider 响应归一化与错误类型。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class ProviderResponse:
    """统一归一化后的 provider 响应。"""

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
    """向 runner 暴露的 provider 请求失败。"""

    message: str
    http_status: int | None
    response_text: str | None
    provider_request_id: str | None

    def __str__(self) -> str:
        return self.message


def extract_message_channels(body: dict[str, Any]) -> tuple[str, str]:
    """从 provider 返回体中提取 assistant 与 reasoning 文本。"""

    choices = body.get("choices") or []
    if not choices:
        return "", ""
    message = choices[0].get("message") or {}
    assistant_text = extract_text_field(message.get("content", ""))
    provider_reasoning_text = extract_text_field(
        message.get("reasoning", message.get("reasoning_content", ""))
    )
    return assistant_text, provider_reasoning_text


def looks_like_provider_soft_rejection(text: str) -> bool:
    """判断 assistant 文本是否像 provider 的软拒答。"""

    normalized = " ".join((text or "").split()).lower()
    if not normalized:
        return False
    return normalized in {
        "the request was rejected because it was considered high risk",
        "request rejected because it was considered high risk",
    }


def sanitize_payload_messages(payload: dict[str, Any]) -> dict[str, Any]:
    """把消息裁剪为更保守的 ASCII 版本，供软拒答后重试。"""

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


def retry_delay_seconds(response, attempt: int) -> float:
    """计算下一次重试前的等待时间。"""

    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
    return min(2**attempt, 8)


def extract_text_field(content: object) -> str:
    """把 provider 的多形态文本字段归一成字符串。"""

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


def extract_finish_reason(body: dict[str, Any]) -> str | None:
    """读取首个 choice 的 finish_reason。"""

    choices = body.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def estimate_usage(payload: dict[str, Any], assistant_text: str) -> dict[str, int]:
    """在 provider 未返回 usage 时构造一个保守估算。"""

    prompt_chars = len(json.dumps(payload, ensure_ascii=False))
    completion_chars = len(assistant_text)
    return {
        "prompt_tokens": max(1, prompt_chars // 4),
        "completion_tokens": max(1, completion_chars // 4),
        "total_tokens": max(2, (prompt_chars + completion_chars) // 4),
    }
