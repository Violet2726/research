"""provider 请求载荷与 token 预留逻辑。"""

from __future__ import annotations

import json
from typing import Any

from research_experiments.core.config import ResolvedModelConfig


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
    apply_thinking_control(config, payload)
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
    """为 completion 预留一个不至于过满的上界。"""

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


def apply_thinking_control(config: ResolvedModelConfig, payload: dict[str, Any]) -> None:
    """按 provider 能力把 reasoning 控制映射到底层字段。"""

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
