from __future__ import annotations

from typing import Any

from research_experiments.core.config import ResolvedModelConfig


def build_payload(
    config: ResolvedModelConfig,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int | None,
    *,
    use_response_format: bool = True,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Map normalized request arguments into a provider payload."""

    payload: dict[str, Any] = {
        "model": config.model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed
    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    apply_thinking_control(config, payload)
    if use_response_format and config.supports_response_format and config.response_format:
        payload["response_format"] = {"type": config.response_format}
    return payload


def apply_thinking_control(config: ResolvedModelConfig, payload: dict[str, Any]) -> None:
    """Map normalized reasoning controls into provider-specific fields."""

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
