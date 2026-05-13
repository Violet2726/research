
"""?? provider ?????usage ????? HTTP ??????"""

from __future__ import annotations

import httpx
import pytest

from research_experiments.core.config import resolve_model_ref
from research_experiments.core.execution.providers import (
    OpenAICompatibleProvider,
    ProviderResponse,
    _extract_message_channels,
    build_payload,
    execute_completion_request,
)
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter


def test_build_payload_maps_thinking_control_by_provider() -> None:
    local_model = resolve_model_ref("local_ollama/qwen3:4b")
    local_payload = build_payload(
        local_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert local_payload["reasoning_effort"] == "none"
    assert "enable_thinking" not in local_payload

    deepseek_model = resolve_model_ref("deepseek/deepseek-v4-flash")
    deepseek_payload = build_payload(
        deepseek_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert deepseek_payload["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in deepseek_payload

    xiaomimimo_model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    xiaomimimo_payload = build_payload(
        xiaomimimo_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert xiaomimimo_payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in xiaomimimo_payload
    assert "enable_thinking" not in xiaomimimo_payload
    assert xiaomimimo_payload["response_format"] == {"type": "json_object"}

    xiaomimimo_payload_no_format = build_payload(
        xiaomimimo_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
        use_response_format=False,
    )
    assert xiaomimimo_payload_no_format["thinking"] == {"type": "disabled"}
    assert "response_format" not in xiaomimimo_payload_no_format

def test_extract_message_channels_supports_reasoning_metadata() -> None:
    assistant_text, provider_reasoning_text = _extract_message_channels(
        {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": '{"final_answer":"yes","reasoning":"short"}'}],
                        "reasoning": "hidden provider reasoning",
                    }
                }
            ]
        }
    )
    assert assistant_text.startswith('{"final_answer":"yes"')
    assert provider_reasoning_text == "hidden provider reasoning"

def test_execute_completion_request_reconciles_usage() -> None:
    class FakeProvider:
        def chat_completion(self, payload: dict[str, object]) -> ProviderResponse:
            return ProviderResponse(
                http_status=200,
                raw_payload={"ok": True},
                assistant_text='{"final_answer": "42", "reasoning": "ok"}',
                provider_reasoning_text="",
                finish_reason="stop",
                usage_reported={"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
                usage_estimated={"prompt_tokens": 8, "completion_tokens": 64, "total_tokens": 72},
                usage_source="reported",
                latency_ms=10.0,
                provider_request_id="req_test",
                response_id="resp_test",
            )

    limiter = SlidingWindowRateLimiter(
        requests_per_minute=None,
        tokens_per_minute=200,
        window_seconds=0.05,
    )
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 512,
    }
    response_payload = execute_completion_request(FakeProvider(), payload, limiter=limiter)
    assert response_payload["request_error"] is None
    assert response_payload["usage_reported"]["total_tokens"] == 20
    assert sum(event.tokens for event in limiter.token_events) == 20

def test_provider_reuses_shared_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[object] = []
    created_http2_flags: list[bool] = []

    class DummyClient:
        def __init__(self, **kwargs: object) -> None:
            created_clients.append(self)
            created_http2_flags.append(bool(kwargs.get("http2")))

        def close(self) -> None:
            return None

    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    monkeypatch.setenv(model.api_key_env, "test-key")
    monkeypatch.setattr("research_experiments.core.execution.providers.httpx.Client", DummyClient)
    OpenAICompatibleProvider._shared_clients.clear()
    provider_a = None
    provider_b = None
    try:
        provider_a = OpenAICompatibleProvider(model)
        provider_b = OpenAICompatibleProvider(model)
        assert len(created_clients) == 1
        assert created_http2_flags == [False]
    finally:
        if provider_a is not None:
            provider_a.close()
        if provider_b is not None:
            provider_b.close()

def test_provider_rotates_shared_http_client_after_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[object] = []

    class DummyClient:
        def __init__(self, **kwargs: object) -> None:
            self.closed = False
            self.should_fail = len(created_clients) == 0
            created_clients.append(self)

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, object], timeout: object) -> httpx.Response:
            if self.should_fail:
                raise httpx.RemoteProtocolError(
                    "Invalid input ConnectionInputs.RECV_WINDOW_UPDATE in state ConnectionState.CLOSED"
                )
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=headers, json=json),
                json={
                    "id": "resp_test",
                    "choices": [
                        {
                            "message": {"content": '{"final_answer": "42"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )

        def close(self) -> None:
            self.closed = True

    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    monkeypatch.setenv(model.api_key_env, "test-key")
    monkeypatch.setattr("research_experiments.core.execution.providers.httpx.Client", DummyClient)
    OpenAICompatibleProvider._shared_clients.clear()
    provider = None
    try:
        provider = OpenAICompatibleProvider(model)
        response = provider.chat_completion(
            {
                "model": model.model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
            }
        )
        assert response.http_status == 200
        assert response.finish_reason == "stop"
        assert len(created_clients) == 2
        assert getattr(created_clients[0], "closed") is True
    finally:
        if provider is not None:
            provider.close()

def test_provider_close_swallows_transport_close_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyClient:
        def __init__(self, **kwargs: object) -> None:
            return None

        def close(self) -> None:
            raise RuntimeError("Invalid input ConnectionInputs.RECV_WINDOW_UPDATE in state ConnectionState.CLOSED")

    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    monkeypatch.setenv(model.api_key_env, "test-key")
    monkeypatch.setattr("research_experiments.core.execution.providers.httpx.Client", DummyClient)
    OpenAICompatibleProvider._shared_clients.clear()
    provider = OpenAICompatibleProvider(model)
    provider.close()
    assert provider._closed is True

