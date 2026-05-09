"""覆盖 `single_agent.runner` 的缓存写入时机。"""

from __future__ import annotations

from typing import Any

import pytest

from experiment_core.foundation.cache import RequestCache, build_request_cache_key
from experiment_core.foundation.providers import ProviderRequestError, ProviderResponse
from experiment_core.foundation.rate_limits import SlidingWindowRateLimiter
from single_agent.runner import CallSpec, _execute_call


class _ProviderStub:
    def __init__(self, outcome: ProviderResponse | Exception) -> None:
        self._outcome = outcome

    def chat_completion(self, payload: dict[str, Any]) -> ProviderResponse:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _build_spec() -> CallSpec:
    payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "seed": 42,
    }
    return CallSpec(
        run_id="run-1",
        dataset="gsm8k",
        split_name="smoke20",
        sample_id="gsm8k-00001",
        sample_order=0,
        method_name="cot_1",
        method_family="cot",
        rerun_index=0,
        replicate_id=0,
        agent_id=None,
        model_name="Demo Model",
        model_id="demo-model",
        provider_name="demo-provider",
        base_url="https://example.invalid/v1",
        prompt_hash="prompt-hash",
        payload=payload,
        cache_key=build_request_cache_key(
            provider="demo-provider",
            request_model="demo-model",
            payload=payload,
        ),
    )


def _response(*, assistant_text: str) -> ProviderResponse:
    return ProviderResponse(
        http_status=200,
        raw_payload={"id": "resp_1"},
        assistant_text=assistant_text,
        provider_reasoning_text="",
        finish_reason="stop",
        usage_reported={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        usage_estimated={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        usage_source="reported",
        latency_ms=12.5,
        provider_request_id="req_1",
        response_id="resp_1",
    )


def _limiter() -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(requests_per_minute=10_000, tokens_per_minute=10_000_000)


def test_execute_call_caches_only_after_successful_parse(tmp_path) -> None:
    spec = _build_spec()
    cache = RequestCache(tmp_path / "requests.sqlite")

    first_row = _execute_call(
        spec,
        _ProviderStub(_response(assistant_text='{"final_answer":"4","reasoning":"basic arithmetic"}')),
        cache,
        _limiter(),
    )

    assert first_row["output_status"] == "ok"
    assert first_row["cache_hit"] is False
    assert cache.get(spec.cache_key) is not None

    second_row = _execute_call(
        spec,
        _ProviderStub(AssertionError("cache hit should skip provider call")),
        cache,
        _limiter(),
    )

    assert second_row["output_status"] == "ok"
    assert second_row["cache_hit"] is True
    cache.close()


def test_execute_call_does_not_cache_request_failures(tmp_path) -> None:
    spec = _build_spec()
    cache = RequestCache(tmp_path / "requests.sqlite")

    row = _execute_call(
        spec,
        _ProviderStub(
            ProviderRequestError(
                message="upstream exploded",
                http_status=503,
                response_text="busy",
                provider_request_id="req_fail",
            )
        ),
        cache,
        _limiter(),
    )

    assert row["output_status"] == "request_fail"
    assert row["cache_hit"] is False
    assert cache.get(spec.cache_key) is None
    cache.close()


def test_execute_call_does_not_cache_schema_failures(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _build_spec()
    cache = RequestCache(tmp_path / "requests.sqlite")

    def _always_fail(*args, **kwargs):
        raise ValueError("schema exploded")

    monkeypatch.setattr("single_agent.runner.validate_or_recover_structured_output", _always_fail)

    row = _execute_call(
        spec,
        _ProviderStub(_response(assistant_text='{"final_answer":"4","reasoning":"basic arithmetic"}')),
        cache,
        _limiter(),
    )

    assert row["output_status"] == "schema_fail"
    assert row["cache_hit"] is False
    assert cache.get(spec.cache_key) is None
    cache.close()
