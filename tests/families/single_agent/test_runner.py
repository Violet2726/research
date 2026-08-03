"""覆盖 `single_agent.runner` 的缓存写入时机。"""

from __future__ import annotations

from typing import Any

import pytest

from research_experiments.core.config import resolve_model_ref
from research_experiments.core.execution.cache import RequestCache, build_request_cache_key
from research_experiments.core.execution.providers import ProviderRequestError, ProviderResponse, build_payload
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.families.single_agent.run.sample import CallSpec, _execute_call


class _ProviderStub:
    def __init__(self, outcome: ProviderResponse | Exception) -> None:
        self._outcome = outcome

    def chat_completion(self, payload: dict[str, Any]) -> ProviderResponse:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _build_spec() -> CallSpec:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    payload = build_payload(
        config=model,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        use_response_format=False,
    )
    return CallSpec(
        run_id="run-1",
        dataset="gsm8k",
        split_name="count20",
        sample_id="gsm8k-00001",
        sample_order=0,
        method_name="cot_1",
        method_family="cot",
        rerun_index=0,
        replicate_id=0,
        agent_id=None,
        model_name=model.name,
        model_id=model.model_id,
        provider_name=model.provider,
        base_url=model.base_url,
        prompt_hash="prompt-hash",
        payload=payload,
        cache_key=build_request_cache_key(
            provider=model.provider,
            request_model=model.model_id,
            payload=payload,
        ),
        backbone=model,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        seed=42,
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


def _throttle() -> RequestThrottle:
    return RequestThrottle(
        max_concurrent_requests=8,
        requests_per_minute=10_000,
    )


def test_execute_call_caches_only_after_successful_parse(tmp_path) -> None:
    spec = _build_spec()
    cache = RequestCache(tmp_path / "requests.sqlite")

    first_row = _execute_call(
        spec,
        _ProviderStub(
            _response(
                assistant_text="REASONING: Add the two numbers.\nFINAL_ANSWER: 4"
            )
        ),
        cache,
        _throttle(),
    )

    assert first_row["output_status"] == "ok"
    assert first_row["cache_hit"] is False
    assert cache.get(spec.cache_key) is not None

    second_row = _execute_call(
        spec,
        _ProviderStub(AssertionError("cache hit should skip provider call")),
        cache,
        _throttle(),
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
        _throttle(),
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

    monkeypatch.setattr(
        "research_experiments.family_runtime.output_protocols.parse_free_text_answer_output",
        _always_fail,
    )

    row = _execute_call(
        spec,
        _ProviderStub(_response(assistant_text="REASONING: Add the two numbers.\nFINAL_ANSWER: 4")),
        cache,
        _throttle(),
    )

    assert row["output_status"] == "protocol_fail"
    assert row["cache_hit"] is False
    assert cache.get(spec.cache_key) is None
    cache.close()

