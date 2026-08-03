"""覆盖共享 runner 的流式批处理原语。"""

from __future__ import annotations

import threading

import pytest

from research_experiments.core.config import resolve_model_ref
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.runner_common import execute_cached_turn, iter_indexed_batch


def test_iter_indexed_batch_yields_completed_sample_before_slow_sample() -> None:
    slow_can_finish = threading.Event()

    def _worker(item: str) -> str:
        if item == "slow":
            slow_can_finish.wait(timeout=1.0)
        return item.upper()

    result_iter = iter_indexed_batch(
        ["slow", "fast"],
        worker=_worker,
        max_concurrent_requests=2,
    )
    first_index, first_result = next(iter(result_iter))
    slow_can_finish.set()
    remaining = list(result_iter)

    assert (first_index, first_result) == (1, "FAST")
    assert remaining == [(0, "SLOW")]


def test_iter_indexed_batch_propagates_worker_errors() -> None:
    def _worker(item: int) -> int:
        if item == 2:
            raise ValueError("boom")
        return item

    with pytest.raises(ValueError, match="boom"):
        list(iter_indexed_batch([2], worker=_worker, max_concurrent_requests=1))


def _response(text: str, *, completion_tokens: int = 5) -> dict:
    return {
        "http_status": 200,
        "finish_reason": "stop",
        "assistant_text": text,
        "provider_reasoning_text": "",
        "request_error": None,
        "usage_reported": {"completion_tokens": completion_tokens},
    }


def test_validated_turn_is_cached_only_after_validator_passes(tmp_path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    backbone = resolve_model_ref("xiaomimimo/mimo-v2.5")
    def validator(text, _reasoning):
        return {"final_answer": text}

    first = execute_cached_turn(
        backbone=backbone,
        provider=object(),  # type: ignore[arg-type]
        cache=cache,
        throttle=None,
        messages=[{"role": "user", "content": "same"}],
        temperature=0.7,
        top_p=1.0,
        seed=42,
        validator=validator,
        max_tokens=65_536,
        request_executor=lambda *_args: _response("A"),
    )
    second = execute_cached_turn(
        backbone=backbone,
        provider=object(),  # type: ignore[arg-type]
        cache=cache,
        throttle=None,
        messages=[{"role": "user", "content": "same"}],
        temperature=0.7,
        top_p=1.0,
        seed=42,
        validator=validator,
        max_tokens=65_536,
        request_executor=lambda *_args: (_ for _ in ()).throw(AssertionError("network should not run")),
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.usage["completion_tokens"] == 5
    cache.close()


def test_validator_failure_never_enters_cache(tmp_path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    backbone = resolve_model_ref("xiaomimimo/mimo-v2.5")

    result = execute_cached_turn(
        backbone=backbone,
        provider=object(),  # type: ignore[arg-type]
        cache=cache,
        throttle=None,
        messages=[{"role": "user", "content": "invalid"}],
        temperature=0.7,
        top_p=1.0,
        seed=42,
        validator=lambda *_args: (_ for _ in ()).throw(ValueError("invalid")),
        request_executor=lambda *_args: _response("bad"),
    )
    assert result.output_status == "schema_fail"
    assert cache.get(result.cache_key) is None
    cache.close()


def test_cached_response_that_exceeds_current_cap_is_a_miss(tmp_path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    backbone = resolve_model_ref("xiaomimimo/mimo-v2.5")
    def validator(text, _reasoning):
        return {"final_answer": text}
    common = {
        "backbone": backbone,
        "provider": object(),
        "cache": cache,
        "throttle": None,
        "messages": [{"role": "user", "content": "long"}],
        "temperature": 0.7,
        "top_p": 1.0,
        "seed": 42,
        "validator": validator,
    }
    execute_cached_turn(
        **common,
        max_tokens=65_536,
        request_executor=lambda *_args: _response("long-answer", completion_tokens=20_000),
    )
    network_calls = 0

    def shorter(*_args):
        nonlocal network_calls
        network_calls += 1
        return _response("short-answer", completion_tokens=100)

    result = execute_cached_turn(
        **common,
        max_tokens=16_384,
        request_executor=shorter,
    )
    assert result.cache_hit is False
    assert network_calls == 1
    cache.close()
