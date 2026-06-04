
"""覆盖请求缓存、分片路由与缓存摘要行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_experiments.core.execution.cache import (
    CachedResponse,
    RequestCache,
    RequestCacheRouter,
    build_request_cache_key,
    cache_successful_response,
    json_dump,
    normalize_payload_for_cache_key,
    resolve_cache_shard_path,
)


def test_request_cache_round_trip(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    record = CachedResponse(
        cache_key="abc",
        payload_json=json_dump({"a": 1}),
        response_json=json_dump({"b": 2}),
        http_status=200,
        latency_ms=12.5,
        provider_request_id="req_1",
    )
    cache.put(record)
    loaded = cache.get("abc")
    cache.close()
    assert loaded == record

def test_build_request_cache_key_depends_only_on_payload() -> None:
    payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) == build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=dict(payload),
    )
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) != build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload={**payload, "temperature": 0.7},
    )


def test_build_request_cache_key_ignores_max_tokens() -> None:
    payload = {
        "model": "demo",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    assert normalize_payload_for_cache_key(payload) == {
        "model": "demo",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) == build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload={**payload, "max_tokens": 512},
    )

def test_cache_successful_response_rejects_failed_request(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    with pytest.raises(ValueError, match="must not be cached"):
        cache_successful_response(
            cache,
            cache_key="abc",
            payload={"model": "demo"},
            response_payload={
                "http_status": 500,
                "assistant_text": "",
                "provider_reasoning_text": "",
                "latency_ms": 0.0,
                "provider_request_id": "req_failed",
                "request_error": "boom",
            },
        )
    assert cache.get("abc") is None
    cache.close()

def test_request_cache_router_shards_by_provider_and_model(tmp_path: Path) -> None:
    router = RequestCacheRouter(tmp_path)
    first = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    second = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    third = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4",
        dataset="gsm8k",
    )
    router.close()

    assert first is second
    assert first.db_path != third.db_path
    assert first.db_path.name == "requests.sqlite"
    assert "deepseek-v4-flash" in first.db_path.parts
    assert "gsm8k" in first.db_path.parts

def test_resolve_cache_shard_path_matches_router(tmp_path: Path) -> None:
    router = RequestCacheRouter(tmp_path)
    cache = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    router.close()

    resolved = resolve_cache_shard_path(
        tmp_path,
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    assert cache.db_path == resolved


def test_resolve_cache_shard_path_preserves_dataset_hierarchy(tmp_path: Path) -> None:
    resolved = resolve_cache_shard_path(
        tmp_path,
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="hotpotqa/validation",
    )

    assert resolved.as_posix().endswith("providers/xiaomimimo/mimo-v2-5/hotpotqa/validation/requests.sqlite")

