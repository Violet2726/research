
"""覆盖请求缓存、分片路由与缓存摘要行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_experiments.core.execution.cache import (
    CACHE_KEY_POLICY_VERSION,
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
        completion_tokens=12,
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


def test_build_request_cache_key_tracks_generation_fields_except_completion_cap() -> None:
    payload = {
        "model": "demo",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    assert normalize_payload_for_cache_key(payload) == payload
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) != build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload={**payload, "seed": 42},
    )
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) != build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload={**payload, "response_format": {"type": "json_object"}},
    )
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) == build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload={**payload, "max_completion_tokens": 65_536},
    )
    assert normalize_payload_for_cache_key({**payload, "max_tokens": 2_048}) == payload
    assert CACHE_KEY_POLICY_VERSION == "request_identity_without_completion_cap_v2"

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


@pytest.mark.parametrize("finish_reason", ["length", "repetition_truncation", "", "tool_calls"])
def test_cache_successful_response_rejects_completion_boundary_results(
    tmp_path: Path,
    finish_reason: str,
) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    with pytest.raises(ValueError, match="must not be cached"):
        cache_successful_response(
            cache,
            cache_key="abc",
            payload={"model": "demo"},
            response_payload={
                "http_status": 200,
                "finish_reason": finish_reason,
                "assistant_text": "REASONING: x\nFINAL_ANSWER: y",
                "latency_ms": 0.0,
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


def test_global_cache_path_and_completion_cap_does_not_change_key(tmp_path: Path) -> None:
    payload = {"model": "mimo-v2.5", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 2048}
    changed_cap = {**payload, "max_completion_tokens": 16384}
    assert build_request_cache_key(provider="xiaomimimo", request_model="mimo-v2.5", payload=payload) == build_request_cache_key(
        provider="xiaomimimo", request_model="mimo-v2.5", payload=changed_cap
    )
    default_path = resolve_cache_shard_path(tmp_path, provider="xiaomimimo", request_model="mimo-v2.5", dataset="bbeh")
    assert "namespaces" not in default_path.parts
    assert default_path.as_posix().endswith("providers/xiaomimimo/mimo-v2-5/bbeh/requests.sqlite")

