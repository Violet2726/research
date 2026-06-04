
"""覆盖请求缓存、分片路由与缓存摘要行为。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research_experiments.core.execution.cache import (
    CachedResponse,
    RequestCache,
    RequestCacheRouter,
    build_request_cache_key,
    cache_successful_response,
    inspect_cache_shard,
    json_dump,
    migrate_cache_keys,
    normalize_payload_for_cache_key,
    resolve_cache_shard_path,
    summarize_cache_root,
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

    shard = inspect_cache_shard(resolved, tmp_path)
    assert shard.dataset == "hotpotqa/validation"

def test_summarize_cache_root_collects_provider_stats(tmp_path: Path) -> None:
    router = RequestCacheRouter(tmp_path)
    deepseek_cache = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    dashscope_cache = router.for_request_target(
        provider="dashscope",
        request_model="qwen-turbo",
        dataset="strategyqa",
    )
    deepseek_cache.put(
        CachedResponse(
            cache_key="a",
            payload_json=json_dump({"request": 1}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=10.0,
            provider_request_id="req_a",
        )
    )
    deepseek_cache.put(
        CachedResponse(
            cache_key="b",
            payload_json=json_dump({"request": 2}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=11.0,
            provider_request_id="req_b",
        )
    )
    dashscope_cache.put(
        CachedResponse(
            cache_key="c",
            payload_json=json_dump({"request": 3}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=12.0,
            provider_request_id="req_c",
        )
    )
    router.close()

    summary = summarize_cache_root(tmp_path)
    assert summary.shard_count == 2
    assert summary.provider_count == 2
    assert summary.total_request_count == 3
    assert summary.total_size_bytes > 0
    assert {item.provider for item in summary.providers} == {"dashscope", "deepseek"}
    assert {item.model_count for item in summary.providers} == {1}
    assert {item.dataset_count for item in summary.providers} == {1}

    shard = inspect_cache_shard(resolve_cache_shard_path(
        tmp_path,
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    ), tmp_path)
    assert shard.exists is True
    assert shard.request_count == 2
    assert shard.provider == "deepseek"
    assert shard.request_model == "deepseek-v4-flash"
    assert shard.dataset == "gsm8k"


def test_migrate_cache_keys_rewrites_legacy_max_tokens_keys(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    shard_path = resolve_cache_shard_path(
        cache_root,
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="gsm8k",
    )
    cache = RequestCache(shard_path)
    payload = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "seed": 42,
    }
    legacy_key = json_dump(
        {
            "provider": "xiaomimimo",
            "request_model": "mimo-v2.5",
            "payload": payload,
        }
    )
    legacy_key = hashlib.sha256(legacy_key.encode("utf-8")).hexdigest()
    canonical_key = build_request_cache_key(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        payload=payload,
    )
    cache.put(
        CachedResponse(
            cache_key=legacy_key,
            payload_json=json_dump(payload),
            response_json=json_dump(
                {
                    "assistant_text": '{"final_answer":"4"}',
                    "provider_reasoning_text": "",
                    "finish_reason": "stop",
                    "usage_reported": {"total_tokens": 16},
                }
            ),
            http_status=200,
            latency_ms=10.0,
            provider_request_id="req_1",
        )
    )
    cache.close()

    result = migrate_cache_keys(cache_root)

    assert result["changed_shard_count"] == 1
    assert result["rewritten_request_count"] == 1
    assert result["deduplicated_request_count"] == 0
    migrated = RequestCache(shard_path)
    assert migrated.get(legacy_key) is None
    assert migrated.get(canonical_key) is not None
    migrated.close()


def test_migrate_cache_keys_deduplicates_by_preferring_complete_response(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    shard_path = resolve_cache_shard_path(
        cache_root,
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="gsm8k",
    )
    cache = RequestCache(shard_path)
    base_payload = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    short_payload = {**base_payload, "max_tokens": 64}
    full_payload = {**base_payload, "max_tokens": 512}

    def _legacy_key(payload: dict[str, object]) -> str:
        fingerprint = {
            "provider": "xiaomimimo",
            "request_model": "mimo-v2.5",
            "payload": payload,
        }
        return hashlib.sha256(json_dump(fingerprint).encode("utf-8")).hexdigest()

    cache.put(
        CachedResponse(
            cache_key=_legacy_key(short_payload),
            payload_json=json_dump(short_payload),
            response_json=json_dump(
                {
                    "assistant_text": '{"final_answer":"',
                    "provider_reasoning_text": "",
                    "finish_reason": "length",
                    "usage_reported": {"total_tokens": 64},
                }
            ),
            http_status=200,
            latency_ms=10.0,
            provider_request_id="req_short",
        )
    )
    cache.put(
        CachedResponse(
            cache_key=_legacy_key(full_payload),
            payload_json=json_dump(full_payload),
            response_json=json_dump(
                {
                    "assistant_text": '{"final_answer":"4","reasoning":"basic arithmetic"}',
                    "provider_reasoning_text": "",
                    "finish_reason": "stop",
                    "usage_reported": {"total_tokens": 72},
                }
            ),
            http_status=200,
            latency_ms=12.0,
            provider_request_id="req_full",
        )
    )
    cache.close()

    result = migrate_cache_keys(cache_root)
    canonical_key = build_request_cache_key(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        payload=short_payload,
    )

    assert result["changed_shard_count"] == 1
    assert result["deduplicated_request_count"] == 1
    migrated = RequestCache(shard_path)
    row = migrated.get(canonical_key)
    migrated.close()
    assert row is not None
    assert "basic arithmetic" in row.response_json

