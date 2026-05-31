"""缓存分片目录归并测试。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research_experiments.core.execution.cache import CachedResponse, RequestCache, json_dump, resolve_cache_shard_path
from research_experiments.workspace.cache_layout_migration import (
    benchmark_cache_namespace_redirects,
    normalize_cache_layout,
)


def test_benchmark_cache_namespace_redirects_include_core_benchmarks() -> None:
    redirects = benchmark_cache_namespace_redirects()
    assert redirects["gsm8k/test"] == "gsm8k"
    assert redirects["hotpotqa/validation-distractor"] == "hotpotqa"
    assert redirects["strategyqa/dev"] == "strategyqa"


def test_normalize_cache_layout_merges_legacy_shard_into_canonical_shard(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    source_path = resolve_cache_shard_path(
        cache_root,
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="gsm8k/test",
    )
    target_path = resolve_cache_shard_path(
        cache_root,
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="gsm8k",
    )

    source_cache = RequestCache(source_path)
    source_cache.put(
        CachedResponse(
            cache_key="legacy-key",
            payload_json=json_dump({"request": "legacy"}),
            response_json=json_dump({"ok": True, "source": "legacy"}),
            http_status=200,
            latency_ms=1.0,
            provider_request_id="req-legacy",
        )
    )
    source_cache.close()

    target_cache = RequestCache(target_path)
    target_cache.put(
        CachedResponse(
            cache_key="canonical-key",
            payload_json=json_dump({"request": "canonical"}),
            response_json=json_dump({"ok": True, "source": "canonical"}),
            http_status=200,
            latency_ms=1.0,
            provider_request_id="req-canonical",
        )
    )
    target_cache.close()

    preview = normalize_cache_layout(cache_root, apply=False)
    assert preview["candidate_count"] == 1
    assert preview["candidates"][0]["source_dataset"] == "gsm8k/test"
    assert preview["candidates"][0]["target_dataset"] == "gsm8k"

    result = normalize_cache_layout(cache_root, apply=True)
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["new_rows_added"] == 1
    assert not source_path.exists()
    assert target_path.exists()

    connection = sqlite3.connect(target_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM requests").fetchone()
    finally:
        connection.close()
    assert row == (2,)
