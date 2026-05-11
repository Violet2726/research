"""cache 最新快照构建与恢复测试。"""

from __future__ import annotations

from pathlib import Path
import sqlite3

from research_experiments.core.foundation.cache import CachedResponse, RequestCacheRouter, json_dump
from research_experiments.core.foundation.cache_snapshots import _build_cache_snapshot_commit_message, build_cache_snapshot, restore_cache_snapshot


def test_build_and_restore_cache_snapshot(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    stage_root = tmp_path / "snapshot"
    restore_root = tmp_path / "restored"

    router = RequestCacheRouter(cache_root)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa",
    )
    cache.put(
        CachedResponse(
            cache_key="abc",
            payload_json=json_dump({"request": 1}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=1.0,
            provider_request_id="req_1",
        )
    )
    router.close()

    payload = build_cache_snapshot(cache_root, staging_root=stage_root)
    restored = restore_cache_snapshot(stage_root, restore_root)

    assert payload["shard_count"] == 1
    assert restored == ("providers/xiaomimimo/mimo-v2-5/strategyqa/requests.sqlite",)
    restored_sqlite = restore_root / restored[0]
    assert restored_sqlite.exists()

    connection = sqlite3.connect(restored_sqlite)
    try:
        row = connection.execute("SELECT COUNT(*) FROM requests").fetchone()
    finally:
        connection.close()
    assert row == (1,)


def test_build_and_restore_cache_snapshot_with_shard_filters(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    stage_root = tmp_path / "snapshot"
    restore_root = tmp_path / "restored"

    router = RequestCacheRouter(cache_root)
    strategyqa = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa",
    )
    strategyqa.put(
        CachedResponse(
            cache_key="strategyqa",
            payload_json=json_dump({"request": "strategyqa"}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=1.0,
            provider_request_id="req_strategyqa",
        )
    )
    gsm8k = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="gsm8k",
    )
    gsm8k.put(
        CachedResponse(
            cache_key="gsm8k",
            payload_json=json_dump({"request": "gsm8k"}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=1.0,
            provider_request_id="req_gsm8k",
        )
    )
    router.close()

    payload = build_cache_snapshot(
        cache_root,
        staging_root=stage_root,
        shard_filters=["providers/xiaomimimo/mimo-v2-5/strategyqa"],
    )
    restored = restore_cache_snapshot(
        stage_root,
        restore_root,
        shard_filters=["providers/xiaomimimo/mimo-v2-5/strategyqa"],
    )

    assert payload["shard_count"] == 1
    assert restored == ("providers/xiaomimimo/mimo-v2-5/strategyqa/requests.sqlite",)
    assert not (restore_root / "providers/xiaomimimo/mimo-v2-5/gsm8k/requests.sqlite").exists()


def test_build_cache_snapshot_commit_message_is_human_readable() -> None:
    message = _build_cache_snapshot_commit_message(
        {
            "shard_count": 6,
            "total_original_size_bytes": 625172480,
            "total_compressed_size_bytes": 39375870,
            "generated_at": "2026-05-10T06:00:00+00:00",
        }
    )
    assert message == "更新 cache 快照 | 6 shards | 596.21 MiB -> 37.55 MiB | 2026-05-10T06:00:00+00:00"
