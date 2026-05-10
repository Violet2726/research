"""cache 最新快照构建与恢复测试。"""

from __future__ import annotations

from pathlib import Path
import sqlite3

from experiment_core.foundation.cache import CachedResponse, RequestCacheRouter, json_dump
from experiment_core.foundation.cache_snapshots import build_cache_snapshot, restore_cache_snapshot


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
