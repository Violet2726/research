"""cache 最新快照构建与恢复测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from research_experiments.core.execution.cache import CachedResponse, RequestCacheRouter, json_dump
from research_experiments.workspace.cache_snapshots import (
    _build_cache_snapshot_commit_message,
    build_cache_snapshot,
    pull_latest_cache_snapshot,
    push_latest_cache_snapshot,
    restore_cache_snapshot,
)


def test_build_and_restore_cache_snapshot(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    stage_root = tmp_path / "snapshot"
    restore_root = tmp_path / "restored"

    router = RequestCacheRouter(cache_root)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa/dev",
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
    assert restored == ("providers/xiaomimimo/mimo-v2-5/strategyqa/dev/requests.sqlite",)
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
        dataset="strategyqa/dev",
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
        shard_filters=["providers/xiaomimimo/mimo-v2-5/strategyqa/dev"],
    )
    restored = restore_cache_snapshot(
        stage_root,
        restore_root,
        shard_filters=["providers/xiaomimimo/mimo-v2-5/strategyqa/dev"],
    )

    assert payload["shard_count"] == 1
    assert restored == ("providers/xiaomimimo/mimo-v2-5/strategyqa/dev/requests.sqlite",)
    assert not (restore_root / "providers/xiaomimimo/mimo-v2-5/gsm8k/requests.sqlite").exists()


def test_build_cache_snapshot_reads_consistent_live_wal_state(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    stage_root = tmp_path / "snapshot"
    restore_root = tmp_path / "restored"

    router = RequestCacheRouter(cache_root)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa/dev",
    )
    for index in range(32):
        cache.put(
            CachedResponse(
                cache_key=f"key-{index}",
                payload_json=json_dump({"request": index}),
                response_json=json_dump({"ok": True, "index": index}),
                http_status=200,
                latency_ms=1.0,
                provider_request_id=f"req_{index}",
            )
        )

    try:
        payload = build_cache_snapshot(cache_root, staging_root=stage_root)
        restored = restore_cache_snapshot(stage_root, restore_root)
    finally:
        router.close()

    assert payload["shard_count"] == 1
    restored_sqlite = restore_root / restored[0]
    connection = sqlite3.connect(restored_sqlite)
    try:
        row = connection.execute("SELECT COUNT(*) FROM requests").fetchone()
    finally:
        connection.close()
    assert row == (32,)


def test_restore_cache_snapshot_replaces_old_sidecars(tmp_path: Path) -> None:
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

    build_cache_snapshot(cache_root, staging_root=stage_root)
    target_dir = restore_root / "providers/xiaomimimo/mimo-v2-5/strategyqa"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "requests.sqlite-wal").write_text("stale", encoding="utf-8")
    (target_dir / "requests.sqlite-shm").write_text("stale", encoding="utf-8")

    restore_cache_snapshot(stage_root, restore_root)

    assert not (target_dir / "requests.sqlite-wal").exists()
    assert not (target_dir / "requests.sqlite-shm").exists()


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


def test_build_cache_snapshot_marks_unchanged_shards_against_previous_manifest(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    first_stage = tmp_path / "snapshot-first"
    second_stage = tmp_path / "snapshot-second"

    router = RequestCacheRouter(cache_root)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa/dev",
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

    first_payload = build_cache_snapshot(cache_root, staging_root=first_stage)
    second_payload = build_cache_snapshot(
        cache_root,
        staging_root=second_stage,
        previous_manifest=json.loads((first_stage / "snapshot_manifest.json").read_text(encoding="utf-8")),
    )

    shard_dir = second_stage / "providers" / "xiaomimimo" / "mimo-v2-5" / "strategyqa" / "dev"
    assert first_payload["shard_count"] == 1
    assert second_payload["uploaded_shard_count"] == 0
    assert second_payload["skipped_shard_count"] == 1
    assert second_payload["upload_required"] is False
    assert not (shard_dir / "requests.sqlite.zst").exists()


def test_push_latest_cache_snapshot_skips_upload_when_remote_manifest_matches(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    stage_root = tmp_path / "seed-remote"

    router = RequestCacheRouter(cache_root)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa/dev",
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

    build_cache_snapshot(cache_root, staging_root=stage_root)
    remote_manifest = json.loads((stage_root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    upload_calls: list[str] = []

    class FakeApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            return None

        def upload_folder(self, **kwargs) -> None:
            upload_calls.append(kwargs["folder_path"])

    monkeypatch.setattr("research_experiments.workspace.cache_snapshots._download_remote_cache_manifest", lambda **kwargs: remote_manifest)
    monkeypatch.setattr("research_experiments.workspace.cache_snapshots.HfApi", FakeApi)

    payload = push_latest_cache_snapshot(cache_root, repo_id="owner/research-cache", create_repo=False)

    assert payload["published"] is False
    assert payload["uploaded_shard_count"] == 0
    assert payload["skipped_shard_count"] == 1
    assert upload_calls == []


def test_pull_latest_cache_snapshot_skips_download_when_local_shard_already_matches(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    stage_root = tmp_path / "seed-remote"

    router = RequestCacheRouter(cache_root)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa/dev",
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

    build_cache_snapshot(cache_root, staging_root=stage_root)
    remote_manifest = json.loads((stage_root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    download_calls: list[list[str] | None] = []

    monkeypatch.setattr("research_experiments.workspace.cache_snapshots._download_remote_cache_manifest", lambda **kwargs: remote_manifest)
    monkeypatch.setattr(
        "research_experiments.workspace.cache_snapshots.snapshot_download",
        lambda **kwargs: download_calls.append(kwargs.get("allow_patterns")),
    )

    payload = pull_latest_cache_snapshot(cache_root, repo_id="owner/research-cache")

    assert payload["restored_shard_count"] == 0
    assert payload["skipped_shard_count"] == 1
    assert download_calls == []

