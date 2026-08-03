from __future__ import annotations

import json
import tempfile
from pathlib import Path

from research_experiments.core.execution.cache import CachedResponse, RequestCacheRouter, json_dump
from research_experiments.workspace.hf import (
    CACHE_HASH_FILENAME,
    pull_cache_from_hub,
    push_cache_to_hub,
    resolve_local_cache_hash,
)
from research_experiments.workspace.hf.cache import _compress_zstd_file, _export_consistent_sqlite_snapshot


def test_resolve_local_cache_hash_generates_and_refreshes_sidecar(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
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
            completion_tokens=1,
        )
    )
    first_sqlite = cache.db_path
    router.close()

    first_hash = resolve_local_cache_hash(first_sqlite)
    first_sidecar = json.loads((first_sqlite.parent / CACHE_HASH_FILENAME).read_text(encoding="utf-8"))
    first_key = next(iter(first_sidecar["hash_by_mtime_ns"]))

    router = RequestCacheRouter(cache_root)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="strategyqa/dev",
    )
    cache.put(
        CachedResponse(
            cache_key="def",
            payload_json=json_dump({"request": 2}),
            response_json=json_dump({"ok": True}),
            completion_tokens=1,
        )
    )
    router.close()

    second_hash = resolve_local_cache_hash(first_sqlite)
    second_sidecar = json.loads((first_sqlite.parent / CACHE_HASH_FILENAME).read_text(encoding="utf-8"))
    second_key = next(iter(second_sidecar["hash_by_mtime_ns"]))

    assert first_hash != second_hash
    assert first_key != second_key
    assert second_sidecar["hash_by_mtime_ns"] == {second_key: second_hash}


def test_push_cache_to_hub_skips_unchanged_shards_against_remote_manifest(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
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
            completion_tokens=1,
        )
    )
    router.close()

    sqlite_path = cache.db_path
    sqlite_sha256, sqlite_size_bytes = _snapshot_hash_and_size(sqlite_path)
    relative_dir = sqlite_path.parent.relative_to(cache_root).as_posix()
    remote_manifest = {
        "schema_version": 1,
        "generated_at": "2026-06-03T10:00:00+00:00",
        "shards": [
            {
                "relative_dir": relative_dir,
                "sqlite_sha256": sqlite_sha256,
                "compressed_name": "requests.sqlite.zst",
                "sqlite_size_bytes": sqlite_size_bytes,
                "compressed_size_bytes": 123,
                "published_at": "2026-06-03T10:00:00+00:00",
            }
        ],
    }
    upload_calls: list[str] = []

    class FakeApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            return None

        def upload_file(self, **kwargs) -> None:
            upload_calls.append(kwargs["path_in_repo"])

        def delete_file(self, **kwargs) -> None:
            raise AssertionError("delete_file should not be called for unchanged shards")

    monkeypatch.setattr("research_experiments.workspace.hf.cache.download_repo_manifest", lambda **kwargs: remote_manifest)
    monkeypatch.setattr("research_experiments.workspace.hf.cache.HfApi", FakeApi)

    payload = push_cache_to_hub(cache_root, repo_id="owner/research-cache", create_repo=False)

    assert payload["published"] is False
    assert payload["published_shard_count"] == 0
    assert payload["skipped_shard_count"] == 1
    assert upload_calls == []


def test_pull_cache_from_hub_skips_matching_local_shards(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
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
            completion_tokens=1,
        )
    )
    router.close()

    sqlite_path = cache.db_path
    sqlite_sha256 = resolve_local_cache_hash(sqlite_path)
    relative_dir = sqlite_path.parent.relative_to(cache_root).as_posix()
    remote_manifest = {
        "schema_version": 1,
        "generated_at": "2026-06-03T10:00:00+00:00",
        "shards": [
            {
                "relative_dir": relative_dir,
                "sqlite_sha256": sqlite_sha256,
                "compressed_name": "requests.sqlite.zst",
                "sqlite_size_bytes": 1,
                "compressed_size_bytes": 1,
                "published_at": "2026-06-03T10:00:00+00:00",
            }
        ],
    }
    download_calls: list[str] = []

    monkeypatch.setattr("research_experiments.workspace.hf.cache.download_repo_manifest", lambda **kwargs: remote_manifest)
    monkeypatch.setattr(
        "research_experiments.workspace.hf.cache.hf_hub_download",
        lambda **kwargs: download_calls.append(kwargs["filename"]),
    )

    payload = pull_cache_from_hub(cache_root, repo_id="owner/research-cache")

    assert payload["fetched_shard_count"] == 0
    assert payload["skipped_shard_count"] == 1
    assert download_calls == []


def test_pull_cache_from_hub_restores_sqlite_and_rewrites_sidecar(monkeypatch, tmp_path: Path) -> None:
    source_cache_root = tmp_path / "source-cache"
    target_cache_root = tmp_path / "target-cache"
    router = RequestCacheRouter(source_cache_root)
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
            completion_tokens=1,
        )
    )
    router.close()

    sqlite_path = cache.db_path
    sqlite_sha256, sqlite_size_bytes = _snapshot_hash_and_size(sqlite_path)
    relative_dir = sqlite_path.parent.relative_to(source_cache_root).as_posix()
    remote_manifest = {
        "schema_version": 1,
        "generated_at": "2026-06-03T10:00:00+00:00",
        "shards": [
            {
                "relative_dir": relative_dir,
                "sqlite_sha256": sqlite_sha256,
                "compressed_name": "requests.sqlite.zst",
                "sqlite_size_bytes": sqlite_size_bytes,
                "compressed_size_bytes": 0,
                "published_at": "2026-06-03T10:00:00+00:00",
            }
        ],
    }

    with tempfile.TemporaryDirectory(prefix="research-hf-cache-remote-") as temp_dir:
        temp_root = Path(temp_dir)
        snapshot_path = temp_root / "requests.snapshot.sqlite"
        _export_consistent_sqlite_snapshot(sqlite_path, snapshot_path)
        compressed_path = temp_root / "requests.sqlite.zst"
        _compress_zstd_file(snapshot_path, compressed_path)

        monkeypatch.setattr("research_experiments.workspace.hf.cache.download_repo_manifest", lambda **kwargs: remote_manifest)
        monkeypatch.setattr(
            "research_experiments.workspace.hf.cache.hf_hub_download",
            lambda **kwargs: compressed_path.as_posix(),
        )

        payload = pull_cache_from_hub(target_cache_root, repo_id="owner/research-cache")

    restored_sqlite = target_cache_root / relative_dir / "requests.sqlite"
    sidecar_payload = json.loads((restored_sqlite.parent / CACHE_HASH_FILENAME).read_text(encoding="utf-8"))
    mtime_key = str(restored_sqlite.stat().st_mtime_ns)

    assert payload["fetched_shard_count"] == 1
    assert restored_sqlite.exists()
    assert sidecar_payload["hash_by_mtime_ns"] == {mtime_key: sqlite_sha256}
    assert resolve_local_cache_hash(restored_sqlite) == sqlite_sha256


def test_pull_then_push_cache_round_trip_skips_unchanged_remote_shards(monkeypatch, tmp_path: Path) -> None:
    source_cache_root = tmp_path / "source-cache"
    target_cache_root = tmp_path / "target-cache"
    router = RequestCacheRouter(source_cache_root)
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
            completion_tokens=1,
        )
    )
    router.close()

    sqlite_path = cache.db_path
    sqlite_sha256, sqlite_size_bytes = _snapshot_hash_and_size(sqlite_path)
    relative_dir = sqlite_path.parent.relative_to(source_cache_root).as_posix()
    remote_manifest = {
        "schema_version": 1,
        "generated_at": "2026-06-03T10:00:00+00:00",
        "shards": [
            {
                "relative_dir": relative_dir,
                "sqlite_sha256": sqlite_sha256,
                "compressed_name": "requests.sqlite.zst",
                "sqlite_size_bytes": sqlite_size_bytes,
                "compressed_size_bytes": 0,
                "published_at": "2026-06-03T10:00:00+00:00",
            }
        ],
    }

    upload_calls: list[str] = []

    class FakeApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            return None

        def upload_file(self, **kwargs) -> None:
            upload_calls.append(kwargs["path_in_repo"])

        def delete_file(self, **kwargs) -> None:
            raise AssertionError("delete_file should not be called for unchanged shards")

    with tempfile.TemporaryDirectory(prefix="research-hf-cache-remote-") as temp_dir:
        temp_root = Path(temp_dir)
        snapshot_path = temp_root / "requests.snapshot.sqlite"
        _export_consistent_sqlite_snapshot(sqlite_path, snapshot_path)
        compressed_path = temp_root / "requests.sqlite.zst"
        _compress_zstd_file(snapshot_path, compressed_path)

        monkeypatch.setattr("research_experiments.workspace.hf.cache.download_repo_manifest", lambda **kwargs: remote_manifest)
        monkeypatch.setattr(
            "research_experiments.workspace.hf.cache.hf_hub_download",
            lambda **kwargs: compressed_path.as_posix(),
        )

        pull_payload = pull_cache_from_hub(target_cache_root, repo_id="owner/research-cache")

    monkeypatch.setattr("research_experiments.workspace.hf.cache.download_repo_manifest", lambda **kwargs: remote_manifest)
    monkeypatch.setattr("research_experiments.workspace.hf.cache.HfApi", FakeApi)

    push_payload = push_cache_to_hub(target_cache_root, repo_id="owner/research-cache", create_repo=False)

    assert pull_payload["fetched_shard_count"] == 1
    assert push_payload["published"] is False
    assert push_payload["published_shard_count"] == 0
    assert push_payload["skipped_shard_count"] == 1
    assert upload_calls == []


def _snapshot_hash_and_size(sqlite_path: Path) -> tuple[str, int]:
    with tempfile.TemporaryDirectory(prefix="research-hf-cache-hash-") as temp_dir:
        snapshot_path = Path(temp_dir) / "requests.snapshot.sqlite"
        _export_consistent_sqlite_snapshot(sqlite_path, snapshot_path)
        return resolve_local_cache_hash(sqlite_path), snapshot_path.stat().st_size
