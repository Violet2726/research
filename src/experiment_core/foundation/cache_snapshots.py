"""cache 最新快照的压缩、发布与恢复。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import tempfile
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from experiment_core.foundation.archive_common import sha256_file
from experiment_core.foundation.cache import collect_cache_shard_summaries
from experiment_core.foundation.workspace import auto_push_cache_snapshot_enabled, default_cache_hf_repo, workspace_layout

import zstandard as zstd


CACHE_SNAPSHOT_MANIFEST = "snapshot_manifest.json"


def push_latest_cache_snapshot(
    cache_root: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    create_repo: bool = True,
    private: bool = True,
) -> dict[str, Any]:
    """压缩当前 cache 根目录并发布到 HF dataset repo。"""
    root = Path(cache_root)
    with tempfile.TemporaryDirectory(prefix="research-cache-publish-") as temp_dir:
        staging_root = Path(temp_dir)
        summary = build_cache_snapshot(root, staging_root=staging_root)
        api = HfApi(token=token)
        if create_repo:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=staging_root.as_posix(),
            path_in_repo="",
            commit_message=f"更新 cache 快照 {summary['generated_at']}",
        )
    return {
        **summary,
        "remote_repo": repo_id,
        "published": True,
        "private_repo": private,
    }


def push_cache_snapshot_if_configured(
    cache_root: str | Path | None = None,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    create_repo: bool = True,
    private: bool = True,
) -> dict[str, Any] | None:
    """按环境约定推送 cache 最新快照；未启用时返回 `None`。"""
    if not auto_push_cache_snapshot_enabled():
        return None
    resolved_repo = repo_id or default_cache_hf_repo()
    if not resolved_repo:
        return None
    resolved_cache_root = cache_root or workspace_layout().cache_root
    return push_latest_cache_snapshot(
        resolved_cache_root,
        repo_id=resolved_repo,
        token=token,
        create_repo=create_repo,
        private=private,
    )


def pull_latest_cache_snapshot(
    target_root: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
) -> dict[str, Any]:
    """从 HF dataset repo 拉取最新 cache 快照，并恢复 sqlite 分库。"""
    destination = Path(target_root)
    with tempfile.TemporaryDirectory(prefix="research-cache-fetch-") as temp_dir:
        temp_root = Path(temp_dir)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=temp_root,
            token=token,
        )
        restored_shards = restore_cache_snapshot(temp_root, destination)
    return {
        "target_root": destination.as_posix(),
        "remote_repo": repo_id,
        "restored_shard_count": len(restored_shards),
        "restored_shards": restored_shards,
    }


def build_cache_snapshot(cache_root: str | Path, *, staging_root: str | Path) -> dict[str, Any]:
    """把 cache 根目录压缩成适合远程同步的最新快照目录。"""
    root = Path(cache_root)
    stage = Path(staging_root)
    stage.mkdir(parents=True, exist_ok=True)
    shards_payload: list[dict[str, Any]] = []
    total_original_size = 0
    total_compressed_size = 0

    for shard in collect_cache_shard_summaries(root):
        if not shard.exists:
            continue
        source_path = shard.shard_path
        relative_dir = source_path.parent.relative_to(root).as_posix()
        target_dir = stage / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        compressed_path = target_dir / "requests.sqlite.zst"
        _compress_zstd_file(source_path, compressed_path)
        digest = sha256_file(compressed_path)
        metadata = {
            "provider": shard.provider,
            "request_model": shard.request_model,
            "dataset": shard.dataset,
            "relative_dir": relative_dir,
            "source_path": source_path.relative_to(root).as_posix(),
            "request_count": shard.request_count,
            "original_size_bytes": shard.file_size_bytes,
            "compressed_size_bytes": compressed_path.stat().st_size,
            "sha256": digest,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (target_dir / "sha256.txt").write_text(digest + "\n", encoding="utf-8")

        shards_payload.append(
            {
                "provider": shard.provider,
                "request_model": shard.request_model,
                "dataset": shard.dataset,
                "relative_dir": relative_dir,
                "compressed_name": "requests.sqlite.zst",
                "metadata_name": "metadata.json",
                "sha256_name": "sha256.txt",
                "original_size_bytes": shard.file_size_bytes,
                "compressed_size_bytes": compressed_path.stat().st_size,
                "sha256": digest,
            }
        )
        total_original_size += shard.file_size_bytes
        total_compressed_size += compressed_path.stat().st_size

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "semantics": "latest_only",
        "cache_root_name": root.name,
        "shard_count": len(shards_payload),
        "total_original_size_bytes": total_original_size,
        "total_compressed_size_bytes": total_compressed_size,
        "shards": shards_payload,
    }
    (stage / CACHE_SNAPSHOT_MANIFEST).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def restore_cache_snapshot(snapshot_root: str | Path, target_root: str | Path) -> tuple[str, ...]:
    """从本地快照目录恢复 requests.sqlite 分库。"""
    source = Path(snapshot_root)
    destination = Path(target_root)
    manifest = _load_json(source / CACHE_SNAPSHOT_MANIFEST)
    restored_shards: list[str] = []
    for row in manifest.get("shards", []):
        if not isinstance(row, dict):
            continue
        relative_dir = str(row.get("relative_dir") or "")
        compressed_name = str(row.get("compressed_name") or "")
        compressed_path = source / relative_dir / compressed_name
        target_sqlite = destination / relative_dir / "requests.sqlite"
        target_sqlite.parent.mkdir(parents=True, exist_ok=True)
        _decompress_zstd_file(compressed_path, target_sqlite)
        restored_shards.append(target_sqlite.relative_to(destination).as_posix())
    return tuple(restored_shards)


def _compress_zstd_file(source_path: Path, target_path: Path) -> None:
    cctx = zstd.ZstdCompressor(level=10)
    with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle:
        with cctx.stream_writer(target_handle) as compressed_handle:
            shutil.copyfileobj(source_handle, compressed_handle)


def _decompress_zstd_file(source_path: Path, target_path: Path) -> None:
    dctx = zstd.ZstdDecompressor()
    with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle:
        with dctx.stream_reader(source_handle) as decompressed_handle:
            shutil.copyfileobj(decompressed_handle, target_handle)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
