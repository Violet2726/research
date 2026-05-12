"""cache 最新快照的压缩、发布与恢复。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import json
import sqlite3
import shutil
import tempfile
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from research_experiments.core.foundation.archive_common import sha256_file
from research_experiments.core.foundation.cache import collect_cache_shard_summaries, repair_cache_shard
from research_experiments.core.foundation.workspace import auto_push_cache_snapshot_enabled, default_cache_hf_repo, workspace_layout

import zstandard as zstd


CACHE_SNAPSHOT_MANIFEST = "snapshot_manifest.json"


def push_latest_cache_snapshot(
    cache_root: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    create_repo: bool = True,
    private: bool = True,
    shard_filters: list[str] | None = None,
) -> dict[str, Any]:
    """压缩当前 cache 根目录并发布到 HF dataset repo。"""
    root = Path(cache_root)
    with tempfile.TemporaryDirectory(prefix="research-cache-publish-") as temp_dir:
        staging_root = Path(temp_dir)
        summary = build_cache_snapshot(root, staging_root=staging_root, shard_filters=shard_filters)
        api = HfApi(token=token)
        if create_repo:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=staging_root.as_posix(),
            path_in_repo="",
            commit_message=_build_cache_snapshot_commit_message(summary),
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
    shard_filters: list[str] | None = None,
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
        shard_filters=shard_filters,
    )


def pull_latest_cache_snapshot(
    target_root: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    shard_filters: list[str] | None = None,
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
        restored_shards = restore_cache_snapshot(temp_root, destination, shard_filters=shard_filters)
    return {
        "target_root": destination.as_posix(),
        "remote_repo": repo_id,
        "restored_shard_count": len(restored_shards),
        "restored_shards": restored_shards,
    }


def build_cache_snapshot(
    cache_root: str | Path,
    *,
    staging_root: str | Path,
    shard_filters: list[str] | None = None,
) -> dict[str, Any]:
    """把 cache 根目录压缩成适合远程同步的最新快照目录。"""
    root = Path(cache_root)
    stage = Path(staging_root)
    stage.mkdir(parents=True, exist_ok=True)
    normalized_filters = _normalize_shard_filters(shard_filters, base_root=root)
    shards_payload: list[dict[str, Any]] = []
    total_original_size = 0
    total_compressed_size = 0

    for shard in collect_cache_shard_summaries(root):
        if not shard.exists:
            continue
        source_path = shard.shard_path
        relative_dir = source_path.parent.relative_to(root).as_posix()
        if normalized_filters and not _matches_shard_filter(relative_dir, normalized_filters):
            continue
        target_dir = stage / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = target_dir / "requests.snapshot.sqlite"
        compressed_path = target_dir / "requests.sqlite.zst"
        snapshot_info = _export_consistent_sqlite_snapshot(source_path, snapshot_path)
        _compress_zstd_file(snapshot_path, compressed_path)
        digest = sha256_file(compressed_path)
        snapshot_size_bytes = snapshot_path.stat().st_size
        metadata = {
            "provider": shard.provider,
            "request_model": shard.request_model,
            "dataset": shard.dataset,
            "relative_dir": relative_dir,
            "source_path": source_path.relative_to(root).as_posix(),
            "request_count": snapshot_info["request_count"],
            "original_size_bytes": snapshot_size_bytes,
            "compressed_size_bytes": compressed_path.stat().st_size,
            "sha256": digest,
            "schema_version": 2,
            "snapshot_strategy": "sqlite_backup",
            "repaired_before_snapshot": snapshot_info["repaired_before_snapshot"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (target_dir / "sha256.txt").write_text(digest + "\n", encoding="utf-8")
        snapshot_path.unlink()

        shards_payload.append(
            {
                "provider": shard.provider,
                "request_model": shard.request_model,
                "dataset": shard.dataset,
                "relative_dir": relative_dir,
                "compressed_name": "requests.sqlite.zst",
                "metadata_name": "metadata.json",
                "sha256_name": "sha256.txt",
                "original_size_bytes": snapshot_size_bytes,
                "compressed_size_bytes": compressed_path.stat().st_size,
                "sha256": digest,
                "snapshot_strategy": "sqlite_backup",
            }
        )
        total_original_size += snapshot_size_bytes
        total_compressed_size += compressed_path.stat().st_size

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "semantics": "latest_only",
        "snapshot_strategy": "sqlite_backup",
        "cache_root_name": root.name,
        "shard_count": len(shards_payload),
        "total_original_size_bytes": total_original_size,
        "total_compressed_size_bytes": total_compressed_size,
        "shards": shards_payload,
    }
    (stage / CACHE_SNAPSHOT_MANIFEST).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def restore_cache_snapshot(
    snapshot_root: str | Path,
    target_root: str | Path,
    *,
    shard_filters: list[str] | None = None,
) -> tuple[str, ...]:
    """从本地快照目录恢复 requests.sqlite 分库。"""
    source = Path(snapshot_root)
    destination = Path(target_root)
    manifest = _load_json(source / CACHE_SNAPSHOT_MANIFEST)
    normalized_filters = _normalize_shard_filters(shard_filters)
    restored_shards: list[str] = []
    for row in manifest.get("shards", []):
        if not isinstance(row, dict):
            continue
        relative_dir = str(row.get("relative_dir") or "")
        if normalized_filters and not _matches_shard_filter(relative_dir, normalized_filters):
            continue
        compressed_name = str(row.get("compressed_name") or "")
        compressed_path = source / relative_dir / compressed_name
        target_sqlite = destination / relative_dir / "requests.sqlite"
        target_sqlite.parent.mkdir(parents=True, exist_ok=True)
        temp_snapshot = target_sqlite.parent / "requests.snapshot.restore.sqlite"
        _decompress_zstd_file(compressed_path, temp_snapshot)
        try:
            _validate_sqlite_snapshot(temp_snapshot)
            _install_sqlite_snapshot(temp_snapshot, target_sqlite)
        finally:
            if temp_snapshot.exists():
                temp_snapshot.unlink()
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


def _export_consistent_sqlite_snapshot(source_path: Path, snapshot_path: Path) -> dict[str, Any]:
    repaired_before_snapshot = False
    try:
        _backup_sqlite_database(source_path, snapshot_path)
    except sqlite3.DatabaseError as exc:
        if "malformed" not in str(exc).lower():
            raise
        repair_cache_shard(source_path)
        repaired_before_snapshot = True
        _backup_sqlite_database(source_path, snapshot_path)
    _validate_sqlite_snapshot(snapshot_path)
    return {
        "request_count": _read_sqlite_request_count(snapshot_path),
        "repaired_before_snapshot": repaired_before_snapshot,
    }


def _install_sqlite_snapshot(snapshot_path: Path, target_sqlite: Path) -> None:
    staged_target = target_sqlite.parent / "requests.sqlite.restore.tmp"
    _cleanup_sqlite_sidecars(staged_target)
    if staged_target.exists():
        staged_target.unlink()
    _backup_sqlite_database(snapshot_path, staged_target)
    _validate_sqlite_snapshot(staged_target)
    _cleanup_sqlite_sidecars(target_sqlite)
    try:
        staged_target.replace(target_sqlite)
    except OSError as exc:
        if staged_target.exists():
            staged_target.unlink()
        raise RuntimeError(
            f"无法安装 cache snapshot 到 {target_sqlite.as_posix()}；请先关闭正在使用该 cache 的进程。"
        ) from exc


def _backup_sqlite_database(source_path: Path, target_path: Path) -> None:
    if target_path.exists():
        target_path.unlink()
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
        target.commit()
        target.execute("PRAGMA journal_mode=WAL")
        target.commit()
    finally:
        source.close()
        target.close()


def _validate_sqlite_snapshot(snapshot_path: Path) -> None:
    connection = sqlite3.connect(snapshot_path)
    try:
        issues = [row[0] for row in connection.execute("PRAGMA integrity_check").fetchall()]
    finally:
        connection.close()
    if issues != ["ok"]:
        raise sqlite3.DatabaseError(f"SQLite snapshot integrity check failed: {issues[:3]}")


def _read_sqlite_request_count(snapshot_path: Path) -> int:
    connection = sqlite3.connect(snapshot_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM requests").fetchone()
    finally:
        connection.close()
    return int(row[0] if row is not None else 0)


def _cleanup_sqlite_sidecars(sqlite_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = sqlite_path.with_name(f"{sqlite_path.name}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_cache_snapshot_commit_message(summary: dict[str, Any]) -> str:
    shard_count = int(summary.get("shard_count") or 0)
    original_size = _format_mib(float(summary.get("total_original_size_bytes") or 0.0))
    compressed_size = _format_mib(float(summary.get("total_compressed_size_bytes") or 0.0))
    generated_at = str(summary.get("generated_at") or "")
    return f"更新 cache 快照 | {shard_count} shards | {original_size} -> {compressed_size} | {generated_at}"


def _format_mib(size_bytes: float) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MiB"


def _normalize_shard_filters(
    shard_filters: list[str] | None,
    *,
    base_root: Path | None = None,
) -> tuple[str, ...]:
    if not shard_filters:
        return ()
    normalized: list[str] = []
    for item in shard_filters:
        value = str(item).strip()
        if not value:
            continue
        candidate = Path(value)
        if base_root is not None and candidate.is_absolute():
            relative = candidate.resolve().relative_to(base_root.resolve()).as_posix()
        elif base_root is not None and (base_root / candidate).exists():
            relative = (base_root / candidate).resolve().relative_to(base_root.resolve()).as_posix()
        else:
            relative = PurePosixPath(value.replace("\\", "/")).as_posix().lstrip("./")
        if relative.endswith("/requests.sqlite"):
            relative = relative[: -len("/requests.sqlite")]
        if relative.endswith("/requests.sqlite.zst"):
            relative = relative[: -len("/requests.sqlite.zst")]
        normalized_path = PurePosixPath(relative).as_posix().strip("/")
        if not normalized_path:
            continue
        if ".." in PurePosixPath(normalized_path).parts:
            raise ValueError(f"cache shard filter must stay within cache root: {item}")
        normalized.append(normalized_path)
    return tuple(sorted(dict.fromkeys(normalized)))


def _matches_shard_filter(relative_dir: str, normalized_filters: tuple[str, ...]) -> bool:
    return any(relative_dir == item or relative_dir.startswith(f"{item}/") for item in normalized_filters)
