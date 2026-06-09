"""重写后的 Hugging Face cache 同步服务。"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_download

from research_experiments.core.io import write_json
from research_experiments.workspace.archive_utils import sha256_file
from research_experiments.workspace.hf.common import (
    CACHE_HASH_FILENAME,
    CACHE_MANIFEST_FILENAME,
    HF_SYNC_SCHEMA_VERSION,
    download_repo_manifest,
    iso_utc_now,
    load_json_if_exists,
    resolve_cache_repo_id,
    run_hf_request,
    upload_manifest,
)
from research_experiments.workspace.layout import (
    auto_push_cache_snapshot_enabled,
    workspace_layout,
)


def push_cache_to_hub(
    cache_root: str | Path | None = None,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    create_repo: bool = True,
    private: bool = True,
) -> dict[str, Any]:
    """Push the full local cache workspace to Hugging Face."""

    resolved_cache_root = Path(cache_root) if cache_root is not None else workspace_layout().cache_root
    resolved_repo_id = resolve_cache_repo_id(repo_id)
    api = HfApi(token=token)
    remote_manifest = download_repo_manifest(
        repo_id=resolved_repo_id,
        filename=CACHE_MANIFEST_FILENAME,
        token=token,
        missing_ok=True,
    )
    remote_rows = _index_cache_rows(remote_manifest)
    manifest_rows: list[dict[str, Any]] = []
    uploaded_shards: list[str] = []
    skipped_shards: list[str] = []

    with tempfile.TemporaryDirectory(prefix="research-hf-cache-push-") as temp_dir:
        stage_root = Path(temp_dir)
        for sqlite_path in _iter_cache_sqlite_paths(resolved_cache_root):
            relative_dir = sqlite_path.parent.relative_to(resolved_cache_root).as_posix()
            remote_row = remote_rows.get(relative_dir)
            row, upload_path = _build_cache_row(
                sqlite_path,
                relative_dir=relative_dir,
                remote_row=remote_row,
                stage_root=stage_root,
            )
            manifest_rows.append(row)
            if upload_path is None:
                skipped_shards.append(relative_dir)
                continue
            if create_repo:
                run_hf_request(
                    lambda: api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=private, exist_ok=True)
                )
                create_repo = False
            run_hf_request(
                lambda upload_path=upload_path, relative_dir=relative_dir, compressed_name=str(row["compressed_name"]): api.upload_file(
                    path_or_fileobj=upload_path,
                    path_in_repo=f"{relative_dir}/{compressed_name}",
                    repo_id=resolved_repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Update cache shard {relative_dir}",
                )
            )
            uploaded_shards.append(relative_dir)

    current_dirs = {row["relative_dir"] for row in manifest_rows}
    deleted_shards: list[str] = []
    for relative_dir, remote_row in sorted(remote_rows.items()):
        if relative_dir in current_dirs:
            continue
        compressed_name = str(remote_row.get("compressed_name") or "requests.sqlite.zst")
        try:
            run_hf_request(
                lambda relative_dir=relative_dir, compressed_name=compressed_name: api.delete_file(
                    path_in_repo=f"{relative_dir}/{compressed_name}",
                    repo_id=resolved_repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Delete stale cache shard {relative_dir}",
                )
            )
            deleted_shards.append(relative_dir)
        except Exception:
            continue

    manifest_payload = {
        "schema_version": HF_SYNC_SCHEMA_VERSION,
        "generated_at": iso_utc_now(),
        "shards": sorted(manifest_rows, key=lambda row: str(row["relative_dir"])),
    }
    if uploaded_shards or deleted_shards or _cache_manifest_signature(remote_manifest) != _cache_manifest_signature(manifest_payload):
        if create_repo:
            run_hf_request(
                lambda: api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=private, exist_ok=True)
            )
        upload_manifest(
            api,
            repo_id=resolved_repo_id,
            filename=CACHE_MANIFEST_FILENAME,
            payload=manifest_payload,
            token=token,
            commit_message="Update cache manifest",
        )
        published = True
    else:
        published = False

    return {
        "cache_root": resolved_cache_root.as_posix(),
        "remote_repo": resolved_repo_id,
        "published": published,
        "published_shard_count": len(uploaded_shards),
        "published_shards": uploaded_shards,
        "skipped_shard_count": len(skipped_shards),
        "skipped_shards": skipped_shards,
        "deleted_shard_count": len(deleted_shards),
        "deleted_shards": deleted_shards,
        "manifest_shard_count": len(manifest_rows),
    }


def pull_cache_from_hub(
    cache_root: str | Path | None = None,
    *,
    repo_id: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Pull the full remote cache workspace from Hugging Face."""

    resolved_cache_root = Path(cache_root) if cache_root is not None else workspace_layout().cache_root
    resolved_repo_id = resolve_cache_repo_id(repo_id)
    manifest = download_repo_manifest(
        repo_id=resolved_repo_id,
        filename=CACHE_MANIFEST_FILENAME,
        token=token,
        missing_ok=False,
    )
    fetched_shards: list[str] = []
    skipped_shards: list[str] = []

    for row in sorted(manifest.get("shards", []), key=lambda item: str(item.get("relative_dir") or "")):
        if not isinstance(row, dict):
            continue
        relative_dir = str(row.get("relative_dir") or "").strip()
        if not relative_dir:
            continue
        sqlite_sha256 = str(row.get("sqlite_sha256") or "").strip()
        if not sqlite_sha256:
            continue
        target_sqlite = resolved_cache_root / relative_dir / "requests.sqlite"
        local_hash = _resolve_local_cache_hash(target_sqlite) if target_sqlite.exists() else None
        if local_hash == sqlite_sha256:
            skipped_shards.append(relative_dir)
            continue

        compressed_name = str(row.get("compressed_name") or "requests.sqlite.zst")
        compressed_path = Path(
            run_hf_request(
                lambda relative_dir=relative_dir, compressed_name=compressed_name: hf_hub_download(
                    repo_id=resolved_repo_id,
                    repo_type="dataset",
                    filename=f"{relative_dir}/{compressed_name}",
                    token=token,
                )
            )
        )
        _restore_sqlite_from_archive(compressed_path, target_sqlite)
        _write_cache_hash_sidecar(target_sqlite, sqlite_sha256)
        fetched_shards.append(relative_dir)

    return {
        "cache_root": resolved_cache_root.as_posix(),
        "remote_repo": resolved_repo_id,
        "fetched_shard_count": len(fetched_shards),
        "fetched_shards": fetched_shards,
        "skipped_shard_count": len(skipped_shards),
        "skipped_shards": skipped_shards,
    }


def push_cache_if_configured(
    cache_root: str | Path | None = None,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    create_repo: bool = True,
    private: bool = True,
) -> dict[str, Any] | None:
    """Push cache when auto-publish is enabled."""

    if not auto_push_cache_snapshot_enabled():
        return None
    return push_cache_to_hub(
        cache_root,
        repo_id=repo_id,
        token=token,
        create_repo=create_repo,
        private=private,
    )


def resolve_local_cache_hash(sqlite_path: str | Path) -> str:
    """Expose local cache hash resolution for tests and callers."""

    return _resolve_local_cache_hash(Path(sqlite_path))


def _iter_cache_sqlite_paths(cache_root: Path) -> tuple[Path, ...]:
    return tuple(sorted(cache_root.rglob("requests.sqlite")))


def _build_cache_row(
    sqlite_path: Path,
    *,
    relative_dir: str,
    remote_row: dict[str, Any] | None,
    stage_root: Path,
) -> tuple[dict[str, Any], Path | None]:
    remote_hash = str((remote_row or {}).get("sqlite_sha256") or "").strip()
    cached_hash = _load_cached_hash_for_current_mtime(sqlite_path)
    if remote_hash and cached_hash == remote_hash:
        return (
            {
                "relative_dir": relative_dir,
                "sqlite_sha256": remote_hash,
                "compressed_name": str((remote_row or {}).get("compressed_name") or "requests.sqlite.zst"),
                "sqlite_size_bytes": int((remote_row or {}).get("sqlite_size_bytes") or 0),
                "compressed_size_bytes": int((remote_row or {}).get("compressed_size_bytes") or 0),
                "published_at": str((remote_row or {}).get("published_at") or ""),
            },
            None,
        )

    target_dir = stage_root / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = target_dir / "requests.snapshot.sqlite"
    _export_consistent_sqlite_snapshot(sqlite_path, snapshot_path)
    sqlite_sha256 = sha256_file(snapshot_path)
    sqlite_size_bytes = snapshot_path.stat().st_size
    _write_cache_hash_sidecar(sqlite_path, sqlite_sha256)

    if remote_hash == sqlite_sha256:
        snapshot_path.unlink()
        return (
            {
                "relative_dir": relative_dir,
                "sqlite_sha256": sqlite_sha256,
                "compressed_name": str((remote_row or {}).get("compressed_name") or "requests.sqlite.zst"),
                "sqlite_size_bytes": sqlite_size_bytes,
                "compressed_size_bytes": int((remote_row or {}).get("compressed_size_bytes") or 0),
                "published_at": str((remote_row or {}).get("published_at") or ""),
            },
            None,
        )

    compressed_path = target_dir / "requests.sqlite.zst"
    _compress_zstd_file(snapshot_path, compressed_path)
    snapshot_path.unlink()
    return (
        {
            "relative_dir": relative_dir,
            "sqlite_sha256": sqlite_sha256,
            "compressed_name": "requests.sqlite.zst",
            "sqlite_size_bytes": sqlite_size_bytes,
            "compressed_size_bytes": compressed_path.stat().st_size,
            "published_at": iso_utc_now(),
        },
        compressed_path,
    )


def _resolve_local_cache_hash(sqlite_path: Path) -> str:
    cached_hash = _load_cached_hash_for_current_mtime(sqlite_path)
    if cached_hash is not None:
        return cached_hash

    with tempfile.TemporaryDirectory(prefix="research-hf-cache-hash-") as temp_dir:
        snapshot_path = Path(temp_dir) / "requests.snapshot.sqlite"
        _export_consistent_sqlite_snapshot(sqlite_path, snapshot_path)
        sqlite_sha256 = sha256_file(snapshot_path)
    _write_cache_hash_sidecar(sqlite_path, sqlite_sha256)
    return sqlite_sha256


def _write_cache_hash_sidecar(sqlite_path: Path, sqlite_sha256: str) -> None:
    payload = {
        "schema_version": HF_SYNC_SCHEMA_VERSION,
        "source_file": sqlite_path.name,
        "hash_by_mtime_ns": {
            str(sqlite_path.stat().st_mtime_ns): sqlite_sha256,
        },
        "updated_at": iso_utc_now(),
    }
    write_json(sqlite_path.parent / CACHE_HASH_FILENAME, payload)


def _load_cached_hash_for_current_mtime(sqlite_path: Path) -> str | None:
    sidecar_path = sqlite_path.parent / CACHE_HASH_FILENAME
    current_mtime_ns = str(sqlite_path.stat().st_mtime_ns)
    payload = load_json_if_exists(sidecar_path)
    cached_hash = payload.get("hash_by_mtime_ns", {}).get(current_mtime_ns) if isinstance(payload.get("hash_by_mtime_ns"), dict) else None
    if isinstance(cached_hash, str) and cached_hash.strip():
        return cached_hash.strip()
    return None


def _index_cache_rows(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in (manifest or {}).get("shards", []):
        if not isinstance(row, dict):
            continue
        relative_dir = str(row.get("relative_dir") or "").strip()
        if relative_dir:
            rows[relative_dir] = row
    return rows


def _cache_manifest_signature(manifest: dict[str, Any] | None) -> tuple[tuple[str, str, str, int, int, str], ...]:
    rows: list[tuple[str, str, str, int, int, str]] = []
    for row in (manifest or {}).get("shards", []):
        if not isinstance(row, dict):
            continue
        rows.append(
            (
                str(row.get("relative_dir") or ""),
                str(row.get("sqlite_sha256") or ""),
                str(row.get("compressed_name") or "requests.sqlite.zst"),
                int(row.get("sqlite_size_bytes") or 0),
                int(row.get("compressed_size_bytes") or 0),
                str(row.get("published_at") or ""),
            )
        )
    return tuple(sorted(rows))


def _compress_zstd_file(source_path: Path, target_path: Path) -> None:
    cctx = zstd.ZstdCompressor(level=10)
    with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle, cctx.stream_writer(target_handle) as compressed_handle:
        shutil.copyfileobj(source_handle, compressed_handle)


def _decompress_zstd_file(source_path: Path, target_path: Path) -> None:
    dctx = zstd.ZstdDecompressor()
    with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle, dctx.stream_reader(source_handle) as decompressed_handle:
        shutil.copyfileobj(decompressed_handle, target_handle)


def _restore_sqlite_from_archive(compressed_path: Path, target_sqlite: Path) -> None:
    target_sqlite.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="research-hf-cache-restore-") as temp_dir:
        snapshot_path = Path(temp_dir) / "requests.snapshot.sqlite"
        _decompress_zstd_file(compressed_path, snapshot_path)
        _install_sqlite_snapshot(snapshot_path, target_sqlite)


def _export_consistent_sqlite_snapshot(source_path: Path, snapshot_path: Path) -> None:
    try:
        _backup_sqlite_database(source_path, snapshot_path)
    except sqlite3.DatabaseError as exc:
        if "malformed" not in str(exc).lower():
            raise
        from research_experiments.core.execution.cache import repair_cache_shard

        repair_cache_shard(source_path)
        _backup_sqlite_database(source_path, snapshot_path)
    _validate_sqlite_snapshot(snapshot_path)


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


def _install_sqlite_snapshot(snapshot_path: Path, target_sqlite: Path) -> None:
    staged_target = target_sqlite.parent / "requests.sqlite.restore.tmp"
    _cleanup_sqlite_sidecars(staged_target)
    if staged_target.exists():
        staged_target.unlink()

    # Preserve the published snapshot bytes exactly so a pull->push round trip
    # keeps the same sqlite_sha256 and does not republish unchanged shards.
    shutil.copyfile(snapshot_path, staged_target)
    _validate_sqlite_snapshot(staged_target)
    _cleanup_sqlite_sidecars(target_sqlite)
    try:
        staged_target.replace(target_sqlite)
    except OSError as exc:
        if staged_target.exists():
            staged_target.unlink()
        raise RuntimeError(f"无法安装 cache snapshot 到 {target_sqlite.as_posix()}") from exc


def _cleanup_sqlite_sidecars(sqlite_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = sqlite_path.with_name(f"{sqlite_path.name}{suffix}")
        if sidecar.exists():
            sidecar.unlink()
