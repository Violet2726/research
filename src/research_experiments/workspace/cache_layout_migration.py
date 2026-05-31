"""缓存分片目录归并工具。"""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.execution.cache import collect_cache_shard_summaries, resolve_cache_shard_path

BENCHMARK_CONFIG_ROOT = Path("configs/core/shared/benchmarks")
REQUESTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    cache_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    payload_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    http_status INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    provider_request_id TEXT
)
"""

REQUESTS_INSERT_FROM_ATTACHED_SQL = """
INSERT OR REPLACE INTO requests (
    cache_key,
    payload_json,
    response_json,
    http_status,
    latency_ms,
    provider_request_id
)
SELECT
    cache_key,
    payload_json,
    response_json,
    http_status,
    latency_ms,
    provider_request_id
FROM source_db.requests
"""


@dataclass(frozen=True)
class CacheLayoutMigrationItem:
    """表示一条待归并的旧缓存分片。"""

    provider: str
    request_model: str
    source_dataset: str
    target_dataset: str
    source_path: Path
    target_path: Path
    source_request_count: int
    target_request_count: int


def normalize_cache_layout(
    cache_root: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """按新的 canonical dataset slug 归并旧缓存分片。"""

    root = Path(cache_root)
    candidates = discover_cache_layout_candidates(root)
    applied_rows: list[dict[str, Any]] = []
    for item in candidates:
        applied_rows.append(
            _apply_single_migration(item) if apply else _plan_single_migration(item)
        )
    return {
        "cache_root": root.as_posix(),
        "apply": apply,
        "candidate_count": len(candidates),
        "candidates": applied_rows,
    }


def discover_cache_layout_candidates(cache_root: str | Path) -> list[CacheLayoutMigrationItem]:
    """扫描 cache root，找出仍使用旧 dataset namespace 的缓存分片。"""

    redirects = benchmark_cache_namespace_redirects()
    candidates: list[CacheLayoutMigrationItem] = []
    for shard in collect_cache_shard_summaries(cache_root):
        target_dataset = redirects.get(shard.dataset)
        if target_dataset is None:
            continue
        target_path = resolve_cache_shard_path(
            cache_root,
            provider=shard.provider,
            request_model=shard.request_model,
            dataset=target_dataset,
        )
        target_request_count = _read_request_count(target_path)
        candidates.append(
            CacheLayoutMigrationItem(
                provider=shard.provider,
                request_model=shard.request_model,
                source_dataset=shard.dataset,
                target_dataset=target_dataset,
                source_path=shard.shard_path,
                target_path=target_path,
                source_request_count=int(shard.request_count or 0),
                target_request_count=target_request_count,
            )
        )
    candidates.sort(
        key=lambda item: (
            item.provider,
            item.request_model,
            item.source_dataset,
            item.source_path.as_posix(),
        )
    )
    return candidates


def benchmark_cache_namespace_redirects(
    config_root: str | Path = BENCHMARK_CONFIG_ROOT,
) -> dict[str, str]:
    """构建旧 cache namespace 到新 dataset slug 的映射表。"""

    redirects: dict[str, str] = {}
    for path in sorted(Path(config_root).rglob("*.toml")):
        benchmark = load_benchmark_config(path)
        source_dataset = _normalize_dataset_shard_key(benchmark.cache_namespace or benchmark.slug)
        target_dataset = _normalize_dataset_shard_key(benchmark.slug)
        if source_dataset == target_dataset:
            continue
        previous = redirects.get(source_dataset)
        if previous is not None and previous != target_dataset:
            raise RuntimeError(
                f"缓存 namespace 重定向冲突: {source_dataset!r} 同时映射到 {previous!r} 和 {target_dataset!r}"
            )
        redirects[source_dataset] = target_dataset
    return redirects


def _plan_single_migration(item: CacheLayoutMigrationItem) -> dict[str, Any]:
    return {
        "provider": item.provider,
        "request_model": item.request_model,
        "source_dataset": item.source_dataset,
        "target_dataset": item.target_dataset,
        "source_path": item.source_path.as_posix(),
        "target_path": item.target_path.as_posix(),
        "source_request_count": item.source_request_count,
        "target_request_count": item.target_request_count,
        "action": "merge_into_canonical_shard",
    }


def _apply_single_migration(item: CacheLayoutMigrationItem) -> dict[str, Any]:
    item.target_path.parent.mkdir(parents=True, exist_ok=True)
    before_target_count = _read_request_count(item.target_path)
    snapshot_path = _export_sqlite_snapshot(item.source_path)
    try:
        _merge_snapshot_into_target(snapshot_path, item.target_path)
    finally:
        if snapshot_path.exists():
            snapshot_path.unlink()
    after_target_count = _read_request_count(item.target_path)
    _delete_sqlite_with_sidecars(item.source_path)
    _prune_empty_parents(item.source_path.parent, stop_at=item.source_path.parents[2])
    return {
        "provider": item.provider,
        "request_model": item.request_model,
        "source_dataset": item.source_dataset,
        "target_dataset": item.target_dataset,
        "source_path": item.source_path.as_posix(),
        "target_path": item.target_path.as_posix(),
        "source_request_count": item.source_request_count,
        "target_request_count_before": before_target_count,
        "target_request_count_after": after_target_count,
        "new_rows_added": max(after_target_count - before_target_count, 0),
        "action": "merged_and_removed_legacy_shard",
    }


def _export_sqlite_snapshot(source_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="research-cache-layout-",
        suffix=".sqlite",
        delete=False,
    ) as handle:
        snapshot_path = Path(handle.name)
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(snapshot_path)
    try:
        source.backup(target)
        target.commit()
    finally:
        source.close()
        target.close()
    return snapshot_path


def _merge_snapshot_into_target(snapshot_path: Path, target_path: Path) -> None:
    connection = sqlite3.connect(target_path)
    try:
        connection.execute(REQUESTS_TABLE_SCHEMA)
        connection.execute("ATTACH DATABASE ? AS source_db", (snapshot_path.as_posix(),))
        try:
            connection.execute(REQUESTS_INSERT_FROM_ATTACHED_SQL)
            connection.commit()
        finally:
            connection.execute("DETACH DATABASE source_db")
    finally:
        connection.close()


def _read_request_count(sqlite_path: Path) -> int:
    if not sqlite_path.exists():
        return 0
    connection = sqlite3.connect(sqlite_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM requests").fetchone()
    finally:
        connection.close()
    return int(row[0] if row is not None else 0)


def _delete_sqlite_with_sidecars(sqlite_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = sqlite_path if not suffix else sqlite_path.with_name(f"{sqlite_path.name}{suffix}")
        if candidate.exists():
            candidate.unlink()


def _prune_empty_parents(directory: Path, *, stop_at: Path) -> None:
    current = directory
    while current != stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _normalize_dataset_shard_key(dataset: str) -> str:
    normalized = str(dataset).replace("\\", "/").strip("/")
    if not normalized:
        return "default"
    return "/".join(_slugify_segment(part) for part in normalized.split("/") if part)


def _slugify_segment(value: str) -> str:
    lowered = value.strip().lower()
    pieces = [character if character.isalnum() else "-" for character in lowered]
    slug = "".join(pieces).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"
