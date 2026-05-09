"""共享请求缓存能力。

缓存体系分成两层：
1. `RequestCache`：单个 SQLite 分库，负责具体读写；
2. `RequestCacheRouter`：按 `provider + request_model + dataset` 路由到对应分库。

目录结构示例：
`cache/providers/xiaomimimo/mimo-v2-5/strategyqa/requests.sqlite`

同一家供应商的同一请求模型在同一数据集内共享缓存，不再区分端点。
缓存键仅由 `provider + request_model + payload` 决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
import threading
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    """表示一条已经标准化并可落盘的缓存记录。"""

    cache_key: str
    payload_json: str
    response_json: str
    http_status: int
    latency_ms: float
    provider_request_id: str | None


@dataclass(frozen=True)
class CacheShardSummary:
    """表示单个缓存分库的统计结果。"""

    shard_path: Path
    provider: str
    request_model: str
    dataset: str
    exists: bool
    file_size_bytes: int
    request_count: int | None
    error: str | None


@dataclass(frozen=True)
class CacheProviderSummary:
    """表示单个 provider 维度下的缓存聚合统计。"""

    provider: str
    model_count: int
    dataset_count: int
    shard_count: int
    total_request_count: int
    total_size_bytes: int


@dataclass(frozen=True)
class CacheRootSummary:
    """表示整个缓存根目录的聚合统计。"""

    cache_root: Path
    shard_count: int
    provider_count: int
    total_request_count: int
    total_size_bytes: int
    providers: tuple[CacheProviderSummary, ...]
    shards: tuple[CacheShardSummary, ...]


class RequestCache:
    """线程安全的单库 SQLite 请求缓存。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
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
        )
        self.connection.commit()
        self._pending_writes = 0
        self._commit_every = 32

    def get(self, cache_key: str) -> CachedResponse | None:
        """按缓存键读取记录；未命中时返回 `None`。"""
        with self._lock:
            row = self.connection.execute(
                """
                SELECT cache_key, payload_json, response_json, http_status, latency_ms, provider_request_id
                FROM requests
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return CachedResponse(*row)

    def put(self, record: CachedResponse) -> None:
        """写入或覆盖一条缓存记录。"""
        with self._lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO requests (
                    cache_key, payload_json, response_json, http_status, latency_ms, provider_request_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.cache_key,
                    record.payload_json,
                    record.response_json,
                    record.http_status,
                    record.latency_ms,
                    record.provider_request_id,
                ),
            )
            self._pending_writes += 1
            if self._pending_writes >= self._commit_every:
                self.connection.commit()
                self._pending_writes = 0

    def close(self) -> None:
        """关闭底层数据库连接。"""
        with self._lock:
            if self._pending_writes > 0:
                self.connection.commit()
                self._pending_writes = 0
            self.connection.close()


class RequestCacheRouter:
    """按供应商、请求模型和数据集路由到对应缓存分库。"""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._caches: dict[str, RequestCache] = {}

    def for_request_target(
        self,
        *,
        provider: str,
        request_model: str,
        dataset: str,
    ) -> RequestCache:
        """返回某个供应商、请求模型和数据集对应的缓存分库。"""
        shard_identity = _shard_identity(
            provider=provider,
            request_model=request_model,
            dataset=dataset,
        )
        with self._lock:
            cache = self._caches.get(shard_identity)
            if cache is not None:
                return cache
            cache = RequestCache(
                resolve_cache_shard_path(
                    cache_root=self.cache_root,
                    provider=provider,
                    request_model=request_model,
                    dataset=dataset,
                )
            )
            self._caches[shard_identity] = cache
            return cache

    def close(self) -> None:
        """关闭当前路由器已打开的全部缓存分库。"""
        with self._lock:
            caches = list(self._caches.values())
            self._caches.clear()
        for cache in caches:
            cache.close()


def json_dump(data: Any) -> str:
    """按稳定规则序列化 JSON，保证哈希可复现。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def build_request_cache_key(
    *,
    provider: str,
    request_model: str,
    payload: dict[str, Any],
) -> str:
    """基于真实请求身份构造缓存键。"""
    fingerprint = {
        "provider": provider,
        "request_model": request_model,
        "payload": payload,
    }
    return sha256(json_dump(fingerprint).encode("utf-8")).hexdigest()


def cache_successful_response(
    cache: RequestCache,
    *,
    cache_key: str,
    payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> None:
    """仅在请求成功且解析成功后落盘缓存记录。"""
    if response_payload.get("request_error") is not None:
        raise ValueError("Request failures must not be cached.")
    cache.put(
        CachedResponse(
            cache_key=cache_key,
            payload_json=json_dump(payload),
            response_json=json_dump(response_payload),
            http_status=int(response_payload.get("http_status") or 0),
            latency_ms=float(response_payload.get("latency_ms") or 0.0),
            provider_request_id=(
                str(response_payload["provider_request_id"])
                if response_payload.get("provider_request_id") is not None
                else None
            ),
        )
    )


def resolve_cache_shard_path(
    cache_root: str | Path,
    *,
    provider: str,
    request_model: str,
    dataset: str,
) -> Path:
    """根据供应商、请求模型和数据集解析缓存分库路径。"""
    root = Path(cache_root)
    return (
        root
        / "providers"
        / _slugify(provider)
        / _slugify(request_model)
        / _slugify(dataset)
        / "requests.sqlite"
    )


def inspect_cache_shard(shard_path: str | Path, cache_root: str | Path) -> CacheShardSummary:
    """读取单个缓存分库的统计信息。"""
    root = Path(cache_root)
    path = Path(shard_path)
    provider, request_model, dataset = _decompose_shard_path(root, path)
    exists = path.exists()
    file_size_bytes = path.stat().st_size if exists else 0
    if not exists:
        return CacheShardSummary(
            shard_path=path,
            provider=provider,
            request_model=request_model,
            dataset=dataset,
            exists=False,
            file_size_bytes=file_size_bytes,
            request_count=0,
            error=None,
        )

    try:
        request_count = _read_request_count(path)
        error = None
    except sqlite3.Error as exc:
        request_count = None
        error = f"{exc.__class__.__name__}: {exc}"

    return CacheShardSummary(
        shard_path=path,
        provider=provider,
        request_model=request_model,
        dataset=dataset,
        exists=True,
        file_size_bytes=file_size_bytes,
        request_count=request_count,
        error=error,
    )


def collect_cache_shard_summaries(cache_root: str | Path) -> list[CacheShardSummary]:
    """扫描缓存根目录下全部缓存分库。"""
    root = Path(cache_root)
    providers_root = root / "providers"
    if not providers_root.exists():
        return []
    return [inspect_cache_shard(path, root) for path in sorted(providers_root.rglob("requests.sqlite"))]


def summarize_cache_root(cache_root: str | Path) -> CacheRootSummary:
    """聚合整个缓存根目录的统计信息。"""
    root = Path(cache_root)
    shards = tuple(collect_cache_shard_summaries(root))

    provider_buckets: dict[str, list[CacheShardSummary]] = {}
    total_request_count = 0
    total_size_bytes = 0
    for shard in shards:
        provider_buckets.setdefault(shard.provider, []).append(shard)
        total_size_bytes += shard.file_size_bytes
        total_request_count += int(shard.request_count or 0)

    providers = tuple(
        sorted(
            (
                CacheProviderSummary(
                    provider=provider,
                    model_count=len({item.request_model for item in items}),
                    dataset_count=len({item.dataset for item in items}),
                    shard_count=len(items),
                    total_request_count=sum(int(item.request_count or 0) for item in items),
                    total_size_bytes=sum(item.file_size_bytes for item in items),
                )
                for provider, items in provider_buckets.items()
            ),
            key=lambda item: (-item.total_size_bytes, item.provider),
        )
    )

    return CacheRootSummary(
        cache_root=root,
        shard_count=len(shards),
        provider_count=len(providers),
        total_request_count=total_request_count,
        total_size_bytes=total_size_bytes,
        providers=providers,
        shards=tuple(sorted(shards, key=lambda item: (-item.file_size_bytes, item.shard_path.as_posix()))),
    )


def format_bytes(num_bytes: int) -> str:
    """把字节数格式化成更易读的容量字符串。"""
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.2f} {units[unit_index]}"


def _shard_identity(*, provider: str, request_model: str, dataset: str) -> str:
    """生成分库身份指纹，用于进程内路由复用。"""
    return json_dump(
        {
            "provider": provider,
            "request_model": request_model,
            "dataset": dataset,
        }
    )


def _slugify(value: str) -> str:
    """把路径片段压缩成适合目录名与文件名的 ASCII 形式。"""
    lowered = value.strip().lower()
    pieces = [character if character.isalnum() else "-" for character in lowered]
    slug = "".join(pieces).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"


def _read_request_count(shard_path: Path) -> int:
    """读取单个缓存分库中的请求条数。"""
    connection = sqlite3.connect(shard_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM requests").fetchone()
    finally:
        connection.close()
    return int(row[0] if row is not None else 0)


def _decompose_shard_path(cache_root: Path, shard_path: Path) -> tuple[str, str, str]:
    """从缓存分库路径中反解出供应商、请求模型和数据集。"""
    providers_root = cache_root / "providers"
    try:
        relative = shard_path.relative_to(providers_root)
    except ValueError:
        return ("unknown", "unknown", "unknown")

    parts = relative.parts
    provider = parts[0] if len(parts) >= 1 else "unknown"
    request_model = parts[1] if len(parts) >= 2 else "unknown"
    dataset = parts[2] if len(parts) >= 3 else "unknown"
    return (provider, request_model, dataset)
