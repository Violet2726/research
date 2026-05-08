"""共享请求缓存能力。

本模块把缓存拆成两层：
1. `RequestCache`：单个 SQLite 分库，负责具体读写；
2. `RequestCacheRouter`：按 `provider + base_url + chat_path` 路由到对应分库。

在此基础上，还提供缓存分库定位与统计能力，方便排查缓存增长、复用命中和磁盘占用。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
import threading
from typing import Any
from urllib.parse import urlparse


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
    host: str
    upstream_path: str
    endpoint_stem: str
    shard_suffix: str
    exists: bool
    file_size_bytes: int
    request_count: int | None
    error: str | None


@dataclass(frozen=True)
class CacheProviderSummary:
    """表示单个 provider 维度下的缓存聚合统计。"""

    provider: str
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
            self.connection.commit()

    def close(self) -> None:
        """关闭底层数据库连接。"""
        with self._lock:
            self.connection.close()


class RequestCacheRouter:
    """按真实接口端点把请求路由到对应缓存分库。"""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._caches: dict[str, RequestCache] = {}

    def for_endpoint(self, *, provider: str, base_url: str, chat_path: str) -> RequestCache:
        """返回某个接口端点对应的缓存分库，必要时懒加载创建。"""
        endpoint_identity = _endpoint_identity(
            provider=provider,
            base_url=base_url,
            chat_path=chat_path,
        )
        with self._lock:
            cache = self._caches.get(endpoint_identity)
            if cache is not None:
                return cache
            cache = RequestCache(
                resolve_cache_shard_path(
                    cache_root=self.cache_root,
                    provider=provider,
                    base_url=base_url,
                    chat_path=chat_path,
                )
            )
            self._caches[endpoint_identity] = cache
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


def build_request_cache_key(payload: dict[str, Any]) -> str:
    """仅基于真实出站 payload 构造缓存键。"""
    return sha256(json_dump(payload).encode("utf-8")).hexdigest()


def resolve_cache_shard_path(
    cache_root: str | Path,
    *,
    provider: str,
    base_url: str,
    chat_path: str,
) -> Path:
    """根据端点身份解析其应落入的缓存分库路径。"""
    return _shard_path(
        cache_root=Path(cache_root),
        provider=provider,
        base_url=base_url,
        chat_path=chat_path,
    )


def inspect_cache_shard(shard_path: str | Path, cache_root: str | Path) -> CacheShardSummary:
    """读取单个缓存分库的统计信息。"""
    root = Path(cache_root)
    path = Path(shard_path)
    provider, host, upstream_path, endpoint_stem, shard_suffix = _decompose_shard_path(root, path)
    exists = path.exists()
    file_size_bytes = path.stat().st_size if exists else 0
    if not exists:
        return CacheShardSummary(
            shard_path=path,
            provider=provider,
            host=host,
            upstream_path=upstream_path,
            endpoint_stem=endpoint_stem,
            shard_suffix=shard_suffix,
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
        host=host,
        upstream_path=upstream_path,
        endpoint_stem=endpoint_stem,
        shard_suffix=shard_suffix,
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

    summaries = [inspect_cache_shard(path, root) for path in sorted(providers_root.rglob("*.sqlite"))]
    return summaries


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


def _endpoint_identity(*, provider: str, base_url: str, chat_path: str) -> str:
    """生成端点身份指纹，用于进程内路由与文件分片。"""
    return json_dump(
        {
            "provider": provider,
            "base_url": base_url.rstrip("/"),
            "chat_path": _normalize_chat_path(chat_path),
        }
    )


def _shard_path(cache_root: Path, *, provider: str, base_url: str, chat_path: str) -> Path:
    """把端点身份映射成稳定的缓存分库路径。"""
    parsed = urlparse(base_url.rstrip("/"))
    host = parsed.netloc or parsed.path or "unknown-host"
    base_path = parsed.path.strip("/")
    normalized_chat_path = _normalize_chat_path(chat_path)
    shard_suffix = sha256(
        _endpoint_identity(
            provider=provider,
            base_url=base_url,
            chat_path=chat_path,
        ).encode("utf-8")
    ).hexdigest()[:12]

    root = cache_root / "providers" / _slugify(provider) / _slugify(host)
    if base_path:
        root = root / _slugify(base_path)

    endpoint_stem = _slugify(normalized_chat_path.strip("/")) or "root"
    return root / f"{endpoint_stem}-{shard_suffix}.sqlite"


def _normalize_chat_path(chat_path: str) -> str:
    """把 chat path 规范化成稳定的绝对路径形态。"""
    normalized = str(chat_path or "").strip()
    if not normalized:
        return "/"
    return normalized if normalized.startswith("/") else f"/{normalized}"


def _slugify(value: str) -> str:
    """把路径片段压缩成适合目录名、文件名的 ASCII 形式。"""
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


def _decompose_shard_path(cache_root: Path, shard_path: Path) -> tuple[str, str, str, str, str]:
    """从缓存分库路径中反解出展示所需的路径信息。"""
    providers_root = cache_root / "providers"
    try:
        relative = shard_path.relative_to(providers_root)
    except ValueError:
        filename = shard_path.stem
        endpoint_stem, shard_suffix = _split_shard_filename(filename)
        return ("unknown", "unknown", "", endpoint_stem, shard_suffix)

    parts = relative.parts
    provider = parts[0] if len(parts) >= 1 else "unknown"
    host = parts[1] if len(parts) >= 2 else "unknown"
    upstream_parts = parts[2:-1] if len(parts) >= 3 else ()
    upstream_path = "/".join(upstream_parts)
    endpoint_stem, shard_suffix = _split_shard_filename(relative.stem)
    return (provider, host, upstream_path, endpoint_stem, shard_suffix)


def _split_shard_filename(filename_stem: str) -> tuple[str, str]:
    """把 `<endpoint>-<hash>` 形式的文件名拆成端点名和分片后缀。"""
    if "-" not in filename_stem:
        return (filename_stem, "")
    endpoint_stem, shard_suffix = filename_stem.rsplit("-", 1)
    return (endpoint_stem, shard_suffix)
