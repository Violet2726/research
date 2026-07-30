"""基于 SQLite 分片的共享请求缓存工具。"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    """A normalized cache record that can be persisted to disk."""

    cache_key: str
    payload_json: str
    response_json: str
    http_status: int
    latency_ms: float
    provider_request_id: str | None


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

REQUESTS_SELECT_COLUMNS = """
rowid, cache_key, payload_json, response_json, http_status, latency_ms, provider_request_id
"""

REQUESTS_INSERT_SQL = """
INSERT OR REPLACE INTO requests (
    cache_key, payload_json, response_json, http_status, latency_ms, provider_request_id
) VALUES (?, ?, ?, ?, ?, ?)
"""


class RequestCache:
    """Thread-safe request cache for one SQLite shard."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._pending_writes = 0
        self._commit_every = 32
        self.connection = self._connect_with_recovery()

    def get(self, cache_key: str) -> CachedResponse | None:
        """Read one record by cache key."""

        with self._lock:
            row = self._execute_with_recovery(
                lambda: self.connection.execute(
                    """
                    SELECT cache_key, payload_json, response_json, http_status, latency_ms, provider_request_id
                    FROM requests
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
            )
        if row is None:
            return None
        return CachedResponse(*row)

    def put(self, record: CachedResponse) -> None:
        """Insert or replace one cache record."""

        with self._lock:

            def _write() -> None:
                self.connection.execute(
                    REQUESTS_INSERT_SQL,
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

            self._execute_with_recovery(_write)

    def delete(self, cache_key: str) -> None:
        """Remove one response that failed a post-request protocol contract."""

        with self._lock:
            def _delete() -> None:
                self.connection.execute("DELETE FROM requests WHERE cache_key = ?", (cache_key,))
                self.connection.commit()
                self._pending_writes = 0

            self._execute_with_recovery(_delete)

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self._lock:
            if self._pending_writes > 0:
                self._execute_with_recovery(self.connection.commit)
                self._pending_writes = 0
            self.connection.close()

    def _connect_with_recovery(self) -> sqlite3.Connection:
        try:
            return _open_cache_connection(self.db_path)
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_sqlite_error(exc):
                raise
            repair_cache_shard(self.db_path)
            return _open_cache_connection(self.db_path)

    def _execute_with_recovery(self, callback):
        try:
            return callback()
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_sqlite_error(exc):
                raise
            self._recover_malformed_shard()
            return callback()

    def _recover_malformed_shard(self) -> None:
        with contextlib.suppress(Exception):
            self.connection.close()
        repair_cache_shard(self.db_path)
        self._pending_writes = 0
        self.connection = _open_cache_connection(self.db_path)


class RequestCacheRouter:
    """Route provider/model/dataset tuples to cache shards."""

    def __init__(self, cache_root: str | Path, *, namespace: str | None = None) -> None:
        self.cache_root = Path(cache_root)
        self.namespace = _normalize_namespace(namespace)
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
        """Return the cache shard for one request target."""

        shard_identity = _shard_identity(
            provider=provider,
            request_model=request_model,
            dataset=dataset,
            namespace=self.namespace,
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
                    namespace=self.namespace,
                )
            )
            self._caches[shard_identity] = cache
            return cache

    def close(self) -> None:
        """Close all currently opened shards."""

        with self._lock:
            caches = list(self._caches.values())
            self._caches.clear()
        for cache in caches:
            cache.close()


def json_dump(data: Any) -> str:
    """Serialize JSON with stable key ordering for reproducible hashes."""

    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def build_request_cache_key(
    *,
    provider: str,
    request_model: str,
    payload: dict[str, Any],
) -> str:
    """Build a cache key from the real request identity."""

    fingerprint = {
        "provider": provider,
        "request_model": request_model,
        "payload": normalize_payload_for_cache_key(payload),
    }
    return sha256(json_dump(fingerprint).encode("utf-8")).hexdigest()


def normalize_payload_for_cache_key(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical payload fingerprint used for cache keys."""

    # Every submitted payload field is semantically relevant, including the
    # provider's completion cap.  Historical caches that elided a cap are not
    # valid confirmation evidence.
    return dict(payload)


def cache_successful_response(
    cache: RequestCache,
    *,
    cache_key: str,
    payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> None:
    """Persist only successful, parseable request results."""

    if response_payload.get("request_error") is not None:
        raise ValueError("Request failures must not be cached.")

    # Reject API rejections (soft rejections)
    assistant_text = str(response_payload.get("assistant_text") or "")
    if looks_like_soft_rejection(assistant_text):
        raise ValueError("API rejection responses must not be cached.")

    # Reject truncated outputs (missing FINAL_ANSWER for free-text protocols)
    if _is_truncated_output(assistant_text):
        raise ValueError("Truncated outputs must not be cached.")

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
    namespace: str | None = None,
) -> Path:
    """Resolve the shard path for a provider/model/dataset tuple."""

    root = Path(cache_root)
    normalized_namespace = _normalize_namespace(namespace)
    if normalized_namespace:
        root = root / "namespaces" / _slugify_segment(normalized_namespace)
    return (
        root
        / "providers"
        / _slugify_segment(provider)
        / _slugify_segment(request_model)
        / _slugify_dataset_path(dataset)
        / "requests.sqlite"
    )


def repair_cache_shard(shard_path: str | Path) -> dict[str, Any]:
    """Repair one malformed cache shard, or reset it to an empty shard."""

    path = Path(shard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        connection = _open_cache_connection(path)
        connection.close()
        return {
            "shard_path": path.as_posix(),
            "status": "created",
            "backup_path": None,
            "recovered_row_count": 0,
            "skipped_rowids": [],
        }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
    recovered_path = path.with_name(f"{path.name}.recovered-{timestamp}")
    _cleanup_sqlite_sidecars(path)

    try:
        recovered_row_count, skipped_rowids = _recover_cache_shard_rows(path, recovered_path)
    except Exception:
        if recovered_path.exists():
            recovered_path.unlink()
        path.replace(backup_path)
        connection = _open_cache_connection(path)
        connection.close()
        return {
            "shard_path": path.as_posix(),
            "status": "reset_to_empty",
            "backup_path": backup_path.as_posix(),
            "recovered_row_count": 0,
            "skipped_rowids": [],
        }

    path.replace(backup_path)
    recovered_path.replace(path)
    return {
        "shard_path": path.as_posix(),
        "status": "recovered",
        "backup_path": backup_path.as_posix(),
        "recovered_row_count": recovered_row_count,
        "skipped_rowids": skipped_rowids,
    }


def _shard_identity(*, provider: str, request_model: str, dataset: str, namespace: str | None) -> str:
    return json_dump(
        {
            "provider": provider,
            "request_model": request_model,
            "dataset": dataset,
            "namespace": namespace,
        }
    )


def _slugify_segment(value: str) -> str:
    lowered = value.strip().lower()
    pieces = [character if character.isalnum() else "-" for character in lowered]
    slug = "".join(pieces).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"


def _slugify_dataset_path(dataset: str) -> Path:
    normalized = str(dataset).replace("\\", "/").strip("/")
    if not normalized:
        return Path("default")
    parts = [part for part in normalized.split("/") if part]
    return Path(*[_slugify_segment(part) for part in parts])


def _normalize_namespace(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("Cache namespace must be a single non-path segment.")
    return normalized


def _open_cache_connection(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(REQUESTS_TABLE_SCHEMA)
    connection.commit()
    return connection


def _is_malformed_sqlite_error(exc: sqlite3.Error) -> bool:
    return "malformed" in str(exc).lower()


def _cleanup_sqlite_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(f"{db_path.name}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


def _recover_cache_shard_rows(source_path: Path, target_path: Path) -> tuple[int, list[int]]:
    if target_path.exists():
        target_path.unlink()

    source = sqlite3.connect(source_path)
    target = _open_cache_connection(target_path)
    recovered_row_count = 0
    skipped_rowids: list[int] = []
    try:
        max_rowid = int(source.execute("SELECT max(rowid) FROM requests").fetchone()[0] or 0)
        for start_rowid in range(0, max_rowid + 1, 2048):
            end_rowid = min(max_rowid, start_rowid + 2048)
            rows, skipped = _read_rows_with_salvage(source, start_rowid, end_rowid)
            if rows:
                target.executemany(
                    REQUESTS_INSERT_SQL,
                    [row[1:] for row in rows],
                )
                recovered_row_count += len(rows)
            skipped_rowids.extend(skipped)
        target.commit()
    finally:
        source.close()
        target.close()
    return recovered_row_count, skipped_rowids


def _read_rows_with_salvage(
    connection: sqlite3.Connection,
    start_rowid: int,
    end_rowid: int,
) -> tuple[list[tuple[Any, ...]], list[int]]:
    if end_rowid <= start_rowid:
        return [], []
    try:
        rows = connection.execute(
            f"""
            SELECT {REQUESTS_SELECT_COLUMNS}
            FROM requests
            WHERE rowid > ? AND rowid <= ?
            ORDER BY rowid
            """,
            (start_rowid, end_rowid),
        ).fetchall()
        return rows, []
    except sqlite3.DatabaseError as exc:
        if not _is_malformed_sqlite_error(exc):
            raise
        if end_rowid - start_rowid == 1:
            return [], [end_rowid]
        midpoint = start_rowid + ((end_rowid - start_rowid) // 2)
        left_rows, left_skipped = _read_rows_with_salvage(connection, start_rowid, midpoint)
        right_rows, right_skipped = _read_rows_with_salvage(connection, midpoint, end_rowid)
        return left_rows + right_rows, left_skipped + right_skipped


def looks_like_soft_rejection(text: str) -> bool:
    """Check if the response is an API rejection (soft rejection)."""

    normalized = " ".join(str(text or "").split()).lower()
    if not normalized:
        return False
    return normalized in {
        "content omitted due to provider safety policy",
        "provider_refused_due_to_policy",
        "the request was rejected because it was considered high risk",
        "request rejected because it was considered high risk",
    }


def _is_truncated_output(text: str) -> bool:
    """Check if the output is truncated (has REASONING but no FINAL_ANSWER)."""

    if not text:
        return False
    text_upper = text.upper()
    # Has REASONING but no FINAL_ANSWER - indicates truncated output
    has_reasoning = "REASONING:" in text_upper
    has_final_answer = bool(re.search(r"\bFINAL[\s_]+ANSWER\s*:", text_upper))
    return has_reasoning and not has_final_answer
