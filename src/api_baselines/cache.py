from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import threading
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    cache_key: str
    payload_json: str
    response_json: str
    http_status: int
    latency_ms: float
    provider_request_id: str | None


class RequestCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
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
        with self.lock:
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
        with self.lock:
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
        with self.lock:
            self.connection.close()


def json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
