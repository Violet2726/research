"""请求级缓存。

本模块使用 SQLite 持久化“同一请求指纹 -> 同一响应结果”的映射关系，
避免重复实验时反复消耗 API 配额，也为复现实验、离线排障和运行恢复
提供稳定的底层支撑。
"""

from __future__ import annotations

from dataclasses import dataclass
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


class RequestCache:
    """线程安全的 SQLite 请求缓存封装。"""

    def __init__(self, db_path: str | Path) -> None:
        """初始化缓存数据库，并在首次运行时创建表结构。"""
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
        """按缓存键读取记录；未命中时返回 `None`。"""
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
        """写入或覆盖一条缓存记录。"""
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
        """关闭底层数据库连接。"""
        with self.lock:
            self.connection.close()


def json_dump(data: Any) -> str:
    """按统一规则序列化 JSON，保证缓存键和值都具备稳定性。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
