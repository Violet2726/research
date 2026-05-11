"""共享的 run 产物写盘辅助。"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, TextIO


class BufferedJsonlWriter:
    """按行写 JSONL，并以小批量方式稳定 flush。"""

    def __init__(
        self,
        handle: TextIO,
        *,
        flush_every: int = 20,
        flush_interval_seconds: float = 2.0,
    ) -> None:
        self.handle = handle
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_seconds = max(0.0, float(flush_interval_seconds))
        self.pending_rows = 0
        self.last_flush_monotonic = time.monotonic()

    def write_row(self, row: dict[str, Any]) -> None:
        """写入一行 JSON 记录。"""
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.pending_rows += 1
        self._maybe_flush()

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        """写入多行 JSON 记录。"""
        for row in rows:
            self.write_row(row)

    def flush(self) -> None:
        """立刻把缓冲内容刷入底层句柄。"""
        if self.pending_rows <= 0:
            return
        self.handle.flush()
        self.pending_rows = 0
        self.last_flush_monotonic = time.monotonic()

    def close(self) -> None:
        """在关闭前确保最后一批内容已落盘。"""
        self.flush()

    def _maybe_flush(self) -> None:
        if self.pending_rows >= self.flush_every:
            self.flush()
            return
        now = time.monotonic()
        if now - self.last_flush_monotonic >= self.flush_interval_seconds:
            self.flush()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """按 UTF-8 规范写出 JSON 文件。"""
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
