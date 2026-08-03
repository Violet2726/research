"""用于 D4 模型 turn 的、每运行独立的追加式完成 ledger。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class D4CompletionLedger:
    """Persist completed D4 turns independently of the shared request cache.

    The shared cache deliberately drops malformed, truncated, and rejected
    responses.  A frozen run must nevertheless remember that such a turn has
    already completed, so an interrupted resume does not silently issue a new
    request for it.  The ledger is append-only and scoped to one run directory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._rows: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    key = self._key_from_row(row)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                self._rows.setdefault(key, row)

    def lookup(
        self,
        *,
        sample_id: str,
        method_name: str,
        role: str,
        agent_id: int,
        seed: int,
    ) -> dict[str, Any] | None:
        """Return the first durable completion for one deterministic turn."""

        key = (str(sample_id), str(method_name), str(role), int(agent_id), int(seed))
        with self._lock:
            row = self._rows.get(key)
            return dict(row) if row is not None else None

    def record(self, row: dict[str, Any]) -> None:
        """Durably append a completed turn once; success and failure both count."""

        key = self._key_from_row(row)
        with self._lock:
            if key in self._rows:
                return
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._rows[key] = dict(row)

    @staticmethod
    def _key_from_row(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
        return (
            str(row["sample_id"]),
            str(row["method_name"]),
            str(row["role"]),
            int(row["agent_id"]),
            int(row["request_seed"]),
        )
