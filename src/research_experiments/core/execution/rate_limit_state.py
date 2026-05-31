"""跨进程限流状态的持久化存储。"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from research_experiments.workspace.layout import workspace_layout


@dataclass
class PersistentTokenEvent:
    """跨进程 token 预留事件。"""

    event_id: str
    timestamp: float
    tokens: int


@dataclass
class PersistentRateLimitState:
    """一个 provider/model 的跨进程限流账本。"""

    request_events: list[float] = field(default_factory=list)
    token_events: list[PersistentTokenEvent] = field(default_factory=list)
    last_request_admission: float | None = None
    not_before_wall: float | None = None
    rate_limit_429_count: int = 0
    last_retry_after_seconds: float | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PersistentRateLimitState:
        token_events = []
        for item in payload.get("token_events", []):
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("id") or "")
            if not event_id:
                continue
            token_events.append(
                PersistentTokenEvent(
                    event_id=event_id,
                    timestamp=float(item.get("timestamp") or 0.0),
                    tokens=max(0, int(item.get("tokens") or 0)),
                )
            )
        return cls(
            request_events=[float(item) for item in payload.get("request_events", [])],
            token_events=token_events,
            last_request_admission=_optional_float(payload.get("last_request_admission")),
            not_before_wall=_optional_float(payload.get("not_before_wall")),
            rate_limit_429_count=max(0, int(payload.get("rate_limit_429_count") or 0)),
            last_retry_after_seconds=_optional_float(payload.get("last_retry_after_seconds")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "request_events": self.request_events,
            "token_events": [
                {
                    "id": event.event_id,
                    "timestamp": event.timestamp,
                    "tokens": event.tokens,
                }
                for event in self.token_events
            ],
            "last_request_admission": self.last_request_admission,
            "not_before_wall": self.not_before_wall,
            "rate_limit_429_count": self.rate_limit_429_count,
            "last_retry_after_seconds": self.last_retry_after_seconds,
        }

    def evict_expired(self, now: float, window_seconds: float) -> None:
        self.request_events = [
            timestamp for timestamp in self.request_events if now - timestamp < window_seconds
        ]
        self.token_events = [
            event for event in self.token_events if now - event.timestamp < window_seconds
        ]


class FileRateLimitStateStore:
    """用文件锁保护的跨进程限流状态存储。"""

    _process_locks: dict[Path, threading.Lock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, *, state_path: Path, lock_path: Path) -> None:
        self.state_path = state_path
        self.lock_path = lock_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_scope(
        cls,
        *,
        scope_key: str,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        window_seconds: float,
    ) -> FileRateLimitStateStore:
        state_root = workspace_layout().cache_root / "_rate_limits"
        state_name = sha256(
            f"{scope_key}|{requests_per_minute}|{tokens_per_minute}|{window_seconds}".encode()
        ).hexdigest()
        return cls(
            state_path=state_root / f"{state_name}.json",
            lock_path=state_root / f"{state_name}.lock",
        )

    @contextmanager
    def edit(self) -> Iterator[PersistentRateLimitState]:
        with self._exclusive_lock():
            state = self._read_state()
            try:
                yield state
            except Exception:
                raise
            else:
                self._write_state(state)

    def read(self) -> PersistentRateLimitState:
        with self._exclusive_lock():
            return self._read_state()

    def _read_state(self) -> PersistentRateLimitState:
        if not self.state_path.exists():
            return PersistentRateLimitState()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return PersistentRateLimitState()
        if not isinstance(payload, dict):
            return PersistentRateLimitState()
        return PersistentRateLimitState.from_payload(payload)

    def _write_state(self, state: PersistentRateLimitState) -> None:
        tmp_path = self.state_path.with_name(f"{self.state_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(state.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        local_lock = self._process_lock(self.lock_path)
        with local_lock, self.lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _process_lock(cls, lock_path: Path) -> threading.Lock:
        resolved = lock_path.resolve()
        with cls._process_locks_guard:
            lock = cls._process_locks.get(resolved)
            if lock is None:
                lock = threading.Lock()
                cls._process_locks[resolved] = lock
            return lock


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
