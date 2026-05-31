"""共享的网络请求限流与并发控制。"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from research_experiments.core.execution.rate_limit_state import (
    FileRateLimitStateStore,
    PersistentRateLimitState,
    PersistentTokenEvent,
)

STANDARD_MAX_CONCURRENT_REQUESTS = 90
STANDARD_REQUESTS_PER_MINUTE_LIMIT = 95
STANDARD_TOKENS_PER_MINUTE_LIMIT = 9000000


def standard_runtime_limits() -> dict[str, int]:
    """返回项目统一执行限流基线。"""

    return {
        "max_concurrent_requests": STANDARD_MAX_CONCURRENT_REQUESTS,
        "requests_per_minute_limit": STANDARD_REQUESTS_PER_MINUTE_LIMIT,
        "tokens_per_minute_limit": STANDARD_TOKENS_PER_MINUTE_LIMIT,
    }


@dataclass
class _TokenEvent:
    """表示仍在限流窗口内的 token 预留事件。"""

    timestamp: float
    tokens: int


@dataclass(frozen=True)
class RateLimitReservation:
    """表示一次已占用配额、后续可按真实 usage 对账的预留。"""

    reserved_tokens: int
    token_event: _TokenEvent
    event_id: str | None = None


class RateLimiter(Protocol):
    """RequestThrottle 依赖的限流器接口。"""

    def acquire(self, estimated_tokens: int) -> RateLimitReservation: ...

    def settle(self, reservation: RateLimitReservation, actual_tokens: int, *, http_status: int | None = None) -> None: ...

    def note_retry_after(self, retry_after_seconds: float | None) -> None: ...

    def snapshot(self) -> dict[str, float | int | None]: ...


@dataclass(frozen=True)
class _SharedThrottleState:
    """同一 provider/model 在进程内共享的并发槽与限流器。"""

    semaphore: threading.BoundedSemaphore
    limiter: RateLimiter


class SlidingWindowRateLimiter:
    """线程安全地约束 RPM/TPM 配额。"""

    def __init__(
        self,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        window_seconds: float = 60.0,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = window_seconds
        self.request_events: deque[float] = deque()
        self.token_events: deque[_TokenEvent] = deque()
        self.condition = threading.Condition()
        self.last_request_admission: float | None = None
        self.rate_limit_429_count = 0
        self.total_wait_seconds = 0.0
        self.last_retry_after_seconds: float | None = None
        self.not_before_monotonic: float | None = None
        self.request_spacing_seconds = _request_spacing_seconds(requests_per_minute, window_seconds)

    def acquire(self, estimated_tokens: int) -> RateLimitReservation:
        """阻塞直到下一次请求可以安全进入本进程窗口。"""

        with self.condition:
            while True:
                now = time.monotonic()
                self._evict_expired(now)
                wait_seconds = max(
                    self._request_wait_seconds(now),
                    self._token_wait_seconds(now, estimated_tokens),
                    self._cooldown_wait_seconds(now),
                )
                if wait_seconds <= 0:
                    token_event = _TokenEvent(
                        timestamp=now,
                        tokens=max(0, int(estimated_tokens)),
                    )
                    self.request_events.append(now)
                    self.token_events.append(token_event)
                    self.last_request_admission = now
                    self.condition.notify_all()
                    return RateLimitReservation(
                        reserved_tokens=token_event.tokens,
                        token_event=token_event,
                    )
                wait_started = time.monotonic()
                self.condition.wait(timeout=wait_seconds)
                self.total_wait_seconds += max(0.0, time.monotonic() - wait_started)

    def settle(self, reservation: RateLimitReservation, actual_tokens: int, *, http_status: int | None = None) -> None:
        """按真实 usage 回写 token 消耗，并记录服务端限流诊断。"""

        with self.condition:
            reservation.token_event.tokens = max(0, int(actual_tokens))
            if http_status == 429:
                self.rate_limit_429_count += 1
            self.condition.notify_all()

    def note_retry_after(self, retry_after_seconds: float | None) -> None:
        """记录 provider 返回的 retry-after 并阻塞后续请求。"""

        if retry_after_seconds is None:
            return
        with self.condition:
            self.last_retry_after_seconds = max(0.0, float(retry_after_seconds))
            self.not_before_monotonic = max(
                self.not_before_monotonic or 0.0,
                time.monotonic() + self.last_retry_after_seconds,
            )
            self.condition.notify_all()

    def snapshot(self) -> dict[str, float | int | None]:
        """返回当前限流状态，供进度心跳展示。"""

        with self.condition:
            return _snapshot_payload(
                target_rpm=self.requests_per_minute,
                local_429_count=self.rate_limit_429_count,
                total_wait_seconds=self.total_wait_seconds,
                local_retry_after=self.last_retry_after_seconds,
                cooldown_remaining=self._cooldown_wait_seconds(time.monotonic()),
            )

    def _evict_expired(self, now: float) -> None:
        while self.request_events and now - self.request_events[0] >= self.window_seconds:
            self.request_events.popleft()
        while self.token_events and now - self.token_events[0].timestamp >= self.window_seconds:
            self.token_events.popleft()

    def _request_wait_seconds(self, now: float) -> float:
        spacing_wait = 0.0
        if self.last_request_admission is not None and self.request_spacing_seconds > 0:
            spacing_wait = max(0.0, self.request_spacing_seconds - (now - self.last_request_admission))
        if not self.requests_per_minute:
            return spacing_wait
        if len(self.request_events) < self.requests_per_minute:
            return spacing_wait
        oldest = self.request_events[0]
        window_wait = max(0.0, self.window_seconds - (now - oldest))
        return max(spacing_wait, window_wait)

    def _cooldown_wait_seconds(self, now: float) -> float:
        if self.not_before_monotonic is None:
            return 0.0
        remaining = self.not_before_monotonic - now
        if remaining <= 0:
            self.not_before_monotonic = None
            return 0.0
        return remaining

    def _token_wait_seconds(self, now: float, estimated_tokens: int) -> float:
        if not self.tokens_per_minute:
            return 0.0
        total_tokens = sum(event.tokens for event in self.token_events)
        if total_tokens + estimated_tokens <= self.tokens_per_minute:
            return 0.0
        excess = total_tokens + estimated_tokens - self.tokens_per_minute
        released = 0
        for event in self.token_events:
            released += event.tokens
            if released >= excess:
                return max(0.0, self.window_seconds - (now - event.timestamp))
        return self.window_seconds


class PersistentSlidingWindowRateLimiter:
    """基于文件账本的跨进程 provider/model 限流器。"""

    def __init__(
        self,
        *,
        scope_key: str,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        window_seconds: float = 60.0,
        store: FileRateLimitStateStore | None = None,
    ) -> None:
        self.scope_key = scope_key
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = window_seconds
        self.request_spacing_seconds = _request_spacing_seconds(requests_per_minute, window_seconds)
        self.store = store or FileRateLimitStateStore.for_scope(
            scope_key=scope_key,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            window_seconds=window_seconds,
        )
        self.rate_limit_429_count = 0
        self.total_wait_seconds = 0.0
        self.last_retry_after_seconds: float | None = None

    def acquire(self, estimated_tokens: int) -> RateLimitReservation:
        """阻塞直到跨进程共享窗口允许下一次请求进入。"""

        event_id = uuid.uuid4().hex
        while True:
            with self.store.edit() as state:
                now = time.time()
                state.evict_expired(now, self.window_seconds)
                wait_seconds = max(
                    self._request_wait_seconds(state, now),
                    self._token_wait_seconds(state, now, estimated_tokens),
                    self._cooldown_wait_seconds(state, now),
                )
                if wait_seconds <= 0:
                    tokens = max(0, int(estimated_tokens))
                    state.request_events.append(now)
                    state.token_events.append(
                        PersistentTokenEvent(
                            event_id=event_id,
                            timestamp=now,
                            tokens=tokens,
                        )
                    )
                    state.last_request_admission = now
                    return RateLimitReservation(
                        reserved_tokens=tokens,
                        token_event=_TokenEvent(timestamp=now, tokens=tokens),
                        event_id=event_id,
                    )
            wait_started = time.monotonic()
            time.sleep(wait_seconds)
            self.total_wait_seconds += max(0.0, time.monotonic() - wait_started)

    def settle(self, reservation: RateLimitReservation, actual_tokens: int, *, http_status: int | None = None) -> None:
        """按真实响应回写跨进程 token 账本。"""

        with self.store.edit() as state:
            now = time.time()
            state.evict_expired(now, self.window_seconds)
            for event in state.token_events:
                if event.event_id == reservation.event_id:
                    event.tokens = max(0, int(actual_tokens))
                    break
            if http_status == 429:
                self.rate_limit_429_count += 1
                state.rate_limit_429_count += 1

    def note_retry_after(self, retry_after_seconds: float | None) -> None:
        """把 provider 的 Retry-After 写入跨进程冷却窗口。"""

        if retry_after_seconds is None:
            return
        with self.store.edit() as state:
            retry_after = max(0.0, float(retry_after_seconds))
            self.last_retry_after_seconds = retry_after
            state.last_retry_after_seconds = retry_after
            state.not_before_wall = max(
                state.not_before_wall or 0.0,
                time.time() + retry_after,
            )

    def snapshot(self) -> dict[str, float | int | None]:
        """返回当前限流状态，供进度心跳展示。"""

        with self.store.edit() as state:
            now = time.time()
            state.evict_expired(now, self.window_seconds)
            cooldown_remaining = self._cooldown_wait_seconds(state, now)
            payload = _snapshot_payload(
                target_rpm=self.requests_per_minute,
                local_429_count=self.rate_limit_429_count,
                total_wait_seconds=self.total_wait_seconds,
                local_retry_after=self.last_retry_after_seconds,
                cooldown_remaining=cooldown_remaining,
            )
            payload["cross_process_rate_limit_429_count"] = state.rate_limit_429_count
            payload["cross_process_last_retry_after_seconds"] = state.last_retry_after_seconds
            return payload

    def _request_wait_seconds(self, state: PersistentRateLimitState, now: float) -> float:
        spacing_wait = 0.0
        if state.last_request_admission is not None and self.request_spacing_seconds > 0:
            spacing_wait = max(0.0, self.request_spacing_seconds - (now - state.last_request_admission))
        if not self.requests_per_minute:
            return spacing_wait
        if len(state.request_events) < self.requests_per_minute:
            return spacing_wait
        oldest = min(state.request_events)
        window_wait = max(0.0, self.window_seconds - (now - oldest))
        return max(spacing_wait, window_wait)

    def _cooldown_wait_seconds(self, state: PersistentRateLimitState, now: float) -> float:
        if state.not_before_wall is None:
            return 0.0
        remaining = state.not_before_wall - now
        if remaining <= 0:
            state.not_before_wall = None
            return 0.0
        return remaining

    def _token_wait_seconds(self, state: PersistentRateLimitState, now: float, estimated_tokens: int) -> float:
        if not self.tokens_per_minute:
            return 0.0
        total_tokens = sum(event.tokens for event in state.token_events)
        if total_tokens + estimated_tokens <= self.tokens_per_minute:
            return 0.0
        excess = total_tokens + estimated_tokens - self.tokens_per_minute
        released = 0
        for event in sorted(state.token_events, key=lambda item: item.timestamp):
            released += event.tokens
            if released >= excess:
                return max(0.0, self.window_seconds - (now - event.timestamp))
        return self.window_seconds


class RequestThrottle:
    """统一控制真实网络请求的并发、RPM 与 TPM。"""

    _shared_states: dict[tuple[str, int, int | None, int | None, float], _SharedThrottleState] = {}
    _shared_states_lock = threading.Lock()

    def __init__(
        self,
        *,
        max_concurrent_requests: int | None,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        window_seconds: float = 60.0,
        scope_key: str | None = None,
    ) -> None:
        max_workers = max(1, int(max_concurrent_requests or 1))
        self.max_concurrent_requests = max_workers
        if scope_key is None:
            self._semaphore = threading.BoundedSemaphore(max_workers)
            self.limiter = SlidingWindowRateLimiter(
                requests_per_minute=requests_per_minute,
                tokens_per_minute=tokens_per_minute,
                window_seconds=window_seconds,
            )
        else:
            state = self._shared_state(
                scope_key=scope_key,
                max_workers=max_workers,
                requests_per_minute=requests_per_minute,
                tokens_per_minute=tokens_per_minute,
                window_seconds=window_seconds,
            )
            self._semaphore = state.semaphore
            self.limiter = state.limiter

    @classmethod
    def for_model(
        cls,
        model_config,
        *,
        max_concurrent_requests: int | None,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        window_seconds: float = 60.0,
    ) -> RequestThrottle:
        """按 provider/model 共享限流窗口，避免矩阵 family 或进程边界重置真实配额。"""

        return cls(
            max_concurrent_requests=max_concurrent_requests,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            window_seconds=window_seconds,
            scope_key=f"{model_config.provider}:{model_config.model_id}",
        )

    @contextmanager
    def reserve(self, estimated_tokens: int) -> Iterator[RateLimitReservation]:
        """为一次真实 provider 请求同时占用并发槽和限流配额。"""

        self._semaphore.acquire()
        reservation = self.limiter.acquire(estimated_tokens)
        try:
            yield reservation
        finally:
            self._semaphore.release()

    def settle(self, reservation: RateLimitReservation, actual_tokens: int, *, http_status: int | None = None) -> None:
        """按真实响应对账本次请求。"""

        self.limiter.settle(reservation, actual_tokens, http_status=http_status)

    def note_retry_after(self, retry_after_seconds: float | None) -> None:
        """记录 provider 的 retry-after 诊断信息。"""

        self.limiter.note_retry_after(retry_after_seconds)

    def snapshot(self) -> dict[str, float | int | None]:
        """返回运行期吞吐控制快照。"""

        return {
            "max_concurrent_requests": self.max_concurrent_requests,
            **self.limiter.snapshot(),
        }

    @classmethod
    def _shared_state(
        cls,
        *,
        scope_key: str,
        max_workers: int,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        window_seconds: float,
    ) -> _SharedThrottleState:
        state_key = (scope_key, max_workers, requests_per_minute, tokens_per_minute, float(window_seconds))
        with cls._shared_states_lock:
            state = cls._shared_states.get(state_key)
            if state is None:
                state = _SharedThrottleState(
                    semaphore=threading.BoundedSemaphore(max_workers),
                    limiter=PersistentSlidingWindowRateLimiter(
                        scope_key=scope_key,
                        requests_per_minute=requests_per_minute,
                        tokens_per_minute=tokens_per_minute,
                        window_seconds=window_seconds,
                    ),
                )
                cls._shared_states[state_key] = state
            return state


def _request_spacing_seconds(requests_per_minute: int | None, window_seconds: float) -> float:
    if not requests_per_minute:
        return 0.0
    return window_seconds / requests_per_minute


def _snapshot_payload(
    *,
    target_rpm: int | None,
    local_429_count: int,
    total_wait_seconds: float,
    local_retry_after: float | None,
    cooldown_remaining: float,
) -> dict[str, float | int | None]:
    return {
        "target_network_rpm": target_rpm,
        "effective_network_rpm_limit": target_rpm,
        "rate_limit_429_count": local_429_count,
        "rate_limit_wait_seconds": round(total_wait_seconds, 2),
        "last_retry_after_seconds": local_retry_after,
        "cooldown_remaining_seconds": round(cooldown_remaining, 2),
    }
