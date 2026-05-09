"""共享的滑动窗口限流器。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time


@dataclass
class _TokenEvent:
    """表示一个仍在限流窗口内的 token 预留事件。"""

    timestamp: float
    tokens: int


@dataclass(frozen=True)
class RateLimitReservation:
    """表示一次已占用配额、后续可按真实 usage 对账的预留。"""

    reserved_tokens: int
    token_event: _TokenEvent


class SlidingWindowRateLimiter:
    """线程安全地约束 RPM/TPM 全局配额。"""

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
        self.request_spacing_seconds = self._compute_request_spacing_seconds()

    def acquire(self, estimated_tokens: int) -> RateLimitReservation:
        """阻塞直到下一次请求可以安全进入全局窗口。"""
        with self.condition:
            while True:
                now = time.monotonic()
                self._evict_expired(now)
                request_wait = self._request_wait_seconds(now)
                token_wait = self._token_wait_seconds(now, estimated_tokens)
                wait_seconds = max(request_wait, token_wait)
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
                self.condition.wait(timeout=wait_seconds)

    def settle(self, reservation: RateLimitReservation, actual_tokens: int) -> None:
        """按真实 usage 回写这次请求的 token 消耗。"""
        actual = max(0, int(actual_tokens))
        with self.condition:
            reservation.token_event.tokens = actual
            self.condition.notify_all()

    def _compute_request_spacing_seconds(self) -> float:
        if not self.requests_per_minute:
            return 0.0
        effective_requests_per_minute = self.requests_per_minute
        if self.requests_per_minute > 10:
            effective_requests_per_minute = max(1, self.requests_per_minute - 2)
        return self.window_seconds / effective_requests_per_minute

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
