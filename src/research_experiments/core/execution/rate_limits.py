"""共享的网络请求限流与并发控制。"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

STANDARD_MAX_CONCURRENT_REQUESTS = 15
STANDARD_REQUESTS_PER_MINUTE_LIMIT = 18
STANDARD_TOKENS_PER_MINUTE_LIMIT = 9000000


def standard_runtime_limits() -> dict[str, int]:
    """Return the project's standard runtime throttle defaults."""

    return {
        "max_concurrent_requests": STANDARD_MAX_CONCURRENT_REQUESTS,
        "requests_per_minute_limit": STANDARD_REQUESTS_PER_MINUTE_LIMIT,
        "tokens_per_minute_limit": STANDARD_TOKENS_PER_MINUTE_LIMIT,
    }


@dataclass
class _TokenEvent:
    """A token reservation that is still inside the active sliding window."""

    timestamp: float
    tokens: int


@dataclass(frozen=True)
class RateLimitReservation:
    """A reserved request slot that can be settled against actual usage later."""

    reserved_tokens: int
    token_event: _TokenEvent


@dataclass(frozen=True)
class _SharedThrottleState:
    """Per-model shared in-process semaphore and limiter state."""

    semaphore: threading.BoundedSemaphore
    limiter: SlidingWindowRateLimiter


class SlidingWindowRateLimiter:
    """Thread-safe RPM/TPM sliding window limiter for one process."""

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
        self.request_spacing_seconds = _request_spacing_seconds(requests_per_minute, window_seconds)

    def acquire(self, estimated_tokens: int) -> RateLimitReservation:
        """Block until the next request fits inside the RPM/TPM window."""

        with self.condition:
            while True:
                now = time.monotonic()
                self._evict_expired(now)
                wait_seconds = max(
                    self._request_wait_seconds(now),
                    self._token_wait_seconds(now, estimated_tokens),
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

    def settle(
        self,
        reservation: RateLimitReservation,
        actual_tokens: int,
        *,
        http_status: int | None = None,
    ) -> None:
        """Replace reserved tokens with actual usage and record any 429."""

        with self.condition:
            reservation.token_event.tokens = max(0, int(actual_tokens))
            if http_status == 429:
                self.rate_limit_429_count += 1
            self.condition.notify_all()

    def snapshot(self) -> dict[str, float | int | None]:
        """Return a progress-compatible limiter snapshot."""

        with self.condition:
            return {
                "target_network_rpm": self.requests_per_minute,
                "effective_network_rpm_limit": self.requests_per_minute,
                "rate_limit_429_count": self.rate_limit_429_count,
                "rate_limit_wait_seconds": round(self.total_wait_seconds, 2),
                "last_retry_after_seconds": None,
                "cooldown_remaining_seconds": 0.0,
            }

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


class RequestThrottle:
    """Control real network request concurrency plus RPM/TPM in one process."""

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
        """Share one in-process limiter window per provider/model pair."""

        return cls(
            max_concurrent_requests=max_concurrent_requests,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            window_seconds=window_seconds,
            scope_key=f"{model_config.provider}:{model_config.model_id}",
        )

    @contextmanager
    def reserve(self, estimated_tokens: int) -> Iterator[RateLimitReservation]:
        """Reserve both a concurrency slot and a rate-limit slot for one request."""

        self._semaphore.acquire()
        reservation = self.limiter.acquire(estimated_tokens)
        try:
            yield reservation
        finally:
            self._semaphore.release()

    def settle(
        self,
        reservation: RateLimitReservation,
        actual_tokens: int,
        *,
        http_status: int | None = None,
    ) -> None:
        """Settle a prior reservation against actual usage."""

        self.limiter.settle(reservation, actual_tokens, http_status=http_status)

    def snapshot(self) -> dict[str, float | int | None]:
        """Return a progress snapshot for the current throttle state."""

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
                    limiter=SlidingWindowRateLimiter(
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
