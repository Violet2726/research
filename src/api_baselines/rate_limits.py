from __future__ import annotations

from collections import deque
import threading
import time


class SlidingWindowRateLimiter:
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
        self.token_events: deque[tuple[float, int]] = deque()
        self.condition = threading.Condition()

    def acquire(self, estimated_tokens: int) -> None:
        with self.condition:
            while True:
                now = time.monotonic()
                self._evict_expired(now)

                request_wait = self._request_wait_seconds(now)
                token_wait = self._token_wait_seconds(now, estimated_tokens)
                wait_seconds = max(request_wait, token_wait)
                if wait_seconds <= 0:
                    self.request_events.append(now)
                    self.token_events.append((now, estimated_tokens))
                    self.condition.notify_all()
                    return
                self.condition.wait(timeout=wait_seconds)

    def _evict_expired(self, now: float) -> None:
        while self.request_events and now - self.request_events[0] >= self.window_seconds:
            self.request_events.popleft()
        while self.token_events and now - self.token_events[0][0] >= self.window_seconds:
            self.token_events.popleft()

    def _request_wait_seconds(self, now: float) -> float:
        if not self.requests_per_minute:
            return 0.0
        if len(self.request_events) < self.requests_per_minute:
            return 0.0
        oldest = self.request_events[0]
        return max(0.0, self.window_seconds - (now - oldest))

    def _token_wait_seconds(self, now: float, estimated_tokens: int) -> float:
        if not self.tokens_per_minute:
            return 0.0
        total_tokens = sum(tokens for _, tokens in self.token_events)
        if total_tokens + estimated_tokens <= self.tokens_per_minute:
            return 0.0

        excess = total_tokens + estimated_tokens - self.tokens_per_minute
        released = 0
        for timestamp, tokens in self.token_events:
            released += tokens
            if released >= excess:
                return max(0.0, self.window_seconds - (now - timestamp))
        return self.window_seconds
