
"""??????????????????"""

from __future__ import annotations

import time

from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter


def test_rate_limiter_without_waiting() -> None:
    limiter = SlidingWindowRateLimiter(requests_per_minute=100, tokens_per_minute=1000)
    started = time.monotonic()
    limiter.acquire(10)
    limiter.acquire(10)
    assert time.monotonic() - started < 1.0

def test_rate_limiter_settle_releases_reserved_tokens() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=None,
        tokens_per_minute=200,
        window_seconds=0.05,
    )
    reservation = limiter.acquire(90)
    limiter.settle(reservation, 10)
    started = time.monotonic()
    limiter.acquire(90)
    assert time.monotonic() - started < 0.02

