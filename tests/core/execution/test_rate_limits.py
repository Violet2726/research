"""Tests for the minimal in-process RPM/TPM limiter."""

from __future__ import annotations

import time
from types import SimpleNamespace

from research_experiments.core.execution.rate_limits import RequestThrottle, SlidingWindowRateLimiter


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


def test_rate_limiter_uses_full_configured_rpm_without_extra_headroom() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=12,
        tokens_per_minute=None,
        window_seconds=6.0,
    )
    limiter.acquire(1)
    started = time.monotonic()
    limiter.acquire(1)
    elapsed = time.monotonic() - started
    assert 0.45 <= elapsed < 0.58


def test_rate_limiter_snapshot_reports_admission_configuration_only() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=100,
        tokens_per_minute=None,
        window_seconds=0.05,
    )
    first = limiter.acquire(1)
    limiter.settle(first, 1)
    first_snapshot = limiter.snapshot()

    second = limiter.acquire(1)
    limiter.settle(second, 1)
    second_snapshot = limiter.snapshot()

    assert first_snapshot["effective_network_rpm_limit"] == 100
    assert second_snapshot["effective_network_rpm_limit"] == 100
    assert "rate_limit_429_count" not in second_snapshot
    assert "rate_limit_wait_seconds" not in second_snapshot
    assert second_snapshot["last_retry_after_seconds"] is None
    assert second_snapshot["cooldown_remaining_seconds"] == 0.0


def test_request_throttle_for_model_shares_provider_window_across_instances() -> None:
    model = SimpleNamespace(provider="demo-provider", model_id=f"demo-model-{time.monotonic_ns()}")
    first = RequestThrottle.for_model(
        model,
        max_concurrent_requests=2,
        requests_per_minute=1,
        tokens_per_minute=None,
        window_seconds=0.05,
    )
    second = RequestThrottle.for_model(
        model,
        max_concurrent_requests=2,
        requests_per_minute=1,
        tokens_per_minute=None,
        window_seconds=0.05,
    )

    with first.reserve(1) as reservation:
        first.settle(reservation, 1)

    started = time.monotonic()
    with second.reserve(1) as reservation:
        second.settle(reservation, 1)

    assert time.monotonic() - started >= 0.04


def test_request_throttle_snapshot_keeps_429_metrics_local_even_when_window_is_shared() -> None:
    model = SimpleNamespace(provider="demo-provider", model_id=f"demo-model-{time.monotonic_ns()}")
    first = RequestThrottle.for_model(
        model,
        max_concurrent_requests=2,
        requests_per_minute=None,
        tokens_per_minute=None,
        window_seconds=0.05,
    )
    second = RequestThrottle.for_model(
        model,
        max_concurrent_requests=2,
        requests_per_minute=None,
        tokens_per_minute=None,
        window_seconds=0.05,
    )

    with first.reserve(1) as reservation:
        first.settle(reservation, 1, http_status=429)

    assert first.snapshot()["rate_limit_429_count"] == 1
    assert second.snapshot()["rate_limit_429_count"] == 0
