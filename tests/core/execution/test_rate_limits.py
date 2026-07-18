"""Tests for the minimal in-process RPM limiter."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from research_experiments.core.execution.rate_limits import RequestThrottle, SlidingWindowRateLimiter


def test_rate_limiter_without_waiting() -> None:
    limiter = SlidingWindowRateLimiter(requests_per_minute=100)
    started = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    assert time.monotonic() - started < 1.0


def test_rate_limiter_without_rpm_admits_immediately() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=None,
        window_seconds=0.05,
    )
    started = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    assert time.monotonic() - started < 0.02


def test_rate_limiter_uses_full_configured_rpm_without_extra_headroom() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=12,
        window_seconds=6.0,
    )
    limiter.acquire()
    started = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - started
    assert 0.45 <= elapsed < 0.58


def test_rate_limiter_snapshot_reports_admission_configuration_only() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=100,
        window_seconds=0.05,
    )
    limiter.acquire()
    first_snapshot = limiter.snapshot()

    limiter.acquire()
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
        window_seconds=0.05,
    )
    second = RequestThrottle.for_model(
        model,
        max_concurrent_requests=2,
        requests_per_minute=1,
        window_seconds=0.05,
    )

    with first.reserve():
        pass

    started = time.monotonic()
    with second.reserve():
        pass

    assert time.monotonic() - started >= 0.04


def test_request_throttle_snapshot_keeps_429_metrics_local_even_when_window_is_shared() -> None:
    model = SimpleNamespace(provider="demo-provider", model_id=f"demo-model-{time.monotonic_ns()}")
    first = RequestThrottle.for_model(
        model,
        max_concurrent_requests=2,
        requests_per_minute=None,
        window_seconds=0.05,
    )
    second = RequestThrottle.for_model(
        model,
        max_concurrent_requests=2,
        requests_per_minute=None,
        window_seconds=0.05,
    )

    with first.reserve():
        pass
    first.settle(http_status=429)

    assert first.snapshot()["rate_limit_429_count"] == 1
    assert second.snapshot()["rate_limit_429_count"] == 0


def test_request_throttle_reports_real_peak_concurrency_and_admission_rpm() -> None:
    throttle = RequestThrottle(
        max_concurrent_requests=4,
        requests_per_minute=None,
        window_seconds=0.1,
    )

    def request() -> None:
        with throttle.reserve():
            time.sleep(0.02)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: request(), range(60)))

    snapshot = throttle.snapshot()
    assert snapshot["peak_active_requests"] == 4
    assert snapshot["active_requests"] == 0
    assert snapshot["queued_requests"] == 0
    assert snapshot["admission_rpm"] > 0
