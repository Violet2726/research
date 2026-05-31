"""覆盖共享 runner 的流式批处理原语。"""

from __future__ import annotations

import threading

import pytest

from research_experiments.core.execution.runner_common import iter_indexed_batch


def test_iter_indexed_batch_yields_completed_sample_before_slow_sample() -> None:
    slow_can_finish = threading.Event()

    def _worker(item: str) -> str:
        if item == "slow":
            slow_can_finish.wait(timeout=1.0)
        return item.upper()

    result_iter = iter_indexed_batch(
        ["slow", "fast"],
        worker=_worker,
        max_concurrent_requests=2,
    )
    first_index, first_result = next(iter(result_iter))
    slow_can_finish.set()
    remaining = list(result_iter)

    assert (first_index, first_result) == (1, "FAST")
    assert remaining == [(0, "SLOW")]


def test_iter_indexed_batch_propagates_worker_errors() -> None:
    def _worker(item: int) -> int:
        if item == 2:
            raise ValueError("boom")
        return item

    with pytest.raises(ValueError, match="boom"):
        list(iter_indexed_batch([2], worker=_worker, max_concurrent_requests=1))
