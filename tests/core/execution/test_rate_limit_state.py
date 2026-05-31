"""覆盖跨进程限流状态存储。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research_experiments.core.execution.rate_limit_state import FileRateLimitStateStore, PersistentTokenEvent


def test_file_rate_limit_state_store_serializes_same_process_writes(tmp_path: Path) -> None:
    store = FileRateLimitStateStore(
        state_path=tmp_path / "scope.json",
        lock_path=tmp_path / "scope.lock",
    )

    def append_event(index: int) -> None:
        with store.edit() as state:
            state.request_events.append(float(index))
            state.token_events.append(
                PersistentTokenEvent(
                    event_id=str(index),
                    timestamp=float(index),
                    tokens=1,
                )
            )

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(append_event, index) for index in range(40)]
        for future in futures:
            future.result()

    state = store.read()
    assert len(state.request_events) == 40
    assert len(state.token_events) == 40
    assert not list(tmp_path.glob("*.tmp"))
