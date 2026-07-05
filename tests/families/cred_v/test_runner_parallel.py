from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from research_experiments.families.cred_v.run import execute as execute_mod


class _CacheRouterStub:
    def for_request_target(self, **_: Any) -> object:
        return object()


def _run_parallel_helper(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slugs: list[str],
    sleeps: dict[str, float],
) -> tuple[list[Any], int]:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_load_selected_samples(benchmark: Any, split_name: str) -> list[Any]:
        return [SimpleNamespace(sample_id=f"{benchmark.slug}:{split_name}")]

    def fake_resolve_split_name(experiment: Any, phase_name: str, benchmark_slug: str) -> str:
        return f"{phase_name}:{benchmark_slug}"

    def fake_run_cred_batch(**kwargs: Any):
        nonlocal active, max_active
        dataset = str(kwargs["benchmark_slug"])
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(sleeps.get(dataset, 0.0))
        with lock:
            active -= 1
        yield (
            0,
            [{"dataset": dataset, "method_name": "cred_stage_a"}],
            [{"dataset": dataset, "message_type": "stage_a"}],
            [{"dataset": dataset, "router": "row"}],
            [{"dataset": dataset, "method_name": "cred_rfs_vote_5_anchor"}],
        )

    monkeypatch.setattr(execute_mod, "load_selected_samples", fake_load_selected_samples)
    monkeypatch.setattr(execute_mod, "resolve_split_name", fake_resolve_split_name)
    monkeypatch.setattr(execute_mod, "run_cred_batch", fake_run_cred_batch)

    benchmarks = [SimpleNamespace(slug=slug) for slug in slugs]
    experiment = SimpleNamespace(
        max_concurrent_requests=8,
        control_methods=[],
        global_seed=42,
        control_prompt_version="single_agent_free_text_v1",
    )
    backbone = SimpleNamespace(provider="provider", model_id="model")

    results = list(
        execute_mod._run_dataset_batches_parallel(
            run_id="run-1",
            benchmarks=benchmarks,
            experiment=experiment,
            phase_name="count100",
            protocol=SimpleNamespace(),
            controls={},
            backbone=backbone,
            provider=object(),
            cache_router=_CacheRouterStub(),
            throttle=object(),
            verifier_providers=[],
        )
    )
    return results, max_active


def test_cred_v_datasets_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    _, max_active = _run_parallel_helper(
        monkeypatch,
        slugs=["math500", "hotpotqa"],
        sleeps={"math500": 0.15, "hotpotqa": 0.15},
    )

    assert max_active == 2


def test_cred_v_dataset_results_preserve_benchmark_order(monkeypatch: pytest.MonkeyPatch) -> None:
    results, _ = _run_parallel_helper(
        monkeypatch,
        slugs=["slow_dataset", "fast_dataset"],
        sleeps={"slow_dataset": 0.15, "fast_dataset": 0.0},
    )

    assert [result.dataset_slug for result in results] == ["slow_dataset", "fast_dataset"]
    assert [result.prediction_rows[0]["dataset"] for result in results] == ["slow_dataset", "fast_dataset"]
