from __future__ import annotations

from dataclasses import replace

import pytest

from research_experiments.families.risk_controlled_trace_mad.config import load_experiment_config
from research_experiments.families.risk_controlled_trace_mad.run import execute as execute_runner


def test_historical_hsgsa_runner_is_locked_before_any_provider_work(tmp_path) -> None:
    source = load_experiment_config("configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml")
    raw = dict(source.raw)
    raw["phases"] = {
        "confirm_seed42": {
            "benchmark_slugs": ["bbeh"],
            "split_overrides": {"bbeh": "count20_seed42"},
            "methods": source.methods,
        }
    }
    experiment = replace(source, name="fake_smoke", raw=raw)
    with pytest.raises(ValueError, match="historical-only"):
        execute_runner.run_experiment(
            experiment,
            "confirm_seed42",
            run_root=tmp_path / "runs",
            cache_root=tmp_path / "cache",
        )
