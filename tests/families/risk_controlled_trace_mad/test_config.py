from __future__ import annotations

from research_experiments.families.risk_controlled_trace_mad.config import (
    load_experiment_config,
    load_protocol_config,
    runtime_for_provider,
)


def test_canonical_phases_and_runtime_limits() -> None:
    experiment = load_experiment_config("configs/families/risk_controlled_trace_mad/experiments/rcta_mad.toml")
    assert set(experiment.raw["phases"]) == {"count20_seed42", "count300_seed42", "full_seed42"}
    assert runtime_for_provider(experiment, "dashscope").requests_per_minute_limit == 1000
    mimo = runtime_for_provider(experiment, "xiaomimimo")
    assert (mimo.requests_per_minute_limit, mimo.max_concurrent_requests) == (18, 8)
    protocol = load_protocol_config(experiment.protocol)
    assert (protocol.stage_a_candidates, protocol.sc_ceiling_candidates, protocol.trace_synthesizer_count) == (5, 9, 1)

