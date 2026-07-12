import pytest

from research_experiments.families.risk_controlled_trace_mad.config import (
    load_experiment_config,
    load_protocol_config,
    load_version_registry,
    phase_methods,
    require_active_version,
)

EXPERIMENT = "configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml"


def test_unified_experiment_has_one_active_version_and_canonical_phases() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    registry = load_version_registry(experiment.version_registry)
    protocol = load_protocol_config(experiment.protocol)
    assert registry.active_version == experiment.active_version == "v4_evf"
    assert [key for key, value in registry.versions.items() if value.status == "active"] == ["v4_evf"]
    assert set(experiment.raw["phases"]) == {"count20_seed42", "count100_seed42", "count300_seed42", "full_seed42"}
    assert protocol.max_logical_calls == 10
    assert phase_methods(experiment, "count20_seed42")[-1] == "evf_mad_1"


def test_retired_versions_cannot_run() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    registry = load_version_registry(experiment.version_registry)
    with pytest.raises(ValueError, match="cannot be run"):
        require_active_version(registry, "v3_rcta")
    assert require_active_version(registry, None).version_id == "v4_evf"
