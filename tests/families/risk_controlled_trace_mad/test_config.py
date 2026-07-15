import pytest

from research_experiments.families.risk_controlled_trace_mad.config import (
    load_experiment_config,
    load_protocol_config,
    load_version_registry,
    phase_methods,
    require_active_version,
)
from research_experiments.families.risk_controlled_trace_mad.run.hsgsa_execute import _require_development_gate

EXPERIMENT = "configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml"


def test_unified_experiment_has_one_active_version_and_canonical_phases() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    registry = load_version_registry(experiment.version_registry)
    protocol = load_protocol_config(experiment.protocol)
    assert registry.active_version == experiment.active_version == "v5_hsgsa"
    assert [key for key, value in registry.versions.items() if value.status == "active"] == ["v5_hsgsa"]
    assert set(experiment.raw["phases"]) == {"replay_dev_seed42", "confirm_seed42"}
    assert protocol.max_logical_calls == 11
    assert protocol.max_network_attempts == 50_000
    assert phase_methods(experiment, "confirm_seed42")[-1] == "hsgsa_unanimous_3"
    assert experiment.primary_model_ref == "xiaomimimo/mimo-v2.5"


def test_retired_versions_cannot_run() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    registry = load_version_registry(experiment.version_registry)
    with pytest.raises(ValueError, match="cannot be run"):
        require_active_version(registry, "v3_rcta")
    with pytest.raises(ValueError, match="cannot be run"):
        require_active_version(registry, "v4_evf")
    assert require_active_version(registry, None).version_id == "v5_hsgsa"


def test_confirmation_is_locked_when_frozen_development_audit_fails() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    with pytest.raises(ValueError, match="pre-registered stop rule"):
        _require_development_gate(dict(experiment.raw["phases"]["confirm_seed42"]))
