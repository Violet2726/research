from __future__ import annotations

from research_experiments.core.data.datasets import load_samples
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
)
from research_experiments.families.contrastive_active_testing.run.execute import _select_phase_samples
from research_experiments.families.registry import get_family_registration

EXPERIMENT = "configs/families/contrastive_active_testing/experiments/catch_gate.toml"


def test_frozen_protocol_and_registration_are_discoverable() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    protocol = load_protocol_config(experiment.protocol)
    registration = get_family_registration("contrastive_active_testing")

    assert protocol.d_min_grid == (2, 3, 4)
    assert protocol.margin_grid == (1, 2)
    assert protocol.role_max_tokens == 4_096
    assert protocol.max_network_attempts == 62_000
    assert registration.prototype == "shared_stage_policy"
    assert registration.artifact_schema.progress_path == "progress.json"
    assert registration.artifact_schema.validation_path == "run_validation.json"


def test_bbeh_100_200_and_confirmation_remainder_are_disjoint() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    benchmark = load_phase_benchmarks(experiment, "development")[0]
    development = _select_phase_samples(benchmark, experiment.raw["phases"]["development"], "development")
    heldout = _select_phase_samples(benchmark, experiment.raw["phases"]["heldout"], "heldout")
    confirmation = _select_phase_samples(benchmark, experiment.raw["phases"]["confirmation"], "confirmation")
    dev_ids = {sample.sample_id for sample in development}
    heldout_ids = {sample.sample_id for sample in heldout}
    confirmation_ids = {sample.sample_id for sample in confirmation}

    assert (len(dev_ids), len(heldout_ids), len(confirmation_ids)) == (100, 200, 4_220)
    assert not dev_ids & heldout_ids
    assert not dev_ids & confirmation_ids
    assert not heldout_ids & confirmation_ids


def test_all_geometric_shapes_samples_have_structured_options_after_rounding_note_fix() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    benchmark = load_phase_benchmarks(experiment, "development")[0]
    samples = [sample for sample in load_samples(benchmark) if sample.metadata.get("task") == "geometric_shapes"]
    assert len(samples) == 200
    assert all(sample.metadata.get("options") for sample in samples)

