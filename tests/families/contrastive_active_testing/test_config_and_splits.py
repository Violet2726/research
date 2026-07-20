from __future__ import annotations

from collections import Counter

from research_experiments.core.data.datasets import load_samples
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
)
from research_experiments.families.contrastive_active_testing.run.execute import (
    _frozen_component_hashes,
    _select_phase_samples,
)
from research_experiments.families.registry import get_family_registration

EXPERIMENT = "configs/families/contrastive_active_testing/experiments/catch_gate.toml"
CERT_V2_EXPERIMENT = "configs/families/contrastive_active_testing/experiments/catch_cert_v2_development.toml"


def test_best_effort_protocol_and_registration_are_discoverable() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    protocol = load_protocol_config(experiment.protocol)
    registration = get_family_registration("contrastive_active_testing")

    assert protocol.d_min_grid == ()
    assert protocol.margin_grid == ()
    assert protocol.max_proposed_tests == 0
    assert protocol.role_max_tokens == 4_096
    assert protocol.max_network_attempts == 62_000
    assert protocol.protocol_version == "catch_v3"
    assert protocol.preflight_sample_count == 0
    assert protocol.coordinates_per_pair == 3
    assert protocol.pair_judge_count == 3
    assert experiment.cache_namespaces["development"] == "catch-dev-v3"
    assert experiment.baseline_cache_namespaces == {"development": "catch-dev-v1"}
    assert registration.prototype == "shared_stage_policy"
    assert registration.artifact_schema.progress_path == "progress.json"
    assert registration.artifact_schema.validation_path == "run_validation.json"
    assert registration.artifact_schema.diagnostic_paths == ()
    frozen_components = _frozen_component_hashes(experiment)
    assert "src/research_experiments/families/contrastive_active_testing/icv.py" in frozen_components
    assert "configs/core/shared/benchmarks/splits/dgcr_dev100/bbeh/bbeh-main-seed42.json" in frozen_components


def test_cert_v2_protocol_namespaces_and_frozen_components_are_versioned() -> None:
    experiment = load_experiment_config(CERT_V2_EXPERIMENT)
    protocol = load_protocol_config(experiment.protocol)
    assert protocol.protocol_version == "catch_cert_v2"
    assert protocol.max_selected_tests == 6
    assert experiment.cache_namespaces["development"] == "catch-dev-cert_v2"
    assert experiment.baseline_cache_namespaces["development"] == "catch-dev-cert_v1"
    assert experiment.readiness_assessment_path.as_posix().endswith(
        "local/analysis/catch_cert_v2_readiness_assessment.json"
    )
    assert experiment.config_warnings == ()
    frozen = _frozen_component_hashes(experiment)
    assert "src/research_experiments/families/contrastive_active_testing/certificates_v2.py" in frozen
    assert "src/research_experiments/families/contrastive_active_testing/cert_prompts_v2.py" in frozen


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


def test_full_bbeh_answer_contract_scan_has_frozen_inline_counts() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    benchmark = load_phase_benchmarks(experiment, "development")[0]
    samples = load_samples(benchmark)
    inline = [sample for sample in samples if sample.metadata["answer_contract"]["source_style"] == "inline"]
    counts = Counter(sample.metadata["task"] for sample in inline)

    assert len(samples) == 4_520
    assert len(inline) == 720
    assert counts == {
        "boolean_expressions": 200,
        "disambiguation_qa": 120,
        "nycc": 200,
        "hyperbaton": 200,
    }
    assert all(
        sample.metadata["answer_contract"]["kind"] == "multi_choice"
        for sample in inline
        if sample.metadata["task"] == "hyperbaton"
    )
