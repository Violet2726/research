from __future__ import annotations

import hashlib
import json
from collections import Counter
from types import SimpleNamespace

from research_experiments.core.data.datasets import load_samples
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
)
from research_experiments.families.contrastive_active_testing.run.execute import (
    _build_comparison_method_audit,
    _frozen_component_hashes,
    _select_kernel_confirmation_strata,
    _select_phase_samples,
    _selected_sample_manifest,
    _validate_kernel_freeze,
)
from research_experiments.families.registry import get_family_registration

EXPERIMENT = "configs/families/contrastive_active_testing/experiments/catch_gate.toml"
CERT_V2_EXPERIMENT = "configs/families/contrastive_active_testing/experiments/catch_cert_v2_development.toml"
KERNEL_EXPERIMENT = "configs/families/contrastive_active_testing/experiments/catch_kernel_d1.toml"


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
    assert experiment.cache_policy == "global_validated_response_v3"
    assert registration.prototype == "shared_stage_policy"
    assert registration.artifact_schema.progress_path == "progress.json"
    assert registration.artifact_schema.validation_path == "run_validation.json"
    assert registration.artifact_schema.diagnostic_paths == ()
    frozen_components = _frozen_component_hashes(experiment)
    assert "src/research_experiments/families/contrastive_active_testing/icv.py" in frozen_components
    assert "configs/core/shared/benchmarks/splits/dgcr_dev100/bbeh/bbeh-main-seed42.json" in frozen_components


def test_cert_v2_global_cache_policy_and_frozen_components_are_versioned() -> None:
    experiment = load_experiment_config(CERT_V2_EXPERIMENT)
    protocol = load_protocol_config(experiment.protocol)
    assert protocol.protocol_version == "catch_cert_v2"
    assert protocol.max_selected_tests == 6
    assert experiment.cache_policy == "global_validated_response_v3"
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


def test_kernel_confirmation_uses_frozen_hash_selection_and_all_comparators() -> None:
    experiment = load_experiment_config(KERNEL_EXPERIMENT)
    protocol = load_protocol_config(experiment.protocol)
    phase = experiment.raw["phases"]["confirmation"]
    benchmark = load_phase_benchmarks(experiment, "confirmation")[0]
    first = _select_phase_samples(benchmark, phase, "confirmation")
    second = _select_phase_samples(benchmark, phase, "confirmation")

    assert phase["hash_sample_selection"] is True
    assert phase["selection_strategy"] == "kernel_confirmation_stratified_sha256"
    assert phase["selection_seed"] == 42
    assert phase["run_direct_judge"] is True
    assert protocol.role_max_tokens == 16_384
    assert protocol.judge_max_tokens == 4_096
    assert protocol.max_selected_tests == 24
    assert experiment.config_warnings == ()
    assert experiment.cache_policy == "global_validated_response_v3"
    assert len(first) == 200
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]


def test_kernel_confirmation_stratifies_musr_and_seqbench_without_gold() -> None:
    musr = [
        SimpleNamespace(sample_id=f"{task}-{index}", metadata={"task": task})
        for task in ("object_placements", "team_allocation", "murder_mysteries")
        for index in range(10)
    ]
    musr_selected = _select_kernel_confirmation_strata(
        musr,
        benchmark_slug="musr",
        phase_name="confirmation",
        seed=42,
        limit=9,
    )
    assert Counter(item.metadata["task"] for item in musr_selected) == {
        "object_placements": 3,
        "team_allocation": 3,
        "murder_mysteries": 3,
    }

    seq = [
        SimpleNamespace(
            sample_id=f"b{backtrack}-n{noise}-{index}",
            metadata={
                "backtracking_count_B": backtrack,
                "noise_ratio_N": noise,
                "logical_depth_L": index + backtrack * 100,
                "task": f"B{backtrack}_N{noise}",
            },
        )
        for backtrack in range(3)
        for noise in (0.0, 1.0)
        for index in range(10)
    ]
    seq_selected = _select_kernel_confirmation_strata(
        seq,
        benchmark_slug="seqbench",
        phase_name="confirmation",
        seed=42,
        limit=18,
    )
    assert {(item.metadata["backtracking_count_B"], item.metadata["noise_ratio_N"]) for item in seq_selected} == {
        (backtrack, noise) for backtrack in range(3) for noise in (0.0, 1.0)
    }
    manifest = _selected_sample_manifest({"seqbench": seq_selected}, phase_name="confirmation")
    assert manifest["seqbench"]["count"] == 18
    assert len(manifest["seqbench"]["sha256"]) == 64


def test_kernel_freeze_validates_components_and_exact_selection_hashes(tmp_path) -> None:
    unsigned = {
        "schema_version": "catch_kernel_d2_freeze_v1",
        "component_sha256": {"kernel.py": "abc"},
        "selected_sample_manifest": {"bbeh": {"sha256": "selection"}},
        "global_seed": 42,
    }
    payload = {
        **unsigned,
        "sha256": hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    valid = _validate_kernel_freeze(
        path,
        component_hashes={"kernel.py": "abc"},
        selection_manifest={"bbeh": {"sha256": "selection"}},
        expected_metadata={"global_seed": 42},
    )
    invalid = _validate_kernel_freeze(
        path,
        component_hashes={"kernel.py": "changed"},
        selection_manifest={"bbeh": {"sha256": "selection"}},
    )
    invalid_seed = _validate_kernel_freeze(
        path,
        component_hashes={"kernel.py": "abc"},
        selection_manifest={"bbeh": {"sha256": "selection"}},
        expected_metadata={"global_seed": 7},
    )
    assert valid["valid"] is True
    assert invalid == {"valid": False, "reason": "component_hash_mismatch", "path": path.as_posix()}
    assert invalid_seed == {
        "valid": False,
        "reason": "freeze_metadata_mismatch:global_seed",
        "path": path.as_posix(),
    }


def test_comparison_audit_detects_missing_methods_after_dataset_metrics_exist() -> None:
    audit = _build_comparison_method_audit(
        {"datasets": {"bbeh": {"methods": {"sc_5": {}, "catch_kernel": {}}}}},
        ["sc_5", "catch_kernel", "catch_cert_v2"],
    )
    assert audit["bbeh"]["available"] == ["catch_kernel", "sc_5"]
    assert audit["bbeh"]["missing"] == ["catch_cert_v2"]
    assert audit["bbeh"]["complete"] is False


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
