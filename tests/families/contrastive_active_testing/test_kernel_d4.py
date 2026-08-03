from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_experiments.core.data.datasets import DatasetSample, load_samples
from research_experiments.core.data.evaluation import score_prediction
from research_experiments.families.contrastive_active_testing.config import (
    CatchProtocolConfig,
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.d4_audit import (
    build_d4_gate_evidence,
    cohen_kappa,
    evaluate_d4_human_audit,
    gwet_ac1,
    validate_d4_gate_evidence,
)
from research_experiments.families.contrastive_active_testing.d4_contract import (
    D4_MAINLINE_PROTOCOL_VERSION,
)
from research_experiments.families.contrastive_active_testing.d4_data import (
    partition_latent_records,
    partition_text_records,
    science_record_eligible,
    sealed_manifest,
    text_sealed_manifest,
    text_sha256,
    validate_sealed_manifest,
    validate_text_sealed_manifest,
    write_latent_sealed_manifest_from_files,
    write_text_sealed_manifest_from_files,
)
from research_experiments.families.contrastive_active_testing.d4_protocol_validation import (
    evaluate_tagged_protocol_validation,
    validate_tagged_protocol_validation_assessment,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    D4RouteDecision,
    D4SolverResult,
    ProofPackageV2,
    SourceIRv3,
    compile_exact_source_ir,
    metamorphic_checks_passed,
    parse_source_ir_v3,
    risk_gate_snapshot,
    route_for_sample,
    run_metamorphic_checks,
    solve_source_ir,
)
from research_experiments.families.contrastive_active_testing.kernel_prompts import (
    build_d4_source_compiler_messages,
)
from research_experiments.families.contrastive_active_testing.run import sample as sample_runner
from research_experiments.families.contrastive_active_testing.run.execute import (
    D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS,
    _require_d4_sealed_manifests,
    _select_phase_samples,
    _selected_sample_manifest,
    _validate_d4_independent_protocol_validation_config,
    d4_source_compiler_smoke_snapshot,
    require_passing_d4_source_compiler_smoke,
    run_experiment,
)


def _mc_sample() -> DatasetSample:
    return DatasetSample(
        "musr",
        "d4-mc",
        "Choose one.",
        "A|||alpha",
        "",
        {
            "task": "object_placements",
            "answer_contract": {
                "kind": "single_choice",
                "options": [{"label": "A", "text": "alpha"}, {"label": "B", "text": "beta"}],
                "selection_mode": "single",
            },
        },
    )


def _shuffled_sample() -> DatasetSample:
    question = (
        "Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: "
        "Alice has a red ball, Bob has a blue ball, and Claire has a green ball. As the game progresses, "
        "pairs of players trade balls. First, Alice and Bob swap balls (let's call it Action 1). "
        "Then, Bob and Claire discuss something (let's call it Action 2). Then, Action 1 repeats. "
        "Finally, Bob and Claire swap balls. At the end of the game, Claire has the\n"
        "Options:\n(A) red ball\n(B) blue ball\n(C) green ball"
    )
    return DatasetSample(
        "bbeh",
        "d4-shuffled",
        question,
        "(B)",
        "",
        {
            "task": "shuffled_objects",
            "answer_contract": {
                "kind": "single_choice",
                "options": [
                    {"label": "A", "text": "red ball"},
                    {"label": "B", "text": "blue ball"},
                    {"label": "C", "text": "green ball"},
                ],
                "block_start": question.index("Options:"),
                "block_end": len(question),
                "selection_mode": "single",
            },
        },
    )


def _truth_sample(*, anchored: bool = True) -> DatasetSample:
    direct = "The person at the park tells the truth. " if anchored else ""
    question = (
        "In this question, assume each person either always tells the truth or always lies. "
        f"{direct}The person at the park says the person at the school lies. "
        "Does the person at the park tell the truth? Does the person at the school tell the truth? "
        "Your answer should be a list of two words, yes or no."
    )
    return DatasetSample("bbeh", "d4-truth", question, "yes, no", "", {"task": "web_of_lies"})




def test_source_ir_v3_has_frozen_public_fields_and_rejects_candidate_leakage() -> None:
    expected = {
        "capability_id",
        "query_operator",
        "entities",
        "facts",
        "events",
        "constraints",
        "query",
        "answer_contract",
        "source_span_map",
        "mandatory_spans",
        "uncovered_spans",
        "canonical_ir_hash",
    }
    assert {field.name for field in fields(SourceIRv3)} == expected
    assert {
        "compiler_vote_hashes",
        "solver_status",
        "solver_trace",
        "candidate_evaluation",
        "concrete_witness_status",
        "metamorphic_transformation_status",
        "reference_checker_status",
        "first_failure_layer",
        "risk_gate_version",
        "source_hash",
        "code_hash",
    }.issubset({field.name for field in fields(ProofPackageV2)})

    sample = _shuffled_sample()
    decision = route_for_sample(sample)
    ir, reason = compile_exact_source_ir(sample, decision)
    assert reason == "ok" and ir is not None
    payload = {
        "entities": list(ir.entities),
        "facts": list(ir.facts),
        "events": list(ir.events),
        "constraints": list(ir.constraints),
        "query": dict(ir.query),
        "uncovered_span_ids": list(ir.uncovered_spans),
    }
    payload["query"]["anchor"] = "A"
    parsed, reason = parse_source_ir_v3(payload, sample=sample, decision=decision)
    assert parsed is None
    assert reason == "source_ir_v3_candidate_leakage:anchor"

    payload = {
        "entities": list(ir.entities),
        "facts": list(ir.facts),
        "events": list(ir.events),
        "constraints": list(ir.constraints),
        "query": dict(ir.query),
        "uncovered_span_ids": [item["span_id"] for item in ir.source_span_map],
    }
    parsed, reason = parse_source_ir_v3(payload, sample=sample, decision=decision)
    assert parsed is None
    assert reason == "source_ir_v3_span_partition_invalid"

    payload = {
        "entities": list(ir.entities),
        "facts": list(ir.facts),
        "events": list(ir.events),
        "constraints": list(ir.constraints),
        "query": dict(ir.query),
        "uncovered_span_ids": list(ir.uncovered_spans),
        "source_span_map": list(ir.source_span_map),
    }
    parsed, reason = parse_source_ir_v3(payload, sample=sample, decision=decision)
    assert parsed is None
    assert reason == "source_ir_v3_keys_invalid"


def test_source_ir_v3_host_binds_contract_and_complete_span_map() -> None:
    sample = _shuffled_sample()
    decision = route_for_sample(sample)
    ir, reason = compile_exact_source_ir(sample, decision)
    assert reason == "ok" and ir is not None
    payload = {
        "entities": list(ir.entities),
        "facts": list(ir.facts),
        "events": list(ir.events),
        "constraints": list(ir.constraints),
        "query": dict(ir.query),
        "uncovered_span_ids": list(ir.uncovered_spans),
    }
    parsed, reason = parse_source_ir_v3(payload, sample=sample, decision=decision)
    assert reason == "ok" and parsed is not None
    assert parsed.capability_id == decision.capability_id
    assert parsed.query_operator == decision.query_operator
    assert parsed.answer_contract == ir.answer_contract
    assert [row["text"] for row in parsed.source_span_map] == [row["text"] for row in ir.source_span_map]
    assert set(parsed.mandatory_spans).isdisjoint(parsed.uncovered_spans)
    assert set(parsed.mandatory_spans) | set(parsed.uncovered_spans) == {
        row["span_id"] for row in parsed.source_span_map
    }


def test_sequence_trace_shuffled_solver_and_metamorphics_are_exact() -> None:
    sample = _shuffled_sample()
    decision = route_for_sample(sample)
    assert decision.capability_id == "sequence.shuffled_swap_v1"
    ir, reason = compile_exact_source_ir(sample, decision)
    assert reason == "ok" and ir is not None
    result = solve_source_ir(sample, decision, ir)
    assert result.status == "UNIQUE"
    assert result.canonical_answer == "B"
    metamorphic = run_metamorphic_checks(ir, result, sample=sample, decision=decision)
    assert metamorphic["option_permutation"] == "PASSED"
    assert metamorphic["entity_renaming"] == "PASSED"
    assert metamorphic["reversible_event"] == "PASSED"
    assert metamorphic_checks_passed(ir, result, metamorphic) is True

    options = list(reversed(sample.metadata["answer_contract"]["options"]))
    permuted = replace(
        sample,
        metadata={
            **sample.metadata,
            "answer_contract": {**sample.metadata["answer_contract"], "options": options},
        },
    )
    permuted_ir, reason = compile_exact_source_ir(permuted, route_for_sample(permuted))
    assert reason == "ok" and permuted_ir is not None
    assert solve_source_ir(permuted, route_for_sample(permuted), permuted_ir).canonical_answer == "B"

    inserted = "Then, everyone passes an object clockwise. "
    unknown_action = replace(
        sample,
        question=sample.question.replace("Finally,", inserted + "Finally,"),
        metadata={
            **sample.metadata,
            "answer_contract": {
                **sample.metadata["answer_contract"],
                "block_start": sample.metadata["answer_contract"]["block_start"] + len(inserted),
                "block_end": sample.metadata["answer_contract"]["block_end"] + len(inserted),
            },
        },
    )
    rejected, reason = compile_exact_source_ir(unknown_action, decision)
    assert rejected is None
    assert reason == "shuffled_action_sentence_unparsed"


def test_constraint_truth_graph_distinguishes_unique_and_multiple() -> None:
    sample = _truth_sample()
    decision = route_for_sample(sample)
    ir, reason = compile_exact_source_ir(sample, decision)
    assert reason == "ok" and ir is not None
    result = solve_source_ir(sample, decision, ir)
    assert result.status == "UNIQUE"
    assert result.answer_text == "yes, no"

    unanchored = _truth_sample(anchored=False)
    decision = route_for_sample(unanchored)
    ir, reason = compile_exact_source_ir(unanchored, decision)
    assert reason == "ok" and ir is not None
    assert solve_source_ir(unanchored, decision, ir).status == "MULTIPLE"


def test_numeric_solver_rejects_ungrounded_constants_and_extra_records() -> None:
    sample = DatasetSample(
        "gpqa_diamond",
        "numeric-fixture",
        "What is 2 + 3?\nA. 4\nB. 5\nC. 6\nD. 7",
        "B",
        "",
        {
            "high_level_domain": "Physics",
            "answer_contract": {
                "kind": "single_choice",
                "selection_mode": "single",
                "options": [
                    {"label": "A", "text": "4"},
                    {"label": "B", "text": "5"},
                    {"label": "C", "text": "6"},
                    {"label": "D", "text": "7"},
                ],
            },
        },
    )
    decision = route_for_sample(sample)
    assert decision.capability_id == "constraint.explicit_calculator_v1"
    ir = SourceIRv3(
        capability_id=decision.capability_id,
        query_operator=decision.query_operator,
        entities=(),
        facts=(),
        events=(),
        constraints=(
            {
                "constraint_id": "C0",
                "kind": "numeric_expression",
                "expression": "2 + 3",
                "source_span_ids": ["S0"],
            },
        ),
        query={"kind": "evaluate_numeric_expression", "source_span_ids": ["S0"]},
        answer_contract=sample.metadata["answer_contract"],
        source_span_map=({"span_id": "S0", "text": "What is 2 + 3?"},),
        mandatory_spans=("S0",),
        uncovered_spans=(),
        canonical_ir_hash="fixture",
    )
    assert solve_source_ir(sample, decision, ir).canonical_answer == "B"
    invented = replace(
        ir,
        constraints=({**ir.constraints[0], "expression": "2 + 4"},),
    )
    assert solve_source_ir(sample, decision, invented).reason == "source_ir_undeclared_numeric_constant"
    extra_fact = replace(ir, facts=({"kind": "invented"},))
    assert solve_source_ir(sample, decision, extra_fact).reason == "numeric_ir_contains_unexpected_records"


def test_new_routes_remain_shadow_until_empirical_gate_passes() -> None:
    inactive = risk_gate_snapshot("sequence.shuffled_swap_v1", route="EXACT_EXECUTABLE")
    assert inactive.route_activation_state == "SHADOW_GATE_NOT_MET"
    passing = {
        "capabilities": {
            "sequence.shuffled_swap_v1": {
                "evidence_frozen": True,
                "override_count": 50,
                "correct_override_count": 50,
                "correction_count": 50,
                "harm_count": 0,
                "audit_sample_count": 60,
                "inter_rater_agreement": 0.9,
                "critical_semantic_error_rate": 0.0,
                "adjudicated_ir_validity": 1.0,
                "unexplained_high_severity_false_pass": 0,
                "metamorphic_pass_rate": 1.0,
                "coverage": 0.1,
            }
        }
    }
    active = risk_gate_snapshot(
        "sequence.shuffled_swap_v1",
        route="EXACT_EXECUTABLE",
        evidence=passing,
    )
    assert active.route_activation_state == "ACTIVE"
    assert active.precision_one_sided_95_lower is not None
    assert active.precision_one_sided_95_lower >= 0.90
    assert active.gate_family_size == 9
    assert active.multiplicity_correction == "bonferroni_fixed_preregistered_capability_family"

    passing["capabilities"]["sequence.shuffled_swap_v1"].update(
        {"override_count": 59, "correct_override_count": 59, "correction_count": 59}
    )
    confirmation = risk_gate_snapshot(
        "sequence.shuffled_swap_v1",
        route="EXACT_EXECUTABLE",
        evidence=passing,
        phase="confirmation",
    )
    assert confirmation.route_activation_state == "ACTIVE"
    assert confirmation.harm_one_sided_95_upper is not None
    assert confirmation.harm_one_sided_95_upper <= 0.05


def test_blind_audit_requires_all_three_kernels_and_sixty_items_each() -> None:
    items = []
    metamorphic = {}
    for kernel_id in (
        "sequence_trace_kernel_v1",
        "event_state_kernel_v1",
        "constraint_calculator_kernel_v1",
    ):
        metamorphic[kernel_id] = {"total": 10, "passed": 10}
        for index in range(60):
            ir_valid = index >= 3
            unique_index = len(items) + 1
            items.append(
                {
                    "item_id": f"{kernel_id}:{index}",
                    "kernel_id": kernel_id,
                    "capability_id": f"{kernel_id}.fixture",
                    "ir_hash": f"{unique_index:064x}",
                    "visible_fields": ["source", "source_ir", "source_span_map"],
                    "annotations": [
                        {"annotator_id": "A", "ir_valid": ir_valid, "critical_error": False},
                        {"annotator_id": "B", "ir_valid": ir_valid, "critical_error": False},
                    ],
                    "adjudicated_ir_valid": ir_valid,
                    "critical_semantic_error": False,
                    "unexplained_high_severity_false_pass": False,
                }
            )
    audit_payload = {
        "schema": "catch_d4_blind_ir_audit_v1",
        "blinding_contract": {
            "hidden": [
                "candidate",
                "candidates",
                "stage_a_candidates",
                "gold",
                "solver_answer",
                "final_accuracy",
                "anchor",
                "vote_counts",
            ],
            "two_independent_annotators": True,
            "third_person_adjudication": True,
        },
        "items": items,
        "metamorphic": metamorphic,
    }
    assessment = evaluate_d4_human_audit(audit_payload)
    assert assessment["passed"] is True
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert gwet_ac1([(True, True)] * 60) == 1.0
    items[0]["visible_fields"].append("gold")
    assert evaluate_d4_human_audit(audit_payload)["passed"] is False

    for item in items:
        item["visible_fields"] = ["source", "source_ir", "source_span_map"]
        for annotation in item["annotations"]:
            annotation["ir_valid"] = True
        item["adjudicated_ir_valid"] = True
    degenerate = evaluate_d4_human_audit(audit_payload)
    assert degenerate["passed"] is True
    assert all(not row["kappa_estimable"] for row in degenerate["kernel_results"])
    assert all(row["gwet_ac1"] == 1.0 for row in degenerate["kernel_results"])
    items[1]["item_id"] = items[0]["item_id"]
    assert evaluate_d4_human_audit(audit_payload)["passed"] is False


def test_musr_x_partition_is_latent_first_balanced_and_disjoint(tmp_path) -> None:
    tasks = ("murder_mysteries", "object_placements", "team_allocation")
    records = [
        {"record_id": f"{task}-{index}", "task": task, "latent_graph": {"task": task, "index": index}}
        for task in tasks
        for index in range(20)
    ]
    partition = partition_latent_records(
        records,
        counts_by_split={"development": 12, "human_audit": 6, "confirmation": 12},
        seed=42,
        strata=tasks,
    )
    rendered_asset = tmp_path / "musr_x.jsonl"
    rendered_asset.write_text('{"sealed":"fixture"}', encoding="utf-8")
    render_audit = tmp_path / "render_audit.json"
    render_audit.write_text('{"status":"passed"}', encoding="utf-8")
    manifest = sealed_manifest(
        partition,
        generator_repository="https://github.com/Zayne-Sprague/MuSR",
        generator_commit="d" * 40,
        generation_lock_sha256="a" * 64,
        narrative_generator_id="independent-generator-v1",
        quality_validation_protocol_sha256="b" * 64,
        rendered_asset_relative_path=rendered_asset.name,
        rendered_asset_sha256=hashlib.sha256(rendered_asset.read_bytes()).hexdigest(),
        render_audit_relative_path=render_audit.name,
        render_audit_sha256=hashlib.sha256(render_audit.read_bytes()).hexdigest(),
        custodian_id="independent-custodian",
        seed=42,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_sealed_manifest(
        manifest,
        expected_counts={"development": 12, "human_audit": 6, "confirmation": 12},
        manifest_path=manifest_path,
    )
    assert validation["passed"] is True
    assert all(
        set(row) == {"record_id", "task", "latent_graph_sha256"}
        for rows in manifest["splits"].values()
        for row in rows
    )
    manifest["splits"]["development"][1] = dict(manifest["splits"]["development"][0])
    assert validate_sealed_manifest(
        manifest,
        expected_counts={"development": 12, "human_audit": 6, "confirmation": 12},
        manifest_path=manifest_path,
    )["passed"] is False


def test_supergpqa_official_science_schema_is_recognized_without_gold() -> None:
    assert science_record_eligible(
        {
            "uuid": "fixture",
            "question": "Compute the result.",
            "options": ["1", "2", "3", "4"],
            "discipline": "Science",
            "field": "Physics",
            "subfield": "Quantum Mechanics",
        }
    )
    assert not science_record_eligible(
        {"options": ["A", "B"], "discipline": "Engineering", "field": "Electrical Engineering"}
    )
    assert not science_record_eligible({"options": ["A", "B"], "field": "Physics"})


def test_text_sealed_manifest_hash_links_dedup_audit_and_disjoint_splits(tmp_path) -> None:
    strata = ("Physics", "Chemistry", "Biology")
    records = [
        {
            "uuid": f"record-{stratum}-{index}",
            "field": stratum,
            "question": f"Question {stratum} {index}",
            "options": ["one", "two"],
            "answer": "one",
        }
        for stratum in strata
        for index in range(5)
    ]
    partition = partition_text_records(
        records,
        counts_by_split={"development": 6, "human_audit": 3, "confirmation": 6},
        seed=42,
        strata=strata,
    )
    audit_path = tmp_path / "dedup_audit.json"
    audit_path.write_text('{"status":"passed"}', encoding="utf-8")
    manifest = text_sealed_manifest(
        partition,
        dataset_id="supergpqa_science",
        source_repository="https://huggingface.co/datasets/m-a-p/SuperGPQA",
        source_revision="e" * 40,
        source_asset_sha256="a" * 64,
        license_id="odc-by",
        partition_protocol_sha256="b" * 64,
        dedup_audit_relative_path=audit_path.name,
        dedup_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        custodian_id="independent-custodian",
        seed=42,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_text_sealed_manifest(
        manifest,
        expected_dataset_id="supergpqa_science",
        expected_counts={"development": 6, "human_audit": 3, "confirmation": 6},
        expected_strata=strata,
        manifest_path=manifest_path,
    )
    assert result["passed"] is True
    assert result["conditions"]["dedup_audit_file"] is True

    manifest["splits"]["confirmation"][0] = dict(manifest["splits"]["development"][0])
    assert validate_text_sealed_manifest(
        manifest,
        expected_dataset_id="supergpqa_science",
        expected_counts={"development": 6, "human_audit": 3, "confirmation": 6},
        expected_strata=strata,
        manifest_path=manifest_path,
    )["passed"] is False


def test_sealed_selection_recomputes_text_and_source_record_hashes(tmp_path, monkeypatch) -> None:
    sample = DatasetSample(
        dataset="bbeh_extension",
        sample_id="ext-001",
        question="A sealed question.",
        reference_answer="A",
        prompt_context="",
        metadata={
            "task": "shuffled_objects",
            "source_record_sha256": "f" * 64,
        },
    )
    source_asset = tmp_path / "extension.jsonl"
    source_asset.write_text('{"sealed":"fixture"}', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "source_asset_sha256": hashlib.sha256(source_asset.read_bytes()).hexdigest(),
        "splits": {
            "confirmation": [
                {
                    "record_id": sample.sample_id,
                    "stratum": "shuffled_objects",
                    "question_sha256": text_sha256(sample.question),
                    "source_record_sha256": "f" * 64,
                }
            ]
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "research_experiments.families.contrastive_active_testing.run.execute.load_samples",
        lambda benchmark: [sample],
    )
    benchmark = SimpleNamespace(
        slug="bbeh_extension",
        cache_namespace="bbeh_extension",
        source_path=source_asset.as_posix(),
    )
    phase = {
        "split_overrides": {"bbeh_extension": "full1_seed42"},
        "selection_strategy": "d4_sealed_manifest_only",
        "sealed_manifest_paths": {"bbeh_extension": manifest_path.as_posix()},
        "sealed_manifest_split": "confirmation",
    }
    assert _select_phase_samples(benchmark, phase, "confirmation") == [sample]

    manifest["splits"]["confirmation"][0]["question_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="question hash mismatch"):
        _select_phase_samples(benchmark, phase, "confirmation")

    manifest["splits"]["confirmation"][0]["question_sha256"] = text_sha256(sample.question)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="preregistered sample limit"):
        _select_phase_samples(
            benchmark,
            {**phase, "sample_limits": {"bbeh_extension": 0}},
            "confirmation",
        )


def test_confirmation_manifest_gate_requires_preregistered_text_expectations(tmp_path) -> None:
    records = [
        {
            "record_id": f"ext-{index}",
            "task": "shuffled_objects",
            "input": f"Question {index}",
            "target": "A",
        }
        for index in range(3)
    ]
    partition = partition_text_records(
        records,
        counts_by_split={"development": 1, "human_audit": 1, "confirmation": 1},
        seed=42,
        strata=("shuffled_objects",),
    )
    audit_path = tmp_path / "dedup.json"
    audit_path.write_text("{}", encoding="utf-8")
    manifest = text_sealed_manifest(
        partition,
        dataset_id="bbeh_extension",
        source_repository="https://example.org/custodian/bbeh-extension",
        source_revision="f" * 40,
        source_asset_sha256="a" * 64,
        license_id="research-only",
        partition_protocol_sha256="b" * 64,
        dedup_audit_relative_path=audit_path.name,
        dedup_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        custodian_id="independent-custodian",
        seed=42,
    )
    manifest_path = tmp_path / "bbeh_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = {"bbeh_extension": manifest_path.as_posix()}
    expectations = {
        "bbeh_extension": {
            "counts": {"development": 1, "human_audit": 1, "confirmation": 1},
            "strata": ["shuffled_objects"],
        }
    }
    result = _require_d4_sealed_manifests(
        paths,
        expected_labels={"bbeh_extension"},
        expectations=expectations,
    )
    assert result["bbeh_extension"]["validation"]["passed"] is True
    with pytest.raises(RuntimeError, match="preregistered expectation"):
        _require_d4_sealed_manifests(paths, expected_labels={"bbeh_extension"})


def test_file_driven_text_and_latent_sealing_workflows_are_recomputable(tmp_path) -> None:
    counts = {"development": 1, "human_audit": 1, "confirmation": 1}
    protocol_path = tmp_path / "split_protocol.md"
    protocol_path.write_text("Frozen before disclosure.", encoding="utf-8")
    dedup_path = tmp_path / "dedup.json"
    dedup_path.write_text('{"status":"passed"}', encoding="utf-8")
    text_records_path = tmp_path / "science.jsonl"
    text_records = [
        {
            "uuid": f"science-{index}",
            "field": "Physics",
            "question": f"Physics question {index}",
            "options": ["one", "two"],
            "answer": "one",
        }
        for index in range(3)
    ]
    text_records_path.write_text(
        "\n".join(json.dumps(row) for row in text_records) + "\n",
        encoding="utf-8",
    )
    text_result = write_text_sealed_manifest_from_files(
        records_path=text_records_path,
        output_path=tmp_path / "science_manifest.json",
        dataset_id="supergpqa_science",
        counts_by_split=counts,
        strata=("Physics",),
        source_repository="https://huggingface.co/datasets/m-a-p/SuperGPQA",
        source_revision="a" * 40,
        license_id="odc-by",
        partition_protocol_path=protocol_path,
        dedup_audit_path=dedup_path,
        custodian_id="independent-custodian",
        seed=42,
    )
    assert text_result["validation"]["passed"] is True

    latent_records_path = tmp_path / "latent.jsonl"
    latent_records_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "record_id": f"latent-{index}",
                    "task": "object_placements",
                    "latent_graph": {"index": index},
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    rendered_path = tmp_path / "musr_x.jsonl"
    rendered_path.write_text("{}\n", encoding="utf-8")
    render_audit_path = tmp_path / "render_audit.json"
    render_audit_path.write_text('{"status":"passed"}', encoding="utf-8")
    generation_lock_path = tmp_path / "generation.lock"
    generation_lock_path.write_text("locked", encoding="utf-8")
    quality_path = tmp_path / "quality_protocol.md"
    quality_path.write_text("quality", encoding="utf-8")
    latent_result = write_latent_sealed_manifest_from_files(
        latent_records_path=latent_records_path,
        rendered_asset_path=rendered_path,
        render_audit_path=render_audit_path,
        output_path=tmp_path / "musr_x_manifest.json",
        counts_by_split=counts,
        strata=("object_placements",),
        generator_repository="https://github.com/Zayne-Sprague/MuSR",
        generator_commit="b" * 40,
        generation_lock_path=generation_lock_path,
        narrative_generator_id="independent-generator-v1",
        quality_validation_protocol_path=quality_path,
        custodian_id="independent-custodian",
        seed=42,
    )
    assert latent_result["validation"]["passed"] is True
    assert latent_result["manifest"]["schema"] == "catch_d4_latent_first_sealed_manifest_v3"


def test_typed_event_state_operator_solves_auditable_musr_object_ledger() -> None:
    sample = _mc_sample()
    decision = D4RouteDecision(
        "SEMANTIC_EXECUTABLE",
        "event_state_kernel_v1",
        "event.musr_object_belief_ledger_v1",
        "belief_state_at_query_time",
        "fixture",
    )
    ir = SourceIRv3(
        capability_id=decision.capability_id,
        query_operator=decision.query_operator,
        entities=({"entity_id": "key", "kind": "object"},),
        facts=({"kind": "object_location_initial", "object": "key", "location": "alpha"},),
        events=({"kind": "object_move", "object": "key", "to": "alpha"},),
        constraints=(),
        query={"kind": "object_location", "object": "key"},
        answer_contract=sample.metadata["answer_contract"],
        source_span_map=(),
        mandatory_spans=(),
        uncovered_spans=(),
        canonical_ir_hash="fixture",
    )
    result = solve_source_ir(sample, decision, ir)
    assert result.status == "UNIQUE"
    assert result.answer_text == "alpha"
    assert result.reference_checker_status == "PASSED_TYPED_EVENT_STATE_CHECKER"


def test_typed_sequence_operators_solve_word_trace_and_temporal_windows() -> None:
    word_sample = DatasetSample(
        "bbeh",
        "word-fixture",
        "Sort words.",
        "2",
        "",
        {"answer_contract": {"kind": "free_text", "options": [], "selection_mode": "none"}},
    )
    word_decision = D4RouteDecision(
        "SEMANTIC_EXECUTABLE",
        "sequence_trace_kernel_v1",
        "sequence.word_sort_error_trace_v1",
        "earliest_trace_divergence",
        "fixture",
    )
    word_ir = SourceIRv3(
        capability_id=word_decision.capability_id,
        query_operator=word_decision.query_operator,
        entities=(),
        facts=({"kind": "word_sort_target", "words": ["alpha", "beta", "gamma"]},),
        events=(
            {"kind": "word_sort_step", "step_index": 1, "observed": ["alpha", "beta", "gamma"], "expected": ["alpha", "beta", "gamma"]},
            {"kind": "word_sort_step", "step_index": 2, "observed": ["alpha", "gamma", "beta"], "expected": ["alpha", "beta", "gamma"]},
        ),
        constraints=(),
        query={"kind": "earliest_trace_divergence", "no_error_value": 0},
        answer_contract=word_sample.metadata["answer_contract"],
        source_span_map=(),
        mandatory_spans=(),
        uncovered_spans=(),
        canonical_ir_hash="fixture",
    )
    word_result = solve_source_ir(word_sample, word_decision, word_ir)
    assert word_result.status == "UNIQUE"
    assert word_result.answer_text == "2"

    temporal_sample = DatasetSample(
        "bbeh",
        "temporal-fixture",
        "Schedule.",
        "150, 1",
        "",
        {"answer_contract": {"kind": "free_text", "options": [], "selection_mode": "none"}},
    )
    temporal_decision = replace(word_decision, capability_id="sequence.temporal_interval_trace_v1", query_operator="longest_feasible_interval")
    temporal_ir = replace(
        word_ir,
        capability_id=temporal_decision.capability_id,
        query_operator=temporal_decision.query_operator,
        facts=(
            {"kind": "feasible_window", "start_min": 0, "end_min": 120},
            {"kind": "feasible_window", "start_min": 0, "end_min": 150},
        ),
        events=(),
        query={"kind": "longest_feasible_interval", "grid_minutes": 30},
        answer_contract=temporal_sample.metadata["answer_contract"],
    )
    temporal_result = solve_source_ir(temporal_sample, temporal_decision, temporal_ir)
    assert temporal_result.status == "UNIQUE"
    assert temporal_result.answer_text == "150, 1"


def test_d4_prompt_is_candidate_blind() -> None:
    sample = _shuffled_sample()
    decision = route_for_sample(sample)
    ir, _ = compile_exact_source_ir(sample, decision)
    assert ir is not None
    messages = build_d4_source_compiler_messages(
        sample,
        source_spans=list(ir.source_span_map),
        answer_contract=ir.answer_contract,
        decision=decision,
    )
    prompt = "\n".join(row["content"] for row in messages)
    assert "Stage-A" in prompt
    assert "anchor_answer" not in prompt
    assert "vote_counts" not in prompt
    assert "reference_answer" not in prompt


def test_d4_runner_emits_exactly_five_main_methods(monkeypatch) -> None:
    sample = _shuffled_sample()
    protocol = CatchProtocolConfig(
        5,
        3,
        2,
        3,
        0,
        0,
        0.7,
        1.0,
        16_384,
        4_096,
        (),
        (),
        62_000,
        protocol_version="catch_kernel_v1",
        pair_judge_count=3,
    )
    experiment = SimpleNamespace(
        global_seed=42,
        raw={
            "kernel_revision": "d4_proof_carrying_v1",
            "d4_output": {"stage_a_protocol": "tagged_text"},
            "d4_risk": {
                "evidence_path": "configs/families/contrastive_active_testing/d4_gate_evidence.example.json",
                "new_exact_override_enabled": False,
                "semantic_override_enabled": False,
            },
            "phases": {"development": {"sample_limits": {"bbeh": 1}}},
        },
    )

    def fake_answer_turn(sample, *, role, agent_id, **_kwargs):
        answer = "B" if agent_id == 1 else "A"
        return {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "role": role,
            "agent_id": agent_id,
            "answer_class_key": answer,
            "prediction": answer,
            "normalized_answer": answer,
            "validated_output": {"reasoning": "fixture", "final_answer": answer},
            "protocol_parse_status": "ok",
            "network_attempt_count": 0,
            "actual_total_tokens": 1,
            "total_tokens": 1,
        }

    monkeypatch.setattr(sample_runner, "_answer_turn", fake_answer_turn)
    stage_rows = tuple(fake_answer_turn(sample, role="stage_a_solver", agent_id=i) for i in range(1, 6))
    _, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="d4-test",
        split_name="fixture",
        experiment=experiment,
        protocol=protocol,
        endpoint=None,
        network_budget=sample_runner.NetworkAttemptBudget(100),
        phase_name="development",
        run_direct_judge=False,
        precomputed_stage_rows=stage_rows,
    )
    main = {row["method_name"] for row in predictions if row.get("main_table_eligible")}
    assert main == {"sc_5", "fixed_sc_8", "catch_d3_exact_only", "ssv_raw", "catch_kernel_d4"}
    assert router["first_failure_layer"] == "RISK_GATE"
    assert next(row for row in predictions if row["method_name"] == "catch_kernel_d4")[
        "logical_calls_per_question"
    ] == 5


def test_semantic_compiler_consensus_requires_three_complete_proof_chains(monkeypatch) -> None:
    sample = _shuffled_sample()
    decision = route_for_sample(sample)
    base_ir, reason = compile_exact_source_ir(sample, decision)
    assert reason == "ok" and base_ir is not None
    irs = [
        (2, replace(base_ir, canonical_ir_hash="ir-two")),
        (1, replace(base_ir, canonical_ir_hash="ir-one")),
        (3, replace(base_ir, canonical_ir_hash="ir-three")),
    ]

    def fake_solver(_sample, _decision, ir):
        answer = "C" if ir.canonical_ir_hash == "ir-three-bad" else "B"
        return D4SolverResult(
            status="UNIQUE",
            canonical_answer=answer,
            answer_text=answer,
            solver_trace=({"ir_hash": ir.canonical_ir_hash},),
            reference_checker_status="PASSED_FIXTURE_REFERENCE_CHECKER",
            concrete_witness_status={"status": "PASSED"},
            reason="fixture",
        )

    monkeypatch.setattr(sample_runner, "d4_solve_source_ir", fake_solver)
    monkeypatch.setattr(sample_runner, "run_metamorphic_checks", lambda *_args, **_kwargs: {"fixture": "PASSED"})
    fallback = D4SolverResult("UNSUPPORTED", None, None, (), "NOT_RUN", {"status": "NOT_AVAILABLE"}, "fallback")
    source_ir, solver, hashes, verifications, agreed, metamorphic = sample_runner.verify_d4_semantic_compiler_consensus(
        sample=sample,
        decision=decision,
        parsed_irs=irs,
        required_compiler_count=3,
        fallback_solver=fallback,
    )
    assert agreed is True
    assert source_ir is not None and source_ir.canonical_ir_hash == "ir-one"
    assert solver.canonical_answer == "B"
    assert hashes == ("ir-one", "ir-two", "ir-three")
    assert all(item["passed"] and item["solver_trace"] for item in verifications)
    assert metamorphic == {"fixture": "PASSED"}

    bad_irs = [*irs[:2], (3, replace(base_ir, canonical_ir_hash="ir-three-bad"))]
    source_ir, solver, _, verifications, agreed, _ = sample_runner.verify_d4_semantic_compiler_consensus(
        sample=sample,
        decision=decision,
        parsed_irs=bad_irs,
        required_compiler_count=3,
        fallback_solver=fallback,
    )
    assert agreed is False
    assert source_ir is None
    assert solver.reason == "compiler_ir_non_agreement_or_parse_failure"
    assert len(verifications) == 3


def test_d4_mainline_config_rejects_retired_protocol_field(tmp_path) -> None:
    source = Path(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d4.toml"
    ).read_text(encoding="utf-8")
    target = tmp_path / "retired.toml"
    target.write_text(source.replace("conflicts_fail_closed = true", 'prompt_variant = "json"'), encoding="utf-8")
    with pytest.raises(ValueError, match="retired configuration fields"):
        load_experiment_config(target)


def test_d4_v3_protocol_rejects_non_d4_kernel_revision(tmp_path) -> None:
    source = Path(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d4.toml"
    ).read_text(encoding="utf-8")
    target = tmp_path / "wrong-kernel.toml"
    target.write_text(
        source.replace('kernel_revision = "d4_proof_carrying_v1"', 'kernel_revision = "d3_source_blind_v1"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be paired"):
        load_experiment_config(target)


def test_d4_mainline_config_has_one_protocol_and_distinct_confirmation_assets() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d4.toml"
    )
    protocol = load_protocol_config(experiment.protocol)
    confirmation = phase_metadata(experiment, "confirmation")
    benchmarks = load_phase_benchmarks(experiment, "confirmation")

    assert experiment.raw["d4_mainline_protocol_version"] == D4_MAINLINE_PROTOCOL_VERSION
    assert experiment.protocol.name == "catch_kernel_d4_v3.toml"
    assert (protocol.solver_max_tokens, protocol.role_max_tokens, protocol.judge_max_tokens) == (
        65_536,
        65_536,
        32_768,
    )
    assert set(confirmation["benchmark_slugs"]) == {
        "bbeh_extension",
        "musr_x",
        "supergpqa_science",
    }
    assert {Path(item.source_path).parent.as_posix() for item in benchmarks} == {"d4_confirmation"}
    assert experiment.cache_policy == "global_validated_response_v3"


def test_hash_linked_passing_smoke_must_match_current_mainline(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    protocol_hash = "a" * 64
    result = {
        "schema": "catch_d4_source_compiler_smoke_v1",
        "d4_mainline_protocol_version": D4_MAINLINE_PROTOCOL_VERSION,
        "protocol_sha256": protocol_hash,
        "passed": True,
        "run_id": "fixture",
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "d4_mainline_protocol_version": D4_MAINLINE_PROTOCOL_VERSION,
                "protocol_sha256": protocol_hash,
                "protocol": {
                    "solver_max_tokens": 65_536,
                    "role_max_tokens": 65_536,
                    "judge_max_tokens": 32_768,
                },
            }
        ),
        encoding="utf-8",
    )
    experiment = SimpleNamespace(
        raw={
            "source_compiler_smoke_status": "passed",
            "source_compiler_smoke_result_path": result_path.as_posix(),
            "source_compiler_smoke_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        }
    )

    snapshot = d4_source_compiler_smoke_snapshot(experiment)
    assert snapshot["artifact_valid"] is True
    assert snapshot["mainline_compatible"] is True
    assert require_passing_d4_source_compiler_smoke(experiment) == snapshot

    result["d4_mainline_protocol_version"] = "retired"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    experiment.raw["source_compiler_smoke_result_sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="compiler smoke"):
        require_passing_d4_source_compiler_smoke(experiment)





def test_independent_tagged_validation_gates_turn_and_sample_quorum_bounds() -> None:
    datasets = sorted(D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS)
    predictions = [
        {
            "dataset": datasets[index // 100],
            "sample_id": f"sample-{index}",
            "method_name": "sc_5",
            "score": 1.0,
            "output_protocol_validation_only": True,
            "logical_calls_per_question": 5,
        }
        for index in range(300)
    ]
    turns = [
        {
            "dataset": prediction["dataset"],
            "sample_id": prediction["sample_id"],
            "role": "stage_a_solver",
            "agent_id": agent_id,
            "protocol_parse_status": "ok",
        }
        for prediction in predictions
        for agent_id in range(1, 6)
    ]
    source = {
        "run_root": "unused",
        "kernel_revision": "d4_proof_carrying_v1",
        "evaluation_role": "d4_output_protocol_independent_validation_tagged_v3",
        "stage_a_protocol": "tagged_text",
        "phase_name": "development",
        "run_id": "independent-run",
        "run_status": "completed",
        "selected_sample_counts": {dataset: 100 for dataset in datasets},
        "expected_selection_sha256": {
            dataset: f"{index + 1:064x}" for index, dataset in enumerate(datasets)
        },
        "selected_selection_sha256": {
            dataset: f"{index + 1:064x}" for index, dataset in enumerate(datasets)
        },
        "cache_policy": "global_validated_response_v3",
        "provider_audit": {"required": True, "status": "passed"},
    }
    result = evaluate_tagged_protocol_validation(
        {"turns": turns, "predictions": predictions, "source": source}
    )
    assert result["independent_validation_passed"] is True
    assert result["summary"]["quorum_failure_count"] == 0
    assert result["summary"]["quorum_failure_one_sided_95_upper"] < 0.01
    assert validate_tagged_protocol_validation_assessment(result)["passed"] is True

    inspected = evaluate_tagged_protocol_validation(
        {
            "turns": turns,
            "predictions": predictions,
            "source": {**source, "evaluation_role": "inspected_development"},
        }
    )
    assert inspected["independent_validation_passed"] is False




def test_independent_protocol_validation_template_is_sealed_and_fail_closed() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/"
        "catch_kernel_d4_protocol_independent_validation_tagged_v3.toml"
    )
    phase = phase_metadata(experiment, "development")
    assert set(phase["benchmark_slugs"]) == D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS
    assert phase["sample_limits"] == {
        dataset: 100 for dataset in D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS
    }
    assert phase["selection_strategy"] == "d4_sealed_manifest_only"
    assert phase["sealed_manifest_split"] == "protocol_validation"
    assert experiment.raw["d4_output"]["stage_a_protocol"] == "tagged_text"
    assert "prompt_variant" not in experiment.raw["d4_output"]
    assert experiment.cache_policy == "global_validated_response_v3"
    assert "cache_namespaces" not in experiment.raw
    assert "baseline_cache_namespaces" not in experiment.raw
    with pytest.raises(RuntimeError, match="fresh sealed design"):
        _validate_d4_independent_protocol_validation_config(
            experiment,
            phase_name="development",
            phase=phase,
        )


def test_independent_validation_uses_global_cache_without_namespace() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/"
        "catch_kernel_d4_protocol_independent_validation_tagged_v3.toml"
    )
    assert experiment.cache_policy == "global_validated_response_v3"
    assert "cache_namespaces" not in experiment.raw


def test_d4_gate_evidence_is_derived_from_shadow_rows_and_hash_linked() -> None:
    predictions = []
    for index in range(50):
        predictions.append(
            {
                "run_id": "development-run",
                "dataset": "bbeh",
                "method_name": "catch_kernel_d4",
                "d4_capability_id": "sequence.shuffled_swap_v1",
                "d4_kernel_id": "sequence_trace_kernel_v1",
                "d4_route": "EXACT_EXECUTABLE",
                "d4_shadow_score": 1.0,
                "d4_shadow_override": True,
                "d4_shadow_correction": True,
                "d4_shadow_harm": False,
                "d4_metamorphic_checks_passed": True,
                "d4_proof_package": {
                    "metamorphic_transformation_status": {
                        "entity_renaming": "PASSED",
                        "irrelevant_text_insertion": "NOT_APPLICABLE",
                    }
                },
                "sample_id": f"sample-{index}",
            }
        )
    evidence = build_d4_gate_evidence(predictions, predictions_sha256="a" * 64)
    assert validate_d4_gate_evidence(evidence)["passed"] is True
    snapshot = risk_gate_snapshot(
        "sequence.shuffled_swap_v1",
        route="EXACT_EXECUTABLE",
        evidence=evidence,
    )
    assert snapshot.route_activation_state == "ACTIVE"
    with pytest.raises(ValueError, match="not in the frozen registry"):
        build_d4_gate_evidence(
            [{**predictions[0], "d4_capability_id": "unknown.fixture"}],
            predictions_sha256="b" * 64,
        )


def test_d4_confirmation_hard_gate_blocks_before_any_api_or_selection_call() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d4.toml"
    )
    with pytest.raises(RuntimeError, match="compiler smoke"):
        run_experiment(experiment, "confirmation", SimpleNamespace())


def test_d4_development_selection_meets_frozen_capability_quotas() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d4.toml"
    )
    phase = phase_metadata(experiment, "development")
    selected = {
        benchmark.slug: _select_phase_samples(benchmark, phase, "development")
        for benchmark in load_phase_benchmarks(experiment, "development")
    }
    manifests = _selected_sample_manifest(selected, phase_name="development")
    assert {key: len(value) for key, value in selected.items()} == {
        "bbeh": 540,
        "musr": 120,
        "gpqa_diamond": 47,
    }
    assert {
        key: row["sha256"] for key, row in manifests.items()
    } == phase["expected_selection_sha256"]


def test_all_shuffled_bbeh_items_compile_and_solve_without_gold_patching() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d4.toml"
    )
    benchmark = next(item for item in load_phase_benchmarks(experiment, "development") if item.slug == "bbeh")
    rows = [item for item in load_samples(benchmark) if item.metadata.get("task") == "shuffled_objects"]
    assert len(rows) == 200
    for sample in rows:
        decision = route_for_sample(sample)
        ir, reason = compile_exact_source_ir(sample, decision)
        assert reason == "ok" and ir is not None
        result = solve_source_ir(sample, decision, ir)
        assert result.status == "UNIQUE"
        assert score_prediction(
            sample.dataset,
            result.canonical_answer,
            sample.reference_answer,
            sample=sample,
        ) == 1.0
