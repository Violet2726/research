from __future__ import annotations

from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import pytest

from research_experiments.core.data.datasets import DatasetSample, load_samples
from research_experiments.core.data.evaluation import score_prediction
from research_experiments.families.contrastive_active_testing.config import (
    CatchProtocolConfig,
    load_experiment_config,
    load_phase_benchmarks,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.d4_audit import (
    build_d4_gate_evidence,
    cohen_kappa,
    evaluate_d4_human_audit,
    gwet_ac1,
    validate_d4_gate_evidence,
)
from research_experiments.families.contrastive_active_testing.d4_data import (
    partition_latent_records,
    science_record_eligible,
    sealed_manifest,
    validate_sealed_manifest,
)
from research_experiments.families.contrastive_active_testing.d4_protocol_ab import (
    evaluate_output_protocol_ab,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    ProofPackageV2,
    SourceIRv2,
    compile_exact_source_ir,
    metamorphic_checks_passed,
    parse_source_ir_v2,
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
    _select_phase_samples,
    _selected_sample_manifest,
    run_experiment,
)
from research_experiments.family_runtime.answer_first_json_protocol import (
    parse_answer_first_json_output,
    parse_reasoning_first_json_output,
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


def test_answer_first_json_is_strict_fail_closed() -> None:
    sample = _mc_sample()
    parsed = parse_answer_first_json_output(sample, '{"final_answer":"A","reasoning":"checked"}')
    assert parsed["canonical_key"] == "A"
    assert parse_reasoning_first_json_output(
        sample, '{"reasoning":"checked","final_answer":"A"}'
    )["canonical_key"] == "A"
    invalid = [
        '{"reasoning":"checked","final_answer":"A"}',
        '{"final_answer":"A","final_answer":"B","reasoning":"duplicate"}',
        '{"final_answer":"A","reasoning":"checked","extra":1}',
        '{"final_answer":"A","reasoning":"截断"',
        '{"final_answer":"","reasoning":"missing"}',
        '```json\n{"final_answer":"A","reasoning":"fenced"}\n```',
    ]
    for raw in invalid:
        with pytest.raises(ValueError):
            parse_answer_first_json_output(sample, raw)
    long_reasoning = "验证" * 20_000
    assert parse_answer_first_json_output(
        sample,
        '{"final_answer":"A","reasoning":"' + long_reasoning + '"}',
    )["canonical_key"] == "A"


def test_source_ir_v2_has_frozen_public_fields_and_rejects_candidate_leakage() -> None:
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
    assert {field.name for field in fields(SourceIRv2)} == expected
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
    payload = asdict(ir)
    payload["canonical_ir_hash"] = ""
    payload["query"]["anchor"] = "A"
    parsed, reason = parse_source_ir_v2(payload, sample=sample, decision=decision)
    assert parsed is None
    assert reason == "source_ir_v2_candidate_leakage:anchor"

    payload = asdict(ir)
    payload["canonical_ir_hash"] = ""
    payload["source_span_map"] = list(payload["source_span_map"])
    payload["mandatory_spans"] = list(payload["mandatory_spans"])
    payload["uncovered_spans"] = list(payload["uncovered_spans"])
    payload["mandatory_spans"] = payload["mandatory_spans"][:-1]
    parsed, reason = parse_source_ir_v2(payload, sample=sample, decision=decision)
    assert parsed is None
    assert reason == "source_ir_v2_span_partition_invalid"

    payload = asdict(ir)
    payload["canonical_ir_hash"] = ""
    payload["source_span_map"] = [*payload["source_span_map"], payload["source_span_map"][0]]
    parsed, reason = parse_source_ir_v2(payload, sample=sample, decision=decision)
    assert parsed is None
    assert reason == "source_ir_v2_span_map_not_exact_source"


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
    ir = SourceIRv2(
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


def test_musr_x_partition_is_latent_first_balanced_and_disjoint() -> None:
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
    manifest = sealed_manifest(
        partition,
        generator_repository="https://github.com/Zayne-Sprague/MuSR",
        generator_commit="d" * 40,
        generation_lock_sha256="a" * 64,
        narrative_generator_id="independent-generator-v1",
        quality_validation_protocol_sha256="b" * 64,
        custodian_id="independent-custodian",
        seed=42,
    )
    validation = validate_sealed_manifest(
        manifest,
        expected_counts={"development": 12, "human_audit": 6, "confirmation": 12},
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
            "d4_output": {"stage_a_protocol": "answer_first_json"},
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

    experiment.raw["phases"]["development"]["evaluation_role"] = (
        "d4_output_protocol_ab_answer_first_json"
    )
    physical, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="d4-ab-test",
        split_name="fixture",
        experiment=experiment,
        protocol=protocol,
        endpoint=None,
        network_budget=sample_runner.NetworkAttemptBudget(100),
        phase_name="development",
        run_direct_judge=False,
        precomputed_stage_rows=stage_rows,
    )
    assert len(physical) == 5
    assert [row["method_name"] for row in predictions] == ["sc_5"]
    assert router["route"] == "OUTPUT_PROTOCOL_AB_ONLY"


def test_protocol_ab_acceptance_is_predeclared() -> None:
    def arm(parse_status: str, score: float) -> dict[str, list[dict[str, object]]]:
        predictions = [
            {
                "dataset": "fixture",
                "sample_id": f"sample-{index}",
                "method_name": "sc_5",
                "score": score,
                "output_protocol_ab_only": True,
                "logical_calls_per_question": 5,
            }
            for index in range(300)
        ]
        return {
            "turns": [
                {
                    "role": "stage_a_solver",
                    "protocol_parse_status": parse_status,
                    "dataset": prediction["dataset"],
                    "sample_id": prediction["sample_id"],
                    "agent_id": agent_id,
                }
                for prediction in predictions
                for agent_id in range(1, 6)
            ],
            "predictions": predictions,
        }

    result = evaluate_output_protocol_ab(
        {
            "tagged_text": arm("ok", 0.5),
            "reasoning_first_json": arm("ok", 0.5),
            "answer_first_json": arm("ok", 0.5),
        }
    )
    assert result["answer_first_json_accepted"] is True
    assert result["answer_first_json_certified"] is True
    assert result["minimum_zero_failure_turns_for_95pct_certification"] == 1497
    assert result["accuracy_check_interpretation"].endswith("not_confidence_certification")


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
    with pytest.raises(RuntimeError, match="sealed_data_ready"):
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
