from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample, load_samples, select_samples
from research_experiments.families.contrastive_active_testing.certificates_v2 import build_source_span_graph
from research_experiments.families.contrastive_active_testing.config import (
    CatchProtocolConfig,
    load_experiment_config,
    load_phase_benchmarks,
)
from research_experiments.families.contrastive_active_testing.kernel_d3 import (
    D3_CAPABILITY_REGISTRY_VERSION,
    D3_IR_SCHEMA,
    D3_IR_VERSION,
    _safe_numeric_value,
    capability_registry,
    evaluate_candidate,
    parse_source_ir,
    route_for_sample,
    solve_exact,
    solve_numeric_ir,
)
from research_experiments.families.contrastive_active_testing.kernel_prompts import (
    build_d3_source_compiler_messages,
)
from research_experiments.families.contrastive_active_testing.run import sample as sample_runner
from research_experiments.families.contrastive_active_testing.run.execute import _d3_data_audit, _select_phase_samples
from research_experiments.families.contrastive_active_testing.run.sample import _d3_primary_metric


def _dyck_sample() -> DatasetSample:
    source = (
        "Input: ( ]\n"
        "Thought 1: inspect\n"
        "Thought 2: stack: empty\n"
        "Thought 3: (; stack: (\n"
        "Thought 4: ]; stack: (\n"
        "Options:\n(A) 4\n(B) 5\n(C) no\n"
    )
    return DatasetSample(
        "bbeh",
        "d3-dyck",
        source,
        "4",
        "",
        {
            "task": "dyck_languages",
            "options": [
                {"label": "A", "text": "4"},
                {"label": "B", "text": "5"},
                {"label": "C", "text": "no"},
            ],
            "answer_contract": {
                "kind": "single_choice",
                "options": [
                    {"label": "A", "text": "4"},
                    {"label": "B", "text": "5"},
                    {"label": "C", "text": "no"},
                ],
                "block_start": source.index("Options:"),
                "block_end": len(source),
                "selection_mode": "single",
            },
        },
    )


def test_d3_exact_route_is_candidate_blind_and_evaluates_candidates_after_solving() -> None:
    sample = _dyck_sample()
    route = route_for_sample(sample)
    assert route.route == "EXACT_EXECUTABLE"
    certificate = solve_exact(sample, route)
    assert certificate.status == "UNIQUE"
    assert certificate.canonical_answer == "A"
    assert evaluate_candidate(sample, "A", certificate).status == "VALID"
    assert evaluate_candidate(sample, "B", certificate).status == "INVALID"


def test_d3_exact_route_option_permutation_and_irrelevant_text_are_invariant() -> None:
    sample = _dyck_sample()
    baseline = solve_exact(sample, route_for_sample(sample))
    options = list(reversed(sample.metadata["answer_contract"]["options"]))
    permuted = replace(
        sample,
        metadata={
            **sample.metadata,
            "options": options,
            "answer_contract": {**sample.metadata["answer_contract"], "options": options},
        },
    )
    assert solve_exact(permuted, route_for_sample(permuted)).canonical_answer == baseline.canonical_answer

    insertion = "\nIrrelevant note: the weather outside is clear.\n"
    block_start = int(sample.metadata["answer_contract"]["block_start"])
    augmented = replace(
        sample,
        question=sample.question[:block_start] + insertion + sample.question[block_start:],
        metadata={
            **sample.metadata,
            "answer_contract": {
                **sample.metadata["answer_contract"],
                "block_start": block_start + len(insertion),
                "block_end": len(sample.question) + len(insertion),
            },
        },
    )
    assert solve_exact(augmented, route_for_sample(augmented)).canonical_answer == baseline.canonical_answer


def test_d3_exact_spatial_route_is_entity_rename_invariant() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d3.toml"
    )
    benchmark = next(item for item in load_phase_benchmarks(experiment, "development") if item.slug == "bbeh")
    sample = next(item for item in load_samples(benchmark) if item.sample_id == "bbeh-spatial_reasoning-0000")
    assert route_for_sample(sample).route == "EXACT_EXECUTABLE"
    baseline = solve_exact(sample, route_for_sample(sample))
    assert baseline.canonical_answer == "southern black widow"
    renamed = replace(
        sample,
        question=sample.question.replace("southern black widow", "renamed arachnid"),
    )
    renamed_certificate = solve_exact(renamed, route_for_sample(renamed))
    assert renamed_certificate.status == "UNIQUE"
    assert renamed_certificate.canonical_answer == "renamed arachnid"


def test_d3_numeric_ir_requires_complete_source_span_partition() -> None:
    sample = DatasetSample(
        "gpqa_diamond",
        "d3-physics",
        "Compute the value of 2 + 2.",
        "A|||4",
        "(A) 4\n(B) 5",
        {
            "high_level_domain": "Physics",
            "options": [{"label": "A", "text": "4"}, {"label": "B", "text": "5"}],
            "answer_contract": {
                "kind": "single_choice",
                "options": [{"label": "A", "text": "4"}, {"label": "B", "text": "5"}],
                "selection_mode": "single",
            },
        },
    )
    route = route_for_sample(sample)
    graph = build_source_span_graph(sample)
    span_ids = [span.span_id for span in graph.spans]
    payload = {
        "schema": D3_IR_SCHEMA,
        "ir_version": D3_IR_VERSION,
        "query": {
            "kind": "evaluate_numeric_expression",
            "source_span_ids": span_ids,
            "constraint_ids": ["C0"],
        },
        "constraints": [
            {
                "constraint_id": "C0",
                "kind": "numeric_expression",
                "expression": "2 + 2",
                "source_span_ids": span_ids,
            }
        ],
        "covered_span_ids": span_ids,
        "uncovered_span_ids": [],
    }
    ir, reason = parse_source_ir(payload, sample=sample, decision=route)
    assert reason == "ok"
    assert ir is not None
    certificate = solve_numeric_ir(sample, route, ir)
    assert certificate.status == "UNIQUE"
    assert certificate.canonical_answer == "A"
    payload["covered_span_ids"] = []
    payload["uncovered_span_ids"] = []
    _, reason = parse_source_ir(payload, sample=sample, decision=route)
    assert reason == "source_ir_span_partition_invalid"


def test_d3_source_compiler_prompt_is_candidate_blind_and_rejects_unbound_constants() -> None:
    sample = DatasetSample(
        "gpqa_diamond",
        "d3-blind",
        "Compute 2 + 2.",
        "A|||4",
        "(A) 4\n(B) 5",
        {
            "high_level_domain": "Physics",
            "answer_contract": {
                "kind": "single_choice",
                "options": [{"label": "A", "text": "4"}, {"label": "B", "text": "5"}],
                "selection_mode": "single",
            },
        },
    )
    route = route_for_sample(sample)
    graph = build_source_span_graph(sample)
    span_ids = [span.span_id for span in graph.spans]
    messages = build_d3_source_compiler_messages(
        sample,
        source_spans=[{"span_id": span.span_id, "text": span.text} for span in graph.spans],
        answer_schema=[{"label": "A", "text": "4"}, {"label": "B", "text": "5"}],
        operation_kind=route.operation_kind,
    )
    prompt = "\n".join(message["content"] for message in messages)
    assert "Stage-A answers" in prompt
    assert "candidate labels" in prompt
    assert "anchor_answer" not in prompt
    payload = {
        "schema": D3_IR_SCHEMA,
        "ir_version": D3_IR_VERSION,
        "query": {
            "kind": "evaluate_numeric_expression",
            "source_span_ids": span_ids,
            "constraint_ids": ["C0"],
        },
        "constraints": [
            {
                "constraint_id": "C0",
                "kind": "numeric_expression",
                "expression": "2 + 2 + 7",
                "source_span_ids": span_ids,
            }
        ],
        "covered_span_ids": span_ids,
        "uncovered_span_ids": [],
    }
    _, reason = parse_source_ir(payload, sample=sample, decision=route)
    assert reason == "source_ir_undeclared_numeric_constant"


def test_d3_numeric_solver_distinguishes_unsat_and_multiple_answer_contracts() -> None:
    def make_sample(options: list[dict[str, str]]) -> DatasetSample:
        return DatasetSample(
            "gpqa_diamond",
            "d3-status",
            "Compute 2 + 2.",
            "A|||4",
            "",
            {
                "high_level_domain": "Physics",
                "answer_contract": {"kind": "single_choice", "options": options, "selection_mode": "single"},
            },
        )

    sample = make_sample([{"label": "A", "text": "4"}, {"label": "B", "text": "4"}])
    route = route_for_sample(sample)
    spans = [span.span_id for span in build_source_span_graph(sample).spans]
    payload = {
        "schema": D3_IR_SCHEMA,
        "ir_version": D3_IR_VERSION,
        "query": {"kind": "evaluate_numeric_expression", "source_span_ids": spans, "constraint_ids": ["C0"]},
        "constraints": [{"constraint_id": "C0", "kind": "numeric_expression", "expression": "2 + 2", "source_span_ids": spans}],
        "covered_span_ids": spans,
        "uncovered_span_ids": [],
    }
    ir, reason = parse_source_ir(payload, sample=sample, decision=route)
    assert reason == "ok" and ir is not None
    assert solve_numeric_ir(sample, route, ir).status == "MULTIPLE"
    sample = make_sample([{"label": "A", "text": "5"}, {"label": "B", "text": "6"}])
    route = route_for_sample(sample)
    spans = [span.span_id for span in build_source_span_graph(sample).spans]
    payload["query"]["source_span_ids"] = spans
    payload["constraints"][0]["source_span_ids"] = spans
    payload["covered_span_ids"] = spans
    ir, reason = parse_source_ir(payload, sample=sample, decision=route)
    assert reason == "ok" and ir is not None
    assert solve_numeric_ir(sample, route, ir).status == "UNSAT"


def test_d3_numeric_checker_accepts_only_typed_arithmetic_ast() -> None:
    assert _safe_numeric_value("2 + 2 * 3")[0] == 8.0
    assert _safe_numeric_value("2 ** -3")[0] == 0.125
    for expression in ("True", "2 if True else 3", "'2'", "[2][0]", "2 == 2"):
        value, reason = _safe_numeric_value(expression)
        assert value is None
        assert reason in {
            "numeric_expression_constant_unsupported",
            "numeric_expression_ast_unsupported",
        }


def test_d3_primary_metric_does_not_call_unseen460_bbeh_full() -> None:
    assert _d3_primary_metric("bbeh", "full4520_seed42", phase_name="confirmation", sample_limit=460) == (
        "bbeh_task_stratified_micro"
    )
    assert _d3_primary_metric("bbeh", "full4520_seed42", phase_name="confirmation", sample_limit=4520) == (
        "bbeh_full_adjusted_harmonic"
    )
    assert _d3_primary_metric("bbeh", "bbeh_mini460_seed42", phase_name="confirmation", sample_limit=460) == (
        "bbeh_mini_micro"
    )


def test_d3_route_variants_keep_abstentions_in_denominator(monkeypatch) -> None:
    protocol = CatchProtocolConfig(
        5, 3, 2, 3, 0, 0, 0.7, 1.0, 16_384, 4_096, (), (), 62_000,
        protocol_version="catch_kernel_v1", pair_judge_count=3,
    )
    experiment = SimpleNamespace(
        global_seed=42,
        raw={
            "kernel_revision": "d3_source_blind_v1",
            "d3_risk": {
                "semantic_override_enabled": False,
                "semantic_precision_gate_passed": False,
                "semantic_metamorphic_suite_passed": False,
                "semantic_human_audit_passed": False,
            },
            "phases": {"development": {"sample_limits": {"gpqa_diamond": 50}}},
        },
    )

    def fake_answer_turn(sample, *, role, agent_id, **_kwargs):
        return {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "role": role,
            "agent_id": agent_id,
            "answer_class_key": "A",
            "prediction": "A",
            "normalized_answer": "A",
            "validated_output": {"reasoning": "fixture", "final_answer": "A"},
            "network_attempt_count": 0,
            "actual_total_tokens": 1,
            "total_tokens": 1,
        }

    monkeypatch.setattr(sample_runner, "_answer_turn", fake_answer_turn)
    stage_rows = tuple(fake_answer_turn(_dyck_sample(), role="stage_a_solver", agent_id=i) for i in range(1, 6))
    _, exact_router, exact_predictions = sample_runner.run_catch_sample(
        _dyck_sample(),
        run_id="d3-exact",
        split_name="catch_kernel_d3_dev50_seed42",
        experiment=experiment,
        protocol=protocol,
        endpoint=None,
        network_budget=sample_runner.NetworkAttemptBudget(100),
        phase_name="development",
        run_direct_judge=False,
        precomputed_stage_rows=stage_rows,
    )
    exact_methods = {row["method_name"] for row in exact_predictions}
    assert exact_router["first_failure_layer"] == "NONE"
    assert {"catch_d3_exact_no_completion", "catch_d3_exact_completion", "solver_direct"}.issubset(exact_methods)
    assert all(row["logical_calls_per_question"] == 5 for row in exact_predictions if row["method_name"] in {
        "catch_kernel", "catch_d3_exact_no_completion", "catch_d3_exact_completion", "solver_direct"
    })

    semantic = DatasetSample(
        "gpqa_diamond", "d3-semantic-abstain", "Compute 2 + 2.", "A|||4", "",
        {
            "high_level_domain": "Physics",
            "answer_contract": {
                "kind": "single_choice",
                "options": [{"label": "A", "text": "4"}, {"label": "B", "text": "5"}],
                "selection_mode": "single",
            },
        },
    )

    def fake_json_turn(sample, **_kwargs):
        return (
            {"dataset": sample.dataset, "sample_id": sample.sample_id, "role": "d3_source_compiler", "agent_id": 1},
            None,
        )

    monkeypatch.setattr(sample_runner, "_json_turn", fake_json_turn)
    semantic_stage = tuple(fake_answer_turn(semantic, role="stage_a_solver", agent_id=i) for i in range(1, 6))
    _, semantic_router, semantic_predictions = sample_runner.run_catch_sample(
        semantic,
        run_id="d3-semantic",
        split_name="catch_kernel_d3_dev50_seed42",
        experiment=experiment,
        protocol=protocol,
        endpoint=None,
        network_budget=sample_runner.NetworkAttemptBudget(100),
        phase_name="development",
        run_direct_judge=False,
        precomputed_stage_rows=semantic_stage,
    )
    compiler = next(row for row in semantic_predictions if row["method_name"] == "catch_d3_semantic_compiler")
    assert semantic_router["first_failure_layer"] == "PARSE"
    assert compiler["prediction"] == "A"
    assert compiler["certificate_abstained"] is True
    assert compiler["solver_status"] == "UNSUPPORTED"


def test_d3_soft_route_is_explicit_for_unsupported_task() -> None:
    sample = DatasetSample("musr", "d3-soft", "Narrative", "A|||x", "(A) x\n(B) y", {"task": "murder_mysteries"})
    route = route_for_sample(sample)
    assert route.route == "SOFT_UNSUPPORTED"


def test_d3_capability_registry_is_frozen_and_candidate_independent() -> None:
    registry = capability_registry()
    assert registry["version"] == D3_CAPABILITY_REGISTRY_VERSION
    assert registry["soft_default"] is True
    assert "bbeh.dyck_languages" in registry["exact"]
    assert "musr" not in str(registry)


def test_d3_count50_assets_and_unseen_confirmation_are_frozen() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d3.toml"
    )
    development = {item.slug: item for item in load_phase_benchmarks(experiment, "development")}
    assert {slug: len(select_samples(benchmark, "catch_kernel_d3_dev50_seed42")) for slug, benchmark in development.items()} == {
        "bbeh": 50,
        "musr": 50,
        "gpqa_diamond": 50,
    }
    phase = experiment.raw["phases"]["confirmation"]
    confirmation = {
        benchmark.slug: _select_phase_samples(benchmark, phase, "confirmation")
        for benchmark in load_phase_benchmarks(experiment, "confirmation")
    }
    assert {slug: len(rows) for slug, rows in confirmation.items()} == {
        "bbeh": 460,
        "musr": 356,
        "gpqa_diamond": 100,
    }
    assert len({str(row.metadata.get("task")) for row in confirmation["bbeh"]}) == 23
    audit = _d3_data_audit(
        experiment,
        benchmarks=load_phase_benchmarks(experiment, "confirmation"),
        selected_by_benchmark=confirmation,
        phase=phase,
        phase_name="confirmation",
    )
    assert audit["official_mini_count"] == 460
    assert audit["official_mini_overlap_with_inspected_count"] == 45
    assert audit["official_mini_text_hash_overlap_with_inspected_count"] == 45
    assert audit["selected_bbeh_inspected_overlap_count"] == 0
    assert audit["selected_bbeh_text_hash_overlap_with_inspected_count"] == 0


def test_d3_nested_dev50_is_task_stratified_and_compatibility_run_is_explicit() -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d3_count50.toml"
    )
    by_slug = {item.slug: item for item in load_phase_benchmarks(experiment, "development")}
    selected = {slug: select_samples(benchmark, "catch_kernel_d3_dev50_seed42") for slug, benchmark in by_slug.items()}
    assert len({str(item.metadata.get("task")) for item in selected["bbeh"]}) == 23
    assert {str(item.metadata.get("task")) for item in selected["musr"]} == {
        "murder_mysteries", "object_placements", "team_allocation"
    }
    compatibility = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_kernel_d3_benchmark_compat.toml"
    )
    assert compatibility.raw["phases"]["confirmation"]["evaluation_role"].startswith(
        "secondary_benchmark_compatibility"
    )
