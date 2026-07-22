from __future__ import annotations

from dataclasses import replace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.algorithms import CandidateClass, StageDecision
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    AdapterResult,
    CertificateVerifierParseResultV2,
    VerifierResultV2,
    build_all_candidate_pairs_v2,
    build_candidate_answer_nodes,
    build_source_span_graph,
    run_deterministic_adapters_v2,
)
from research_experiments.families.contrastive_active_testing.kernel import (
    bind_verifier_capabilities,
    build_proof_results,
    build_task_semantics,
    compile_answer_obligation_graph,
    compile_local_certificate_bank,
    compile_typed_obligations,
    decide_with_proof_kernel,
    kernel_decision_to_decode,
    semantics_requires_designer,
    task_contract_from_semantics,
    validate_kernel_certificate_bank,
)
from research_experiments.families.contrastive_active_testing.kernel_adapters import run_kernel_adapters


def _stage(*answers: str) -> StageDecision:
    candidates = tuple(
        CandidateClass(answer, answer, len(answers) - index, f"Reasoning concludes {answer}.", f"hash-{index}")
        for index, answer in enumerate(answers)
    )
    return StageDecision(
        anchor_key=answers[0],
        anchor_answer=answers[0],
        candidates=candidates,
        vote_counts={answer: len(answers) - index for index, answer in enumerate(answers)},
        valid_count=5,
    )


def _choice_contract(options: list[str]) -> dict[str, object]:
    return {
        "kind": "single_choice",
        "options": [{"label": chr(ord("A") + index), "text": text} for index, text in enumerate(options)],
    }


def test_musr_registry_encodes_official_question_semantics() -> None:
    options = ["desk", "Michael's room"]
    object_sample = DatasetSample(
        "musr",
        "object",
        "Which location would Alice look in for the stapler?",
        "A",
        "",
        {"task": "object_placements", "answer_contract": _choice_contract(options)},
    )
    object_semantics = build_task_semantics(object_sample, build_source_span_graph(object_sample))
    assert object_semantics.state_model == "observer_belief_not_objective_world_state"
    assert {item.operation_kind for item in object_semantics.mandatory_obligation_templates} == {"object_belief_state"}
    assert semantics_requires_designer(object_semantics) is False

    team_sample = DatasetSample("musr", "team", "Which team is best?", "A", "", {"task": "team_allocation"})
    team = build_task_semantics(team_sample, build_source_span_graph(team_sample))
    assert team.query_operator == "argmax"
    assert team.state_model == "skill_plus_teamwork_utility"

    murder_sample = DatasetSample("musr", "murder", "Who is guilty?", "A", "", {"task": "murder_mysteries"})
    murder = build_task_semantics(murder_sample, build_source_span_graph(murder_sample))
    assert [item.operation_kind for item in murder.mandatory_obligation_templates] == [
        "murder_means",
        "murder_motive",
        "murder_opportunity",
    ]


def test_bbeh_registry_distinguishes_global_operators_instead_of_using_one_semantic_default() -> None:
    cases = {
        "boolean_expressions": ("multiple_choice_truth", {"candidate_validity"}),
        "web_of_lies": ("exact_sequence", {"candidate_validity", "global_completeness"}),
        "movie_recommendation": ("argmax", {"candidate_validity", "comparative_dominance"}),
        "object_counting": ("point_value", {"candidate_validity", "global_completeness"}),
        "object_properties": (
            "final_state",
            {"candidate_validity", "final_state", "global_completeness"},
        ),
        "temporal_sequence": (
            "argmax",
            {"candidate_validity", "comparative_dominance", "global_completeness"},
        ),
        "time_arithmetic": (
            "exact_set",
            {"candidate_validity", "final_state", "global_completeness"},
        ),
    }
    for task, (operator, kinds) in cases.items():
        sample = DatasetSample("bbeh", task, "Task source", "x", "", {"task": task})
        semantics = build_task_semantics(sample, build_source_span_graph(sample))
        assert semantics.query_operator == operator
        assert {item.kind for item in semantics.mandatory_obligation_templates} == kinds


def test_word_sorting_registry_separates_earliest_trace_from_direct_custom_order() -> None:
    trace = DatasetSample(
        "bbeh",
        "sort-trace",
        "Identify the first step that was a mistake in reasoning. Thought 1: valid.",
        "1",
        "",
        {"task": "word_sorting"},
    )
    direct = DatasetSample(
        "bbeh",
        "sort-direct",
        "Consider a new alphabet. Sort the following words with the new alphabet: a, b.",
        "a,b",
        "",
        {"task": "word_sorting"},
    )
    trace_semantics = build_task_semantics(trace, build_source_span_graph(trace))
    direct_semantics = build_task_semantics(direct, build_source_span_graph(direct))

    assert trace_semantics.query_operator == "earliest"
    assert {item.kind for item in trace_semantics.mandatory_obligation_templates} == {
        "candidate_validity",
        "prefix_validity",
    }
    assert {item.guarantee_level for item in trace_semantics.mandatory_obligation_templates} == {"diagnostic_only"}
    assert semantics_requires_designer(trace_semantics) is False
    assert direct_semantics.query_operator == "exact_sequence"
    assert {item.operation_kind for item in direct_semantics.mandatory_obligation_templates} == {"custom_sort_order"}


def test_custom_alphabet_is_executed_and_can_override_without_model_calls() -> None:
    correct = (
        "administrations,built,characterizations,poland,reconfigurations,swab,"
        "thunderclap,implementations,interpretations"
    )
    conventional = (
        "administrations,built,characterizations,implementations,interpretations,"
        "poland,reconfigurations,swab,thunderclap"
    )
    sample = DatasetSample(
        "bbeh",
        "sort",
        "Consider a new alphabet whose letters have the same order as the English alphabet, except that t and i "
        "are the last two letters. Sort the following words with the new alphabet and separate them with comma: "
        "built, thunderclap, swab, poland, administrations, implementations, interpretations, reconfigurations, "
        "characterizations",
        correct,
        "",
        {"task": "word_sorting"},
    )
    stage = _stage(conventional, correct)
    mapping = {"H0": conventional, "H1": correct}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    contract = task_contract_from_semantics(semantics, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="sort")
    validation = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    adapters = run_kernel_adapters(
        sample,
        contract=contract,
        tests=validation.tests,
        answer_nodes=nodes,
        pairs=pairs,
    )
    assert {item.execution_status for item in adapters.values()} == {"EXECUTED"}
    bindings = bind_verifier_capabilities(semantics, validation)
    obligations = compile_typed_obligations(semantics, validation)
    proofs = build_proof_results(
        stage=stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        bindings=bindings,
        adapter_results=adapters,
        panels=(),
    )
    kernel = decide_with_proof_kernel(
        stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        obligations=obligations,
        proofs=proofs,
    )
    decision = kernel_decision_to_decode(stage, kernel, public_to_key=mapping)
    assert decision.override_accepted is True
    assert decision.answer_key == correct


def test_circular_spatial_path_is_executed_without_gold_or_model_fallback() -> None:
    source = (
        "You have been given a circular path consisting of 4 connected dots. At the start, you are positioned on "
        "the dot that is located at the top of the path, where you find a keyboard. Moving in a clockwise direction "
        "from the keyboard, the elements on the path are a library, a mailbox, and a plate. Starting from the "
        "mailbox, you move around the ring by 1 step in a counter-clockwise direction, and you move around the ring "
        "by 2 steps in a clockwise direction. What will you find?"
    )
    sample = DatasetSample("bbeh", "ring", source, "plate", "", {"task": "spatial_reasoning"})
    stage = _stage("library", "plate")
    mapping = {"H0": "library", "H1": "plate"}
    graph = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, graph)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="ring")
    validation = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=graph,
        pairs=pairs,
    )
    frozen_v2 = run_deterministic_adapters_v2(
        sample,
        contract=task_contract_from_semantics(semantics, graph),
        tests=validation.tests,
        answer_nodes=nodes,
        pairs=pairs,
    )
    adapters = run_kernel_adapters(
        sample,
        contract=task_contract_from_semantics(semantics, graph),
        tests=validation.tests,
        answer_nodes=nodes,
        pairs=pairs,
    )
    proofs = build_proof_results(
        stage=stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        bindings=bind_verifier_capabilities(semantics, validation),
        adapter_results=adapters,
        panels=(),
    )
    decision = decide_with_proof_kernel(
        stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        obligations=compile_typed_obligations(semantics, validation),
        proofs=proofs,
    )

    assert {item.execution_status for item in frozen_v2.values()} == {"UNSUPPORTED"}
    assert {item.execution_status for item in adapters.values()} == {"EXECUTED"}
    assert decision.decision == "OVERRIDE"
    assert kernel_decision_to_decode(stage, decision, public_to_key=mapping).answer_key == "plate"


def test_dyck_adapter_checks_candidate_error_and_complete_prefix() -> None:
    sample = DatasetSample(
        "bbeh",
        "dyck-runtime-golden",
        "Thought 1: [ ; stack: [\nThought 2: ] ; stack: [",
        "2",
        "",
        {"task": "dyck_languages"},
    )
    stage = _stage("1", "2")
    mapping = {"H0": "1", "H1": "2"}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    validation = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    results = run_kernel_adapters(
        sample,
        contract=task_contract_from_semantics(semantics, source),
        tests=validation.tests,
        answer_nodes=nodes,
        pairs=pairs,
    )
    assert results
    assert all(item.execution_status == "EXECUTED" for item in results.values())
    assert all(item.observed_outcome == "O1" for item in results.values())


def test_executable_conflict_never_falls_back_to_model_verifier() -> None:
    sample = DatasetSample(
        "seqbench",
        "seq",
        "Room A1 and A2 are connected by an open door. Bob is in A1. Alice is in A2.",
        '["start: A1","move_to: A2","rescue: Alice"]',
        "",
        {"seqbench_instance_metadata": {"agent_name": "Bob", "target_name": "Alice"}},
    )
    stage = _stage('["start: A1"]', '["start: A2"]')
    mapping = {"H0": stage.candidates[0].key, "H1": stage.candidates[1].key}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="seq")
    validation = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    conflicts = {
        test.test_id: AdapterResult(test.test_id, None, "CONFLICT", "both candidates invalid", "trace")
        for test in validation.tests
    }
    bindings = bind_verifier_capabilities(semantics, validation)
    proofs = build_proof_results(
        stage=stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        bindings=bindings,
        adapter_results=conflicts,
        panels=(),
    )
    kernel = decide_with_proof_kernel(
        stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        obligations=compile_typed_obligations(semantics, validation),
        proofs=proofs,
    )
    assert {item.status for item in proofs} == {"CONFLICT"}
    assert kernel.decision == "ABSTAIN"
    assert kernel.failure_layer == "adapter"
    assert kernel.reason_code == "adapter_conflict"


def test_kernel_schema_does_not_accept_model_authored_operation_kind() -> None:
    sample = DatasetSample("musr", "team", "Which team is best?", "A", "", {"task": "team_allocation"})
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="team")
    skeleton = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    payload = {
        "tests": [
            {
                "test_id": "T0",
                "pair_id": pairs[0].pair_id,
                "obligation_ids": [item.obligation_id for item in semantics.mandatory_obligation_templates],
                "operation_kind": "model_invented_operation",
                "finite_outcomes": [
                    {"outcome_id": "O0", "text": "left"},
                    {"outcome_id": "O1", "text": "right"},
                ],
                "expected_outcome_by_candidate": {
                    pairs[0].left_candidate: "O0",
                    pairs[0].right_candidate: "O1",
                },
                "source_span_ids": [source.spans[0].span_id],
                "typed_payload": {},
            }
        ],
        "certificates": [
            {
                "candidate_key_anon": "H1",
                "answer_hash": nodes["H1"].answer_hash,
                "required_test_ids": ["T0"],
            }
        ],
    }
    result = validate_kernel_certificate_bank(
        payload,
        semantics=semantics,
        skeleton=skeleton,
    )
    assert result.protocol_error == "kernel_payload_top_level_schema_failure"


def test_ungrounded_permutation_is_atomic_semantic_instead_of_model_authored_execution() -> None:
    sample = DatasetSample(
        "bbeh",
        "shuffle-slots",
        "Alice and Bob swap their objects. Which object does Alice have?",
        "B",
        "",
        {"task": "shuffled_objects", "answer_contract": _choice_contract(["book", "pen"])},
    )
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    skeleton = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    assert semantics_requires_designer(semantics) is False
    assert semantics.authorized_verifier_kinds == ("model.bounded_semantic_panel",)
    assert len(skeleton.tests) == 4
    assert all(test.deterministic_payload == {} for test in skeleton.tests)
    assert all(
        {outcome.outcome_id for outcome in test.finite_outcomes} == {"TRUE", "FALSE", "UNKNOWN"}
        for test in skeleton.tests
    )
    assert sum(test.test_id.endswith("_support") for test in skeleton.tests) == 2
    assert sum(test.test_id.endswith("_refute_anchor") for test in skeleton.tests) == 2


def test_any_challenger_conflict_is_terminal_even_when_another_challenger_passes() -> None:
    sample = DatasetSample(
        "bbeh",
        "multi-conflict",
        "Consider a new alphabet. Sort the following words: a, b.",
        "a,b",
        "",
        {"task": "word_sorting"},
    )
    stage = _stage("b,a", "a,b", "a,a")
    mapping = {"H0": "b,a", "H1": "a,b", "H2": "a,a"}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    validation = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    public_by_key = {key: public for public, key in mapping.items()}
    adapters = {}
    for test in validation.tests:
        pair = pair_by_id[test.pair_id]
        challenger_public = public_by_key[pair.challenger_key]
        if pair.challenger_key == "a,b":
            adapters[test.test_id] = AdapterResult(
                test.test_id,
                test.expected_outcome_by_candidate[challenger_public],
                "EXECUTED",
                "valid",
                "trace-pass",
            )
        else:
            adapters[test.test_id] = AdapterResult(
                test.test_id,
                None,
                "CONFLICT",
                "ambiguous adapter result",
                "trace-conflict",
            )
    proofs = build_proof_results(
        stage=stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        bindings=bind_verifier_capabilities(semantics, validation),
        adapter_results=adapters,
        panels=(),
    )
    decision = decide_with_proof_kernel(
        stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        obligations=compile_typed_obligations(semantics, validation),
        proofs=proofs,
    )
    assert decision.decision == "ABSTAIN"
    assert decision.reason_code == "adapter_conflict"


def test_kernel_requires_certificates_for_every_challenger() -> None:
    sample = DatasetSample("musr", "coverage", "Which option is true?", "B", "", {"task": "team_allocation"})
    stage = _stage("A", "B", "C")
    mapping = {"H0": "A", "H1": "B", "H2": "C"}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    complete = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    incomplete = replace(
        complete,
        certificates=complete.certificates[:1],
        eligible_challengers=complete.eligible_challengers[:1],
        answer_link_coverage=0.5,
    )
    decision = decide_with_proof_kernel(
        stage,
        semantics=semantics,
        validation=incomplete,
        public_to_key=mapping,
        obligations=compile_typed_obligations(semantics, incomplete),
        proofs=(),
    )
    assert decision.decision == "ABSTAIN"
    assert decision.failure_layer == "answer_link"
    assert decision.reason_code == "all_challenger_certificates_required"


def test_bounded_semantic_proof_requires_both_panels_to_refute_anchor() -> None:
    sample = DatasetSample(
        "musr",
        "team-proof",
        "Team Red has utility 3. Team Blue has utility 5. Which team is best?",
        "B",
        "",
        {"task": "team_allocation", "answer_contract": _choice_contract(["Team Red", "Team Blue"])},
    )
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="team-proof")
    validation = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    assert validation.protocol_error is None

    def panel(*, refute_anchor: bool) -> CertificateVerifierParseResultV2:
        results = {}
        for test in validation.tests:
            challenger_expected = test.expected_outcome_by_candidate["H1"]
            anchor_expected = test.expected_outcome_by_candidate["H0"]
            results[test.test_id] = VerifierResultV2(
                test.test_id,
                challenger_expected,
                "ENTAILED",
                (source.spans[0].span_id,),
                (anchor_expected,) if refute_anchor else (),
                "ok",
            )
        return CertificateVerifierParseResultV2(
            True,
            results,
            len(results),
            len(results),
            (),
            0,
        )

    bindings = bind_verifier_capabilities(semantics, validation)
    incomplete = build_proof_results(
        stage=stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        bindings=bindings,
        adapter_results={},
        panels=(panel(refute_anchor=True), panel(refute_anchor=False)),
    )
    complete = build_proof_results(
        stage=stage,
        semantics=semantics,
        validation=validation,
        public_to_key=mapping,
        bindings=bindings,
        adapter_results={},
        panels=(panel(refute_anchor=True), panel(refute_anchor=True)),
    )
    assert {item.status for item in incomplete} == {"FAIL"}
    assert {item.status for item in complete} == {"PASS"}


def test_local_kernel_is_invariant_to_public_label_swap() -> None:
    sample = DatasetSample(
        "bbeh",
        "dyck",
        "Thought 1: [ ; stack: [\nThought 2: ] ; stack: [",
        "2",
        "",
        {"task": "dyck_languages"},
    )
    stage = _stage("1", "2")
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    outcomes = []
    for mapping in ({"H0": "1", "H1": "2"}, {"H0": "2", "H1": "1"}):
        nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
        pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="dyck")
        validation = compile_local_certificate_bank(
            sample=sample,
            semantics=semantics,
            stage=stage,
            public_to_key=mapping,
            answer_nodes=nodes,
            source_graph=source,
            pairs=pairs,
        )
        adapters = run_deterministic_adapters_v2(
            sample,
            contract=task_contract_from_semantics(semantics, source),
            tests=validation.tests,
            answer_nodes=nodes,
            pairs=pairs,
        )
        proofs = build_proof_results(
            stage=stage,
            semantics=semantics,
            validation=validation,
            public_to_key=mapping,
            bindings=bind_verifier_capabilities(semantics, validation),
            adapter_results=adapters,
            panels=(),
        )
        decision = decide_with_proof_kernel(
            stage,
            semantics=semantics,
            validation=validation,
            public_to_key=mapping,
            obligations=compile_typed_obligations(semantics, validation),
            proofs=proofs,
        )
        outcomes.append(kernel_decision_to_decode(stage, decision, public_to_key=mapping).answer_key)
    assert outcomes == ["2", "2"]


def test_answer_obligation_graph_records_dependencies_and_stable_hash() -> None:
    sample = DatasetSample(
        "musr",
        "graph",
        "Who has means, motive, and opportunity?",
        "B",
        "",
        {"task": "murder_mysteries"},
    )
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    validation = compile_local_certificate_bank(
        sample=sample,
        semantics=semantics,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    obligations = compile_typed_obligations(semantics, validation)
    first = compile_answer_obligation_graph(semantics, obligations)
    second = compile_answer_obligation_graph(semantics, obligations)
    assert first.graph_hash == second.graph_hash
    assert len(first.nodes) == 6
    assert len(first.dependency_edges) == 5
