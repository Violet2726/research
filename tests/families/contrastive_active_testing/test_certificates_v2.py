from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import validate_seqbench_plan
from research_experiments.families.contrastive_active_testing.algorithms import CandidateClass, StageDecision
from research_experiments.families.contrastive_active_testing.cert_prompts_v2 import (
    build_certificate_designer_messages_v2,
)
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    CertificateVerifierParseResultV2,
    VerifierResultV2,
    build_all_candidate_pairs_v2,
    build_candidate_answer_nodes,
    build_certificate_verifier_packet_v2,
    build_source_span_graph,
    build_task_contract_v2,
    decode_certificates_v2,
    parse_certificate_verifier_v2,
    run_deterministic_adapters_v2,
    validate_certificate_bank_v2,
)


def _choice_contract(options: list[str]) -> dict[str, object]:
    return {
        "kind": "single_choice",
        "options": [{"label": chr(ord("A") + index), "text": text} for index, text in enumerate(options)],
        "block_start": None,
        "block_end": None,
    }


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


def _gpqa() -> DatasetSample:
    options = ["wrong compound", "six-membered pyran product", "third product", "fourth product"]
    return DatasetSample(
        "gpqa_diamond",
        "gpqa-unit",
        "Which reaction product is formed?",
        "B|||six-membered pyran product",
        "",
        {
            "options": options,
            "answer_contract": _choice_contract(options),
            "high_level_domain": "Chemistry",
            "subdomain": "Organic Chemistry",
        },
    )


def _valid_payload(contract, nodes, pairs, source, *, candidate: str) -> dict[str, object]:
    pair = next(pair for pair in pairs if candidate in {pair.left_candidate, pair.right_candidate})
    other = pair.right_candidate if candidate == pair.left_candidate else pair.left_candidate
    obligations = [item.obligation_id for item in contract.mandatory_obligations]
    return {
        "tests": [
            {
                "test_id": "T0",
                "pair_id": pair.pair_id,
                "obligation_ids": obligations,
                "operation_kind": contract.adapter_kind,
                "question_or_operation": "Which finite result satisfies every question obligation?",
                "finite_outcomes": [
                    {"outcome_id": "O0", "text": "first result"},
                    {"outcome_id": "O1", "text": "second result"},
                ],
                "expected_outcome_by_candidate": {candidate: "O1", other: "O0"},
                "source_span_ids": [source.spans[0].span_id],
                "deterministic_payload": {},
            }
        ],
        "certificates": [
            {
                "candidate_key_anon": candidate,
                "answer_hash": nodes[candidate].answer_hash,
                "required_test_ids": ["T0"],
            }
        ],
    }


def test_answer_nodes_expose_option_meaning_without_gold_or_votes() -> None:
    sample = _gpqa()
    stage = _stage("A", "B")
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key={"H0": "A", "H1": "B"})
    assert nodes["H1"].answer_type == "choice"
    assert nodes["H1"].rendered_content == "six-membered pyran product"
    rendered = repr(nodes)
    assert "vote" not in rendered.casefold()
    assert "gold" not in rendered.casefold()


def test_source_span_graph_is_exactly_reversible_and_keeps_latex_expression_intact() -> None:
    source = "Compute $x = 2.5 + y!$ exactly.\nThen choose A."
    sample = DatasetSample("gpqa_diamond", "symbolic-spans", source, "A", "", {"answer_contract": _choice_contract(["yes", "no"])})
    graph = build_source_span_graph(sample)
    assert "".join(span.text for span in graph.spans) == source
    assert any("$x = 2.5 + y!$ exactly." in span.text for span in graph.spans)


def test_gpqa_contract_uses_domain_metadata() -> None:
    sample = _gpqa()
    contract = build_task_contract_v2(sample, build_source_span_graph(sample))
    assert contract.query_operator == "multiple_choice_truth"
    assert contract.adapter_kind == "gpqa_chemistry"
    assert contract.family == "semantic"
    assert {item.kind for item in contract.mandatory_obligations} == {
        "candidate_validity",
        "constraint_satisfaction",
    }


def test_all_stage_candidates_are_paired_with_anchor() -> None:
    stage = _stage("A", "B", "C", "D", "E")
    mapping = {f"H{index}": answer for index, answer in enumerate(("A", "B", "C", "D", "E"))}
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="unit")
    assert len(pairs) == 4
    assert {pair.challenger_key for pair in pairs} == {"B", "C", "D", "E"}


def test_dynamic_prompt_uses_real_pair_candidates_and_answer_hash() -> None:
    sample = _gpqa()
    stage = _stage("A", "B", "C")
    mapping = {"H2": "A", "H4": "B", "H7": "C"}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    messages = build_certificate_designer_messages_v2(
        sample,
        contract=contract,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
        reasoning_claims={key: () for key in nodes},
    )
    prompt = messages[1]["content"]
    first = pairs[0]
    assert f'"{first.left_candidate}"' in prompt
    assert f'"{first.right_candidate}"' in prompt
    assert nodes[first.left_candidate].answer_hash in prompt


def test_mandatory_earliest_prefix_obligation_cannot_be_omitted() -> None:
    sample = DatasetSample(
        "bbeh",
        "dyck-unit",
        "Thought 1: [ ; stack: [\nThought 2: ] ; stack: [",
        "2",
        "",
        {"task": "dyck_languages"},
    )
    stage = _stage("no", "2")
    mapping = {"H0": "no", "H1": "2"}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    payload = _valid_payload(contract, nodes, pairs, source, candidate="H1")
    payload["tests"][0]["obligation_ids"] = [contract.mandatory_obligations[0].obligation_id]
    invalid = validate_certificate_bank_v2(
        payload,
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    assert invalid.certificates == ()
    assert any(row["reason"] == "mandatory_obligation_missing" for row in invalid.dropped)


def test_invalid_test_removal_rebuilds_certificate_instead_of_cascading() -> None:
    sample = _gpqa()
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    payload = _valid_payload(contract, nodes, pairs, source, candidate="H1")
    payload["tests"].append({"test_id": "bad"})
    payload["certificates"][0]["required_test_ids"] = ["bad", "T0"]
    result = validate_certificate_bank_v2(
        payload,
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    assert result.protocol_error is None
    assert result.certificates[0].required_test_ids == ("T0",)


def test_verifier_requires_real_source_ids_and_repairs_unique_enum_typo() -> None:
    sample = _gpqa()
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    validation = validate_certificate_bank_v2(
        _valid_payload(contract, nodes, pairs, source, candidate="H1"),
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    packet = build_certificate_verifier_packet_v2(
        validation.tests,
        source_graph=source,
        seed=42,
        sample_id=sample.sample_id,
        panel_index=1,
    )
    public_outcome = next(iter(packet.public_outcome_to_internal["Q0"]))
    parsed = parse_certificate_verifier_v2(
        {
            "results": [
                {
                    "test_id": "Q0",
                    "observed_outcome": public_outcome,
                    "support_status": "ENTILED",
                    "source_span_ids": ["S0"],
                    "ruled_out_outcomes": [],
                }
            ]
        },
        packet=packet,
    )
    assert parsed.valid_test_count == 1
    assert parsed.format_repair_count == 1
    invalid = parse_certificate_verifier_v2(
        {
            "results": [
                {
                    "test_id": "Q0",
                    "observed_outcome": public_outcome,
                    "support_status": "ENTAILED",
                    "source_span_ids": ["invented"],
                    "ruled_out_outcomes": [],
                }
            ]
        },
        packet=packet,
    )
    assert invalid.valid_test_count == 0


def test_underdetermined_may_omit_source_but_never_overrides() -> None:
    sample = _gpqa()
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    validation = validate_certificate_bank_v2(
        _valid_payload(contract, nodes, pairs, source, candidate="H1"),
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    result = VerifierResultV2("T0", "O1", "UNDERDETERMINED", (), (), "ok")
    panel = CertificateVerifierParseResultV2(True, {"T0": result}, 1, 1, (), 0)
    decision = decode_certificates_v2(
        stage,
        validation=validation,
        public_to_key=mapping,
        panels=(panel, panel),
        adapter_results={},
    )
    assert decision.override_accepted is False


def test_decoder_no_longer_requires_counterexample_for_entailed_outcome() -> None:
    sample = _gpqa()
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    validation = validate_certificate_bank_v2(
        _valid_payload(contract, nodes, pairs, source, candidate="H1"),
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    result = VerifierResultV2("T0", "O1", "ENTAILED", ("S0",), ("O0",), "ok")
    panel = CertificateVerifierParseResultV2(True, {"T0": result}, 1, 1, (), 0)
    decision = decode_certificates_v2(
        stage,
        validation=validation,
        public_to_key=mapping,
        panels=(panel, panel),
        adapter_results={},
    )
    assert decision.override_accepted is True
    assert decision.answer_key == "B"


def test_seqbench_partial_valid_plan_is_incomplete() -> None:
    sample = DatasetSample(
        "seqbench",
        "seq-unit",
        "Room A1 and A2 are connected by an open door. Bob is in room A1. Alice is in room A2.",
        '["start: A1","move_to: A2","rescue: Alice"]',
        "",
        {"seqbench_instance_metadata": {"agent_name": "Bob", "target_name": "Alice"}},
    )
    partial = validate_seqbench_plan('["start: A1","move_to: A2"]', sample=sample)
    complete = validate_seqbench_plan(sample.reference_answer, sample=sample)
    assert partial.complete is False
    assert partial.first_failure == "missing_final_rescue"
    assert complete.complete is True


def test_seqbench_adapter_can_override_without_model_panels() -> None:
    sample = DatasetSample(
        "seqbench",
        "seq-unit",
        "Room A1 and A2 are connected by an open door. Bob is in room A1. Alice is in room A2.",
        '["start: A1","move_to: A2","rescue: Alice"]',
        "",
        {"seqbench_instance_metadata": {"agent_name": "Bob", "target_name": "Alice"}},
    )
    incomplete = '["start: A1","move_to: A2"]'
    complete = sample.reference_answer
    stage = _stage(incomplete, complete)
    mapping = {"H0": incomplete, "H1": complete}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    payload = _valid_payload(contract, nodes, pairs, source, candidate="H1")
    validation = validate_certificate_bank_v2(
        payload,
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    adapters = run_deterministic_adapters_v2(
        sample,
        contract=contract,
        tests=validation.tests,
        answer_nodes=nodes,
        pairs=pairs,
    )
    assert adapters["T0"].execution_status == "EXECUTED"
    decision = decode_certificates_v2(
        stage,
        validation=validation,
        public_to_key=mapping,
        panels=(),
        adapter_results=adapters,
    )
    assert decision.override_accepted is True


def test_constraint_adapter_checks_only_explicit_witness() -> None:
    sample = DatasetSample("musr", "team", "Who can join?", "B", "", {"task": "team_allocation"})
    stage = _stage("A", "B")
    mapping = {"H0": "A", "H1": "B"}
    source = build_source_span_graph(sample)
    contract = build_task_contract_v2(sample, source)
    nodes = build_candidate_answer_nodes(sample, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id=sample.sample_id)
    payload = _valid_payload(contract, nodes, pairs, source, candidate="H1")
    payload["tests"][0]["deterministic_payload"] = {
        "assignments_by_candidate": {
            "H0": {"alice": "red", "bob": "red"},
            "H1": {"alice": "red", "bob": "blue"},
        },
        "constraints": [{"left": "alice", "operator": "!=", "right": "bob"}],
    }
    validation = validate_certificate_bank_v2(
        payload,
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    adapters = run_deterministic_adapters_v2(
        sample,
        contract=contract,
        tests=validation.tests,
        answer_nodes=nodes,
        pairs=pairs,
    )
    assert adapters["T0"].execution_status == "EXECUTED"
    assert adapters["T0"].observed_outcome == "O1"


def test_permutation_and_object_state_adapters_execute_explicit_payloads() -> None:
    permutation = DatasetSample("bbeh", "perm", "Track swaps.", "2", "", {"task": "shuffled_objects"})
    stage = _stage("1", "2")
    mapping = {"H0": "1", "H1": "2"}
    source = build_source_span_graph(permutation)
    contract = build_task_contract_v2(permutation, source)
    nodes = build_candidate_answer_nodes(permutation, stage, public_to_key=mapping)
    pairs = build_all_candidate_pairs_v2(stage, public_to_key=mapping, seed=42, sample_id="perm")
    payload = _valid_payload(contract, nodes, pairs, source, candidate="H1")
    payload["tests"][0]["deterministic_payload"] = {
        "initial_order": ["alice", "bob"],
        "swaps": [["alice", "bob"]],
        "query_item": "alice",
    }
    validation = validate_certificate_bank_v2(
        payload,
        contract=contract,
        stage=stage,
        public_to_key=mapping,
        answer_nodes=nodes,
        source_graph=source,
        pairs=pairs,
    )
    result = run_deterministic_adapters_v2(
        permutation,
        contract=contract,
        tests=validation.tests,
        answer_nodes=nodes,
        pairs=pairs,
    )["T0"]
    assert result.execution_status == "EXECUTED"
    assert result.observed_outcome == "O1"

    object_sample = DatasetSample(
        "musr",
        "object",
        "The ball starts in the kitchen and is moved to the garden.",
        "garden",
        "",
        {"task": "object_placements"},
    )
    object_stage = _stage("kitchen", "garden")
    object_mapping = {"H0": "kitchen", "H1": "garden"}
    object_source = build_source_span_graph(object_sample)
    object_contract = build_task_contract_v2(object_sample, object_source)
    object_nodes = build_candidate_answer_nodes(object_sample, object_stage, public_to_key=object_mapping)
    object_pairs = build_all_candidate_pairs_v2(
        object_stage,
        public_to_key=object_mapping,
        seed=42,
        sample_id="object",
    )
    object_payload = _valid_payload(object_contract, object_nodes, object_pairs, object_source, candidate="H1")
    object_payload["tests"][0]["deterministic_payload"] = {
        "initial_locations": {"ball": "kitchen"},
        "events": [{"entity": "ball", "to": "garden"}],
        "query_entity": "ball",
    }
    object_validation = validate_certificate_bank_v2(
        object_payload,
        contract=object_contract,
        stage=object_stage,
        public_to_key=object_mapping,
        answer_nodes=object_nodes,
        source_graph=object_source,
        pairs=object_pairs,
    )
    object_result = run_deterministic_adapters_v2(
        object_sample,
        contract=object_contract,
        tests=object_validation.tests,
        answer_nodes=object_nodes,
        pairs=object_pairs,
    )["T0"]
    assert object_result.execution_status == "EXECUTED"
    assert object_result.observed_outcome == "O1"
