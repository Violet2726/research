from __future__ import annotations

from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample, load_split_ids
from research_experiments.core.data.evaluation import evaluate_seqbench_prediction
from research_experiments.families.contrastive_active_testing.algorithms import (
    CandidateClass,
    StageDecision,
)
from research_experiments.families.contrastive_active_testing.certificates import (
    AnswerCertificate,
    CertificateBankValidation,
    CertificateOutcome,
    CertificateTest,
    CertificateVerifierParseResult,
    VerifierResult,
    build_claim_graphs,
    build_task_contract,
    decode_certificates,
    validate_certificate_bank,
)
from research_experiments.families.contrastive_active_testing.run.execute import (
    CatchSampleJob,
    _select_cert_disagreement_jobs,
)


def _stage() -> StageDecision:
    return StageDecision(
        anchor_key="A",
        anchor_answer="A",
        candidates=(
            CandidateClass("A", "A", 3, "A follows from premise one.", "a"),
            CandidateClass("B", "B", 2, "B follows from premise two.", "b"),
        ),
        vote_counts={"A": 3, "B": 2},
        valid_count=5,
    )


def _sample() -> DatasetSample:
    return DatasetSample(
        dataset="musr",
        sample_id="unit",
        question="Which candidate is supported?",
        reference_answer="A",
        prompt_context="",
        metadata={"task": "murder_mysteries"},
    )


def test_certificate_contract_graph_and_gold_free_validation() -> None:
    sample = _sample()
    contract = build_task_contract(sample)
    stage = _stage()
    public_to_key = {"H0": "A", "H1": "B"}
    graphs = build_claim_graphs(stage, public_to_key=public_to_key)
    payload = {
        "certificates": [
            {
                "candidate_key_anon": "H1",
                "required_conditions": ["T0"],
                "predicted_outcome": "B is supported",
                "refutation_condition": "T0",
                "evidence_refs": ["H1:N0"],
                "dependency_refs": [],
            }
        ],
        "tests": [
            {
                "test_id": "T0",
                "pair_id": "P0",
                "question_or_operation": "Which premise is sufficient for the queried conclusion?",
                "finite_outcomes": [
                    {"outcome_id": "O0", "text": "premise one"},
                    {"outcome_id": "O1", "text": "premise two"},
                ],
                "expected_outcome_by_candidate": {"H0": "O0", "H1": "O1"},
                "provenance_refs": ["H0:N0", "H1:N0"],
                "task_family": "semantic",
            }
        ],
    }
    result = validate_certificate_bank(
        payload,
        contract=contract,
        stage=stage,
        public_to_key=public_to_key,
        graphs=graphs,
        pair_candidates={"P0": ("H0", "H1")},
    )
    assert result.protocol_error is None
    assert len(result.tests) == 1
    assert result.eligible_challengers == ("B",)
    assert result.leakage_count == 0


def test_claim_graph_spans_hashes_and_dependencies_are_stable() -> None:
    stage = _stage()
    mapping = {"H0": "A", "H1": "B"}
    first = build_claim_graphs(stage, public_to_key=mapping)
    second = build_claim_graphs(stage, public_to_key=mapping)
    assert first == second
    assert first["H0"].trace_hash == "a"
    assert first["H0"].nodes[0].source_span_refs[0].start == 0
    assert len(first["H0"].nodes[0].source_span_refs[0].sha256) == 64


def test_certificate_decoder_requires_both_panels_and_anchor_refutation() -> None:
    stage = _stage()
    test = CertificateTest(
        "T0",
        "P0",
        "finite check",
        (CertificateOutcome("O0", "A"), CertificateOutcome("O1", "B")),
        {"H0": "O0", "H1": "O1"},
        ("H0:N0", "H1:N0"),
        "semantic",
    )
    certificate = AnswerCertificate("H1", ("T0",), "B", "T0", ("H1:N0",), (), "hash")
    support_panel = CertificateVerifierParseResult(
        True,
        {
            "T0": VerifierResult("T0", "O1", "ENTAILED", "", ("source",), "ok"),
        },
        1,
        1,
        (),
    )
    refutation_panel = CertificateVerifierParseResult(
        True,
        {
            "T0": VerifierResult("T0", "O1", "ENTAILED", "anchor condition is contradicted", ("source",), "ok"),
        },
        1,
        1,
        (),
    )
    validation = CertificateBankValidation((certificate,), (test,), (), None, 0, (), ("B",))
    decision = decode_certificates(
        stage,
        validation=validation,
        public_to_key={"H0": "A", "H1": "B"},
        panels=(support_panel, refutation_panel),
    )
    assert decision.override_accepted is True
    assert decision.answer_key == "B"


def test_certificate_leakage_is_rejected() -> None:
    sample = _sample()
    stage = _stage()
    graphs = build_claim_graphs(stage, public_to_key={"H0": "A", "H1": "B"})
    result = validate_certificate_bank(
        {
            "certificates": [],
            "tests": [
                {
                    "test_id": "T0",
                    "pair_id": "P0",
                    "question_or_operation": "Which candidate H1 is correct?",
                    "finite_outcomes": [
                        {"outcome_id": "O0", "text": "one"},
                        {"outcome_id": "O1", "text": "two"},
                    ],
                    "expected_outcome_by_candidate": {"H0": "O0", "H1": "O1"},
                    "provenance_refs": ["H0:N0", "H1:N0"],
                    "task_family": "semantic",
                }
            ],
        },
        contract=build_task_contract(sample),
        stage=stage,
        public_to_key={"H0": "A", "H1": "B"},
        graphs=graphs,
        pair_candidates={"P0": ("H0", "H1")},
    )
    assert result.tests == ()
    assert result.leakage_count == 1


def test_equation_and_proof_adapters_reject_explicit_conflicts() -> None:
    stage = _stage()
    public_to_key = {"H0": "A", "H1": "B"}
    graphs = build_claim_graphs(stage, public_to_key=public_to_key)

    def payload(family: str, operation: str, outcomes: tuple[str, str]) -> dict[str, object]:
        return {
            "certificates": [
                {
                    "candidate_key_anon": "H1",
                    "required_conditions": ["T0"],
                    "predicted_outcome": "candidate outcome",
                    "refutation_condition": "T0",
                    "evidence_refs": ["H1:N0"],
                    "dependency_refs": [],
                }
            ],
            "tests": [
                {
                    "test_id": "T0",
                    "pair_id": "P0",
                    "question_or_operation": operation,
                    "finite_outcomes": [
                        {"outcome_id": "O0", "text": outcomes[0]},
                        {"outcome_id": "O1", "text": outcomes[1]},
                    ],
                    "expected_outcome_by_candidate": {"H0": "O0", "H1": "O1"},
                    "provenance_refs": ["H0:N0", "H1:N0"],
                    "task_family": family,
                }
            ],
        }

    equation_sample = DatasetSample("bbeh", "eq", "q", "4", "", {"task": "multistep_arithmetic"})
    equation = validate_certificate_bank(
        payload("equation", "CHECK: 2 + 2 == 5", ("CONSISTENT", "INCONSISTENT")),
        contract=build_task_contract(equation_sample),
        stage=stage,
        public_to_key=public_to_key,
        graphs=graphs,
        pair_candidates={"P0": ("H0", "H1")},
    )
    assert equation.adapter_conflicts[0]["reason"] == "equation_residual_nonzero"

    proof_sample = DatasetSample("bbeh", "proof", "q", "yes", "", {"task": "boardgame_qa"})
    proof = validate_certificate_bank(
        payload("proof_state", "Determine the proof state", ("YES", "NO")),
        contract=build_task_contract(proof_sample),
        stage=stage,
        public_to_key=public_to_key,
        graphs=graphs,
        pair_candidates={"P0": ("H0", "H1")},
    )
    assert proof.adapter_conflicts[0]["reason"] == "proof_outcome_schema_missing"


def test_seqbench_metrics_include_exact_progress_precision_recall_and_execution_prefix() -> None:
    sample = DatasetSample(
        dataset="seqbench",
        sample_id="seq-unit",
        question=(
            "Room A1 and A2 are connected by an open door. Room A2 and B2 are connected by an open door. "
            "Bob is in room A1. Alice is in room B2."
        ),
        reference_answer='["start: A1","move_to: A2","move_to: B2","rescue: Alice"]',
        prompt_context="",
        metadata={"task": "B0_N0", "seqbench_instance_metadata": {"agent_name": "Bob", "target_name": "Alice"}},
    )
    exact = evaluate_seqbench_prediction(sample.reference_answer, sample.reference_answer, sample=sample)
    assert exact.exact_match == 1.0
    assert exact.progress_ratio == 1.0
    assert exact.execution_prefix_ratio == 1.0
    invalid = evaluate_seqbench_prediction(
        '["start: A1","move_to: B2","rescue: Alice"]',
        sample.reference_answer,
        sample=sample,
    )
    assert invalid.exact_match == 0.0
    assert invalid.progress_ratio == 0.25
    assert invalid.first_invalid_action_reason == "non_adjacent_or_locked_move"


def test_cert_disagreement_selection_is_gold_blind_and_capped_per_dataset() -> None:
    samples = [
        DatasetSample("bbeh", f"b{i}", "q", "A", "", {"task": "boardgame_qa"})
        for i in range(3)
    ] + [
        DatasetSample("musr", f"m{i}", "q", "A", "", {"task": "murder_mysteries"})
        for i in range(2)
    ]
    jobs = [CatchSampleJob(i, sample, "split", None) for i, sample in enumerate(samples)]
    stages = {
        (sample.dataset, sample.sample_id): ([], SimpleNamespace(triggered=sample.sample_id not in {"b1", "m1"}))
        for sample in samples
    }
    selected = _select_cert_disagreement_jobs(jobs, stages, cap_per_dataset=1, seed=42)
    assert len(selected) == 2
    assert {job.sample.dataset for job in selected} == {"bbeh", "musr"}


def test_cert_development_holdout_and_boundary_splits_are_disjoint() -> None:
    specs = (
        ("bbeh/bbeh-main", "catch_cert_dev200_seed42", "catch_cert_holdout200_seed42"),
        ("musr/all", "catch_cert_dev200_seed42", "catch_cert_holdout200_seed42"),
        ("seqbench/seqBench_compact.jsonl", "catch_cert_dev200_seed42", "catch_cert_holdout200_seed42"),
        ("gpqa/dataset", "catch_cert_dev98_seed42", "catch_cert_holdout0_seed42"),
    )
    for dataset_key, development, heldout in specs:
        dev_ids = set(load_split_ids(dataset_key, development))
        heldout_ids = set(load_split_ids(dataset_key, heldout))
        boundary_ids = set(load_split_ids(dataset_key, "count100_seed42"))
        assert dev_ids.isdisjoint(heldout_ids)
        assert dev_ids.isdisjoint(boundary_ids)
        assert heldout_ids.isdisjoint(boundary_ids)
