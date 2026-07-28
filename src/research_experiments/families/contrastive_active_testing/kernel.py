"""CATCH-Kernel：类型化义务、验证辖域与确定性证明内核。

The model is treated as an untrusted proof producer.  Task semantics and the
meaning of every operation are selected locally from a versioned registry;
model output may only fill the declared slots.  The deterministic kernel never
falls back from an executable jurisdiction to a model verifier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.families.contrastive_active_testing.algorithms import DecodeDecision, StageDecision
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    AdapterResult,
    AnswerCertificateV2,
    CandidateAnswerNode,
    CandidatePairV2,
    CertificateBankValidationV2,
    CertificateOutcomeV2,
    CertificateTestV2,
    CertificateVerifierParseResultV2,
    QuestionObligation,
    SourceSpanGraph,
    TaskContractV2,
)
from research_experiments.families.contrastive_active_testing.kernel_adapters import (
    CandidateAdapterResult,
    compile_local_typed_payload,
    validate_typed_payload,
)

GuaranteeLevel = Literal["executable", "bounded_semantic", "diagnostic_only"]
ProofStatus = Literal["PASS", "FAIL", "UNKNOWN", "CONFLICT", "UNSUPPORTED"]
KernelAction = Literal["KEEP", "OVERRIDE", "ABSTAIN"]
PayloadSource = Literal["local", "model_slots", "none"]

KERNEL_SCHEMA_VERSION = "catch_kernel_typed_obligations_v3"
KERNEL_SEMANTICS_VERSION = "catch_kernel_task_semantics_v3"
KERNEL_CAPABILITY_VERSION = "catch_kernel_verifier_capabilities_v3"
KERNEL_DECODER_VERSION = "catch_kernel_proof_decoder_v3"
KERNEL_D2_DECODER_VERSION = "catch_kernel_unary_exact_decoder_v1"


@dataclass(frozen=True)
class ObligationTemplate:
    obligation_id: str
    kind: str
    operation_kind: str
    description: str
    guarantee_level: GuaranteeLevel
    payload_source: PayloadSource
    required: bool = True


@dataclass(frozen=True)
class TaskSemantics:
    task_family: str
    query_operator: str
    answer_type: str
    state_model: str
    mandatory_obligation_templates: tuple[ObligationTemplate, ...]
    authorized_verifier_kinds: tuple[str, ...]
    unsupported_behavior: str
    semantics_version: str = KERNEL_SEMANTICS_VERSION


@dataclass(frozen=True)
class TypedObligation:
    obligation_id: str
    test_id: str
    pair_id: str
    operation_kind: str
    typed_payload: dict[str, Any]
    candidate_commitments: dict[str, str]
    source_span_ids: tuple[str, ...]
    temporal_scope: str
    completeness_scope: str
    required: bool


@dataclass(frozen=True)
class AnswerObligationGraph:
    semantics_version: str
    nodes: tuple[TypedObligation, ...]
    dependency_edges: tuple[tuple[str, str, str], ...]
    required_obligation_ids: tuple[str, ...]
    graph_hash: str


@dataclass(frozen=True)
class VerifierCapability:
    verifier_kind: str
    supported_operation_kinds: tuple[str, ...]
    guarantee_level: GuaranteeLevel
    possible_results: tuple[str, ...]
    failure_semantics: str
    capability_version: str = KERNEL_CAPABILITY_VERSION


@dataclass(frozen=True)
class VerifierBinding:
    test_id: str
    operation_kind: str
    verifier_kind: str
    guarantee_level: GuaranteeLevel
    binding_status: Literal["BOUND", "UNSUPPORTED"]


@dataclass(frozen=True)
class ProofResult:
    obligation_id: str
    test_id: str
    candidate_key_anon: str
    status: ProofStatus
    observed_value: str | None
    ruled_out_values: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    provenance_valid: bool
    entailment_valid: bool
    obligation_valid: bool
    sufficiency_valid: bool
    execution_trace_hash: str
    verifier_kind: str
    detail: str


@dataclass(frozen=True)
class KernelDecision:
    anchor_id: str
    challenger_id: str | None
    required_obligations: tuple[str, ...]
    covered_obligations: tuple[str, ...]
    accepted_proofs: tuple[str, ...]
    decision: KernelAction
    failure_layer: str
    reason_code: str
    diagnostics: tuple[dict[str, Any], ...]


_DETERMINISTIC_CAPABILITIES = (
    VerifierCapability(
        "deterministic.seq_plan",
        ("seq_plan",),
        "executable",
        ("PASS", "FAIL", "CONFLICT", "UNSUPPORTED"),
        "Execution failure or incompleteness is final; never fall back to a model.",
    ),
    VerifierCapability(
        "deterministic.stack_trace",
        ("stack_trace",),
        "executable",
        ("PASS", "FAIL", "CONFLICT", "UNSUPPORTED"),
        "The earliest-error result includes complete-prefix validity.",
    ),
    VerifierCapability(
        "deterministic.grid_path",
        ("grid_path",),
        "executable",
        ("PASS", "FAIL", "CONFLICT", "UNSUPPORTED"),
        "The final coordinate is computed from every ordered move.",
    ),
    VerifierCapability(
        "deterministic.custom_sort_order",
        ("custom_sort_order",),
        "executable",
        ("PASS", "FAIL", "CONFLICT", "UNSUPPORTED"),
        "The source-defined alphabet and the complete output sequence are checked.",
    ),
)

_MODEL_CAPABILITY = VerifierCapability(
    "model.bounded_semantic_panel",
    (
        "multiple_choice_truth",
        "proof_state",
        "object_belief_state",
        "team_utility_argmax",
        "murder_means",
        "murder_motive",
        "murder_opportunity",
        "boolean_expression_truth",
        "causal_proposition",
        "scientific_proposition",
        "permutation_trace",
    ),
    "bounded_semantic",
    ("PASS", "FAIL", "UNKNOWN"),
    "Two blinded panels must agree on one finite outcome with valid provenance.",
)

VERIFIER_CAPABILITIES: tuple[VerifierCapability, ...] = (*_DETERMINISTIC_CAPABILITIES, _MODEL_CAPABILITY)


def build_task_semantics(sample: DatasetSample, source_graph: SourceSpanGraph) -> TaskSemantics:
    """Return official-task semantics without consulting a reference answer."""

    del source_graph  # The graph is intentionally not interpreted as an answer source.
    task = str(sample.metadata.get("task") or sample.metadata.get("domain") or "").strip().casefold()
    domain = str(sample.metadata.get("high_level_domain") or "").strip().casefold()
    source = question_without_answer_contract(sample).casefold()

    if sample.dataset == "seqbench":
        return _semantics(
            "state_transition",
            "exact_sequence",
            "sequence",
            "ordered_navigation_with_keys_doors_and_rescue",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "seq_plan",
                    "Every action is syntactically valid and executable.",
                ),
                (
                    "final_state",
                    "final_state",
                    "seq_plan",
                    "The complete ordered execution reaches the queried rescue state.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "seq_plan",
                    "The plan has no omitted or extra terminal action.",
                ),
            ),
            payload_source="local",
        )
    if task == "dyck_languages":
        return _semantics(
            "state_transition",
            "earliest",
            "scalar",
            "typed_stack_trace",
            (
                ("candidate_validity", "candidate_validity", "stack_trace", "The candidate step is erroneous."),
                ("prefix_validity", "prefix_validity", "stack_trace", "Every earlier stack transition is valid."),
            ),
            payload_source="local",
        )
    if task == "spatial_reasoning":
        return _semantics(
            "state_transition",
            "final_state",
            "entity",
            "ordered_grid_or_circular_path",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "grid_path",
                    "The candidate names the entity at the queried coordinate.",
                ),
                ("final_state", "final_state", "grid_path", "Every ordered move contributes to the final coordinate."),
            ),
            payload_source="local",
        )
    if task == "word_sorting":
        if "identify the first step" in source or "mistake in thought" in source:
            return _semantics(
                "state_transition",
                "earliest",
                "scalar",
                "source_defined_sort_trace",
                (
                    (
                        "candidate_validity",
                        "candidate_validity",
                        "sort_trace_earliest",
                        "The candidate step contains a sorting error.",
                    ),
                    (
                        "prefix_validity",
                        "prefix_validity",
                        "sort_trace_earliest",
                        "Every earlier sorting thought is valid and complete.",
                    ),
                ),
                guarantee="diagnostic_only",
                payload_source="local",
            )
        return _semantics(
            "set_count",
            "exact_sequence",
            "sequence",
            "source_defined_alphabet",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "custom_sort_order",
                    "The candidate follows the source-defined alphabet.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "custom_sort_order",
                    "Every source word occurs exactly once.",
                ),
            ),
            payload_source="local",
        )
    if task == "shuffled_objects":
        return _semantics(
            "state_transition",
            "final_state",
            "entity",
            "ordered_permutation",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "permutation_trace",
                    "The candidate matches the queried value after every source-declared swap and repeat.",
                ),
                (
                    "final_state",
                    "final_state",
                    "permutation_trace",
                    "Every source-declared swap, named action repeat, and no-op is interpreted in order.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if task == "multistep_arithmetic":
        return _semantics(
            "equation",
            "point_value",
            "scalar",
            "typed_arithmetic",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "arithmetic_dsl",
                    "The declared equation evaluates to the candidate value.",
                ),
                (
                    "unit_consistency",
                    "unit_consistency",
                    "arithmetic_dsl",
                    "Units and numeric residual are consistent.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="model_slots",
        )
    if task == "object_placements":
        return _semantics(
            "state_transition",
            "final_state",
            "choice",
            "observer_belief_not_objective_world_state",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "object_belief_state",
                    "The option matches the queried observer's belief.",
                ),
                (
                    "final_state",
                    "final_state",
                    "object_belief_state",
                    "Use only moves observed by that person, preserving the last observed location.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if task == "team_allocation":
        return _semantics(
            "set_count",
            "argmax",
            "choice",
            "skill_plus_teamwork_utility",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "team_utility_argmax",
                    "The option's stated utility is supported.",
                ),
                (
                    "comparative_dominance",
                    "comparative_dominance",
                    "team_utility_argmax",
                    "The option dominates every alternative under the task utility.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if task == "murder_mysteries":
        return _semantics(
            "semantic",
            "multiple_choice_truth",
            "choice",
            "culprit_requires_means_motive_and_opportunity",
            (
                (
                    "means",
                    "constraint_satisfaction",
                    "murder_means",
                    "The candidate has the means required by the story.",
                ),
                ("motive", "constraint_satisfaction", "murder_motive", "The candidate has a source-supported motive."),
                (
                    "opportunity",
                    "constraint_satisfaction",
                    "murder_opportunity",
                    "The candidate has the opportunity required by the story.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if task == "movie_recommendation":
        return _semantics(
            "set_count",
            "argmax",
            "choice",
            "group_preference_similarity_utility",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "movie_similarity_argmax",
                    "The option's within-set similarity score is supported.",
                ),
                (
                    "comparative_dominance",
                    "comparative_dominance",
                    "movie_similarity_argmax",
                    "The option dominates every alternative under the stated similarity objective.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "object_counting":
        return _semantics(
            "set_count",
            "point_value",
            "scalar",
            "typed_category_filter_and_count",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "set_count_query",
                    "The requested category arithmetic yields the candidate count.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "set_count_query",
                    "Every source item in the requested categories is included exactly once.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "object_properties":
        return _semantics(
            "state_transition",
            "final_state",
            "scalar",
            "ordered_collection_updates_and_property_query",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "object_property_state",
                    "The final queried property count equals the candidate.",
                ),
                (
                    "final_state",
                    "final_state",
                    "object_property_state",
                    "Every collection update is applied in source order.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "object_property_state",
                    "The final disjunctive property filter has neither omissions nor double counting.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "buggy_tables":
        return _semantics(
            "equation",
            "point_value",
            "scalar",
            "table_repair_filter_and_aggregation",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "table_aggregation",
                    "The repaired and filtered table aggregation equals the candidate.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "table_aggregation",
                    "Every eligible row and null-handling rule is accounted for.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "temporal_sequence":
        return _semantics(
            "state_transition",
            "argmax",
            "sequence",
            "schedule_intersection_and_longest_meeting",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "schedule_argmax",
                    "The candidate duration and option count are feasible.",
                ),
                (
                    "comparative_dominance",
                    "comparative_dominance",
                    "schedule_argmax",
                    "No longer feasible meeting exists under all scheduling constraints.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "schedule_argmax",
                    "All starts attaining the maximum duration are counted exactly once.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "time_arithmetic":
        return _semantics(
            "state_transition",
            "exact_set",
            "set",
            "date_arithmetic_and_ordered_selection",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "temporal_composition",
                    "The composed cutoff and selected entities match the candidate.",
                ),
                (
                    "final_state",
                    "final_state",
                    "temporal_composition",
                    "All intermediate dates and offsets are applied in the declared order.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "temporal_composition",
                    "The ordered answer includes every and only entity satisfying the cutoff.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "sportqa":
        return _semantics(
            "semantic",
            "exact_sequence",
            "sequence",
            "rule_application_across_main_and_subquestions",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "rule_based_exact_set",
                    "Every selected choice is licensed by the stated sport rule.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "rule_based_exact_set",
                    "Every correct choice for every subquestion is included in the required order.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "sarc_triples":
        return _semantics(
            "semantic",
            "exact_sequence",
            "sequence",
            "three_independent_sarcasm_labels",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "semantic_sequence",
                    "Each candidate label matches its corresponding post-reply pair.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "semantic_sequence",
                    "Exactly three labels are present in source order.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task == "linguini":
        return _semantics(
            "semantic",
            "exact_sequence",
            "sequence",
            "morphological_rule_completion",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "morphology_sequence",
                    "Each filled form follows the induced morphology rule.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "morphology_sequence",
                    "Every blank is filled exactly once in source order.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if task in {"zebra_puzzles"}:
        return _semantics(
            "set_count",
            "exact_set",
            "choice",
            "explicit_constraint_witness",
            (
                (
                    "constraint_satisfaction",
                    "constraint_satisfaction",
                    "constraint_witness",
                    "The supplied assignment satisfies every typed constraint.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="model_slots",
        )
    if task == "boardgame_qa":
        return _semantics(
            "proof_state",
            "three_way_entailment",
            "proposition",
            "entailed_contradicted_unknown",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "proof_state",
                    "The source entails the candidate proof state.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if task == "boolean_expressions":
        return _semantics(
            "proof_state",
            "multiple_choice_truth",
            "choice",
            "boolean_expression_evaluation",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "boolean_expression_truth",
                    "The anonymous option expression evaluates to true.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if task == "causal_understanding":
        return _semantics(
            "semantic",
            "multiple_choice_truth",
            "proposition",
            "counterfactual_causal_judgment",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "causal_proposition",
                    "The candidate causal judgment follows from the complete scenario.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if task == "web_of_lies":
        return _semantics(
            "proof_state",
            "exact_sequence",
            "sequence",
            "global_truth_liar_assignment",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "truth_assignment_sequence",
                    "Every reported truth value is consistent with the global assignment.",
                ),
                (
                    "global_completeness",
                    "global_completeness",
                    "truth_assignment_sequence",
                    "All queried people appear exactly once in source order.",
                ),
            ),
            guarantee="diagnostic_only",
            payload_source="none",
        )
    if sample.dataset == "gpqa_diamond":
        state_model = f"scientific_option_truth:{domain or 'unknown_domain'}"
        return _semantics(
            "equation" if domain == "physics" else "semantic",
            "multiple_choice_truth",
            "choice",
            state_model,
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "scientific_proposition",
                    "The anonymous option proposition answers the complete scientific question.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    if _has_choice_contract(sample):
        return _semantics(
            "semantic",
            "multiple_choice_truth",
            "choice",
            "bounded_option_proposition",
            (
                (
                    "candidate_validity",
                    "candidate_validity",
                    "multiple_choice_truth",
                    "The anonymous option proposition answers the complete source query.",
                ),
            ),
            guarantee="bounded_semantic",
            payload_source="none",
        )
    return _semantics(
        "semantic",
        "point_value",
        "proposition",
        "unsupported_open_ended_semantics",
        (
            (
                "candidate_validity",
                "candidate_validity",
                "semantic_proposition",
                "The candidate proposition answers the complete source query.",
            ),
        ),
        guarantee="diagnostic_only",
        payload_source="none",
    )


def task_contract_from_semantics(semantics: TaskSemantics, source_graph: SourceSpanGraph) -> TaskContractV2:
    span_ids = tuple(span.span_id for span in source_graph.spans)
    obligations = tuple(
        QuestionObligation(
            obligation_id=template.obligation_id,
            kind=template.kind,  # type: ignore[arg-type]
            scope=f"{semantics.query_operator}:{template.description}",
            required=template.required,
            source_span_ids=span_ids,
        )
        for template in semantics.mandatory_obligation_templates
    )
    primary = semantics.mandatory_obligation_templates[0].operation_kind
    return TaskContractV2(
        family=semantics.task_family,  # type: ignore[arg-type]
        query_operator=semantics.query_operator,  # type: ignore[arg-type]
        adapter_kind=primary,
        answer_schema=semantics.answer_type,
        mandatory_obligations=obligations,
        tolerance_policy={"absolute": 1e-9, "relative": 1e-6},
        adapter_version=semantics.semantics_version,
    )


def semantics_requires_designer(semantics: TaskSemantics) -> bool:
    return any(
        template.payload_source == "model_slots" and template.guarantee_level != "diagnostic_only"
        for template in semantics.mandatory_obligation_templates
    )


def compile_local_certificate_bank(
    *,
    sample: DatasetSample,
    semantics: TaskSemantics,
    stage: StageDecision,
    public_to_key: dict[str, str],
    answer_nodes: dict[str, CandidateAnswerNode],
    source_graph: SourceSpanGraph,
    pairs: tuple[CandidatePairV2, ...],
) -> CertificateBankValidationV2:
    """Compile all pair commitments and outcome meanings locally.

    Model-filled payloads remain empty placeholders until validated by
    :func:`validate_kernel_certificate_bank`; the model never authors tests,
    certificates, candidate commitments, or finite outcome meanings.
    """

    public_by_key = {key: public for public, key in public_to_key.items()}
    source_ids = tuple(span.span_id for span in source_graph.spans)
    tests: list[CertificateTestV2] = []
    certificates: list[AnswerCertificateV2] = []
    errors: list[str] = []
    for pair_index, pair in enumerate(pairs):
        required_test_ids: list[str] = []
        refutation_test_ids: list[str] = []
        challenger_public = public_by_key[pair.challenger_key]
        anchor_public = public_by_key[stage.anchor_key]
        for obligation_index, template in enumerate(semantics.mandatory_obligation_templates):
            if template.guarantee_level == "bounded_semantic":
                for proposition_role, proposition_candidate in (
                    ("support", challenger_public),
                    ("refute_anchor", anchor_public),
                ):
                    test_id = f"KT{pair_index}_{obligation_index}_{proposition_role}"
                    proposition = answer_nodes[proposition_candidate].rendered_content
                    expected = (
                        {challenger_public: "TRUE", anchor_public: "FALSE"}
                        if proposition_role == "support"
                        else {anchor_public: "TRUE", challenger_public: "FALSE"}
                    )
                    tests.append(
                        CertificateTestV2(
                            test_id=test_id,
                            pair_id=pair.pair_id,
                            obligation_ids=(template.obligation_id,),
                            operation_kind=template.operation_kind,
                            question_or_operation=(
                                f"{template.description} Evaluate only this anonymous proposition: {proposition}"
                            ),
                            finite_outcomes=(
                                CertificateOutcomeV2("TRUE", "The proposition is true."),
                                CertificateOutcomeV2("FALSE", "The proposition is false."),
                                CertificateOutcomeV2("UNKNOWN", "The indexed source is insufficient."),
                            ),
                            expected_outcome_by_candidate=expected,
                            source_span_ids=source_ids,
                            deterministic_payload={},
                        )
                    )
                    required_test_ids.append(test_id)
                    if proposition_role == "refute_anchor":
                        refutation_test_ids.append(test_id)
                continue
            test_id = f"KT{pair_index}_{obligation_index}"
            payload: dict[str, Any] = {}
            if template.payload_source == "local":
                compiled, error = compile_local_typed_payload(sample, template.operation_kind)
                if compiled is None:
                    errors.append(f"{test_id}:{error or 'local_payload_compile_failed'}")
                else:
                    payload = compiled
                    valid, reason = validate_typed_payload(
                        template.operation_kind,
                        payload,
                        candidate_ids={pair.left_candidate, pair.right_candidate},
                        payload_source=template.payload_source,
                    )
                    if not valid:
                        errors.append(f"{test_id}:{reason}")
            left = answer_nodes[pair.left_candidate].rendered_content
            right = answer_nodes[pair.right_candidate].rendered_content
            tests.append(
                CertificateTestV2(
                    test_id=test_id,
                    pair_id=pair.pair_id,
                    obligation_ids=(template.obligation_id,),
                    operation_kind=template.operation_kind,
                    question_or_operation=template.description,
                    finite_outcomes=(
                        CertificateOutcomeV2("O0", f"Proposition 0: {left}"),
                        CertificateOutcomeV2("O1", f"Proposition 1: {right}"),
                    ),
                    expected_outcome_by_candidate={pair.left_candidate: "O0", pair.right_candidate: "O1"},
                    source_span_ids=source_ids,
                    deterministic_payload=payload,
                )
            )
            required_test_ids.append(test_id)
            refutation_test_ids.append(test_id)
        canonical = {
            "candidate_key_anon": challenger_public,
            "answer_hash": answer_nodes[challenger_public].answer_hash,
            "required_test_ids": tuple(required_test_ids),
            "derived_refutation_test_ids": tuple(refutation_test_ids),
        }
        certificates.append(AnswerCertificateV2(**canonical, certificate_hash=_json_sha256(canonical)))
    protocol_error = f"kernel_local_payload_invalid:{'|'.join(errors)}" if errors else None
    return CertificateBankValidationV2(
        certificates=tuple(certificates),
        tests=tuple(tests),
        dropped=(),
        protocol_error=protocol_error if tests else "no_certificate_tests",
        adapter_conflicts=(),
        eligible_challengers=tuple(pair.challenger_key for pair in pairs),
        obligation_coverage=1.0 if tests else 0.0,
        answer_link_coverage=1.0 if tests else 0.0,
    )


def validate_kernel_certificate_bank(
    payload: dict[str, Any] | None,
    *,
    semantics: TaskSemantics,
    skeleton: CertificateBankValidationV2,
    max_tests: int = 6,
) -> CertificateBankValidationV2:
    """Merge strictly typed model payload slots into a compiler-owned skeleton."""

    del max_tests
    if skeleton.protocol_error is not None:
        return skeleton
    if not isinstance(payload, dict) or set(payload) != {"payloads"} or not isinstance(payload["payloads"], list):
        return replace(skeleton, protocol_error="kernel_payload_top_level_schema_failure")
    templates = {item.obligation_id: item for item in semantics.mandatory_obligation_templates}
    test_by_id = {test.test_id: test for test in skeleton.tests}
    expected_ids = {
        test.test_id for test in skeleton.tests if templates[test.obligation_ids[0]].payload_source == "model_slots"
    }
    rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for raw in payload["payloads"]:
        if not isinstance(raw, dict) or set(raw) != {"test_id", "typed_payload"}:
            errors.append("payload_row_schema_invalid")
            continue
        test_id = str(raw.get("test_id") or "")
        if test_id not in expected_ids or test_id in rows:
            errors.append(f"payload_test_id_invalid:{test_id}")
            continue
        test = test_by_id[test_id]
        template = templates[test.obligation_ids[0]]
        valid, reason = validate_typed_payload(
            test.operation_kind,
            raw["typed_payload"],
            candidate_ids=set(test.expected_outcome_by_candidate),
            payload_source=template.payload_source,
        )
        if not valid:
            errors.append(f"{test_id}:{reason}")
            continue
        rows[test_id] = dict(raw["typed_payload"])
    missing = expected_ids - set(rows)
    if missing:
        errors.append(f"payload_tests_missing:{','.join(sorted(missing))}")
    if errors:
        return replace(skeleton, protocol_error=f"kernel_typed_payload_invalid:{'|'.join(errors)}")
    merged = tuple(
        replace(test, deterministic_payload=rows.get(test.test_id, test.deterministic_payload))
        for test in skeleton.tests
    )
    return replace(skeleton, tests=merged)


def compile_typed_obligations(
    semantics: TaskSemantics,
    validation: CertificateBankValidationV2,
) -> tuple[TypedObligation, ...]:
    templates = {item.obligation_id: item for item in semantics.mandatory_obligation_templates}
    obligations: list[TypedObligation] = []
    for test in validation.tests:
        for obligation_id in test.obligation_ids:
            template = templates.get(obligation_id)
            if template is None or test.operation_kind != template.operation_kind:
                continue
            obligations.append(
                TypedObligation(
                    obligation_id=obligation_id,
                    test_id=test.test_id,
                    pair_id=test.pair_id,
                    operation_kind=template.operation_kind,
                    typed_payload=dict(test.deterministic_payload),
                    candidate_commitments=dict(test.expected_outcome_by_candidate),
                    source_span_ids=test.source_span_ids,
                    temporal_scope=semantics.state_model,
                    completeness_scope=semantics.query_operator,
                    required=template.required,
                )
            )
    return tuple(obligations)


def compile_answer_obligation_graph(
    semantics: TaskSemantics,
    obligations: tuple[TypedObligation, ...],
) -> AnswerObligationGraph:
    """Materialize answer-critical dependencies instead of retaining a flat ID set."""

    template_order = {item.obligation_id: index for index, item in enumerate(semantics.mandatory_obligation_templates)}
    by_pair: dict[str, list[TypedObligation]] = {}
    for obligation in obligations:
        by_pair.setdefault(obligation.pair_id, []).append(obligation)
    edges: list[tuple[str, str, str]] = []
    for pair_obligations in by_pair.values():
        ordered = sorted(pair_obligations, key=lambda item: template_order.get(item.obligation_id, 10_000))
        for left, right in zip(ordered, ordered[1:], strict=False):
            edges.append((left.test_id, right.test_id, "required_before"))
    canonical = {
        "semantics_version": semantics.semantics_version,
        "nodes": [asdict(item) for item in obligations],
        "dependency_edges": edges,
        "required_obligation_ids": [
            item.obligation_id for item in semantics.mandatory_obligation_templates if item.required
        ],
    }
    return AnswerObligationGraph(
        semantics.semantics_version,
        obligations,
        tuple(edges),
        tuple(canonical["required_obligation_ids"]),
        _json_sha256(canonical),
    )


def bind_verifier_capabilities(
    semantics: TaskSemantics,
    validation: CertificateBankValidationV2,
) -> dict[str, VerifierBinding]:
    templates = {item.obligation_id: item for item in semantics.mandatory_obligation_templates}
    bindings: dict[str, VerifierBinding] = {}
    for test in validation.tests:
        selected = [templates.get(item) for item in test.obligation_ids]
        selected = [item for item in selected if item is not None]
        if not selected or any(item.operation_kind != test.operation_kind for item in selected):
            bindings[test.test_id] = VerifierBinding(
                test.test_id, test.operation_kind, "diagnostic.none", "diagnostic_only", "UNSUPPORTED"
            )
            continue
        guarantee = selected[0].guarantee_level
        if any(item.guarantee_level != guarantee for item in selected):
            bindings[test.test_id] = VerifierBinding(
                test.test_id, test.operation_kind, "diagnostic.none", "diagnostic_only", "UNSUPPORTED"
            )
            continue
        capability = next(
            (
                item
                for item in VERIFIER_CAPABILITIES
                if item.guarantee_level == guarantee and test.operation_kind in item.supported_operation_kinds
            ),
            None,
        )
        bindings[test.test_id] = VerifierBinding(
            test.test_id,
            test.operation_kind,
            capability.verifier_kind if capability else "diagnostic.none",
            guarantee if capability else "diagnostic_only",
            "BOUND" if capability else "UNSUPPORTED",
        )
    return bindings


def build_proof_results(
    *,
    stage: StageDecision,
    semantics: TaskSemantics,
    validation: CertificateBankValidationV2,
    public_to_key: dict[str, str],
    bindings: dict[str, VerifierBinding],
    adapter_results: dict[str, AdapterResult],
    panels: tuple[CertificateVerifierParseResultV2, ...],
) -> tuple[ProofResult, ...]:
    test_by_id = {test.test_id: test for test in validation.tests}
    public_by_key = {key: public for public, key in public_to_key.items()}
    template_ids = {item.obligation_id for item in semantics.mandatory_obligation_templates if item.required}
    results: list[ProofResult] = []
    for challenger in validation.eligible_challengers:
        challenger_public = public_by_key[challenger]
        anchor_public = public_by_key[stage.anchor_key]
        certificate = next(item for item in validation.certificates if item.candidate_key_anon == challenger_public)
        for test_id in certificate.required_test_ids:
            test = test_by_id[test_id]
            binding = bindings.get(test_id)
            expected = test.expected_outcome_by_candidate.get(challenger_public)
            anchor_expected = test.expected_outcome_by_candidate.get(anchor_public)
            observed: str | None = None
            ruled_out: tuple[str, ...] = ()
            source_span_ids: tuple[str, ...] = ()
            provenance_valid = False
            entailment_valid = False
            trace_hash = ""
            detail = ""
            status: ProofStatus = "UNSUPPORTED"
            if binding is None or binding.binding_status != "BOUND":
                detail = "no_authorized_verifier"
            elif binding.guarantee_level == "executable":
                adapter = adapter_results.get(test_id)
                if adapter is None or adapter.execution_status == "UNSUPPORTED":
                    status = "UNSUPPORTED"
                    detail = adapter.residual_or_first_failure if adapter else "adapter_result_missing"
                elif adapter.execution_status in {"CONFLICT", "INVALID"}:
                    status = "CONFLICT"
                    detail = adapter.residual_or_first_failure
                else:
                    observed = adapter.observed_outcome
                    trace_hash = adapter.execution_trace_hash
                    source_span_ids = test.source_span_ids
                    provenance_valid = True
                    entailment_valid = True
                    status = "PASS" if observed == expected and expected != anchor_expected else "FAIL"
                    detail = adapter.residual_or_first_failure
            elif binding.guarantee_level == "bounded_semantic":
                panel_results = [panel.results.get(test_id) for panel in panels] if len(panels) == 2 else []
                if len(panel_results) != 2 or any(result is None for result in panel_results):
                    status = "UNKNOWN"
                    detail = "two_complete_panels_required"
                elif any(result.support_status != "ENTAILED" for result in panel_results if result is not None):
                    status = "UNKNOWN"
                    detail = "panel_not_entailed"
                elif panel_results[0].observed_outcome != panel_results[1].observed_outcome:  # type: ignore[union-attr]
                    status = "UNKNOWN"
                    detail = "panel_disagreement"
                else:
                    observed = panel_results[0].observed_outcome  # type: ignore[union-attr]
                    entailment_valid = True
                    provenance_valid = all(
                        bool(result.source_span_ids) for result in panel_results if result is not None
                    )
                    source_span_ids = tuple(
                        dict.fromkeys(
                            span_id
                            for result in panel_results
                            if result is not None
                            for span_id in result.source_span_ids
                        )
                    )
                    anchor_refuted_by_both = bool(anchor_expected) and all(
                        anchor_expected in result.ruled_out_outcomes for result in panel_results if result is not None
                    )
                    ruled_out = tuple(
                        sorted(
                            {
                                value
                                for result in panel_results
                                if result is not None
                                for value in result.ruled_out_outcomes
                            }
                        )
                    )
                    status = (
                        "PASS"
                        if provenance_valid
                        and anchor_refuted_by_both
                        and observed == expected
                        and expected != anchor_expected
                        else "FAIL"
                    )
                    detail = (
                        "dual_panel_agreement_and_refutation"
                        if anchor_refuted_by_both
                        else "anchor_outcome_not_ruled_out_by_both_panels"
                    )
            covered = set(test.obligation_ids)
            results.append(
                ProofResult(
                    obligation_id="+".join(test.obligation_ids),
                    test_id=test_id,
                    candidate_key_anon=challenger_public,
                    status=status,
                    observed_value=observed,
                    ruled_out_values=ruled_out,
                    source_span_ids=source_span_ids,
                    provenance_valid=provenance_valid,
                    entailment_valid=entailment_valid,
                    obligation_valid=covered.issubset(template_ids),
                    sufficiency_valid=template_ids.issubset(
                        {
                            item
                            for candidate_test_id in certificate.required_test_ids
                            for item in test_by_id[candidate_test_id].obligation_ids
                        }
                    ),
                    execution_trace_hash=trace_hash,
                    verifier_kind=binding.verifier_kind if binding else "diagnostic.none",
                    detail=detail,
                )
            )
    return tuple(results)


def decide_with_proof_kernel(
    stage: StageDecision,
    *,
    semantics: TaskSemantics,
    validation: CertificateBankValidationV2,
    public_to_key: dict[str, str],
    obligations: tuple[TypedObligation, ...],
    proofs: tuple[ProofResult, ...],
) -> KernelDecision:
    required = tuple(item.obligation_id for item in semantics.mandatory_obligation_templates if item.required)
    anchor_public = next(public for public, key in public_to_key.items() if key == stage.anchor_key)
    if validation.protocol_error is not None:
        return KernelDecision(
            anchor_public, None, required, (), (), "ABSTAIN", "compiler", validation.protocol_error, ()
        )
    expected_challengers = {candidate.key for candidate in stage.candidates if candidate.key != stage.anchor_key}
    if set(validation.eligible_challengers) != expected_challengers:
        return KernelDecision(
            anchor_public,
            None,
            required,
            (),
            (),
            "ABSTAIN",
            "answer_link",
            "all_challenger_certificates_required",
            (),
        )
    test_by_id = {test.test_id: test for test in validation.tests}
    proof_by_candidate_test = {(item.candidate_key_anon, item.test_id): item for item in proofs}
    passing: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    diagnostics: list[dict[str, Any]] = []
    failure_statuses: list[ProofStatus] = []
    for challenger in validation.eligible_challengers:
        challenger_public = next(public for public, key in public_to_key.items() if key == challenger)
        certificate = next(item for item in validation.certificates if item.candidate_key_anon == challenger_public)
        covered = tuple(
            dict.fromkeys(
                obligation
                for test_id in certificate.required_test_ids
                for obligation in test_by_id[test_id].obligation_ids
            )
        )
        candidate_proofs = [
            proof_by_candidate_test.get((challenger_public, test_id)) for test_id in certificate.required_test_ids
        ]
        complete = set(required).issubset(covered)
        linked = all(item is not None for item in candidate_proofs)
        passed = (
            complete
            and linked
            and all(
                item.status == "PASS"
                and item.provenance_valid
                and item.entailment_valid
                and item.obligation_valid
                and item.sufficiency_valid
                for item in candidate_proofs
                if item is not None
            )
        )
        refutes_anchor = all(
            test_by_id[test_id].expected_outcome_by_candidate.get(challenger_public)
            != test_by_id[test_id].expected_outcome_by_candidate.get(anchor_public)
            for test_id in certificate.derived_refutation_test_ids
        )
        passed = passed and bool(certificate.derived_refutation_test_ids) and refutes_anchor
        failure_statuses.extend(item.status for item in candidate_proofs if item is not None and item.status != "PASS")
        diagnostics.append(
            {
                "challenger_key": challenger,
                "challenger_id": challenger_public,
                "covered_obligations": list(covered),
                "required_obligations": list(required),
                "proof_statuses": [item.status if item is not None else "MISSING" for item in candidate_proofs],
                "complete": complete,
                "refutes_anchor": refutes_anchor,
                "passed": passed,
            }
        )
        if passed:
            passing.append(
                (
                    challenger,
                    covered,
                    tuple(item.test_id for item in candidate_proofs if item is not None),
                )
            )
    if "CONFLICT" in failure_statuses:
        reason, layer = "adapter_conflict", "adapter"
    elif "UNSUPPORTED" in failure_statuses:
        reason, layer = "jurisdiction_unsupported", "jurisdiction"
    elif "UNKNOWN" in failure_statuses:
        reason, layer = "verifier_ambiguous", "verifier"
    elif len(passing) == 1:
        challenger, covered, accepted = passing[0]
        challenger_public = next(public for public, key in public_to_key.items() if key == challenger)
        return KernelDecision(
            anchor_public,
            challenger_public,
            required,
            covered,
            accepted,
            "OVERRIDE",
            "none",
            "unique_complete_proof",
            tuple(diagnostics),
        )
    elif len(passing) > 1:
        reason, layer = "multiple_complete_proofs", "proof_kernel"
    else:
        reason, layer = "proof_incomplete", "proof_kernel"
    covered = tuple(dict.fromkeys(item.obligation_id for item in obligations))
    return KernelDecision(anchor_public, None, required, covered, (), "ABSTAIN", layer, reason, tuple(diagnostics))


def decide_with_unary_proof_kernel(
    stage: StageDecision,
    *,
    semantics: TaskSemantics,
    validation: CertificateBankValidationV2,
    public_to_key: dict[str, str],
    candidate_results: dict[str, CandidateAdapterResult],
) -> KernelDecision:
    """D2 decision rule: override only from a unique exact candidate verdict.

    Model panels remain logged as diagnostics, but bounded-semantic evidence is
    deliberately incapable of changing the answer in this revision.
    """

    required = tuple(item.obligation_id for item in semantics.mandatory_obligation_templates if item.required)
    anchor_public = next(public for public, key in public_to_key.items() if key == stage.anchor_key)
    if validation.protocol_error is not None:
        return KernelDecision(
            anchor_public, None, required, (), (), "ABSTAIN", "compiler", validation.protocol_error, ()
        )
    if any(
        item.required and item.guarantee_level != "executable"
        for item in semantics.mandatory_obligation_templates
    ):
        return KernelDecision(
            anchor_public,
            None,
            required,
            (),
            (),
            "ABSTAIN",
            "jurisdiction",
            "bounded_semantic_diagnostic_only",
            (),
        )
    expected_challengers = {candidate.key for candidate in stage.candidates if candidate.key != stage.anchor_key}
    if set(validation.eligible_challengers) != expected_challengers:
        return KernelDecision(
            anchor_public,
            None,
            required,
            (),
            (),
            "ABSTAIN",
            "answer_link",
            "all_challenger_certificates_required",
            (),
        )
    expected_public = set(public_to_key)
    if set(candidate_results) != expected_public:
        return KernelDecision(
            anchor_public, None, required, (), (), "ABSTAIN", "adapter", "candidate_verdicts_incomplete", ()
        )
    diagnostics = tuple(
        {
            "candidate_id": public,
            "candidate_key": public_to_key[public],
            "is_anchor": public == anchor_public,
            "status": result.status,
            "detail": result.detail,
            "execution_trace_hash": result.execution_trace_hash,
        }
        for public, result in sorted(candidate_results.items())
    )
    if any(item.status == "UNSUPPORTED" for item in candidate_results.values()):
        return KernelDecision(
            anchor_public, None, required, (), (), "ABSTAIN", "jurisdiction", "jurisdiction_unsupported", diagnostics
        )
    anchor_result = candidate_results[anchor_public]
    valid_challengers = [
        public
        for public, result in candidate_results.items()
        if public != anchor_public and result.status == "VALID"
    ]
    if anchor_result.status == "VALID":
        reason, layer = "anchor_verified_valid", "proof_kernel"
    elif anchor_result.status != "INVALID":
        reason, layer = "anchor_not_refuted", "proof_kernel"
    elif len(valid_challengers) > 1:
        reason, layer = "multiple_valid_candidates", "proof_kernel"
    elif not valid_challengers:
        reason, layer = "no_valid_challenger", "proof_kernel"
    else:
        winner = valid_challengers[0]
        other_challengers = [
            result
            for public, result in candidate_results.items()
            if public not in {anchor_public, winner}
        ]
        if all(result.status == "INVALID" for result in other_challengers):
            return KernelDecision(
                anchor_public,
                winner,
                required,
                required,
                (f"unary:{winner}",),
                "OVERRIDE",
                "none",
                "unique_exact_candidate",
                diagnostics,
            )
        reason, layer = "challenger_verdicts_incomplete", "adapter"
    return KernelDecision(anchor_public, None, required, required, (), "ABSTAIN", layer, reason, diagnostics)


def kernel_decision_to_decode(
    stage: StageDecision,
    decision: KernelDecision,
    *,
    public_to_key: dict[str, str],
) -> DecodeDecision:
    if decision.decision != "OVERRIDE" or decision.challenger_id is None:
        return DecodeDecision(
            stage.anchor_answer,
            stage.anchor_key,
            False,
            decision.reason_code,
            (),
            decision.diagnostics,
        )
    winner_key = public_to_key[decision.challenger_id]
    winner = next(candidate for candidate in stage.candidates if candidate.key == winner_key)
    return DecodeDecision(
        winner.answer,
        winner.key,
        True,
        "kernel_verified_override",
        (winner.key,),
        decision.diagnostics,
    )


def task_semantics_to_dict(value: TaskSemantics) -> dict[str, Any]:
    return asdict(value)


def typed_obligation_to_dict(value: TypedObligation) -> dict[str, Any]:
    return asdict(value)


def answer_obligation_graph_to_dict(value: AnswerObligationGraph) -> dict[str, Any]:
    return asdict(value)


def verifier_binding_to_dict(value: VerifierBinding) -> dict[str, Any]:
    return asdict(value)


def proof_result_to_dict(value: ProofResult) -> dict[str, Any]:
    return asdict(value)


def kernel_decision_to_dict(value: KernelDecision) -> dict[str, Any]:
    return asdict(value)


def _semantics(
    task_family: str,
    query_operator: str,
    answer_type: str,
    state_model: str,
    rows: tuple[tuple[str, str, str, str], ...],
    *,
    guarantee: GuaranteeLevel = "executable",
    payload_source: PayloadSource,
) -> TaskSemantics:
    templates = tuple(
        ObligationTemplate(
            obligation_id=f"Q{index}_{name}",
            kind=kind,
            operation_kind=operation,
            description=description,
            guarantee_level=guarantee,
            payload_source=payload_source,
        )
        for index, (name, kind, operation, description) in enumerate(rows)
    )
    authorized = tuple(
        dict.fromkeys(
            capability.verifier_kind
            for template in templates
            for capability in VERIFIER_CAPABILITIES
            if capability.guarantee_level == template.guarantee_level
            and template.operation_kind in capability.supported_operation_kinds
        )
    )
    return TaskSemantics(
        task_family,
        query_operator,
        answer_type,
        state_model,
        templates,
        authorized,
        "abstain_without_cross_jurisdiction_fallback",
    )


def _has_choice_contract(sample: DatasetSample) -> bool:
    contract = sample.metadata.get("answer_contract")
    return isinstance(contract, dict) and str(contract.get("kind") or "") in {"single_choice", "multi_choice"}


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
