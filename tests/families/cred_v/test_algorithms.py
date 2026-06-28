from __future__ import annotations

from types import SimpleNamespace

from research_experiments.families.cred_v.algorithms import (
    aggregate_adaptive_candidate_search,
    aggregate_pairwise_selection,
    aggregate_reasoning_first_selection,
    aggregate_safe_verification,
    aggregate_safe_select_v3,
    aggregate_shadow_select_v4,
    aggregate_stage_a_vote,
    aggregate_task_verification,
    build_router_decision,
    choice_permutation,
    map_shuffled_choice_answer,
    select_verification_targets,
)


def test_task_verifier_promotes_supported_challenger() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.62),
        _row("gpqa_diamond", "A", confidence=0.61),
        _row("gpqa_diamond", "A", confidence=0.60),
        _row("gpqa_diamond", "B", confidence=0.74),
        _row("gpqa_diamond", "B", confidence=0.73),
    ]
    verifier_rows = [
        _row(
            "gpqa_diamond",
            "B",
            confidence=0.84,
            payload={
                "promote": True,
                "leader_score": 0.25,
                "challenger_score": 0.88,
                "key_evidence": "option B matches the decisive concept while option A misses the stated constraint",
            },
        )
    ]

    decision = aggregate_task_verification(
        dataset="gpqa_diamond",
        stage_rows=stage_rows,
        verifier_rows=verifier_rows,
        stage_winner="A",
        promotion_confidence_min=0.72,
        promotion_score_margin=0.15,
        concrete_evidence_min_chars=12,
    )

    assert decision.final_answer == "B"
    assert decision.changed is True
    assert decision.resolver == "cred_v_task_verify_promoted"


def test_task_verifier_rejects_weak_certificate() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A"),
        _row("gpqa_diamond", "A"),
        _row("gpqa_diamond", "A"),
        _row("gpqa_diamond", "B"),
        _row("gpqa_diamond", "B"),
    ]
    verifier_rows = [
        _row(
            "gpqa_diamond",
            "B",
            confidence=0.90,
            payload={
                "promote": True,
                "leader_score": 0.70,
                "challenger_score": 0.78,
                "key_evidence": "thin",
            },
        )
    ]

    decision = aggregate_task_verification(
        dataset="gpqa_diamond",
        stage_rows=stage_rows,
        verifier_rows=verifier_rows,
        stage_winner="A",
        promotion_confidence_min=0.72,
        promotion_score_margin=0.15,
        concrete_evidence_min_chars=12,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_v_task_verify_rejected"


def test_select_verification_targets_excludes_current_leader() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.90),
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "B", confidence=0.80),
        _row("gpqa_diamond", "C", confidence=0.65),
    ]

    targets = select_verification_targets(
        dataset="gpqa_diamond",
        rows=stage_rows,
        leading_answer=aggregate_stage_a_vote(stage_rows).final_answer,
        max_verifications=1,
    )

    assert [row["normalized_answer"] for row in targets] == ["B"]


def test_safe_verification_blocks_same_model_promotion() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.80),
        _row("gpqa_diamond", "A", confidence=0.75),
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "B", confidence=0.90),
        _row("gpqa_diamond", "B", confidence=0.85),
    ]
    same_model_verifier = [
        _row(
            "gpqa_diamond",
            "B",
            confidence=0.95,
            payload={
                "promote": True,
                "leader_pass": False,
                "challenger_pass": True,
                "key_evidence": "independent verifier claims option B passes and option A fails",
            },
        )
    ]

    decision = aggregate_safe_verification(
        dataset="gpqa_diamond",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        verifier_rows=same_model_verifier,
        hetero_verifier_rows=[],
        stage_winner="A",
        verification_modes=("hetero_verified",),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_verify_safe_rejected"


def test_safe_math_equivalence_repair_promotes_canonical_interval() -> None:
    stage_rows = [
        _row("math500", "(2,infinity)", confidence=0.70),
        _row("math500", "(2,infinity)", confidence=0.69),
        _row("math500", "(2,infinity)", confidence=0.68),
        _row("math500", "(2,\\infty)", confidence=0.90),
        _row("math500", "(2,\\infty)", confidence=0.89),
    ]

    decision = aggregate_safe_verification(
        dataset="math500",
        question="Solve the interval.",
        context="",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="(2,infinity)",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert decision.final_answer == "(2,\\infty)"
    assert decision.changed is True
    assert decision.resolver == "cred_verify_safe_deterministic_repair"


def test_safe_hotpot_span_repair_requires_context_supported_complete_span() -> None:
    stage_rows = [
        _row("hotpotqa", "John Underhill", confidence=0.72),
        _row("hotpotqa", "John Underhill", confidence=0.71),
        _row("hotpotqa", "John Underhill", confidence=0.70),
        _row("hotpotqa", "Captain John Underhill", confidence=0.88),
    ]

    decision = aggregate_safe_verification(
        dataset="hotpotqa",
        question="Who led the expedition?",
        context="The expedition was led by Captain John Underhill before the colony changed command.",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="John Underhill",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert decision.final_answer == "Captain John Underhill"
    assert decision.changed is True

    unsupported = aggregate_safe_verification(
        dataset="hotpotqa",
        question="Who led the expedition?",
        context="The expedition was led by John Underhill before the colony changed command.",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="John Underhill",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert unsupported.final_answer == "John Underhill"
    assert unsupported.changed is False


def test_safe_mmlu_requires_hetero_agreement() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.75),
        _row("mmlu_pro", "A", confidence=0.74),
        _row("mmlu_pro", "A", confidence=0.73),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    hetero_verifier = [
        _row(
            "mmlu_pro",
            "B",
            confidence=0.92,
            payload={
                "promote": True,
                "leader_pass": False,
                "challenger_pass": True,
                "key_evidence": "heterogeneous verifier confirms option B and identifies the failed clue in option A",
            },
        )
    ]

    without_hetero = aggregate_safe_verification(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="A",
        verification_modes=("hetero_verified",),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )
    with_hetero = aggregate_safe_verification(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=hetero_verifier,
        stage_winner="A",
        verification_modes=("hetero_verified",),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert without_hetero.final_answer == "A"
    assert without_hetero.changed is False
    assert with_hetero.final_answer == "B"
    assert with_hetero.changed is True
    assert with_hetero.resolver == "cred_verify_safe_hetero_promoted"


def test_acs_blocks_single_pro_candidate_promotion() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    expansion_rows = [
        _row("mmlu_pro", "B", confidence=0.95, method_name="cred_acs_expansion", expansion_mode="mc_choice_shuffle")
    ]

    decision = aggregate_adaptive_candidate_search(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=expansion_rows,
        stage_winner="A",
        expansion_modes=("mc_choice_shuffle",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.0,
        strong_majority_count=4,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_acs_single_pro_blocked"


def test_acs_promotes_only_with_two_independent_expansion_votes() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    expansion_rows = [
        _row("mmlu_pro", "B", confidence=0.95, method_name="cred_acs_expansion", expansion_mode="mc_choice_shuffle"),
        _row("mmlu_pro", "B", confidence=0.94, method_name="cred_acs_expansion", expansion_mode="mc_choice_shuffle"),
    ]

    decision = aggregate_adaptive_candidate_search(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=expansion_rows,
        stage_winner="A",
        expansion_modes=("mc_choice_shuffle",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.0,
        strong_majority_count=4,
    )

    assert decision.final_answer == "B"
    assert decision.changed is True
    assert decision.resolver == "cred_acs_candidate_promoted"


def test_acs_strategyqa_requires_two_expansion_votes() -> None:
    stage_rows = [
        _row("strategyqa", "yes", confidence=0.70),
        _row("strategyqa", "yes", confidence=0.69),
        _row("strategyqa", "yes", confidence=0.68),
        _row("strategyqa", "no", confidence=0.90),
        _row("strategyqa", "no", confidence=0.89),
    ]
    one_no = [_row("strategyqa", "no", confidence=0.95, method_name="cred_acs_expansion", expansion_mode="strategyqa_dual_polarity")]
    two_no = [
        *one_no,
        _row("strategyqa", "no", confidence=0.94, method_name="cred_acs_expansion", expansion_mode="strategyqa_dual_polarity"),
    ]

    blocked = aggregate_adaptive_candidate_search(
        dataset="strategyqa",
        question="Is the statement true?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=one_no,
        stage_winner="yes",
        expansion_modes=("strategyqa_dual_polarity",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.0,
        strong_majority_count=4,
    )
    promoted = aggregate_adaptive_candidate_search(
        dataset="strategyqa",
        question="Is the statement true?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=two_no,
        stage_winner="yes",
        expansion_modes=("strategyqa_dual_polarity",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.0,
        strong_majority_count=4,
    )

    assert blocked.final_answer == "yes"
    assert blocked.changed is False
    assert promoted.final_answer == "no"
    assert promoted.changed is True


def test_math_repair_does_not_collapse_trig_function_spacing() -> None:
    stage_rows = [
        _row("math500", "cot x", confidence=0.72),
        _row("math500", "cot x", confidence=0.71),
        _row("math500", "cot x", confidence=0.70),
        _row("math500", "\\cot x", confidence=0.88),
    ]
    equivalent = aggregate_safe_verification(
        dataset="math500",
        question="Find the answer.",
        context="",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="cot x",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )
    unsafe = aggregate_safe_verification(
        dataset="math500",
        question="Find the answer.",
        context="",
        stage_rows=[*stage_rows[:3], _row("math500", "cotx", confidence=0.90)],
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="cot x",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert equivalent.final_answer == "cot x"
    assert equivalent.changed is False
    assert unsafe.final_answer == "cot x"
    assert unsafe.changed is False


def test_choice_shuffle_maps_back_to_original_letter() -> None:
    assert choice_permutation(4, 1) == [3, 2, 1, 0]
    assert map_shuffled_choice_answer("A", [3, 2, 1, 0]) == "D"
    assert map_shuffled_choice_answer("choice C", [3, 2, 1, 0]) == "B"


def test_router_clean_skip_and_false_consensus_probe() -> None:
    protocol = SimpleNamespace(strong_majority_count=4, false_consensus_probe=True)
    clean = build_router_decision(
        [
            _row("mmlu_pro", "A", confidence=0.80),
            _row("mmlu_pro", "A", confidence=0.79),
            _row("mmlu_pro", "A", confidence=0.78),
            _row("mmlu_pro", "A", confidence=0.77),
            _row("mmlu_pro", "B", confidence=0.55),
        ],
        protocol=protocol,
    )
    probed = build_router_decision(
        [
            _row("mmlu_pro", "A", confidence=0.50),
            _row("mmlu_pro", "A", confidence=0.50),
            _row("mmlu_pro", "A", confidence=0.50),
            _row("mmlu_pro", "A", confidence=0.50),
            _row(
                "mmlu_pro",
                "B",
                confidence=0.90,
                payload={"key_evidence": "option B because the decisive clue supports B over A"},
            ),
        ],
        protocol=protocol,
    )

    assert clean.triggered is False
    assert clean.trigger_bucket == "clean_skip"
    assert probed.triggered is True
    assert probed.trigger_bucket == "false_consensus_probe"


def test_rfs_v2_router_keeps_clean_anchor_skip_and_probes_high_value_minority() -> None:
    protocol = SimpleNamespace(
        strong_majority_count=4,
        leader_lock_count=4,
        trigger_buckets=("weak_split_select", "deterministic_repair_only", "minority_probe"),
        false_consensus_probe=True,
        selection_modes=("mc_blind_pairwise_duel",),
    )
    clean = build_router_decision(
        [
            _row("mmlu_pro", "A", confidence=0.80),
            _row("mmlu_pro", "A", confidence=0.79),
            _row("mmlu_pro", "A", confidence=0.78),
            _row("mmlu_pro", "A", confidence=0.77),
            _row("mmlu_pro", "B", confidence=0.76),
        ],
        protocol=protocol,
    )
    probed = build_router_decision(
        [
            _row("mmlu_pro", "A", confidence=0.80),
            _row("mmlu_pro", "A", confidence=0.79),
            _row("mmlu_pro", "A", confidence=0.78),
            _row("mmlu_pro", "A", confidence=0.77),
            _row(
                "mmlu_pro",
                "B",
                confidence=0.95,
                payload={"key_evidence": "option B has a decisive mechanistic clue that option A lacks"},
            ),
        ],
        protocol=protocol,
    )

    assert clean.triggered is False
    assert clean.trigger_bucket == "clean_anchor_skip"
    assert probed.triggered is True
    assert probed.trigger_bucket == "minority_probe"


def test_rfs_v3_router_uses_weak_split_select_without_minority_probe() -> None:
    protocol = SimpleNamespace(
        strong_majority_count=4,
        leader_lock_count=4,
        trigger_buckets=("weak_split_select", "deterministic_repair_only"),
        false_consensus_probe=False,
        selection_modes=("gpqa_unanimous_pairwise_duel",),
    )
    split = build_router_decision(
        [
            _row("gpqa_diamond", "A", confidence=0.80),
            _row("gpqa_diamond", "A", confidence=0.79),
            _row("gpqa_diamond", "A", confidence=0.78),
            _row("gpqa_diamond", "B", confidence=0.95),
            _row("gpqa_diamond", "B", confidence=0.94),
        ],
        protocol=protocol,
    )
    locked = build_router_decision(
        [
            _row("gpqa_diamond", "A", confidence=0.80),
            _row("gpqa_diamond", "A", confidence=0.79),
            _row("gpqa_diamond", "A", confidence=0.78),
            _row("gpqa_diamond", "A", confidence=0.77),
            _row(
                "gpqa_diamond",
                "B",
                confidence=0.99,
                payload={"key_evidence": "option B has a decisive mechanistic clue that option A lacks"},
            ),
        ],
        protocol=protocol,
    )

    assert split.triggered is True
    assert split.trigger_bucket == "weak_split_select"
    assert locked.triggered is False
    assert locked.trigger_bucket == "clean_anchor_skip"


def test_acs_ignores_failed_expansion_rows() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    expansion_rows = [
        _row(
            "mmlu_pro",
            "B",
            confidence=0.95,
            method_name="cred_acs_expansion",
            expansion_mode="mc_choice_shuffle",
            request_status="failed",
        ),
        _row("mmlu_pro", "B", confidence=0.94, method_name="cred_acs_expansion", expansion_mode="mc_choice_shuffle"),
    ]

    decision = aggregate_adaptive_candidate_search(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=expansion_rows,
        stage_winner="A",
        expansion_modes=("mc_choice_shuffle",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.0,
        strong_majority_count=4,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False


def test_rfs_strong_majority_lock_blocks_expansion_flip() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.80),
        _row("mmlu_pro", "A", confidence=0.79),
        _row("mmlu_pro", "A", confidence=0.78),
        _row("mmlu_pro", "A", confidence=0.77),
        _row("mmlu_pro", "B", confidence=0.95),
    ]
    expansion_rows = [
        _row("mmlu_pro", "B", confidence=0.95, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
        _row("mmlu_pro", "B", confidence=0.94, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
        _row("mmlu_pro", "B", confidence=0.93, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
    ]

    decision = aggregate_reasoning_first_selection(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=expansion_rows,
        stage_winner="A",
        expansion_modes=("mc_choice_shuffle",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.25,
        leader_lock_count=4,
        mc_shuffle_min_agreement=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_strong_majority_locked"


def test_rfs_blocks_challenger_without_stage_a_support() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.67),
        _row("mmlu_pro", "B", confidence=0.66),
    ]
    expansion_rows = [
        _row("mmlu_pro", "C", confidence=0.95, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
        _row("mmlu_pro", "C", confidence=0.94, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
        _row("mmlu_pro", "C", confidence=0.93, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
    ]

    decision = aggregate_reasoning_first_selection(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=expansion_rows,
        stage_winner="A",
        expansion_modes=("mc_choice_shuffle",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.25,
        leader_lock_count=4,
        mc_shuffle_min_agreement=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False


def test_rfs_promotes_mc_with_stage_support_and_shuffle_agreement() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    expansion_rows = [
        _row("mmlu_pro", "B", confidence=0.95, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
        _row("mmlu_pro", "B", confidence=0.94, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
        _row("mmlu_pro", "B", confidence=0.93, method_name="cred_rfs_expansion", expansion_mode="mc_choice_shuffle"),
    ]

    decision = aggregate_reasoning_first_selection(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=expansion_rows,
        stage_winner="A",
        expansion_modes=("mc_choice_shuffle",),
        promotion_min_independent_support=2,
        promotion_margin_min=1.25,
        leader_lock_count=4,
        mc_shuffle_min_agreement=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "B"
    assert decision.changed is True
    assert decision.resolver == "cred_rfs_candidate_promoted"


def test_rfs_strategyqa_promotion_disabled() -> None:
    stage_rows = [
        _row("strategyqa", "yes", confidence=0.70),
        _row("strategyqa", "yes", confidence=0.69),
        _row("strategyqa", "yes", confidence=0.68),
        _row("strategyqa", "no", confidence=0.90),
        _row("strategyqa", "no", confidence=0.89),
    ]
    expansion_rows = [
        _row("strategyqa", "no", confidence=0.95, method_name="cred_rfs_expansion", expansion_mode="rfs_extra_solver"),
        _row("strategyqa", "no", confidence=0.94, method_name="cred_rfs_expansion", expansion_mode="rfs_extra_solver"),
    ]

    decision = aggregate_reasoning_first_selection(
        dataset="strategyqa",
        question="Is the statement true?",
        context="",
        stage_rows=stage_rows,
        expansion_rows=expansion_rows,
        stage_winner="yes",
        expansion_modes=(),
        promotion_min_independent_support=2,
        promotion_margin_min=1.25,
        leader_lock_count=4,
        mc_shuffle_min_agreement=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "yes"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_strategyqa_promotion_disabled"


def test_rfs_v2_pairwise_duel_promotes_stage_supported_challenger_with_two_wins() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    selection_rows = [
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="A"),
    ]

    decision = aggregate_pairwise_selection(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("mc_blind_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "B"
    assert decision.changed is True
    assert decision.resolver == "cred_rfs_v2_pairwise_promoted"


def test_rfs_v2_pairwise_duel_rejects_one_win() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    selection_rows = [
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="A"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="A"),
    ]

    decision = aggregate_pairwise_selection(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("mc_blind_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v2_pairwise_rejected"


def test_rfs_v2_pairwise_blocks_challenger_without_stage_support() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.67),
        _row("mmlu_pro", "B", confidence=0.66),
    ]
    selection_rows = [
        _duel_row("mmlu_pro", leader="A", challenger="C", winner="C"),
        _duel_row("mmlu_pro", leader="A", challenger="C", winner="C"),
        _duel_row("mmlu_pro", leader="A", challenger="C", winner="C"),
    ]

    decision = aggregate_pairwise_selection(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("mc_blind_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False


def test_rfs_v2_strong_majority_requires_unanimous_pairwise_probe() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.80),
        _row("mmlu_pro", "A", confidence=0.79),
        _row("mmlu_pro", "A", confidence=0.78),
        _row("mmlu_pro", "A", confidence=0.77),
        _row("mmlu_pro", "B", confidence=0.95),
    ]
    selection_rows = [
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="A"),
    ]

    decision = aggregate_pairwise_selection(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("mc_blind_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v2_strong_majority_locked"


def test_rfs_v2_strategyqa_minority_probe_requires_two_extra_votes() -> None:
    stage_rows = [
        _row("strategyqa", "yes", confidence=0.80),
        _row("strategyqa", "yes", confidence=0.79),
        _row("strategyqa", "yes", confidence=0.78),
        _row("strategyqa", "yes", confidence=0.77),
        _row("strategyqa", "no", confidence=0.95),
    ]
    selection_rows = [
        _row("strategyqa", "no", method_name="cred_rfs_expansion", expansion_mode="strategyqa_minority_resample"),
        _row("strategyqa", "no", method_name="cred_rfs_expansion", expansion_mode="strategyqa_minority_resample"),
    ]

    decision = aggregate_pairwise_selection(
        dataset="strategyqa",
        question="Is the statement true?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="yes",
        selection_modes=("strategyqa_minority_resample",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "no"
    assert decision.changed is True
    assert decision.resolver == "cred_rfs_v2_strategyqa_minority_promoted"


def test_rfs_v2_hotpot_blocks_non_answer_candidate() -> None:
    stage_rows = [
        _row("hotpotqa", "Paris", confidence=0.80),
        _row("hotpotqa", "Paris", confidence=0.79),
        _row("hotpotqa", "Paris", confidence=0.78),
        _row("hotpotqa", "not stated in context", confidence=0.77),
        _row("hotpotqa", "Paris", confidence=0.76),
    ]

    decision = aggregate_pairwise_selection(
        dataset="hotpotqa",
        question="What city?",
        context="Paris is stated in the context.",
        stage_rows=stage_rows,
        selection_rows=[],
        stage_winner="Paris",
        selection_modes=("hotpot_context_span_repair",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=2,
        require_stage_a_challenger_support=True,
    )

    assert decision.final_answer == "Paris"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v2_non_answer_blocked"


def test_rfs_v3_gpqa_unanimous_duel_promotes_stage_supported_challenger() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "A", confidence=0.69),
        _row("gpqa_diamond", "A", confidence=0.68),
        _row("gpqa_diamond", "B", confidence=0.90),
        _row("gpqa_diamond", "B", confidence=0.89),
    ]
    selection_rows = [
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
    ]

    decision = aggregate_safe_select_v3(
        dataset="gpqa_diamond",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=3,
        pairwise_allowed_datasets=("gpqa_diamond",),
        pairwise_option_count_max=4,
        option_count=4,
        require_stage_a_challenger_support=True,
        allow_strong_majority_pairwise_promotion=False,
    )

    assert decision.final_answer == "B"
    assert decision.changed is True
    assert decision.resolver == "cred_rfs_v3_gpqa_unanimous_pairwise_promoted"


def test_rfs_v3_gpqa_two_of_three_duel_does_not_promote() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "A", confidence=0.69),
        _row("gpqa_diamond", "A", confidence=0.68),
        _row("gpqa_diamond", "B", confidence=0.90),
        _row("gpqa_diamond", "B", confidence=0.89),
    ]
    selection_rows = [
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="A", mode="gpqa_unanimous_pairwise_duel"),
    ]

    decision = aggregate_safe_select_v3(
        dataset="gpqa_diamond",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=3,
        pairwise_allowed_datasets=("gpqa_diamond",),
        pairwise_option_count_max=4,
        option_count=4,
        require_stage_a_challenger_support=True,
        allow_strong_majority_pairwise_promotion=False,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v3_pairwise_rejected"


def test_rfs_v3_mmlu_unanimous_duel_is_blocked() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.70),
        _row("mmlu_pro", "A", confidence=0.69),
        _row("mmlu_pro", "A", confidence=0.68),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    selection_rows = [
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("mmlu_pro", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
    ]

    decision = aggregate_safe_select_v3(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=3,
        pairwise_allowed_datasets=("gpqa_diamond",),
        pairwise_option_count_max=4,
        option_count=4,
        require_stage_a_challenger_support=True,
        allow_strong_majority_pairwise_promotion=False,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v3_pairwise_dataset_blocked"


def test_rfs_v3_strategyqa_minority_probe_is_disabled() -> None:
    stage_rows = [
        _row("strategyqa", "yes", confidence=0.80),
        _row("strategyqa", "yes", confidence=0.79),
        _row("strategyqa", "yes", confidence=0.78),
        _row("strategyqa", "no", confidence=0.95),
        _row("strategyqa", "no", confidence=0.94),
    ]
    selection_rows = [
        _row("strategyqa", "no", method_name="cred_rfs_expansion", expansion_mode="strategyqa_minority_resample"),
        _row("strategyqa", "no", method_name="cred_rfs_expansion", expansion_mode="strategyqa_minority_resample"),
    ]

    decision = aggregate_safe_select_v3(
        dataset="strategyqa",
        question="Is the statement true?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="yes",
        selection_modes=("deterministic_repair",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=3,
        pairwise_allowed_datasets=("gpqa_diamond",),
        pairwise_option_count_max=4,
        option_count=0,
        require_stage_a_challenger_support=True,
        allow_strong_majority_pairwise_promotion=False,
    )

    assert decision.final_answer == "yes"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v3_pairwise_disabled"


def test_rfs_v3_strong_majority_not_overridden_by_unanimous_duel() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.80),
        _row("gpqa_diamond", "A", confidence=0.79),
        _row("gpqa_diamond", "A", confidence=0.78),
        _row("gpqa_diamond", "A", confidence=0.77),
        _row("gpqa_diamond", "B", confidence=0.95),
    ]
    selection_rows = [
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
    ]

    decision = aggregate_safe_select_v3(
        dataset="gpqa_diamond",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=3,
        pairwise_allowed_datasets=("gpqa_diamond",),
        pairwise_option_count_max=4,
        option_count=4,
        require_stage_a_challenger_support=True,
        allow_strong_majority_pairwise_promotion=False,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v3_strong_majority_locked"


def test_rfs_v3_protocol_failure_duel_does_not_count_as_unanimous() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "A", confidence=0.69),
        _row("gpqa_diamond", "A", confidence=0.68),
        _row("gpqa_diamond", "B", confidence=0.90),
        _row("gpqa_diamond", "B", confidence=0.89),
    ]
    failed = _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel")
    failed["protocol_parse_status"] = "failed"
    failed["pairwise_validation_pass"] = False
    failed["expansion_validation_pass"] = False
    selection_rows = [
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        failed,
    ]

    decision = aggregate_safe_select_v3(
        dataset="gpqa_diamond",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=3,
        pairwise_allowed_datasets=("gpqa_diamond",),
        pairwise_option_count_max=4,
        option_count=4,
        require_stage_a_challenger_support=True,
        allow_strong_majority_pairwise_promotion=False,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v3_pairwise_rejected"


def test_rfs_v4_shadow_wrapper_does_not_promote_two_of_three_pairwise() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "A", confidence=0.69),
        _row("gpqa_diamond", "A", confidence=0.68),
        _row("gpqa_diamond", "B", confidence=0.90),
        _row("gpqa_diamond", "B", confidence=0.89),
    ]
    selection_rows = [
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="A", mode="gpqa_unanimous_pairwise_duel"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_2of3_retry_shadow"),
        _duel_row("gpqa_diamond", leader="A", challenger="B", winner="B", mode="gpqa_2of3_retry_shadow"),
    ]

    decision = aggregate_shadow_select_v4(
        dataset="gpqa_diamond",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner="A",
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        leader_lock_count=4,
        pairwise_duel_replicates=3,
        pairwise_promotion_min_wins=3,
        pairwise_allowed_datasets=("gpqa_diamond",),
        pairwise_option_count_max=4,
        option_count=4,
        require_stage_a_challenger_support=True,
        allow_strong_majority_pairwise_promotion=False,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_rfs_v4_shadow_no_promotion"


def _row(
    dataset: str,
    answer: str,
    *,
    confidence: float = 0.5,
    payload: dict | None = None,
    method_name: str = "cred_stage_a",
    expansion_mode: str = "",
    request_status: str = "ok",
) -> dict:
    validated_output = {
        "answer": answer,
        "final_answer": answer,
        "confidence": confidence,
        "key_evidence": "option clue supports the answer",
        "risk_level": "none",
    }
    if payload:
        validated_output.update(payload)
    return {
        "dataset": dataset,
        "method_name": method_name,
        "prediction": answer,
        "normalized_answer": answer,
        "confidence_value": confidence,
        "key_evidence": validated_output.get("key_evidence", ""),
        "risk_level": validated_output.get("risk_level", "none"),
        "request_status": request_status,
        "output_status": "ok" if request_status == "ok" else "request_fail",
        "protocol_parse_status": "ok",
        "expansion_mode": expansion_mode,
        "expansion_validation_pass": request_status == "ok",
        "validated_output": validated_output,
    }


def _duel_row(
    dataset: str,
    *,
    leader: str,
    challenger: str,
    winner: str,
    request_status: str = "ok",
    mode: str = "mc_blind_pairwise_duel",
) -> dict:
    row = _row(dataset, winner, method_name="cred_rfs_expansion", expansion_mode=mode, request_status=request_status)
    row["pairwise_leader_answer"] = leader
    row["pairwise_challenger_answer"] = challenger
    row["pairwise_winner_answer"] = winner
    row["pairwise_leader_family"] = f"mc:{leader}"
    row["pairwise_challenger_family"] = f"mc:{challenger}"
    row["pairwise_winner_family"] = f"mc:{winner}"
    row["pairwise_validation_pass"] = request_status == "ok"
    row["expansion_validation_pass"] = request_status == "ok"
    return row
