from __future__ import annotations

from budget_comm.config import ContextViewConfig
from budget_comm.dataset_views import build_context_views
from budget_comm.logic import assign_density_tiers, solve_knapsack
from experiment_core.datasets import DatasetSample


def test_assign_density_tiers_three_way_split() -> None:
    tiers = assign_density_tiers({1: 0.1, 2: 0.2, 3: 0.3})
    assert tiers == {1: "keywords", 2: "summary", 3: "full"}


def test_knapsack_prefers_lower_cost_then_lexicographic_order() -> None:
    decision = solve_knapsack(
        [
            {"agent_id": 1, "score": 1.0, "cost": 2},
            {"agent_id": 2, "score": 1.0, "cost": 1},
            {"agent_id": 3, "score": 1.0, "cost": 1},
        ],
        budget_tokens=2,
    )
    assert decision.winner_agent_ids == (2, 3)


def test_build_strategyqa_split_views_cover_all_facts() -> None:
    sample = DatasetSample(
        dataset="strategyqa",
        sample_id="s1",
        question="Question?",
        reference_answer="yes",
        prompt_context="",
        metadata={
            "description": "desc",
            "facts": ["fact_a", "fact_b", "fact_c"],
            "decomposition": ["step_1", "step_2"],
        },
    )
    config = ContextViewConfig(
        track_name="split_context",
        strategyqa_mode="facts_even_odd_plus_decomposition",
        hotpotqa_mode="supporting_paragraph_shards",
        allow_full_context=False,
    )
    views = build_context_views(sample, config, agent_count=3)
    covered = set()
    required = set()
    for view in views:
        covered.update(view.coverage_items)
        required.update(view.required_coverage_items)
        assert view.includes_full_context is False
    assert required.issubset(covered)


def test_build_hotpot_split_views_do_not_leak_full_context() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="h1",
        question="Question?",
        reference_answer="answer",
        prompt_context="[A] para_a\n[B] para_b\n[C] para_c\n[D] para_d",
        metadata={
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
            "raw_context": {
                "title": ["A", "B", "C", "D"],
                "sentences": [["para_a"], ["para_b"], ["para_c"], ["para_d"]],
            },
        },
    )
    config = ContextViewConfig(
        track_name="split_context",
        strategyqa_mode="facts_even_odd_plus_decomposition",
        hotpotqa_mode="supporting_paragraph_shards",
        allow_full_context=False,
    )
    views = build_context_views(sample, config, agent_count=3)
    covered = set()
    for view in views:
        assert view.includes_full_context is False
        assert view.view_context_hash != view.full_context_hash
        covered.update(view.coverage_items)
    assert covered == {"A", "B"}
