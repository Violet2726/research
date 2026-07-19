from __future__ import annotations

import json
from collections import Counter

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import load_samples, question_without_answer_contract
from research_experiments.core.data.evaluation import canonicalize_answer, score_prediction
from research_experiments.core.prompts.dataset_contracts import dataset_instruction
from research_experiments.families.contrastive_active_testing.boundary import (
    boundary_sample_view,
    boundary_stratum,
    select_screening_samples,
    verify_source_asset,
)


def test_musr_official_loader_counts_contract_and_frozen_screening() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/musr/all.toml")
    assert verify_source_asset(benchmark)["sha256"] == benchmark.source_sha256
    samples = load_samples(benchmark)
    assert len(samples) == 756
    assert Counter(sample.metadata["domain"] for sample in samples) == {
        "murder_mysteries": 250,
        "object_placements": 256,
        "team_allocation": 250,
    }
    selected = select_screening_samples("musr", samples, count=100, seed=42)
    assert Counter(sample.metadata["domain"] for sample in selected) == {
        "murder_mysteries": 34,
        "object_placements": 33,
        "team_allocation": 33,
    }
    assert [sample.sample_id for sample in selected] == [
        sample.sample_id for sample in select_screening_samples("musr", samples, count=100, seed=42)
    ]
    view = boundary_sample_view(selected[0])
    assert "Options:" in view.question
    assert "Options:" not in question_without_answer_contract(view)
    assert score_prediction("musr", view.metadata["answer_letter"], view.reference_answer, sample=view) == 1.0
    option = view.metadata["answer_contract"]["options"][0]
    assert canonicalize_answer(view, option["text"]).key == option["label"]
    assert not canonicalize_answer(view, f"({option['label']}) definitely not the exact option").valid


def test_seqbench_official_loader_counts_strata_and_strict_sequences() -> None:
    benchmark = load_benchmark_config(
        "configs/core/shared/benchmarks/seqbench/seqBench_compact.jsonl.toml"
    )
    assert verify_source_asset(benchmark)["sha256"] == benchmark.source_sha256
    samples = load_samples(benchmark)
    assert len(samples) == 7_079
    selected = select_screening_samples("seqbench", samples, count=100, seed=42)
    assert len(selected) == len({sample.sample_id for sample in selected}) == 100
    assert len({boundary_stratum(sample) for sample in selected}) == 42
    actions = json.loads(samples[0].reference_answer)
    python_literal = repr(actions)
    canonical = canonicalize_answer(samples[0], python_literal)
    assert canonical.valid
    assert canonical.key == samples[0].reference_answer
    assert score_prediction("seqbench", python_literal, samples[0].reference_answer, sample=samples[0]) == 1.0
    assert not canonicalize_answer(samples[0], f"The sequence is {python_literal}").valid
    assert not canonicalize_answer(samples[0], '["ok", 3]').valid


def test_seqbench_prompt_contains_official_action_schema_and_three_examples() -> None:
    prompt = dataset_instruction("seqbench")
    for action in (
        "start:",
        "move_to:",
        "pick_up_key:",
        "use_key:",
        "unlock_and_open_door_to:",
        "rescue:",
    ):
        assert action in prompt
    assert "Official-format example 1" in prompt
    assert "Official-format example 2" in prompt
    assert "Official-format example 3" in prompt


def test_gpqa_boundary_view_preserves_shuffle_and_stratifies_domains() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/gpqa/dataset.toml")
    samples = load_samples(benchmark)
    selected = select_screening_samples("gpqa_diamond", samples, count=100, seed=42)
    all_domains = {str(sample.metadata.get("high_level_domain") or "unknown") for sample in samples}
    selected_domains = {str(sample.metadata.get("high_level_domain") or "unknown") for sample in selected}
    assert selected_domains == all_domains
    view = boundary_sample_view(selected[0])
    assert view.prompt_context == ""
    assert view.metadata["task"] == view.metadata["high_level_domain"]
    assert question_without_answer_contract(view) == selected[0].question
    assert [item["text"] for item in view.metadata["answer_contract"]["options"]] == selected[0].metadata["options"]
