from __future__ import annotations

import re
from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.disagreement_guided_crux_reconstruction.config import DgcrProtocolConfig
from research_experiments.families.disagreement_guided_crux_reconstruction.run import sample as sample_runner


def test_disagreement_path_keeps_adaptive_and_dgcr_logical_costs_aligned(monkeypatch) -> None:
    sample = DatasetSample(
        dataset="bbeh",
        sample_id="dgcr-unit",
        question="abcdefgh decisive premise\nOptions:\n(A) alpha\n(B) beta",
        reference_answer="B",
        prompt_context="",
        metadata={"task": "unit", "options": [{"label": "A", "text": "alpha"}, {"label": "B", "text": "beta"}]},
    )
    protocol = DgcrProtocolConfig(5, 3, 2, 0.7, 1.0, 16_384, 2_048, 8, 256, True)
    calls: list[tuple[str, str, str]] = []
    stage_answers = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B"}

    def answer_turn(_sample, *, method_name, role, agent_id, **_kwargs):
        answer = stage_answers[agent_id] if role == "stage_a_solver" else "B"
        calls.append((role, method_name, answer))
        return {"answer_class_key": answer, "normalized_answer": answer, "prediction": answer, "total_tokens": 1, "network_attempt_count": 1}

    def json_turn(_sample, *, role, method_name, messages, **_kwargs):
        calls.append((role, method_name, messages[-1]["content"]))
        if role == "crux_proposer":
            assert "support" not in messages[-1]["content"].lower()
            return {"total_tokens": 1, "network_attempt_count": 1}, {"start_char": 0, "end_char": 8}
        label_to_key = dict(re.findall(r"Candidate ([A-Z]): ([A-Z])", messages[-1]["content"]))
        return (
            {"total_tokens": 1, "network_attempt_count": 1},
            {"reconstructions": {label: "abcdefgh" if key == "B" else "wrong" for label, key in label_to_key.items()}},
        )

    monkeypatch.setattr(sample_runner, "_answer_turn", answer_turn)
    monkeypatch.setattr(sample_runner, "_json_turn", json_turn)
    turns, router, predictions = sample_runner.run_dgcr_sample(
        sample,
        run_id="run",
        split_name="dev",
        experiment=SimpleNamespace(global_seed=42),
        protocol=protocol,
        endpoint=SimpleNamespace(cache_namespace="dgcr-dev-v1"),
    )

    by_method = {row["method_name"]: row for row in predictions}
    assert len(turns) == 11
    assert router["triggered"] and router["override_accepted"]
    assert by_method["adaptive_sc_8"]["calls_per_question"] == 8
    assert by_method["dgcr"]["calls_per_question"] == 8
    assert by_method["adaptive_sc_8"]["total_tokens_per_question"] == 8
    assert by_method["dgcr"]["total_tokens_per_question"] == 8
    assert by_method["adaptive_sc_8"]["actual_intervention_calls_per_question"] == 3
    assert by_method["dgcr"]["actual_intervention_calls_per_question"] == 3
    assert by_method["dgcr"]["prediction"] == "B"
