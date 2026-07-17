from __future__ import annotations

import json
import re
from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.config import CatchProtocolConfig
from research_experiments.families.contrastive_active_testing.run import sample as sample_runner


def test_development_grid_keeps_each_catch_variant_at_five_plus_three_calls(monkeypatch) -> None:
    sample = DatasetSample(
        dataset="bbeh",
        sample_id="catch-unit",
        question="The quantity is either even or odd.\nOptions:\n(A) even\n(B) odd",
        reference_answer="B",
        prompt_context="",
        metadata={
            "task": "unit",
            "options": [{"label": "A", "text": "even"}, {"label": "B", "text": "odd"}],
        },
    )
    protocol = CatchProtocolConfig(5, 3, 2, 3, 6, 4, 0.7, 1.0, 16_384, 4_096, (2, 3, 4), (1, 2), 62_000)
    stage_answers = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B"}
    calls: list[str] = []

    def answer_turn(_sample, *, role, agent_id, **_kwargs):
        answer = stage_answers[agent_id] if role == "stage_a_solver" else "B"
        calls.append(role)
        reasoning = "even fact first; parity check second; final relation third" if answer == "A" else "odd fact first; parity check second; final relation third"
        return {
            "answer_class_key": answer,
            "normalized_answer": answer,
            "prediction": answer,
            "validated_output": {"reasoning": reasoning, "final_answer": answer},
            "total_tokens": 1,
            "actual_total_tokens": 1,
            "network_attempt_count": 1,
        }

    def json_turn(_sample, *, role, messages, **_kwargs):
        calls.append(role)
        if role == "test_designer":
            hypotheses = json.loads(re.search(r"Anonymous hypotheses:\n(.+?)\n\nProduce", messages[-1]["content"], re.S).group(1))
            commitments = {}
            for hypothesis in hypotheses:
                outcome = "O0" if hypothesis["answer"] == "A" else "O1"
                commitments[hypothesis["id"]] = {"outcome_id": outcome, "trace_start": 0, "trace_end": 9}
            tests = []
            for index, start in enumerate((0, 16, 37)):
                per_test = {
                    hypothesis["id"]: {
                        "outcome_id": "O0" if hypothesis["answer"] == "A" else "O1",
                        "trace_start": start,
                        "trace_end": min(start + 9, len(hypothesis["reasoning"])),
                    }
                    for hypothesis in hypotheses
                }
                tests.append(
                    {
                        "test_id": f"T{index}",
                        "question": f"Is parity diagnostic fact {index} even?",
                        "outcomes": [{"id": "O0", "text": "even"}, {"id": "O1", "text": "odd"}],
                        "commitments": per_test,
                    }
                )
            return {"total_tokens": 1, "actual_total_tokens": 1, "network_attempt_count": 1}, {"tests": tests}
        if role == "blinded_witness":
            rendered = json.loads(re.search(r"Diagnostic tests:\n(.+?)\n\nReturn", messages[-1]["content"], re.S).group(1))
            answers = []
            for test in rendered:
                odd = next(outcome for outcome in test["outcomes"] if outcome["text"] == "odd")
                answers.append({"test_id": test["test_id"], "outcome_id": odd["id"], "check": "odd"})
            return {"total_tokens": 1, "actual_total_tokens": 1, "network_attempt_count": 1}, {"answers": answers}
        hypothesis_id = re.search(r'"id": "(H\d+)"', messages[-1]["content"]).group(1)
        return {"total_tokens": 1, "actual_total_tokens": 1, "network_attempt_count": 1}, {"selected_id": hypothesis_id, "check": "ok"}

    monkeypatch.setattr(sample_runner, "_answer_turn", answer_turn)
    monkeypatch.setattr(sample_runner, "_json_turn", json_turn)
    turns, router, predictions = sample_runner.run_catch_sample(
        sample,
        run_id="run",
        split_name="dev",
        experiment=SimpleNamespace(global_seed=42),
        protocol=protocol,
        endpoint=SimpleNamespace(cache_namespace="catch-dev-v1"),
        network_budget=sample_runner.NetworkAttemptBudget(62_000),
        phase_name="development",
        frozen_decoding=None,
        run_direct_judge=True,
    )

    variants = [row for row in predictions if row["method_name"].startswith("catch_d")]
    assert len(turns) == 18
    assert len(variants) == 6
    assert all(row["logical_calls_per_question"] == 8 for row in variants)
    assert all(row["actual_intervention_calls_per_question"] == 3 for row in variants)
    assert router["candidate_oracle_correct"] is True
    assert any(row["override_accepted"] and row["prediction"] == "B" for row in variants)

