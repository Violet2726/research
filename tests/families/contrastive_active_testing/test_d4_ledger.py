from __future__ import annotations

from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.run import sample as sample_runner
from research_experiments.families.contrastive_active_testing.run.d4_ledger import D4CompletionLedger


def _row(*, answer: str, status: str) -> dict[str, object]:
    return {
        "sample_id": "fixture-1",
        "method_name": "catch_stage_a_shared",
        "role": "stage_a_solver",
        "agent_id": 1,
        "request_seed": 42_001,
        "assistant_text": "<final_answer>A</final_answer>",
        "prediction": answer,
        "protocol_parse_status": status,
        "request_error": "timeout" if status == "failed" else None,
    }


def test_d4_completion_ledger_replays_success_and_failure_without_a_request(tmp_path) -> None:
    path = tmp_path / "d4_completion_ledger.jsonl"
    ledger = D4CompletionLedger(path)
    failed = _row(answer="", status="failed")
    ledger.record(failed)
    # The first durable result is authoritative; a later retry cannot replace it.
    ledger.record(_row(answer="A", status="ok"))

    reopened = D4CompletionLedger(path)
    assert reopened.lookup(
        sample_id="fixture-1",
        method_name="catch_stage_a_shared",
        role="stage_a_solver",
        agent_id=1,
        seed=42_001,
    ) == failed

    sample = DatasetSample("bbeh", "fixture-1", "Return A.", "A", "", {"task": "fixture"})
    replayed = sample_runner._answer_turn(
        sample,
        run_id="same-frozen-run",
        split_name="development",
        endpoint=SimpleNamespace(completion_ledger=reopened),
        network_budget=sample_runner.NetworkAttemptBudget(1),
        method_name="catch_stage_a_shared",
        role="stage_a_solver",
        agent_id=1,
        seed=42_001,
        max_tokens=65_536,
    )
    assert replayed == failed
