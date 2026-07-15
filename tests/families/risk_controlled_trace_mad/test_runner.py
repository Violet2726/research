from __future__ import annotations

import json
from dataclasses import replace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.risk_controlled_trace_mad.config import load_experiment_config
from research_experiments.families.risk_controlled_trace_mad.run import execute as execute_runner
from research_experiments.families.risk_controlled_trace_mad.run import hsgsa_execute


def test_fake_provider_end_to_end_smoke_writes_and_validates_artifacts(monkeypatch, tmp_path) -> None:
    benchmark = tmp_path / "bbeh.toml"
    benchmark.write_text(
        "\n".join(
            [
                'name = "BBEH smoke"',
                'slug = "bbeh"',
                'loader = "bbeh_json_bundle"',
                'source_path = "unused.zip"',
                'source_split = "test"',
                'sample_id_prefix = "bbeh"',
                'question_field = "input"',
                'answer_field = "target"',
                "smoke_size = 1",
                "pilot_size = 1",
                "main_size = 1",
                "random_seed = 42",
                'notes = "fake smoke"',
            ]
        ),
        encoding="utf-8",
    )
    source = load_experiment_config("configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml")
    raw = dict(source.raw)
    raw["phases"] = {
        "confirm_seed42": {
            "benchmark_slugs": ["bbeh"],
            "split_overrides": {"bbeh": "count20_seed42"},
            "methods": source.methods,
        }
    }
    experiment = replace(source, name="fake_smoke", benchmark_configs=[benchmark], raw=raw)
    sample = DatasetSample("bbeh", "s", "Is this true?", "yes", "", {"task": "task_a"})
    monkeypatch.setattr(hsgsa_execute, "load_selected_samples", lambda benchmark, split_name: [sample])
    monkeypatch.setattr(hsgsa_execute, "run_hsgsa_batch", _fake_batch)
    monkeypatch.setattr(hsgsa_execute, "OpenAICompatibleProvider", _FakeProvider)
    monkeypatch.setenv("RESEARCH_REPORTS_ROOT", str(tmp_path / "reports"))

    run_dir = execute_runner.run_experiment(
        experiment,
        "confirm_seed42",
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )
    validation = json.loads((run_dir / "run_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"]
    assert (run_dir / "diagnostics" / "hsgsa_diagnostics.json").exists()
    assert (run_dir / "exports" / "hsgsa_comparison.json").exists()


def _fake_batch(**kwargs):
    turns = []
    for agent in range(1, 6):
        turns.append(
            {
                "run_id": kwargs["run_id"],
                "dataset": kwargs["dataset"],
                "split": kwargs["split_name"],
                "sample_id": "s",
                "method_name": "hsgsa_stage_a_shared",
                "method_type": "homogeneous_stage_a",
                "round_index": 0,
                "agent_id": agent,
                "role": "homogeneous_solver",
                "model_lineage": "mimo",
                "model_name": "mimo-v2.5",
                "output_status": "ok",
                "protocol_parse_status": "ok",
                "reason_present": True,
                "prompt_hash": f"mimo-{agent}",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "latency_ms": 1,
                "network_attempt_count": 0,
                "cache_hit": True,
                "request_started_at_events": [],
                "normalized_answer": "yes",
                "answer_class_key": "yes",
                "prediction": "yes",
                "payload": {"seed": agent},
                "validated_output": {"final_answer": "yes", "reasoning": "ok"},
            }
        )
    predictions = []
    limits = {
        "cot_1": 1,
        "sc_3": 3,
        "sc_5": 5,
        "adaptive_sc_8": 5,
        "conditional_resample_3": 5,
        "blind_gsa_1": 5,
        "blind_gsa_quorum_3": 5,
        "hsgsa_unanimous_3": 5,
    }
    for method in kwargs["active_methods"]:
        calls = limits[method]
        predictions.append(
            {
                "run_id": kwargs["run_id"],
                "dataset": kwargs["dataset"],
                "split": kwargs["split_name"],
                "sample_id": "s",
                "task": "task_a",
                "method_name": method,
                "method_type": "homogeneous_support_blind_sgsa",
                "model_name": "mimo-v2.5",
                "prediction": "yes",
                "answer_class_key": "yes",
                "initial_answer_class_key": "yes",
                "gold": "yes",
                "score": 1.0,
                "initial_vote_prediction": "yes",
                "initial_vote_score": 1.0,
                "initial_vote_counts": {"yes": 5},
                "initial_consensus": True,
                "final_vote_prediction": "yes",
                "final_vote_score": 1.0,
                "final_vote_counts": {"yes": 1},
                "total_tokens_per_question": calls * 2,
                "prompt_tokens_per_question": calls,
                "completion_tokens_per_question": calls,
                "latency_ms_per_question": calls,
                "logical_calls_per_question": calls,
                "calls_per_question": calls,
                "network_attempts_per_question": 0,
                "shared_physical_network_attempts_per_question": 0,
                "provider_abstentions_per_question": 0,
                "protocol_failures_per_question": 0,
                "request_failures_per_question": 0,
                "reviewer_calls_per_question": 0,
                "reviewer_valid_picks_per_question": 0,
                "reviewer_protocol_failures_per_question": 0,
                "triggered": False,
                "initial_disagreement": False,
                "override_accepted": False,
                "vote_flipped": False,
                "corrected_by_debate": False,
                "harmed_by_debate": False,
                "unchanged_correct": True,
                "unchanged_wrong": False,
                "candidate_oracle_correct": True,
                "resolver": "no_answer_class_disagreement",
                "stage_a_prompt_hashes": ["x"] * 5,
                "novel_answer": False,
            }
        )
    yield 0, turns, [], [
        {
            "run_id": kwargs["run_id"],
            "dataset": kwargs["dataset"],
            "split": kwargs["split_name"],
            "sample_id": "s",
            "policy_name": "homogeneous_support_blind_sgsa_v5",
            "triggered": False,
        }
    ], predictions


class _FakeProvider:
    def __init__(self, backbone) -> None:
        self.backbone = backbone

    def close(self) -> None:
        return None
