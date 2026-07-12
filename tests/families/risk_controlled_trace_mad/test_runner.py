from __future__ import annotations

import json
from dataclasses import replace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.risk_controlled_trace_mad.config import load_experiment_config
from research_experiments.families.risk_controlled_trace_mad.run import execute as execute_runner


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
                'notes = "烟雾测试"',
            ]
        ),
        encoding="utf-8",
    )
    source = load_experiment_config("configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml")
    raw = dict(source.raw)
    raw["phases"] = {
        "count20_seed42": {
            "benchmark_slugs": ["bbeh"],
            "split_overrides": {"bbeh": "count20_seed42"},
            "methods": source.raw["phases"]["count20_seed42"]["methods"],
        }
    }
    experiment = replace(source, name="fake_smoke", benchmark_configs=[benchmark], raw=raw)
    sample = DatasetSample("bbeh", "s", "Is this true?", "yes", "", {"task": "task_a"})
    monkeypatch.setattr(execute_runner, "load_selected_samples", lambda benchmark, split_name: [sample])
    monkeypatch.setattr(execute_runner, "estimate_work", lambda *args: (18, 7))
    monkeypatch.setattr(execute_runner, "run_batch", _fake_batch)
    monkeypatch.setattr(execute_runner, "OpenAICompatibleProvider", _FakeProvider)
    monkeypatch.setenv("RESEARCH_REPORTS_ROOT", str(tmp_path / "reports"))

    run_dir = execute_runner.run_experiment(
        experiment,
        "count20_seed42",
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )
    validation = json.loads((run_dir / "run_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"]
    assert (run_dir / "diagnostics" / "evf_diagnostics.json").exists()
    assert (run_dir / "exports" / "evf_comparison.json").exists()


def _fake_batch(**kwargs):
    methods = kwargs["active_methods"]
    turns = []
    for lineage in ("qwen", "mimo"):
        for agent in range(1, 10):
            turns.append(
                {
                    "run_id": kwargs["run_id"],
                    "dataset": kwargs["dataset"],
                    "split": kwargs["split_name"],
                    "sample_id": "s",
                    "method_name": "evf_stage_a_shared",
                    "method_type": "evf_stage_a",
                    "round_index": 0,
                    "agent_id": agent,
                    "role": f"{lineage}_solver",
                    "model_lineage": lineage,
                    "output_status": "ok",
                    "protocol_parse_status": "ok",
                    "reason_present": True,
                    "prompt_hash": f"{lineage}-{agent}",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "latency_ms": 1,
                    "network_attempt_count": 0,
                    "cache_hit": True,
                    "request_started_at_events": [],
                    "normalized_answer": "yes",
                    "prediction": "yes",
                    "payload": {"seed": agent},
                    "validated_output": {"final_answer": "yes", "reasoning": "ok"},
                }
            )
    predictions = []
    limits = {
        "cot_1": 1,
        "qwen_sc_5": 5,
        "qwen_sc_9": 9,
        "mimo_sc_5": 5,
        "mimo_sc_9": 9,
        "heterogeneous_mv_5": 5,
        "evf_mad_1": 5,
    }
    for method in methods:
        calls = limits[method]
        predictions.append(
            {
                "run_id": kwargs["run_id"],
                "dataset": kwargs["dataset"],
                "split": kwargs["split_name"],
                "sample_id": "s",
                "task": "task_a",
                "method_name": method,
                "method_type": "mad_innovation",
                "model_name": "fake-compound",
                "prediction": "yes",
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
                "provider_abstentions_per_question": 0,
                "protocol_failures_per_question": 0,
                "request_failures_per_question": 0,
                "triggered": False,
                "initial_disagreement": False,
                "override_accepted": False,
                "vote_flipped": False,
                "corrected_by_debate": False,
                "harmed_by_debate": False,
                "unchanged_correct": True,
                "unchanged_wrong": False,
                "resolver": "unanimous_no_trigger",
                "stage_a_prompt_hashes": ["x"] * min(calls, 5),
                "evf_gate_passed": False,
                "novel_answer": False,
            }
        )
    yield 0, turns, [], [{"sample_id": "s", "policy_name": "evf_executable_falsification_v4"}], predictions


class _FakeProvider:
    def __init__(self, backbone) -> None:
        self.backbone = backbone

    def close(self) -> None:
        return None
