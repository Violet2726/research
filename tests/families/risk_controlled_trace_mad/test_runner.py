from __future__ import annotations

import json

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.risk_controlled_trace_mad.config import RctaExperimentConfig
from research_experiments.families.risk_controlled_trace_mad.run import execute as execute_runner
from research_experiments.families.risk_controlled_trace_mad.run import sample as sample_runner
from research_experiments.family_runtime.config_helpers import resolve_model


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
    experiment = RctaExperimentConfig(
        name="smoke",
        description="fake provider smoke",
        benchmark_configs=[benchmark],
        protocol="configs/families/risk_controlled_trace_mad/protocols/rcta_v1.toml",
        control_catalog="configs/families/single_agent/methods/common.toml",
        control_methods=["sc_5", "sc_9"],
        rcta_methods=["adaptive_sc_9", "gsa_trace_1", "rcta_certificate_shadow_1"],
        method_order=["sc_5", "sc_9", "adaptive_sc_9", "gsa_trace_1", "rcta_certificate_shadow_1"],
        global_seed=42,
        control_prompt_version="single_agent_free_text_v1",
        primary_model_ref="dashscope/qwen-flash",
        max_concurrent_requests=1,
        requests_per_minute_limit=18,
        runtime_profiles={},
        raw={
            "phases": {
                "count20_seed42": {
                    "benchmark_slugs": ["bbeh"],
                    "split_overrides": {"bbeh": "count20_seed42"},
                    "rcta_methods": ["adaptive_sc_9", "gsa_trace_1", "rcta_certificate_shadow_1"],
                }
            }
        },
    )
    sample = DatasetSample("bbeh", "s", "Is this true?", "yes", "", {"task": "task_a"})
    monkeypatch.setattr(sample_runner, "_run_stage_pool", _fake_stage_pool)
    monkeypatch.setattr(sample_runner, "_execute_synthesis_turn", _fake_synthesis)
    monkeypatch.setattr(execute_runner, "load_selected_samples", lambda benchmark, split_name: [sample])
    monkeypatch.setattr(execute_runner, "estimate_work", lambda *args: (10, 5))
    monkeypatch.setattr(execute_runner, "OpenAICompatibleProvider", _FakeProvider)
    monkeypatch.setenv("RESEARCH_REPORTS_ROOT", str(tmp_path / "reports"))

    run_dir = execute_runner.run_experiment(
        experiment,
        "count20_seed42",
        resolve_model("dashscope/qwen-flash"),
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )

    validation = json.loads((run_dir / "run_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"]
    assert (run_dir / "diagnostics" / "rcta_diagnostics.json").exists()
    assert (run_dir / "exports" / "rcta_comparison.json").exists()


def _fake_stage_pool(**kwargs) -> list[dict]:
    del kwargs
    answers = ["yes", "yes", "yes", "no", "no", "yes", "no", "maybe", "yes"]
    return [
        {
            "method_name": "rcta_stage_a_shared",
            "method_type": "rcta_stage_a",
            "agent_id": agent_id,
            "normalized_answer": answer,
            "prediction": answer,
            "validated_output": {"reasoning": f"reason {agent_id}", "final_answer": answer},
            "assistant_text": f"REASONING: reason {agent_id}\nFINAL_ANSWER: {answer}",
            "output_status": "ok",
            "protocol_parse_status": "ok",
            "reason_present": True,
            "prompt_hash": f"hash-{agent_id}",
            "prompt_tokens": 10.0,
            "completion_tokens": 5.0,
            "total_tokens": 15.0,
            "latency_ms": 2.0,
            "network_attempt_count": 0,
            "cache_hit": True,
            "payload": {"seed": 41 + agent_id},
        }
        for agent_id, answer in enumerate(answers, start=1)
    ]


def _fake_synthesis(**kwargs) -> dict:
    del kwargs
    payload = {
        "reasoning_summary": "The majority trace is consistent.",
        "final_answer": "yes",
        "source_trace_ids": ["T1", "T2"],
        "decisive_claim": "direct check",
        "certificate_type": "unsupported",
        "certificate_payload": {},
    }
    return {
        "method_name": "rcta_trace_synthesizer",
        "method_type": "rcta_synthesis",
        "agent_id": 1,
        "normalized_answer": "yes",
        "prediction": "yes",
        "validated_output": payload,
        "assistant_text": json.dumps(payload),
        "output_status": "ok",
        "protocol_parse_status": "ok",
        "reason_present": True,
        "prompt_hash": "synth-hash",
        "prompt_tokens": 10.0,
        "completion_tokens": 5.0,
        "total_tokens": 15.0,
        "latency_ms": 2.0,
        "network_attempt_count": 0,
        "cache_hit": True,
        "payload": {"seed": 20042},
    }


class _FakeProvider:
    def __init__(self, backbone) -> None:
        del backbone

    def close(self) -> None:
        return None
