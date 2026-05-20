from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.dmad.config import load_experiment_config
from research_experiments.families.dmad.run.execute import run_experiment
from research_experiments.families.dmad.run.report import render_report
from research_experiments.families.dmad.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_dmad_render_report_outputs_markdown_and_figures(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-19T12:00:00+00:00",
            "experiment_name": "dmad_reasoning_main",
            "phase_name": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "dmad_v1_json",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "single_agent_cot",
                    "method_type": "single_agent",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "accuracy_mean": 0.60,
                    "total_tokens_mean": 100.0,
                    "communication_tokens_mean": 0.0,
                    "calls_per_question_mean": 1.0,
                    "correction_rate": 0.0,
                    "degradation_rate": 0.0,
                    "accuracy_per_1k_tokens": 6.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "single_agent_reflection_r1",
                    "method_type": "reflection",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "accuracy_mean": 0.67,
                    "total_tokens_mean": 160.0,
                    "communication_tokens_mean": 0.0,
                    "calls_per_question_mean": 3.0,
                    "correction_rate": 0.08,
                    "degradation_rate": 0.01,
                    "accuracy_per_1k_tokens": 4.1875,
                },
                {
                    "dataset": "overall",
                    "method_name": "vanilla_mad_r1",
                    "method_type": "mad",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "accuracy_mean": 0.70,
                    "total_tokens_mean": 300.0,
                    "communication_tokens_mean": 120.0,
                    "calls_per_question_mean": 13.0,
                    "correction_rate": 0.10,
                    "degradation_rate": 0.02,
                    "accuracy_per_1k_tokens": 2.3333,
                },
                {
                    "dataset": "overall",
                    "method_name": "persona_diverse_mad_r1",
                    "method_type": "mad",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "accuracy_mean": 0.72,
                    "total_tokens_mean": 310.0,
                    "communication_tokens_mean": 120.0,
                    "calls_per_question_mean": 13.0,
                    "correction_rate": 0.11,
                    "degradation_rate": 0.01,
                    "accuracy_per_1k_tokens": 2.3226,
                },
                {
                    "dataset": "overall",
                    "method_name": "dmad_strategy_diverse_r1",
                    "method_type": "mad",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "accuracy_mean": 0.74,
                    "total_tokens_mean": 305.0,
                    "communication_tokens_mean": 120.0,
                    "calls_per_question_mean": 13.0,
                    "correction_rate": 0.13,
                    "degradation_rate": 0.01,
                    "accuracy_per_1k_tokens": 2.4262,
                },
                {
                    "dataset": "math500",
                    "method_name": "dmad_strategy_diverse_r1",
                    "method_type": "mad",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "accuracy_mean": 0.80,
                    "total_tokens_mean": 320.0,
                    "communication_tokens_mean": 120.0,
                    "calls_per_question_mean": 13.0,
                    "correction_rate": 0.20,
                    "degradation_rate": 0.00,
                    "accuracy_per_1k_tokens": 2.5,
                    "gain_over_vanilla_mad": 0.05,
                    "gain_over_persona_diverse": 0.03,
                },
            ]
        },
    )
    write_json(
        tmp_path / "strategy_diagnostics.json",
        {
            "rows": [
                {
                    "dataset": "overall",
                    "method_name": "vanilla_mad_r1",
                    "diversity_mode": "homogeneous",
                    "strategy_name": "cot",
                    "initial_disagreement_rate": 0.30,
                    "final_consensus_rate": 0.70,
                    "changed_after_debate_rate": 0.20,
                    "correction_rate": 0.10,
                    "degradation_rate": 0.02,
                    "accuracy_mean": 0.70,
                    "calls_per_question_mean": 13.0,
                    "gain_over_vanilla_mad": None,
                    "gain_over_persona_diverse": None,
                },
                {
                    "dataset": "overall",
                    "method_name": "persona_diverse_mad_r1",
                    "diversity_mode": "persona_diverse",
                    "strategy_name": "cot",
                    "initial_disagreement_rate": 0.35,
                    "final_consensus_rate": 0.68,
                    "changed_after_debate_rate": 0.22,
                    "correction_rate": 0.11,
                    "degradation_rate": 0.01,
                    "accuracy_mean": 0.72,
                    "calls_per_question_mean": 13.0,
                    "gain_over_vanilla_mad": 0.02,
                    "gain_over_persona_diverse": None,
                },
                {
                    "dataset": "overall",
                    "method_name": "dmad_strategy_diverse_r1",
                    "diversity_mode": "strategy_diverse",
                    "strategy_name": "cot, pot_l2m, sbp",
                    "initial_disagreement_rate": 0.45,
                    "final_consensus_rate": 0.75,
                    "changed_after_debate_rate": 0.25,
                    "correction_rate": 0.13,
                    "degradation_rate": 0.01,
                    "accuracy_mean": 0.74,
                    "calls_per_question_mean": 13.0,
                    "gain_over_vanilla_mad": 0.04,
                    "gain_over_persona_diverse": 0.02,
                },
            ]
        },
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "DMAD 多样化多智能体辩论复现报告" in local_report
    assert "dmad_strategy_diverse_r1" in local_report
    assert "策略异质化是否优于表面 persona 多样化" in local_report
    assert Path(payload["figure_manifest"]).exists()


def test_dmad_validate_run_accepts_complete_artifacts(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "demo", "experiment_name": "dmad_reasoning_main"})
    write_jsonl(
        tmp_path / "agent_turns.jsonl",
        [
            {
                "dataset": "math500",
                "sample_id": "s1",
                "method_name": "dmad_strategy_diverse_r1",
                "output_status": "ok",
            }
        ],
    )
    write_jsonl(tmp_path / "debate_messages.jsonl", [{"dataset": "math500", "sample_id": "s1"}])
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "dataset": "math500",
                "sample_id": "s1",
                "method_name": "dmad_strategy_diverse_r1",
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "strategy_diagnostics.json", {"rows": [{"dataset": "overall"}]})
    write_json(tmp_path / "cost_breakdown.json", {"rows": []})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True
    assert payload["request_failures_total"] == 0


def test_dmad_run_experiment_executes_minimal_flow(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "math500"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test.jsonl").write_text(
        json.dumps({"problem": "What is 2 + 2?", "answer": "4"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "math500.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "MATH500"',
                'slug = "math500"',
                'loader = "math500_jsonl"',
                'source_path = "math500/test.jsonl"',
                'source_split = "test"',
                'sample_id_prefix = "math500"',
                'question_field = "problem"',
                'answer_field = "answer"',
                "smoke_size = 1",
                "pilot_size = 1",
                "main_size = 1",
                "random_seed = 42",
                'notes = ""',
                'split_presets = [{ name = "count20_seed42", strategy = "ordered", size = 1 }]',
            ]
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "dmad_reasoning_main.toml"
    experiment_path.write_text(
        "\n".join(
            [
                'name = "dmad_reasoning_main"',
                'description = "test"',
                f'benchmark_configs = ["{benchmark_path.as_posix()}"]',
                'protocol = "configs/families/dmad/protocols/dmad_3a_r1.toml"',
                'control_catalog = "configs/core/shared/controls/no_comm_equal_budget.toml"',
                "global_seed = 42",
                'prompt_version = "dmad_v1_json"',
                "max_concurrent_requests = 90",
                "requests_per_minute_limit = 95",
                "tokens_per_minute_limit = 9000000",
                'primary_model_ref = "mock/mock-model"',
                "",
                "[[methods]]",
                'name = "single_agent_cot"',
                'mode = "single_agent_cot"',
                "",
                "[[methods]]",
                'name = "single_agent_reflection_r1"',
                'mode = "single_agent_reflection_r1"',
                "",
                "[[methods]]",
                'name = "mv_6"',
                'mode = "mv_6"',
                "",
                "[[methods]]",
                'name = "vanilla_mad_r1"',
                'mode = "vanilla_mad_r1"',
                'roster = "configs/families/dmad/rosters/homogeneous_3agent.toml"',
                "",
                "[[methods]]",
                'name = "persona_diverse_mad_r1"',
                'mode = "persona_diverse_mad_r1"',
                'roster = "configs/families/dmad/rosters/persona_diverse_3agent.toml"',
                "",
                "[[methods]]",
                'name = "dmad_strategy_diverse_r1"',
                'mode = "dmad_strategy_diverse_r1"',
                'roster = "configs/families/dmad/rosters/strategy_diverse_3agent.toml"',
                "",
                "[phases.count20]",
                'split_overrides = { math500 = "count20_seed42" }',
            ]
        ),
        encoding="utf-8",
    )

    experiment = load_experiment_config(experiment_path)
    backbone = ResolvedModelConfig(
        name="mock/mock-model",
        provider="mock",
        model_id="mock-model",
        base_url="https://example.invalid",
        api_key_env="API_KEY",
        chat_path="/chat/completions",
        default_temperature=0.0,
        default_top_p=1.0,
        default_max_output_tokens=256,
        reasoning_effort="none",
        supports_response_format=True,
        response_format="json_object",
        timeout_seconds=30,
        max_retries=1,
        tags=["test"],
    )
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("RESEARCH_DATASETS_ROOT", (tmp_path / "datasets").as_posix())
    monkeypatch.setattr(
        "research_experiments.families.dmad.run.execute._load_selected_samples",
        lambda benchmark, split_name: __import__("research_experiments.core.data.datasets", fromlist=["load_samples"]).load_samples(benchmark),
    )
    monkeypatch.setattr(
        "research_experiments.families.dmad.run.execute._estimate_work",
        lambda experiment, phase_name, benchmarks, protocol, methods, rosters, controls: (27, 6),
    )
    monkeypatch.setattr(
        "research_experiments.families.dmad.run.execute.render_report",
        lambda run_dir: __import__(
            "research_experiments.families.dmad.run.report",
            fromlist=["render_report"],
        ).render_report(run_dir, publish_dir=tmp_path / "published"),
    )

    def _fake_completion(provider, payload, limiter=None):
        return {
            "http_status": 200,
            "raw_payload": {"id": "mock"},
            "assistant_text": '{"final_answer":"4","reasoning":"2 plus 2 equals 4."}',
            "provider_reasoning_text": "",
            "finish_reason": "stop",
            "usage_reported": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "usage_estimated": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "usage_source": "reported",
            "latency_ms": 1.0,
            "provider_request_id": "req-mock",
            "response_id": "resp-mock",
            "request_error": None,
        }

    monkeypatch.setattr("research_experiments.core.execution.runner_common.execute_completion_request", _fake_completion)

    run_dir = run_experiment(
        experiment=experiment,
        phase_name="count20",
        backbone=backbone,
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )

    validation = validate_run(run_dir)
    predictions = [json.loads(line) for line in (run_dir / "final_predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    assert validation["passed"] is True
    assert len(predictions) == 6
    assert (run_dir / "strategy_diagnostics.json").exists()
