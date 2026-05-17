from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.colmad.config import load_experiment_config
from research_experiments.families.colmad.run.execute import run_experiment
from research_experiments.families.colmad.run.report import render_report
from research_experiments.families.colmad.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_colmad_render_report_outputs_markdown_and_figures(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-17T12:00:00+00:00",
            "experiment_name": "colmad_realmistake_main",
            "experiment_kind": "paper",
            "phase_name": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "colmad_paper_v1",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "single_agent_detector",
                    "accuracy_mean": 0.55,
                    "total_tokens_mean": 200.0,
                    "communication_tokens_mean": 0.0,
                    "calls_per_question_mean": 1.0,
                    "correct_to_wrong_shift_rate": 0.0,
                    "wrong_to_correct_shift_rate": 0.0,
                    "competitive_hacking_rate": 0.0,
                    "supportive_critique_rate": 0.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "copmad_competitive",
                    "accuracy_mean": 0.50,
                    "total_tokens_mean": 600.0,
                    "communication_tokens_mean": 450.0,
                    "calls_per_question_mean": 5.0,
                    "correct_to_wrong_shift_rate": 0.20,
                    "wrong_to_correct_shift_rate": 0.05,
                    "competitive_hacking_rate": 0.30,
                    "supportive_critique_rate": 0.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "colmad_collaborative",
                    "accuracy_mean": 0.62,
                    "total_tokens_mean": 680.0,
                    "communication_tokens_mean": 470.0,
                    "calls_per_question_mean": 5.0,
                    "correct_to_wrong_shift_rate": 0.05,
                    "wrong_to_correct_shift_rate": 0.18,
                    "competitive_hacking_rate": 0.08,
                    "supportive_critique_rate": 0.80,
                },
            ]
        },
    )
    write_json(
        tmp_path / "protocol_diagnostics.json",
        {
            "summary_rows": [
                {
                    "dataset": "overall",
                    "method_name": "copmad_competitive",
                    "competitive_hacking_rate": 0.30,
                    "supportive_critique_rate": 0.00,
                    "correct_to_wrong_shift_rate": 0.20,
                    "wrong_to_correct_shift_rate": 0.05,
                    "judge_disagreement_rate": 0.20,
                    "evidence_complementarity_rate": 0.05,
                    "fake_evidence_rate": 0.10,
                    "overconfident_claim_rate": 0.15,
                    "fallacious_argument_rate": 0.05,
                },
                {
                    "dataset": "overall",
                    "method_name": "colmad_collaborative",
                    "competitive_hacking_rate": 0.08,
                    "supportive_critique_rate": 0.80,
                    "correct_to_wrong_shift_rate": 0.05,
                    "wrong_to_correct_shift_rate": 0.18,
                    "judge_disagreement_rate": 0.10,
                    "evidence_complementarity_rate": 0.72,
                    "fake_evidence_rate": 0.02,
                    "overconfident_claim_rate": 0.04,
                    "fallacious_argument_rate": 0.02,
                },
            ]
        },
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "ColMAD 协作监督式多智能体辩论复现报告" in local_report
    assert "colmad_collaborative" in local_report
    assert Path(payload["figure_manifest"]).exists()


def test_colmad_validate_run_accepts_complete_artifacts(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "demo", "experiment_name": "colmad_realmistake_main"})
    write_jsonl(tmp_path / "debate_trace.jsonl", [{"method_name": "copmad_competitive", "output_status": "ok"}])
    write_jsonl(
        tmp_path / "judge_trace.jsonl",
        [
            {
                "method_name": "colmad_collaborative",
                "debate_protocol": "collaborative",
                "verdict": "contains_error",
                "observed_failure_modes": [],
            }
        ],
    )
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "sample_id": "demo",
                "task_name": "math_problem_generation",
                "candidate_response_model": "gpt-4-0613",
                "gold": "error",
                "final_verdict": "contains_error",
                "single_agent_verdict": "contains_no_error",
                "copmad_verdict": "contains_error",
                "colmad_verdict": "contains_error",
                "changed_after_debate": True,
                "shift_direction": "wrong_to_correct",
                "judge_confidence": 0.8,
                "debate_protocol": "collaborative",
                "method_name": "colmad_collaborative",
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "protocol_diagnostics.json", {"summary_rows": []})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True


def test_colmad_run_experiment_executes_minimal_flow(monkeypatch, tmp_path: Path) -> None:
    benchmark_path = tmp_path / "realmistake_math.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "ReaLMistake Math Problem Generation"',
                'slug = "realmistake_math_problem_generation"',
                'loader = "realmistake_error_detection_zip"',
                'source_path = "realmistake/data.zip"',
                'source_split = "math_word_problem_generation"',
                'sample_id_prefix = "realmistake_math"',
                'question_field = "input"',
                'answer_field = "error_label"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
                'archive_member = "data/math_word_problem_generation"',
                'cache_namespace_override = "realmistake/math_problem_generation"',
                'split_presets = [{ name = "count20_seed42", strategy = "ordered", size = 1 }]',
            ]
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "colmad.toml"
    experiment_path.write_text(
        "\n".join(
            [
                'name = "colmad_realmistake_main"',
                'description = "test"',
                'experiment_kind = "paper"',
                f'benchmark_configs = ["{benchmark_path.as_posix()}"]',
                'protocol = "configs/families/colmad/protocols/paper_main.toml"',
                "global_seed = 42",
                'prompt_version = "colmad_paper_v1"',
                "max_concurrent_requests = 1",
                "requests_per_minute_limit = 50",
                "tokens_per_minute_limit = 1000000",
                'primary_model_ref = "mock/mock-model"',
                "",
                "[[methods]]",
                'name = "single_agent_detector"',
                'mode = "single_agent_detector"',
                "",
                "[[methods]]",
                'name = "copmad_competitive"',
                'mode = "copmad_competitive"',
                "",
                "[[methods]]",
                'name = "colmad_collaborative"',
                'mode = "colmad_collaborative"',
                "",
                "[phases.count20]",
                'split_suffix = "count20_seed42"',
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
    monkeypatch.setattr(
        "research_experiments.families.colmad.run.execute._load_selected_samples",
        lambda benchmark, split_name: [],
    )
    monkeypatch.setattr(
        "research_experiments.families.colmad.run.execute._estimate_work",
        lambda experiment, phase_name, benchmarks, methods: (11, 3),
    )
    monkeypatch.setattr(
        "research_experiments.families.colmad.run.execute._run_sample_batch",
        lambda **kwargs: [
            __import__("research_experiments.families.colmad.run.sample", fromlist=["SampleResult"]).SampleResult(
                debate_rows=[
                    {
                        "method_name": "single_agent_detector",
                        "output_status": "ok",
                        "prompt_tokens": 10.0,
                        "completion_tokens": 2.0,
                        "total_tokens": 12.0,
                        "latency_ms": 5.0,
                    }
                ],
                judge_rows=[],
                prediction_rows=[
                    {
                        "sample_id": "demo",
                        "dataset": "realmistake_math_problem_generation",
                        "method_name": "single_agent_detector",
                        "task_name": "math_problem_generation",
                        "candidate_response_model": "gpt-4-0613",
                        "gold": "error",
                        "final_verdict": "contains_error",
                        "single_agent_verdict": "contains_error",
                        "copmad_verdict": "contains_error",
                        "colmad_verdict": "contains_error",
                    }
                ],
            )
        ],
    )
    monkeypatch.setattr(
        "research_experiments.families.colmad.run.execute.render_report",
        lambda run_dir: __import__(
            "research_experiments.families.colmad.run.report",
            fromlist=["render_report"],
        ).render_report(run_dir, publish_dir=tmp_path / "published"),
    )

    run_dir = run_experiment(experiment, "count20", backbone, run_root=tmp_path / "runs", cache_root=tmp_path / "cache")

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "protocol_diagnostics.json").exists()
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "colmad_realmistake_main"
