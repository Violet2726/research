from __future__ import annotations

import json
from pathlib import Path
import zipfile

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.dmad.config import load_experiment_config
from research_experiments.families.dmad.run.execute import run_experiment
from research_experiments.families.dmad.run.report import render_report
from research_experiments.families.dmad.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_dmad_render_report_uses_new_paper_scope_and_tables(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-19T12:00:00+00:00",
            "experiment_name": "dmad_reasoning_main",
            "phase_name": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "dmad_v1_json",
            "evaluation_scope": "paper_main",
            "paper_alignment_version": "dmad_iclr2025_llm_text_v1",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "cot",
                    "method_type": "single_agent",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "configured_strategy_name": "cot",
                    "effective_strategy_name": "cot",
                    "accuracy_mean": 0.60,
                    "total_tokens_mean": 100.0,
                    "communication_tokens_mean": 0.0,
                    "calls_per_question_mean": 1.0,
                    "correction_rate": 0.0,
                    "degradation_rate": 0.0,
                    "accuracy_per_1k_tokens": 6.0,
                    "gain_over_best_fixed_mad": -0.1,
                },
                {
                    "dataset": "overall",
                    "method_name": "mad_all_cot",
                    "method_type": "mad",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "configured_strategy_name": "cot",
                    "effective_strategy_name": "cot",
                    "accuracy_mean": 0.70,
                    "total_tokens_mean": 300.0,
                    "communication_tokens_mean": 120.0,
                    "calls_per_question_mean": 12.0,
                    "correction_rate": 0.10,
                    "degradation_rate": 0.02,
                    "accuracy_per_1k_tokens": 2.3333,
                    "gain_over_best_fixed_mad": None,
                },
                {
                    "dataset": "overall",
                    "method_name": "mad_all_sbp",
                    "method_type": "mad",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "configured_strategy_name": "sbp",
                    "effective_strategy_name": "sbp",
                    "accuracy_mean": 0.68,
                    "total_tokens_mean": 290.0,
                    "communication_tokens_mean": 120.0,
                    "calls_per_question_mean": 12.0,
                    "correction_rate": 0.08,
                    "degradation_rate": 0.01,
                    "accuracy_per_1k_tokens": 2.3448,
                    "gain_over_best_fixed_mad": -0.02,
                },
                {
                    "dataset": "overall",
                    "method_name": "dmad_cot_sbp_pot",
                    "method_type": "mad",
                    "model_name": "xiaomimimo/mimo-v2.5",
                    "configured_strategy_name": "cot, pot, sbp",
                    "effective_strategy_name": "competition_math=cot, pot, sbp; gpqa_diamond=cot, l2m, sbp",
                    "accuracy_mean": 0.74,
                    "total_tokens_mean": 305.0,
                    "communication_tokens_mean": 120.0,
                    "calls_per_question_mean": 12.0,
                    "correction_rate": 0.13,
                    "degradation_rate": 0.01,
                    "accuracy_per_1k_tokens": 2.4262,
                    "gain_over_best_fixed_mad": 0.04,
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
                    "method_name": "mad_all_cot",
                    "diversity_mode": "homogeneous_cot",
                    "strategy_name": "cot",
                    "configured_strategy_name": "cot",
                    "effective_strategy_name": "cot",
                    "initial_disagreement_rate": 0.30,
                    "final_consensus_rate": 0.70,
                    "changed_after_debate_rate": 0.20,
                    "correction_rate": 0.10,
                    "degradation_rate": 0.02,
                    "accuracy_mean": 0.70,
                    "calls_per_question_mean": 12.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "dmad_cot_sbp_pot",
                    "diversity_mode": "strategy_diverse",
                    "strategy_name": "cot, pot, sbp",
                    "configured_strategy_name": "cot, pot, sbp",
                    "effective_strategy_name": "competition_math=cot, pot, sbp; gpqa_diamond=cot, l2m, sbp",
                    "initial_disagreement_rate": 0.45,
                    "final_consensus_rate": 0.75,
                    "changed_after_debate_rate": 0.25,
                    "correction_rate": 0.13,
                    "degradation_rate": 0.01,
                    "accuracy_mean": 0.74,
                    "calls_per_question_mean": 12.0,
                },
            ],
            "paper_main_gap_rows": [
                {
                    "dataset": "competition_math",
                    "sample_id": "test/counting_and_probability/191.json",
                    "dmad_prediction": "108",
                    "dmad_score": 0.0,
                    "best_fixed_method_name": "mad_all_cot",
                    "best_fixed_prediction": "48",
                    "best_fixed_score": 1.0,
                    "persona_d_prediction": "48",
                    "persona_d_score": 1.0,
                }
            ],
        },
    )
    write_json(
        tmp_path / "paper_tables.json",
        {
            "evaluation_scope": "paper_main",
            "math_subject_rows": [
                {"method_name": "dmad_cot_sbp_pot", "group_name": "algebra", "question_count": 10, "accuracy_mean": 0.8}
            ],
            "gpqa_domain_rows": [
                {"method_name": "dmad_cot_sbp_pot", "group_name": "Biology", "question_count": 10, "accuracy_mean": 0.7}
            ],
            "overall_rows": [
                {"method_name": "dmad_cot_sbp_pot", "group_name": "overall", "question_count": 20, "accuracy_mean": 0.75}
            ],
            "appendix_rows": [],
            "extended_dataset_rows": [],
        },
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "DMAD 论文主线高保真复现报告" in local_report
    assert "dmad_cot_sbp_pot" in local_report
    assert "paper_main" in local_report
    assert Path(payload["figure_manifest"]).exists()


def test_dmad_validate_run_accepts_complete_artifacts(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "demo", "experiment_name": "dmad_reasoning_main"})
    write_jsonl(
        tmp_path / "agent_turns.jsonl",
        [
            {
                "dataset": "competition_math",
                "sample_id": "s1",
                "method_name": "dmad_cot_sbp_pot",
                "output_status": "ok",
            }
        ],
    )
    write_jsonl(tmp_path / "debate_messages.jsonl", [{"dataset": "competition_math", "sample_id": "s1"}])
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "dataset": "competition_math",
                "sample_id": "s1",
                "method_name": "dmad_cot_sbp_pot",
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall"}]})
    write_json(tmp_path / "strategy_diagnostics.json", {"rows": [{"dataset": "overall"}]})
    write_json(tmp_path / "cost_breakdown.json", {"rows": []})
    write_json(tmp_path / "paper_tables.json", {"overall_rows": []})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True
    assert payload["request_failures_total"] == 0


def test_dmad_run_experiment_executes_minimal_flow(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "competition_math"
    dataset_root.mkdir(parents=True)
    zip_path = dataset_root / "MATH.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "MATH/test/algebra/example.json",
            json.dumps(
                {
                    "problem": "What is 2 + 2?",
                    "answer": "4",
                    "solution": "2 + 2 = 4",
                    "level": "Level 1",
                    "type": "algebra",
                },
                ensure_ascii=False,
            ),
        )

    benchmark_path = tmp_path / "competition_math.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "MATH"',
                'slug = "competition_math"',
                'loader = "competition_math_zip"',
                'source_path = "competition_math/MATH.zip"',
                'source_split = "test"',
                'cache_namespace_override = "tests/dmad/minimal_competition_math"',
                'sample_id_prefix = "competition_math"',
                'question_field = "problem"',
                'answer_field = "answer"',
                "smoke_size = 1",
                "pilot_size = 1",
                "main_size = 1",
                "random_seed = 0",
                'notes = ""',
                "",
                "[[split_presets]]",
                'name = "count20_seed0"',
                'strategy = "stratified"',
                'field = "subject"',
                "size = 1",
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
                'evaluation_scope = "paper_main"',
                'paper_alignment_version = "dmad_iclr2025_llm_text_v1"',
                f'benchmark_configs = ["{benchmark_path.as_posix()}"]',
                'protocol = "configs/families/dmad/protocols/dmad_3a_r2.toml"',
                "global_seed = 42",
                'prompt_version = "dmad_v1_json"',
                "max_concurrent_requests = 90",
                "requests_per_minute_limit = 95",
                "tokens_per_minute_limit = 9000000",
                'primary_model_ref = "mock/mock-model"',
                "",
                "[[methods]]",
                'name = "cot"',
                'mode = "single_cot"',
                "",
                "[[methods]]",
                'name = "self_refine"',
                'mode = "self_refine_cot"',
                "",
                "[[methods]]",
                'name = "mad_all_cot"',
                'mode = "mad_all_cot"',
                'roster = "configs/families/dmad/rosters/mad_all_cot_3agent.toml"',
                "",
                "[[methods]]",
                'name = "dmad_cot_sbp_pot"',
                'mode = "dmad_cot_sbp_pot"',
                'roster = "configs/families/dmad/rosters/dmad_cot_sbp_pot_3agent.toml"',
                "",
                "[phases.count20]",
                'split_overrides = { competition_math = "count20_seed0" }',
            ]
        ),
        encoding="utf-8",
    )

    experiment = load_experiment_config(experiment_path)
    splits_root = tmp_path / "splits"
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
        "research_experiments.families.dmad.run.execute._estimate_work",
        lambda experiment, phase_name, benchmarks, protocol, methods, rosters, controls, splits_root: (28, 4),
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
        splits_root=splits_root,
    )

    validation = validate_run(run_dir)
    predictions = [json.loads(line) for line in (run_dir / "final_predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    assert validation["passed"] is True
    assert len(predictions) == 4
    assert (run_dir / "paper_tables.json").exists()
    assert any(row["method_name"] == "dmad_cot_sbp_pot" for row in predictions)
    assert (splits_root / "count20" / "tests" / "dmad" / "minimal_competition_math-seed0.json").exists()
