from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.econ.config import load_experiment_config
from research_experiments.families.econ.run.execute import run_experiment
from research_experiments.families.econ.run.report import render_report
from research_experiments.families.econ.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_econ_render_report_outputs_markdown_and_figures(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-16T12:00:00+00:00",
            "experiment_name": "econ_same_context_main",
            "phase_name": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "econ_bne_v1",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "vote_mv3",
                    "accuracy_mean": 0.40,
                    "total_tokens_mean": 120.0,
                    "communication_tokens_mean": 0.0,
                    "calls_per_question_mean": 3.0,
                    "correction_rate": 0.0,
                    "degradation_rate": 0.0,
                    "gain_over_vote_mv3": None,
                    "token_ratio_over_full_comm": 0.6,
                    "keep_local_rate": 0.0,
                    "adopt_vote_rate": 1.0,
                    "query_best_peer_rate": 0.0,
                    "query_two_peers_rate": 0.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "econ_full_comm_r1",
                    "accuracy_mean": 0.50,
                    "total_tokens_mean": 220.0,
                    "communication_tokens_mean": 60.0,
                    "calls_per_question_mean": 6.0,
                    "correction_rate": 0.1,
                    "degradation_rate": 0.0,
                    "gain_over_vote_mv3": 0.1,
                    "token_ratio_over_full_comm": None,
                    "keep_local_rate": 0.0,
                    "adopt_vote_rate": 0.0,
                    "query_best_peer_rate": 0.0,
                    "query_two_peers_rate": 0.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "econ_bne_main",
                    "accuracy_mean": 0.48,
                    "total_tokens_mean": 170.0,
                    "communication_tokens_mean": 25.0,
                    "calls_per_question_mean": 4.0,
                    "correction_rate": 0.08,
                    "degradation_rate": 0.02,
                    "gain_over_vote_mv3": 0.08,
                    "token_ratio_over_full_comm": 0.772727,
                    "keep_local_rate": 0.2,
                    "adopt_vote_rate": 0.2,
                    "query_best_peer_rate": 0.6,
                    "query_two_peers_rate": 0.0,
                },
                {
                    "dataset": "gsm8k",
                    "method_name": "econ_bne_main",
                    "accuracy_mean": 0.50,
                    "correction_rate": 0.1,
                    "degradation_rate": 0.0,
                    "gain_over_vote_mv3": 0.1,
                    "keep_local_rate": 0.0,
                    "adopt_vote_rate": 0.0,
                    "query_best_peer_rate": 1.0,
                    "query_two_peers_rate": 0.0,
                },
            ]
        },
    )
    write_jsonl(
        tmp_path / "belief_trace.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-0",
                "method_name": "econ_bne_main",
                "agreement_ratio": 0.67,
                "rationale_conflict": 0.30,
            }
        ],
    )
    write_jsonl(
        tmp_path / "equilibrium_trace.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-0",
                "method_name": "econ_bne_main",
                "selected_action": "query_best_peer",
                "belief_score": 0.42,
                "expected_gain": 0.35,
                "communication_cost": 0.18,
            }
        ],
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "ECON 低通信协调复现报告" in local_report
    assert "econ_bne_main" in local_report
    assert "vote_mv3" in local_report
    assert Path(payload["figure_manifest"]).exists()


def test_econ_validate_run_accepts_complete_artifacts(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "demo", "experiment_name": "econ_same_context_main"})
    write_jsonl(tmp_path / "agent_turns.jsonl", [{"method_name": "shared_stage_a", "output_status": "ok"}])
    write_jsonl(tmp_path / "belief_trace.jsonl", [{"method_name": "econ_bne_main"}])
    write_jsonl(tmp_path / "equilibrium_trace.jsonl", [{"method_name": "econ_bne_main"}])
    write_jsonl(tmp_path / "communication_trace.jsonl", [{"method_name": "econ_full_comm_r1"}])
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "method_name": "econ_bne_main",
                "initial_answer": "4",
                "final_answer": "4",
                "selected_action": "query_best_peer",
                "belief_score": 0.42,
                "expected_gain": 0.35,
                "communication_cost": 0.18,
                "changed_after_coordination": False,
                "coordination_mode": "belief_equilibrium",
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True


def test_econ_run_experiment_executes_minimal_flow(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "gsm8k"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test.jsonl").write_text(
        json.dumps(
            {
                "question": "what is 2 + 2?",
                "answer": "reasoning\n#### 4",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "gsm8k.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "GSM8K"',
                'slug = "gsm8k"',
                'loader = "gsm8k_jsonl"',
                'source_path = "gsm8k/test.jsonl"',
                'source_split = "test"',
                'sample_id_prefix = "gsm8k"',
                'question_field = "question"',
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
    experiment_path = tmp_path / "econ_same_context_main.toml"
    experiment_path.write_text(
        "\n".join(
            [
                'name = "econ_same_context_main"',
                'description = "test"',
                f'benchmark_configs = ["{benchmark_path.as_posix()}"]',
                'protocol = "configs/families/econ/protocols/paper_main.toml"',
                "global_seed = 42",
                'prompt_version = "econ_bne_v1"',
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
                'name = "vote_mv3"',
                'mode = "vote_mv3"',
                "",
                "[[methods]]",
                'name = "econ_full_comm_r1"',
                'mode = "econ_full_comm_r1"',
                "",
                "[[methods]]",
                'name = "econ_bne_main"',
                'mode = "econ_bne_main"',
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
    monkeypatch.setenv("RESEARCH_DATASETS_ROOT", (tmp_path / "datasets").as_posix())
    monkeypatch.setattr(
        "research_experiments.families.econ.run.execute._load_selected_samples",
        lambda benchmark, split_name: __import__("research_experiments.core.data.datasets", fromlist=["load_samples"]).load_samples(benchmark),
    )
    monkeypatch.setattr(
        "research_experiments.families.econ.run.execute._estimate_work",
        lambda experiment, phase_name, benchmarks, protocol, methods: (10, 4),
    )
    monkeypatch.setattr(
        "research_experiments.families.econ.run.execute.render_report",
        lambda run_dir: __import__(
            "research_experiments.families.econ.run.report",
            fromlist=["render_report"],
        ).render_report(run_dir, publish_dir=tmp_path / "published"),
    )

    def _fake_completion(provider, payload, limiter=None):
        user_text = str(payload["messages"][-1]["content"])
        if "Selected coordination action" in user_text:
            if "agent_1" in user_text:
                assistant_text = '{"changed_answer": true, "new_answer": "4", "confidence_delta": 0.10, "reason_for_change": "peer shows the missing step", "remaining_disagreement": ""}'
            else:
                assistant_text = '{"changed_answer": false, "new_answer": "4", "confidence_delta": 0.00, "reason_for_change": "", "remaining_disagreement": ""}'
        elif "single strong baseline" in user_text:
            assistant_text = '{"final_answer":"4","reasoning_trace":"2 plus 2 is 4.","claim_span":"2 + 2 = 4","key_evidence":"basic arithmetic","keyword_clues":["2+2","arithmetic"],"confidence_raw":0.95,"uncertain_point":"none"}'
        elif "agent_1" in user_text:
            assistant_text = '{"final_answer":"5","reasoning_trace":"I miscounted.","claim_span":"2 + 2 = 5","key_evidence":"counted wrongly","keyword_clues":["count"],"confidence_raw":0.55,"uncertain_point":"calculation"}'
        else:
            assistant_text = '{"final_answer":"4","reasoning_trace":"2 plus 2 is 4.","claim_span":"2 + 2 = 4","key_evidence":"basic arithmetic","keyword_clues":["2+2","arithmetic"],"confidence_raw":0.85,"uncertain_point":"none"}'
        return {
            "http_status": 200,
            "raw_payload": {"id": "mock"},
            "assistant_text": assistant_text,
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
    assert len(predictions) == 4
    bne_row = next(row for row in predictions if row["method_name"] == "econ_bne_main")
    assert bne_row["score"] == 1.0
    assert bne_row["selected_action"] in {"adopt_vote", "query_best_peer", "query_two_peers"}
