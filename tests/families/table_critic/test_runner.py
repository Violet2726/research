from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.table_critic.config import load_experiment_config
from research_experiments.families.table_critic.run.execute import run_experiment
from research_experiments.families.table_critic.run.report import render_report
from research_experiments.families.table_critic.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_table_critic_render_report_outputs_markdown_and_figures(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-15T12:00:00+00:00",
            "experiment_name": "table_critic_main",
            "experiment_kind": "paper",
            "phase_name": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "table_critic_paper_v1",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "chain_of_table",
                    "accuracy_mean": 0.31,
                    "total_tokens_mean": 800.0,
                    "calls_per_question_mean": 1.0,
                    "refinement_round_count_mean": 0.0,
                    "correction_rate": 0.0,
                    "degradation_rate": 0.0,
                    "template_reuse_rate": 0.0,
                    "gain_over_chain_of_table": None,
                },
                {
                    "dataset": "overall",
                    "method_name": "table_critic_paper",
                    "accuracy_mean": 0.44,
                    "total_tokens_mean": 1500.0,
                    "calls_per_question_mean": 5.0,
                    "refinement_round_count_mean": 1.2,
                    "correction_rate": 0.4,
                    "degradation_rate": 0.1,
                    "template_reuse_rate": 0.6,
                    "gain_over_chain_of_table": 0.13,
                },
                {
                    "dataset": "wikitq",
                    "method_name": "table_critic_paper",
                    "accuracy_mean": 0.50,
                    "refinement_round_count_mean": 1.1,
                    "correction_rate": 0.5,
                    "degradation_rate": 0.0,
                    "template_reuse_rate": 0.7,
                    "gain_over_chain_of_table": 0.10,
                },
            ]
        },
    )
    write_json(
        tmp_path / "error_analysis.json",
        {
            "summary_rows": [
                {
                    "dataset": "overall",
                    "method_name": "table_critic_paper",
                    "judge_error_detected_rate": 0.8,
                    "correction_rate": 0.4,
                    "degradation_rate": 0.1,
                    "template_reuse_rate": 0.6,
                    "top_error_step": "Calculation Error",
                }
            ],
            "error_step_rows": [],
        },
    )
    write_json(
        tmp_path / "template_tree.json",
        {
            "datasets": {
                "wikitq": {
                    "name": "ROOT",
                    "path": "ROOT",
                    "summaries": [],
                    "children": {
                        "Final Query Error": {
                            "name": "Final Query Error",
                            "path": "ROOT/Final Query Error",
                            "summaries": [
                                {
                                    "template_id": "tpl-00001",
                                    "path": "ROOT/Final Query Error",
                                    "template_title": "Subtotal mismatch",
                                    "pattern_summary": "Arithmetic aggregation used the wrong subtotal.",
                                    "reuse_hint": "Recompute the subtotal before the final claim.",
                                    "usage_count": 2,
                                    "success_count": 1,
                                }
                            ],
                            "children": {},
                        }
                    },
                }
            }
        },
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "## 摘要" in local_report
    assert "Table-Critic 原论文主线复现报告" in local_report
    assert "table_critic_paper" in local_report
    assert "chain_of_table" in local_report
    assert Path(payload["figure_manifest"]).exists()


def test_table_critic_validate_run_accepts_complete_artifacts(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "demo", "experiment_name": "table_critic_main", "experiment_kind": "paper"})
    write_jsonl(tmp_path / "agent_turns.jsonl", [{"method_name": "table_critic_paper", "output_status": "ok"}])
    write_jsonl(
        tmp_path / "critic_trace.jsonl",
        [
            {
                "sample_id": "nu-0",
                "method_name": "table_critic_paper",
                "judge_error_detected": True,
                "judge_error_step": "Calculation Error",
                "critic_feedback": "The subtotal is wrong.",
                "template_ids_used": ["tpl-00001"],
                "template_reuse_count": 1,
            }
        ],
    )
    write_jsonl(
        tmp_path / "refinement_trace.jsonl",
        [
            {
                "sample_id": "nu-0",
                "method_name": "table_critic_paper",
                "round_index": 1,
                "initial_answer": "refuted",
                "previous_answer": "refuted",
                "refined_answer": "entailed",
                "stopped_reason": "judge_passed",
            }
        ],
    )
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "sample_id": "nu-0",
                "method_name": "table_critic_paper",
                "initial_answer": "refuted",
                "final_answer": "entailed",
                "judge_error_detected": True,
                "judge_error_step": "Calculation Error",
                "critic_feedback": "The subtotal is wrong.",
                "refinement_round_count": 1,
                "stopped_reason": "judge_passed",
                "template_ids_used": ["tpl-00001"],
                "table_id": "table-1",
                "question_type": "aggregation",
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "wikitq"}]})
    write_json(tmp_path / "template_tree.json", {"datasets": {}})
    write_json(tmp_path / "error_analysis.json", {"summary_rows": []})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True


def test_table_critic_run_experiment_executes_minimal_flow(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "tabfact"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test.jsonl").write_text(
        json.dumps(
            {
                "statement": "the wildcats kept the opposing team scoreless in four games",
                "label": 1,
                "table_caption": "1947 kentucky wildcats football team",
                "table_text": [
                    ["game", "date", "opponent", "opponents"],
                    ["1", "sept 20", "ole miss", "14"],
                    ["2", "sept 27", "cincinnati", "0"],
                ],
                "table_id": "1-24560733-1.html.csv",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "raw2clean.jsonl").write_text(
        json.dumps(
            {
                "statement": "the wildcats kept the opposing team scoreless in four games",
                "cleaned_statement": "the wildcat keep the oppose team scoreless in 4 game",
                "label": 1,
                "table_caption": "1947 kentucky wildcats football team",
                "table_text": [],
                "table_id": "1-24560733-1.html.csv",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "tabfact.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "TabFact"',
                'slug = "tabfact"',
                'loader = "tabfact_jsonl"',
                'source_path = "tabfact/test.jsonl"',
                'source_split = "test"',
                'sample_id_prefix = "tabfact"',
                'question_field = "statement"',
                'answer_field = "label"',
                "smoke_size = 1",
                "pilot_size = 1",
                "main_size = 1",
                "random_seed = 42",
                'notes = ""',
                'split_presets = [{ name = "count20_seed42", strategy = "ordered", size = 1 }, { name = "full_seed42", strategy = "full" }]',
            ]
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "table_critic_main.toml"
    experiment_path.write_text(
        "\n".join(
            [
                'name = "table_critic_main"',
                'description = "test"',
                'experiment_kind = "paper"',
                f'benchmark_configs = ["{benchmark_path.as_posix()}"]',
                'protocol = "configs/families/table_critic/protocols/paper_main.toml"',
                "global_seed = 42",
                'prompt_version = "table_critic_paper_v1"',
                "max_concurrent_requests = 1",
                "requests_per_minute_limit = 50",
                "tokens_per_minute_limit = 1000000",
                'primary_model_ref = "mock/mock-model"',
                "",
                "[[methods]]",
                'name = "end_to_end_qa"',
                'mode = "direct_qa"',
                "",
                "[[methods]]",
                'name = "few_shot_qa"',
                'mode = "few_shot_qa"',
                "",
                "[[methods]]",
                'name = "chain_of_table"',
                'mode = "chain_of_table"',
                "",
                "[[methods]]",
                'name = "critic_cot"',
                'mode = "critic_cot"',
                "",
                "[[methods]]",
                'name = "table_critic_paper"',
                'mode = "table_critic_paper"',
                'matched_controls = ["chain_of_table"]',
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
        "research_experiments.families.table_critic.run.execute._load_selected_samples",
        lambda benchmark, split_name: __import__("research_experiments.core.data.datasets", fromlist=["load_samples"]).load_samples(benchmark),
    )
    monkeypatch.setattr(
        "research_experiments.families.table_critic.run.execute._estimate_work",
        lambda experiment, phase_name, benchmarks, protocol, methods: (20, 5),
    )
    monkeypatch.setattr(
        "research_experiments.families.table_critic.run.execute.render_report",
        lambda run_dir: __import__(
            "research_experiments.families.table_critic.run.report",
            fromlist=["render_report"],
        ).render_report(run_dir, publish_dir=tmp_path / "published"),
    )

    def _fake_completion(provider, payload, limiter=None):
        user_text = str(payload["messages"][-1]["content"])
        if "Judge signal" in user_text and "Critic feedback" in user_text:
            assistant_text = '{"reasoning":"After rechecking the table, four scoreless games are supported.","final_answer":"entailed"}'
        elif "Summarize this correction pattern for reuse" in user_text:
            assistant_text = '{"pattern_summary":"Verify zero-opponent rows before making a count claim.","reuse_hint":"Count the rows whose opponent score is 0.","template_title":"Zero-score count"}'
        elif "Give a structured critique" in user_text:
            assistant_text = '{"critic_feedback":"The current reasoning undercounts the rows where the opponent scored 0.","conflicting_evidence":["row 2 shows opponent score 0"],"repair_hint":"Count all rows where the opponent column is 0 before answering.","error_category":"Calculation Error"}'
        elif "Decide whether the current answer should pass" in user_text:
            if "Current answer:\nentailed" in user_text:
                assistant_text = '{"passed": true, "error_detected": false, "error_step": "None", "node_path": ["ROOT"], "rationale": "The revised answer matches the table evidence."}'
            else:
                assistant_text = '{"passed": false, "error_detected": true, "error_step": "Calculation Error", "node_path": ["ROOT", "Final Query Error", "Calculation Error"], "rationale": "The count does not use every scoreless row."}'
        elif "Chain-of-Table style" in str(payload["messages"][0]["content"]):
            assistant_text = '{"reasoning":"The current row selection is incomplete, so the statement looks unsupported.","final_answer":"refuted"}'
        else:
            assistant_text = '{"reasoning":"The table supports the statement.","final_answer":"entailed"}'
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
    template_tree = json.loads((run_dir / "template_tree.json").read_text(encoding="utf-8"))

    assert validation["passed"] is True
    assert len(predictions) == 5
    paper_row = next(row for row in predictions if row["method_name"] == "table_critic_paper")
    assert paper_row["score"] == 1.0
    assert paper_row["refinement_round_count"] >= 1
    assert template_tree["datasets"]["tabfact"]["children"]["Final Query Error"]["children"]["Calculation Error"]["summaries"]
