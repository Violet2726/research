from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.dog_graph.run.execute import run_experiment
from research_experiments.families.dog_graph.config import load_experiment_config
from research_experiments.families.dog_graph.run.report import render_report
from research_experiments.families.dog_graph.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_dog_graph_render_report_outputs_paper_markdown_and_figures(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "created_at": "2026-05-14T12:00:00+00:00",
            "experiment_name": "dog_graph_main",
            "experiment_kind": "paper",
            "phase_name": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "dog_paper_v1",
        },
    )
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "tog_iterative_baseline",
                    "accuracy_mean": 0.31,
                    "total_tokens_mean": 1600.0,
                    "calls_per_question_mean": 4.0,
                    "retrieval_hops_mean": 2.0,
                    "simplification_success_rate": 0.0,
                    "direct_fallback_rate": 0.2,
                    "false_positive_relation_rate": 0.4,
                    "gain_over_baseline": None,
                },
                {
                    "dataset": "overall",
                    "method_name": "dog_graph_paper",
                    "accuracy_mean": 0.44,
                    "total_tokens_mean": 2400.0,
                    "calls_per_question_mean": 7.0,
                    "retrieval_hops_mean": 2.1,
                    "simplification_success_rate": 0.6,
                    "direct_fallback_rate": 0.1,
                    "false_positive_relation_rate": 0.2,
                    "gain_over_baseline": 0.13,
                },
                {
                    "dataset": "webquestions_paper_test",
                    "method_name": "dog_graph_paper",
                    "accuracy_mean": 0.5,
                    "gain_over_baseline": 0.1,
                    "retrieval_hops_mean": 2.0,
                    "simplification_success_rate": 0.7,
                    "direct_fallback_rate": 0.1,
                },
            ]
        },
    )
    write_json(
        tmp_path / "graph_diagnostics.json",
        {
            "summary_rows": [
                {
                    "dataset": "overall",
                    "method_name": "dog_graph_paper",
                    "accuracy_mean": 0.44,
                    "retrieval_hops_mean": 2.1,
                    "simplification_success_rate": 0.6,
                    "direct_fallback_rate": 0.1,
                    "false_positive_relation_rate": 0.2,
                    "enough_answer_yes_rate": 0.5,
                }
            ],
            "view_rows": [
                {
                    "dataset": "overall",
                    "method_name": "dog_graph_paper",
                    "graph_view_kind": "dynamic_retrieval",
                    "turn_count": 20,
                    "subgraph_node_count_mean": 5.0,
                    "subgraph_edge_count_mean": 3.0,
                    "available_triple_count_mean": 3.0,
                    "grounded_turn_rate": 1.0,
                }
            ],
        },
    )

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "## 摘要" in local_report
    assert "DoG 原论文高保真复现报告" in local_report
    assert "dog_graph_paper" in local_report
    assert "tog_iterative_baseline" in local_report
    assert Path(payload["figure_manifest"]).exists()


def test_dog_graph_validate_run_accepts_complete_paper_artifacts(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {"run_id": "demo", "experiment_name": "dog_graph_main", "experiment_kind": "paper"},
    )
    write_jsonl(
        tmp_path / "agent_turns.jsonl",
        [{"dataset": "webquestions_paper_test", "sample_id": "wq-1", "method_name": "dog_graph_paper", "output_status": "ok"}],
    )
    write_jsonl(
        tmp_path / "debate_messages.jsonl",
        [{"dataset": "webquestions_paper_test", "sample_id": "wq-1", "method_name": "dog_graph_paper"}],
    )
    write_jsonl(
        tmp_path / "graph_trace.jsonl",
        [{"graph_view_kind": "dynamic_retrieval", "sample_id": "wq-1", "method_name": "dog_graph_paper"}],
    )
    write_jsonl(
        tmp_path / "retrieval_trace.jsonl",
        [
            {
                "sample_id": "wq-1",
                "method_name": "dog_graph_paper",
                "hop_index": 1,
                "head_entities": ["Jamaica"],
                "selected_relations": ["location.country.languages_spoken"],
                "tail_entities": ["Jamaican English"],
                "reasoning_triples": ["(Jamaica, location.country.languages_spoken, Jamaican English)"],
                "retrieval_backend": "freebase_virtuoso",
            }
        ],
    )
    write_jsonl(
        tmp_path / "relation_selection_trace.jsonl",
        [
            {
                "sample_id": "wq-1",
                "method_name": "dog_graph_paper",
                "candidate_relations": ["location.country.languages_spoken"],
                "selected_relations": ["location.country.languages_spoken"],
                "selector_raw_text": "relation_1",
            }
        ],
    )
    write_jsonl(
        tmp_path / "simplification_trace.jsonl",
        [
            {
                "sample_id": "wq-1",
                "method_name": "dog_graph_paper",
                "original_question": "what does jamaican people speak?",
                "simplified_question": "",
                "reasoning_triples": ["(Jamaica, location.country.languages_spoken, Jamaican English)"],
                "role_outputs": {},
            }
        ],
    )
    write_jsonl(
        tmp_path / "answer_attempt_trace.jsonl",
        [
            {
                "sample_id": "wq-1",
                "method_name": "dog_graph_paper",
                "attempt_kind": "enough_answer",
                "decision": "yes",
                "raw_text": "Yes {Jamaican English}",
                "reasoning_triples": ["(Jamaica, location.country.languages_spoken, Jamaican English)"],
            }
        ],
    )
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "sample_id": "wq-1",
                "method_name": "dog_graph_paper",
                "topic_entity_id": "m.03_r3",
                "hop_index": 1,
                "selected_relations": ["location.country.languages_spoken"],
                "reasoning_triples": ["(Jamaica, location.country.languages_spoken, Jamaican English)"],
                "enough_answer_decision": "yes",
                "simplified_question": "",
                "used_direct_fallback": False,
                "retrieval_backend": "freebase_virtuoso",
            }
        ],
    )
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "webquestions_paper_test"}]})
    write_json(tmp_path / "graph_diagnostics.json", {"summary_rows": [{"dataset": "webquestions_paper_test"}]})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True
    assert payload["experiment_kind"] == "paper"


def test_dog_graph_validate_run_rejects_missing_paper_trace_fields(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {"run_id": "demo", "experiment_name": "dog_graph_main", "experiment_kind": "paper"},
    )
    write_jsonl(tmp_path / "agent_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(tmp_path / "debate_messages.jsonl", [{}])
    write_jsonl(tmp_path / "graph_trace.jsonl", [{"graph_view_kind": "full_subgraph"}])
    write_jsonl(tmp_path / "retrieval_trace.jsonl", [{"hop_index": 1}])
    write_jsonl(tmp_path / "relation_selection_trace.jsonl", [{"candidate_relations": []}])
    write_jsonl(tmp_path / "simplification_trace.jsonl", [{"original_question": "q"}])
    write_jsonl(tmp_path / "answer_attempt_trace.jsonl", [{"attempt_kind": "enough_answer"}])
    write_jsonl(tmp_path / "final_predictions.jsonl", [{"topic_entity_id": "m.1"}])
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "webquestions_paper_test"}]})
    write_json(tmp_path / "graph_diagnostics.json", {"summary_rows": [{"dataset": "webquestions_paper_test"}]})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is False
    assert payload["missing_prediction_fields"]
    assert payload["missing_retrieval_fields"]


def test_dog_graph_run_experiment_executes_minimal_paper_metaqa_flow(monkeypatch, tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    kb_root = datasets_root / "metaqa"
    (kb_root / "2-hop").mkdir(parents=True)
    (kb_root / "kb.txt").write_text(
        "\n".join(
            [
                "Kismet|directed_by|William Dieterle",
                "William Dieterle|born_in|Berlin",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (kb_root / "2-hop" / "test.txt").write_text(
        "what city was the director of [Kismet] born in\tBerlin\n",
        encoding="utf-8",
    )

    benchmark_path = tmp_path / "metaqa_2hop.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "DoG MetaQA 2-hop"',
                'slug = "metaqa_2hop"',
                'loader = "metaqa_txt"',
                'source_path = "metaqa/2-hop/test.txt"',
                'source_split = "test"',
                'sample_id_prefix = "metaqa2"',
                'question_field = "question"',
                'answer_field = "answers"',
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
    experiment_path = tmp_path / "dog_graph_main.toml"
    experiment_path.write_text(
        "\n".join(
            [
                'name = "dog_graph_main"',
                'description = "test"',
                'experiment_kind = "paper"',
                f'benchmark_configs = ["{benchmark_path.as_posix()}"]',
                'protocol = "configs/families/dog_graph/protocols/dog_graph_paper.toml"',
                "global_seed = 42",
                'prompt_version = "dog_paper_v1"',
                "max_concurrent_requests = 1",
                "requests_per_minute_limit = 50",
                "tokens_per_minute_limit = 1000000",
                'primary_model_ref = "mock/mock-model"',
                "",
                "[[methods]]",
                'name = "tog_iterative_baseline"',
                'mode = "iterative_baseline"',
                "",
                "[[methods]]",
                'name = "dog_graph_paper"',
                'mode = "paper_dog"',
                'matched_controls = ["tog_iterative_baseline"]',
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
    monkeypatch.setenv("RESEARCH_DATASETS_ROOT", datasets_root.as_posix())
    monkeypatch.setattr(
        "research_experiments.families.dog_graph.run.execute._load_selected_samples",
        lambda experiment, benchmark, split_name: __import__("research_experiments.core.data.datasets", fromlist=["load_samples"]).load_samples(benchmark),
    )
    monkeypatch.setattr(
        "research_experiments.families.dog_graph.run.execute.render_report",
        lambda run_dir: __import__(
            "research_experiments.families.dog_graph.run.report",
            fromlist=["render_report"],
        ).render_report(run_dir, publish_dir=tmp_path / "published"),
    )

    def _fake_completion(provider, payload, limiter=None):
        text = str(payload["messages"][-1]["content"])
        if "Relations:" in text or "relation_set:" in text:
            assistant_text = "relation_1"
        elif "Given a question and the associated retrieved knowledge graph triples" in text:
            assistant_text = "Yes. Therefore, the answer is {Berlin}."
        elif "Question Simplifying Expert" in text:
            assistant_text = "The resolved entity is William Dieterle. The remaining question is what city was William Dieterle born in"
        elif "You are a serious critic" in text:
            assistant_text = "The simplification is valid. what city was William Dieterle born in"
        elif "simplified question:" in text or "You just need to output the correct simplified problem" in text:
            assistant_text = "simplified question: what city was William Dieterle born in"
        else:
            assistant_text = "Berlin"
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
    assert len(predictions) == 2
    dog_row = next(row for row in predictions if row["method_name"] == "dog_graph_paper")
    assert dog_row["score"] == 1.0
    assert dog_row["retrieval_backend"] == "metaqa_kb"
