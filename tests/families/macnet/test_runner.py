from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.macnet.config import load_experiment_config
from research_experiments.families.macnet.run.execute import run_experiment
from research_experiments.families.macnet.run.validate import validate_run
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_macnet_validate_run_accepts_complete_artifacts(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "demo", "experiment_name": "macnet_paper_main"})
    write_jsonl(tmp_path / "artifact_trace.jsonl", [{"method_name": "single_agent_cot", "output_status": "ok"}])
    write_jsonl(tmp_path / "instruction_trace.jsonl", [{"method_name": "macnet_chain", "output_status": "ok"}])
    write_jsonl(
        tmp_path / "final_predictions.jsonl",
        [
            {
                "method_name": "macnet_chain",
                "topology_type": "chain",
                "node_scale": 4,
                "dataset": "mmlu",
                "sample_id": "mmlu-00000",
                "initial_artifact": "draft",
                "final_artifact": "final",
                "final_answer": "C",
                "artifact_revision_count": 3,
                "inbound_instruction_count": 3,
                "max_context_tokens_observed": 64.0,
                "topology_direction_mode": "divergent",
            }
        ],
    )
    write_json(tmp_path / "topology_manifest.json", {"topologies": []})
    write_json(tmp_path / "scaling_summary.json", {"series": []})
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "mmlu"}]})
    touch_figure_contract(tmp_path)

    payload = validate_run(tmp_path)

    assert payload["passed"] is True


def test_macnet_run_experiment_executes_minimal_flow(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "mmlu"
    dataset_root.mkdir(parents=True)
    benchmark_path = tmp_path / "mmlu.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "MMLU"',
                'slug = "mmlu"',
                'loader = "mmlu_parquet"',
                'source_path = "mmlu/test.parquet"',
                'source_split = "test"',
                'sample_id_prefix = "mmlu"',
                'question_field = "question"',
                'answer_field = "answer"',
                'options_field = "choices"',
                'answer_index_field = "answer"',
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
    experiment_path = tmp_path / "macnet.toml"
    experiment_path.write_text(
        "\n".join(
            [
                'name = "macnet_paper_main"',
                'description = "test"',
                'experiment_kind = "paper"',
                f'benchmark_configs = ["{benchmark_path.as_posix()}"]',
                'protocol = "configs/families/macnet/protocols/paper_main.toml"',
                "global_seed = 42",
                'prompt_version = "macnet_paper_v1"',
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
                'name = "macnet_chain"',
                'mode = "macnet_topology"',
                'topology_type = "chain"',
                "",
                "[phases.count20]",
                'split_suffix = "count20_seed42"',
                "node_scale = 2",
                'direction_mode = "divergent"',
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
        "research_experiments.families.macnet.run.execute._load_selected_samples",
        lambda benchmark, split_name: [
            __import__("research_experiments.core.data.datasets", fromlist=["DatasetSample"]).DatasetSample(
                dataset="mmlu",
                sample_id="mmlu-00000",
                question="2 + 2 = ?",
                reference_answer="C|||4",
                prompt_context="Options:\nA. 1\nB. 2\nC. 4\nD. 8",
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(
        "research_experiments.families.macnet.run.execute._estimate_work",
        lambda experiment, phase_name, benchmarks, protocol, methods: (8, 2),
    )
    monkeypatch.setattr(
        "research_experiments.families.macnet.run.execute.render_report",
        lambda run_dir: __import__(
            "research_experiments.families.macnet.run.report",
            fromlist=["render_report"],
        ).render_report(run_dir, publish_dir=tmp_path / "published"),
    )

    def _fake_completion(provider, payload, limiter=None):
        user_text = str(payload["messages"][-1]["content"])
        if "Return exactly one JSON object with keys \"instruction\"" in user_text:
            assistant_text = '{"instruction":"Adopt the correct option after checking arithmetic.","focus_risk":"wrong choice","preserve_strength":"keep concise reasoning"}'
        elif "Role: actor_node_1" in user_text:
            assistant_text = '{"artifact":"Revised reasoning: 2 + 2 = 4.","final_answer":"C","reasoning_trace":"follow the instruction and choose C","confidence_raw":0.9}'
        else:
            assistant_text = '{"artifact":"Initial reasoning draft.","final_answer":"C","reasoning_trace":"2 + 2 = 4","confidence_raw":0.8}'
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
    chain_row = next(row for row in predictions if row["method_name"] == "macnet_chain")
    assert chain_row["score"] == 1.0
    assert chain_row["inbound_instruction_count"] == 1
