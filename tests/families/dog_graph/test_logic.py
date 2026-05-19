from __future__ import annotations

from pathlib import Path

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.dog_graph.config import load_benchmarks, load_experiment_config, load_protocol_config
from research_experiments.families.dog_graph.paper_backend import EntityRef, MetaqaGraphBackend
from research_experiments.families.dog_graph.paper_prompts import (
    parse_enough_answer_output,
    parse_selected_relations,
    parse_simplified_question,
)
from research_experiments.families.dog_graph.run.execute import _cache_dataset_key
from research_experiments.families.dog_graph.run.paper import _build_metrics as _build_paper_metrics
from research_experiments.families.dog_graph.run.sample import _build_metrics as _build_static_metrics
from research_experiments.families.dog_graph.run.sample import _ground_graph_payload, _validate_graph_answer_payload
from research_experiments.families.dog_graph.dataset_views import GraphView


def test_load_dog_graph_main_config_reads_paper_methods_and_protocol() -> None:
    experiment = load_experiment_config("configs/families/dog_graph/experiments/dog_graph_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.name == "dog_graph_main"
    assert experiment.experiment_kind == "paper"
    assert [method.name for method in experiment.methods] == [
        "tog_iterative_baseline",
        "dog_graph_paper",
    ]
    assert protocol.protocol_kind == "paper"
    assert protocol.max_hops == 3
    assert experiment.methods[-1].matched_controls == ["tog_iterative_baseline"]


def test_load_dog_graph_static_config_preserves_legacy_methods() -> None:
    experiment = load_experiment_config("configs/families/dog_graph/experiments/dog_graph_static_ablation.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.experiment_kind == "static"
    assert [method.name for method in experiment.methods] == [
        "single_graph_solver",
        "graph_mv3",
        "dog_graph_r1",
        "dog_graph_r2",
    ]
    assert protocol.protocol_kind == "static"
    assert protocol.agent_count == 3


def test_cache_dataset_key_uses_real_dataset_name_for_paper_line() -> None:
    paper_experiment = load_experiment_config("configs/families/dog_graph/experiments/dog_graph_main.toml")
    static_experiment = load_experiment_config("configs/families/dog_graph/experiments/dog_graph_static_ablation.toml")
    paper_benchmarks = load_benchmarks(paper_experiment)
    static_benchmarks = load_benchmarks(static_experiment)

    assert _cache_dataset_key("paper", next(item for item in paper_benchmarks if item.slug == "webquestions_paper_test")) == "webquestions/paper_test"
    assert _cache_dataset_key("paper", next(item for item in paper_benchmarks if item.slug == "grailqa_test")) == "grailqa/test"
    assert _cache_dataset_key("paper", next(item for item in paper_benchmarks if item.slug == "metaqa_3hop")) == "metaqa/3-hop/test"
    assert _cache_dataset_key("static", next(item for item in static_benchmarks if item.slug == "webquestions")) == "webquestions/test"


def test_parse_selected_relations_accepts_relation_slot_and_literal() -> None:
    relation_mapping = {
        "relation_1": "location.country.languages_spoken",
        "relation_2": "location.country.currency_used",
    }
    assert parse_selected_relations("A: relation_1", relation_mapping, limit=2) == ["location.country.languages_spoken"]
    assert parse_selected_relations("choose location.country.currency_used", relation_mapping, limit=2) == ["location.country.currency_used"]


def test_parse_enough_answer_output_extracts_decision_and_answer() -> None:
    payload = parse_enough_answer_output("Yes. Therefore, the answer is {Kenyan shilling}.")

    assert payload["decision"] == "yes"
    assert payload["answer_text"] == "Kenyan shilling"


def test_parse_simplified_question_rejects_unchanged_question() -> None:
    original = "What is the predominant religion where the leader is Ovadia Yosef?"
    assert parse_simplified_question("simplified question: What is the predominant religion where the leader is Ovadia Yosef?", original) == ""
    assert parse_simplified_question("simplified question: What is the predominant religion in Israel?", original) == "What is the predominant religion in Israel?"


def test_metaqa_backend_loads_forward_and_reverse_relations(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.txt"
    kb_path.write_text(
        "\n".join(
            [
                "Kismet|directed_by|William Dieterle",
                "Kismet|written_by|Edward Knoblock",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    backend = MetaqaGraphBackend(kb_path)

    relations = backend.list_relations([EntityRef("Kismet", "Kismet")])
    reverse_relations = backend.list_relations([EntityRef("Edward Knoblock", "Edward Knoblock")])

    assert "directed_by" in relations
    assert "~written_by" in reverse_relations


def test_validate_graph_answer_payload_reads_graph_fields() -> None:
    payload = _validate_graph_answer_payload(
        '{"final_answer":"Set Designer","reasoning":"supported by graph","evidence_triples":["(A, rel, B)"],"answer_path":["A -> B"]}',
        "",
        dataset="grailqa",
    )

    assert payload["final_answer"] == "Set Designer"
    assert payload["evidence_triples"] == ["(A, rel, B)"]
    assert payload["answer_path"] == ["A -> B"]


def test_ground_graph_payload_fills_missing_evidence_from_visible_view() -> None:
    view = GraphView(
        agent_id=2,
        view_kind="entity_neighborhood_view",
        context_text="graph",
        node_ids=["node:1"],
        node_labels=["Indianapolis, Indiana"],
        edge_keys=["node:1|linked_entity|node:?answer"],
        node_count=1,
        edge_count=1,
        visible_triples=["(topic, candidate_answer_relation, ?answer)"],
        structured_triples=[("topic", "candidate_answer_relation", "?answer")],
    )

    grounded = _ground_graph_payload(
        {
            "final_answer": "Indianapolis, Indiana",
            "reasoning": "No explicit location triple is visible.",
            "evidence_triples": [],
            "answer_path": [],
        },
        view,
        method_mode="debate",
    )

    assert grounded["evidence_triples"] == ["(topic, candidate_answer_relation, ?answer)"]
    assert grounded["answer_path"] == ["(topic, candidate_answer_relation, ?answer)"]


def test_build_static_metrics_reports_graph_specific_fields() -> None:
    experiment = load_experiment_config("configs/families/dog_graph/experiments/dog_graph_static_ablation.toml")
    metrics = _build_static_metrics(
        [
            {
                "dataset": "webquestions",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "graph_mv3",
                "score": 0.5,
                "prompt_tokens_per_question": 12.0,
                "completion_tokens_per_question": 4.0,
                "total_tokens_per_question": 16.0,
                "debate_total_tokens_per_question": 0.0,
                "latency_ms_per_question": 18.0,
                "calls_per_question": 3,
                "debate_rounds": 0,
                "subgraph_node_count": 6,
                "subgraph_edge_count": 5,
                "evidence_triples": ["(topic, rel, ?answer)"],
                "answer_path": ["topic -> ?answer"],
                "communication_grounded": False,
            },
            {
                "dataset": "webquestions",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "dog_graph_r2",
                "score": 1.0,
                "prompt_tokens_per_question": 20.0,
                "completion_tokens_per_question": 6.0,
                "total_tokens_per_question": 26.0,
                "debate_total_tokens_per_question": 8.0,
                "latency_ms_per_question": 25.0,
                "calls_per_question": 9,
                "debate_rounds": 2,
                "subgraph_node_count": 6,
                "subgraph_edge_count": 5,
                "evidence_triples": ["(topic, rel, ?answer)", "(entity, Designers, role)"],
                "answer_path": ["topic -> entity", "entity -> role"],
                "communication_grounded": True,
            },
        ],
        experiment.methods,
    )

    overall_mv3 = next(row for row in metrics["summary"] if row["dataset"] == "overall" and row["method_name"] == "graph_mv3")
    overall_r2 = next(row for row in metrics["summary"] if row["dataset"] == "overall" and row["method_name"] == "dog_graph_r2")
    assert overall_mv3["grounded_communication_rate"] == 0.0
    assert overall_r2["grounded_communication_rate"] == 1.0
    assert overall_r2["matched_vote_control"] == "graph_mv3"
    assert overall_r2["debate_gain_over_vote"] == 0.5


def test_build_paper_metrics_reports_process_fields() -> None:
    experiment = load_experiment_config("configs/families/dog_graph/experiments/dog_graph_main.toml")
    metrics = _build_paper_metrics(
        [
            {
                "dataset": "webquestions_paper_test",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "tog_iterative_baseline",
                "score": 0.0,
                "prompt_tokens_per_question": 40.0,
                "completion_tokens_per_question": 10.0,
                "total_tokens_per_question": 50.0,
                "latency_ms_per_question": 100.0,
                "calls_per_question": 2,
                "retrieval_hops": 1,
                "reasoning_triple_count": 1,
                "tail_entity_count": 1,
                "used_direct_fallback": False,
                "simplification_success": False,
                "question_changed": False,
                "enough_answer_decision": "no",
            },
            {
                "dataset": "webquestions_paper_test",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "dog_graph_paper",
                "score": 1.0,
                "prompt_tokens_per_question": 80.0,
                "completion_tokens_per_question": 20.0,
                "total_tokens_per_question": 100.0,
                "latency_ms_per_question": 200.0,
                "calls_per_question": 5,
                "retrieval_hops": 2,
                "reasoning_triple_count": 2,
                "tail_entity_count": 2,
                "used_direct_fallback": False,
                "simplification_success": True,
                "question_changed": True,
                "enough_answer_decision": "yes",
            },
        ],
        experiment.methods,
    )

    overall_dog = next(row for row in metrics["summary"] if row["dataset"] == "overall" and row["method_name"] == "dog_graph_paper")
    assert overall_dog["simplification_success_rate"] == 1.0
    assert overall_dog["enough_answer_yes_rate"] == 1.0
    assert overall_dog["matched_control"] == "tog_iterative_baseline"
