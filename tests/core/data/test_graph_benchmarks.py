"""覆盖图问答 benchmark 的 loader 与评分约束。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import load_samples
from research_experiments.core.data.evaluation import score_prediction


def test_webquestions_loader_builds_static_candidate_graph(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "webquestions"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test.json").write_text(
        json.dumps(
            [
                {
                    "qId": "wqs000000",
                    "qText": "what does jamaican people speak?",
                    "url": "http://www.freebase.com/view/en/jamaica",
                    "answers": ["Jamaican English", "Jamaican Creole English Language"],
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (dataset_root / "freebase_key_test.json").write_text(
        json.dumps([{"qId": "wqs000000", "freebaseKey": "jamaica"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (dataset_root / "freebase_mids_test.json").write_text(
        json.dumps(
            [
                {
                    "qId": "wqs000000",
                    "freebaseMids": [{"concept": "Jamaica", "mid": "m.03_r3"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (dataset_root / "relation_paths_test.json").write_text(
        json.dumps(
            [
                {
                    "qId": "wqs000000",
                    "relPaths": [[["/location/country/languages_spoken"], 2]],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (dataset_root / "entities_test.json").write_text(
        json.dumps([{"qId": "wqs000000", "entities": [["jamaican people", "NP"]]}], ensure_ascii=False),
        encoding="utf-8",
    )

    benchmark_path = tmp_path / "webquestions.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "WebQuestions"',
                'slug = "webquestions"',
                'loader = "webquestions_json"',
                f'source_path = "{(dataset_root / "test.json").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "webquestions"',
                'question_field = "qText"',
                'answer_field = "answers"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )

    samples = load_samples(load_benchmark_config(benchmark_path))
    sample = samples[0]

    assert sample.dataset == "webquestions"
    assert json.loads(sample.reference_answer) == ["Jamaican English", "Jamaican Creole English Language"]
    assert "Candidate graph:" in sample.prompt_context
    assert "languages spoken" in sample.prompt_context
    assert sample.metadata["candidate_subgraph"]["edges"]


def test_grailqa_loader_reads_graph_query_into_prompt_context(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "grailqa"
    dataset_root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "qid": "3202959008000",
                    "question": "what is the role of opera designer gig who designed the telephone / the medium?",
                    "answer": {
                        "answer_type": ["Entity"],
                        "answer_argument": ["m.0b787yg"],
                        "entity_name": ["Set Designer"],
                    },
                    "function": "none",
                    "num_node": 3,
                    "num_edge": 2,
                    "graph_query": {
                        "nodes": {
                            "nid": [0, 1, 2],
                            "node_type": ["class", "class", "entity"],
                            "id": ["opera.opera_designer_role", "opera.opera_designer_gig", "m.0pm2fgf"],
                            "class": ["opera.opera_designer_role", "opera.opera_designer_gig", "opera.opera_production"],
                            "friendly_name": ["Opera Designer Role", "Opera Designer Gig", "The Telephone / The Medium"],
                            "question_node": [1, 0, 0],
                            "function": ["none", "none", "none"],
                        },
                        "edges": {
                            "start": [1, 2],
                            "end": [0, 1],
                            "relation": ["opera.opera_designer_gig.design_role", "opera.opera_production.designers"],
                            "friendly_name": ["Design Role", "Designers"],
                        },
                    },
                    "sparql_query": "SELECT ?x WHERE { ?x :relation :value }",
                    "domains": ["opera"],
                    "level": "i.i.d.",
                    "s_expression": "(AND opera.opera_designer_role ...)",
                }
            ]
        ),
        dataset_root / "validation.parquet",
    )

    benchmark_path = tmp_path / "grailqa.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "GrailQA"',
                'slug = "grailqa"',
                'loader = "grailqa_parquet"',
                f'source_path = "{(dataset_root / "validation.parquet").as_posix()}"',
                'source_split = "validation"',
                'sample_id_prefix = "grailqa"',
                'question_field = "question"',
                'answer_field = "answer"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )

    samples = load_samples(load_benchmark_config(benchmark_path))
    sample = samples[0]

    assert sample.dataset == "grailqa"
    assert json.loads(sample.reference_answer) == ["Set Designer"]
    assert "Candidate graph:" in sample.prompt_context
    assert "Design Role" in sample.prompt_context
    assert "Query sketch:" in sample.prompt_context
    assert sample.metadata["candidate_subgraph"]["edges"]


def test_graph_qa_scoring_accepts_multiple_gold_aliases() -> None:
    gold = json.dumps(["Jamaican English", "Jamaican Creole English Language"], ensure_ascii=False)
    assert score_prediction("webquestions", "jamaican english", gold) == 1.0
    assert score_prediction("webquestions", "English", gold) > 0.0
    assert score_prediction("grailqa", "set designer", json.dumps(["Set Designer"], ensure_ascii=False)) == 1.0


def test_dog_webquestions_loader_reads_official_topic_entity(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "dog-freebase"
    dataset_root.mkdir(parents=True)
    (dataset_root / "WebQuestions.json").write_text(
        json.dumps(
            [
                {
                    "question": "what does jamaican people speak?",
                    "answers": ["Jamaican English", "Jamaican Creole English Language"],
                    "topic_entity": {"m.03_r3": "Jamaica"},
                    "qid_topic_entity": {"Q766": "Jamaica"},
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "dog_webquestions.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "DoG WebQuestions"',
                'slug = "dog_webquestions"',
                'loader = "dog_webquestions_json"',
                f'source_path = "{(dataset_root / "WebQuestions.json").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "dog-wq"',
                'question_field = "question"',
                'answer_field = "answers"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    sample = load_samples(load_benchmark_config(benchmark_path))[0]

    assert sample.dataset == "dog_webquestions"
    assert sample.metadata["paper_dataset_name"] == "WebQuestions"
    assert sample.metadata["topic_entity_id"] == "m.03_r3"
    assert json.loads(sample.reference_answer) == ["Jamaican English", "Jamaican Creole English Language"]


def test_dog_metaqa_loader_extracts_topic_entity_and_answers(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "dog-metaqa" / "2-hop"
    dataset_root.mkdir(parents=True)
    (dataset_root / "qa_test.txt").write_text(
        "what films did [Michelle Trachtenberg] star in\tInspector Gadget|Ice Princess\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "dog_metaqa.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "DoG MetaQA 2-hop"',
                'slug = "dog_metaqa_2hop"',
                'loader = "dog_metaqa_txt"',
                f'source_path = "{(dataset_root / "qa_test.txt").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "dog-metaqa2"',
                'question_field = "question"',
                'answer_field = "answers"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    sample = load_samples(load_benchmark_config(benchmark_path))[0]

    assert sample.dataset == "dog_metaqa_2hop"
    assert sample.metadata["dog_task_family"] == "metaqa"
    assert sample.metadata["hop_count"] == 2
    assert sample.metadata["topic_entity_name"] == "Michelle Trachtenberg"
    assert json.loads(sample.reference_answer) == ["Inspector Gadget", "Ice Princess"]


def test_dog_paper_scoring_uses_alias_exact_match() -> None:
    gold = json.dumps(["Belmont University"], ensure_ascii=False)
    assert score_prediction("dog_webqsp", "The answer is Belmont University.", gold) == 1.0
    assert score_prediction("dog_webqsp", "Belmont", gold) == 0.0
