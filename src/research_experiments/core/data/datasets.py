"""数据集加载与冻结 split 工具。

本模块把不同 benchmark 的原始样本统一抽象为 `DatasetSample`，并负责：
1. 从底层文件格式中加载全量样本；
2. 生成固定随机种子的冻结 split 清单；
3. 根据冻结 split 选出某轮实验真正要跑的样本。

这样各实验包只需要关心“本轮有哪些题”，不需要重复理解 JSONL、JSON 或 Parquet 的细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import random
import re
from typing import Any
import zipfile

import pyarrow.parquet as pq

from research_experiments.core.config import BenchmarkConfig
from research_experiments.workspace.layout import workspace_layout


@dataclass(frozen=True)
class DatasetSample:
    """统一后的单条样本表示。

    无论底层 benchmark 来自哪种格式，进入实验 runner 之后都会收敛为这一结构，
    从而保证提示构造、评测与日志写出逻辑能够跨数据集复用。
    """

    dataset: str
    sample_id: str
    question: str
    reference_answer: str
    prompt_context: str
    metadata: dict[str, Any]


def load_samples(config: BenchmarkConfig) -> list[DatasetSample]:
    """按 `loader` 类型读取 benchmark 的全量样本。"""
    loader_map = {
        "commongen_hard_json": _load_commongen_hard,
        "humaneval_parquet": _load_humaneval,
        "gsm8k_jsonl": _load_gsm8k,
        "math500_jsonl": _load_math500,
        "mmlu_parquet": _load_mmlu,
        "strategyqa_json": _load_strategyqa,
        "hotpotqa_parquet": _load_hotpotqa,
        "wikitq_jsonl": _load_wikitq,
        "tabfact_jsonl": _load_tabfact,
        "webquestions_json": _load_webquestions,
        "grailqa_parquet": _load_grailqa,
        "dog_webquestions_json": _load_dog_webquestions,
        "dog_grailqa_json": _load_dog_grailqa,
        "dog_webqsp_json": _load_dog_webqsp,
        "dog_cwq_json": _load_dog_cwq,
        "dog_metaqa_txt": _load_dog_metaqa,
        "mmlu_pro_parquet": _load_mmlu_pro,
        "gpqa_zip_csv": _load_gpqa_zip_csv,
        "gsm_symbolic_jsonl": _load_gsm_symbolic,
    }
    return loader_map[config.loader](config)


def generate_split_manifests(
    benchmark_configs: list[BenchmarkConfig],
    output_dir: str | Path,
) -> list[Path]:
    """为多个 benchmark 生成冻结后的 split 清单。

    这里生成的 JSON 清单只记录样本 ID 列表，而不复制样本正文。
    这样既能保持 split 可复现，也能避免在切分阶段复制整份原始数据。
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for config in benchmark_configs:
        samples = load_samples(config)
        split_specs = _resolve_split_specs(config, samples)
        dataset_key = config.cache_namespace or config.slug

        for split_name, sample_ids in split_specs:
            path = resolve_split_manifest_path(
                dataset_key,
                split_name,
                splits_root=output,
                random_seed=config.random_seed,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset": config.slug,
                "split_name": split_name,
                "source_split": config.source_split,
                "sample_count": len(sample_ids),
                "sample_ids": sample_ids,
                "random_seed": config.random_seed,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            created.append(path)

    return created


def _resolve_split_specs(
    config: BenchmarkConfig,
    samples: list[DatasetSample],
) -> list[tuple[str, list[str]]]:
    if config.split_presets:
        return [
            (str(preset["name"]), _build_split_ids(config, samples, preset))
            for preset in config.split_presets
        ]

    indexed_ids = [sample.sample_id for sample in samples]
    shuffled = indexed_ids[:]
    random.Random(config.random_seed).shuffle(shuffled)
    split_specs = [
        ("count20_seed42", shuffled[: min(config.smoke_size, len(shuffled))]),
        ("count100_seed42", shuffled[: min(config.pilot_size, len(shuffled))]),
    ]
    if len(indexed_ids) > 100 and len(indexed_ids) > 300:
        split_specs.append(("count300_seed42", shuffled[: min(config.main_size, len(shuffled))]))
    if len(indexed_ids) > 500:
        split_specs.append(("count500_seed42", shuffled[:500]))
    split_specs.append((f"full{len(indexed_ids)}_seed42", indexed_ids[:]))
    return split_specs


def _build_split_ids(
    config: BenchmarkConfig,
    samples: list[DatasetSample],
    preset: dict[str, Any],
) -> list[str]:
    strategy = str(preset.get("strategy") or "shuffle")
    size = int(preset.get("size") or len(samples))
    indexed_ids = [sample.sample_id for sample in samples]
    if strategy == "full":
        return indexed_ids[:]
    if strategy == "ordered":
        return indexed_ids[: min(size, len(indexed_ids))]
    if strategy == "shuffle":
        shuffled = indexed_ids[:]
        random.Random(config.random_seed).shuffle(shuffled)
        return shuffled[: min(size, len(shuffled))]
    if strategy == "stratified":
        field_name = str(preset.get("field") or "").strip()
        if not field_name:
            raise ValueError(f"Stratified split preset for {config.slug} requires a non-empty field.")
        return _stratified_sample_ids(samples, field_name=field_name, size=size, seed=config.random_seed)
    raise ValueError(f"Unsupported split preset strategy {strategy!r} for benchmark {config.slug}.")


def _stratified_sample_ids(
    samples: list[DatasetSample],
    *,
    field_name: str,
    size: int,
    seed: int,
) -> list[str]:
    grouped: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        value = sample.metadata.get(field_name)
        group_key = str(value if value not in {None, ""} else "unknown")
        grouped.setdefault(group_key, []).append(sample)

    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)

    total = len(samples)
    if size >= total:
        return [sample.sample_id for sample in samples]

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for group_key, rows in grouped.items():
        ideal = size * len(rows) / total
        base = min(len(rows), int(ideal))
        allocations[group_key] = base
        allocated += base
        remainders.append((ideal - base, group_key))

    remaining = size - allocated
    for _, group_key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        capacity = len(grouped[group_key]) - allocations[group_key]
        if capacity <= 0:
            continue
        allocations[group_key] += 1
        remaining -= 1

    while remaining > 0:
        progressed = False
        for group_key in sorted(grouped):
            capacity = len(grouped[group_key]) - allocations[group_key]
            if capacity <= 0:
                continue
            allocations[group_key] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    selected = [
        sample.sample_id
        for group_key in sorted(grouped)
        for sample in grouped[group_key][: allocations[group_key]]
    ]
    rng.shuffle(selected)
    return selected


def load_split_ids(
    dataset_slug: str,
    split_name: str,
    splits_root: str | Path = "configs/core/shared/benchmarks/splits",
    random_seed: int = 42,
) -> list[str]:
    """读取某个冻结 split 中的样本 ID 列表。"""
    payload = json.loads(
        resolve_split_manifest_path(
            dataset_slug,
            split_name,
            splits_root=splits_root,
            random_seed=random_seed,
        ).read_text(encoding="utf-8")
    )
    return payload["sample_ids"]


def select_samples(
    benchmark: BenchmarkConfig,
    split_name: str,
    splits_root: str | Path = "configs/core/shared/benchmarks/splits",
) -> list[DatasetSample]:
    """按冻结 split 从全量 benchmark 中选出本轮样本。"""
    split_ids = load_split_ids(
        benchmark.cache_namespace or benchmark.slug,
        split_name,
        splits_root=splits_root,
        random_seed=benchmark.random_seed,
    )
    sample_map = {sample.sample_id: sample for sample in load_samples(benchmark)}
    return [sample_map[sample_id] for sample_id in split_ids if sample_id in sample_map]


def resolve_split_manifest_path(
    dataset_slug: str,
    split_name: str,
    *,
    splits_root: str | Path = "configs/core/shared/benchmarks/splits",
    random_seed: int = 42,
) -> Path:
    """把 split 名解析成统一目录化后的 manifest 路径。"""

    split_dir_name, seed = _split_directory_and_seed(split_name, random_seed)
    dataset_path = Path(str(dataset_slug).replace("\\", "/"))
    return Path(splits_root) / split_dir_name / dataset_path.parent / f"{dataset_path.name}-seed{seed}.json"


def _split_directory_and_seed(split_name: str, fallback_seed: int) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<name>.+?)_seed(?P<seed>\d+)", split_name)
    if match:
        split_dir_name = _normalize_split_directory_name(match.group("name"))
        return split_dir_name, int(match.group("seed"))
    return split_name, int(fallback_seed)


def _normalize_split_directory_name(split_name_without_seed: str) -> str:
    if re.fullmatch(r"full\d+", split_name_without_seed):
        return "full"
    return split_name_without_seed


def resolve_dataset_source_path(source_path: str | Path) -> Path:
    """把 benchmark 的 source_path 解析为当前生效的数据集资产路径。"""
    path = Path(source_path)
    if path.is_absolute():
        return path
    return workspace_layout().datasets_root / path


def _load_gsm8k(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 GSM8K JSONL，并抽取 `####` 后的标准数字答案。"""
    path = resolve_dataset_source_path(config.source_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=f"{config.sample_id_prefix}-{index:05d}",
                    question=record["question"].strip(),
                    reference_answer=_extract_gsm8k_gold(record["answer"]),
                    prompt_context="",
                    metadata={"raw_index": index},
                )
            )
    return samples


def _load_math500(config: BenchmarkConfig) -> list[DatasetSample]:
    path = resolve_dataset_source_path(config.source_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            unique_id = str(record.get("unique_id") or f"{config.sample_id_prefix}-{index:05d}")
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=unique_id,
                    question=str(record["problem"]).strip(),
                    reference_answer=str(record["answer"]).strip(),
                    prompt_context="",
                    metadata={
                        "raw_index": index,
                        "subject": record.get("subject"),
                        "level": record.get("level"),
                        "unique_id": unique_id,
                        "solution": record.get("solution"),
                    },
                )
            )
    return samples


def _load_mmlu(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 MMLU 聚合 parquet，并把选项渲染成统一多选提示。"""

    table = pq.read_table(resolve_dataset_source_path(config.source_path))
    payload = table.to_pylist()
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        options = [str(item).strip() for item in record.get("choices", [])]
        answer_index = int(record.get("answer") or 0)
        answer_letter = _choice_letter(answer_index)
        answer_text = options[answer_index] if 0 <= answer_index < len(options) else ""
        sample_id = f"{config.sample_id_prefix}-{index:05d}"
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("question") or "").strip(),
                reference_answer=f"{answer_letter}|||{answer_text}",
                prompt_context=_render_multiple_choice_options(options),
                metadata={
                    "raw_index": index,
                    "subject": record.get("subject"),
                    "choices": options,
                    "answer_index": answer_index,
                    "answer_letter": answer_letter,
                    "answer_text": answer_text,
                },
            )
        )
    return samples


def _load_strategyqa(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 StrategyQA JSON，并把答案规范化为 `yes / no`。"""
    path = resolve_dataset_source_path(config.source_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        answer = "yes" if bool(record["answer"]) else "no"
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=str(record.get("qid") or f"{config.sample_id_prefix}-{index:05d}"),
                question=record["question"].strip(),
                reference_answer=answer,
                prompt_context="",
                metadata={
                    "raw_index": index,
                    "term": record.get("term"),
                    "description": record.get("description"),
                    "facts": list(record.get("facts", [])),
                    "decomposition": list(record.get("decomposition", [])),
                    "evidence": record.get("evidence"),
                },
            )
        )
    return samples


def _load_humaneval(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 HumanEval parquet，并把 prompt 与测试契约编码进参考答案。"""

    table = pq.read_table(resolve_dataset_source_path(config.source_path))
    payload = table.to_pylist()
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        prompt = str(record.get("prompt") or "")
        test_code = str(record.get("test") or "")
        entry_point = str(record.get("entry_point") or "").strip()
        canonical_solution = str(record.get("canonical_solution") or "")
        sample_id = str(record.get("task_id") or f"{config.sample_id_prefix}-{index:05d}")
        reference_answer = json.dumps(
            {
                "prompt": prompt,
                "test": test_code,
                "entry_point": entry_point,
                "canonical_solution": canonical_solution,
            },
            ensure_ascii=False,
        )
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=prompt.rstrip(),
                reference_answer=reference_answer,
                prompt_context="Return only the Python completion for the unfinished function.",
                metadata={
                    "raw_index": index,
                    "task_id": sample_id,
                    "entry_point": entry_point,
                    "canonical_solution": canonical_solution,
                    "test": test_code,
                },
            )
        )
    return samples


def _load_commongen_hard(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 CommonGen-Hard JSON，并把 concept set 编码为稳定参考。"""

    payload = json.loads(resolve_dataset_source_path(config.source_path).read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        concept_set = [str(item).strip() for item in record.get("concept_set", []) if str(item).strip()]
        reference_answer = json.dumps(
            {
                "concept_set": concept_set,
                "id": str(record.get("id") or f"{config.sample_id_prefix}-{index:05d}"),
            },
            ensure_ascii=False,
        )
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=str(record.get("id") or f"{config.sample_id_prefix}-{index:05d}"),
                question=str(record.get("instruction") or "").strip(),
                reference_answer=reference_answer,
                prompt_context="Required concepts:\n- " + "\n- ".join(concept_set),
                metadata={
                    "raw_index": index,
                    "concept_set": concept_set,
                    "human_annotations": list(record.get("human_annotations") or []),
                },
            )
        )
    return samples


def _load_hotpotqa(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 HotpotQA Parquet，并把上下文段落渲染成 prompt 可直接使用的文本。"""
    table = pq.read_table(resolve_dataset_source_path(config.source_path))
    payload = table.to_pylist()
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        sample_id = record.get("id") or f"{config.sample_id_prefix}-{index:05d}"
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record["question"]).strip(),
                reference_answer=str(record["answer"]).strip(),
                prompt_context=_render_hotpot_context(record["context"]),
                metadata={
                    "raw_index": index,
                    "type": record.get("type"),
                    "level": record.get("level"),
                    "supporting_facts": record.get("supporting_facts"),
                    "raw_context": record.get("context"),
                },
            )
            )
    return samples


def _load_wikitq(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 Table-Critic 官方仓提供的 WikiTQ 测试 JSONL。"""

    path = resolve_dataset_source_path(config.source_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            answers = [str(item).strip() for item in record.get("answer", []) if str(item).strip()]
            table_rows = record.get("table_text") or []
            sample_id = str(record.get("ids") or f"{config.sample_id_prefix}-{index:05d}")
            question = str(record.get("statement") or record.get("question") or "").strip()
            prompt_context = _render_table_context(
                table_rows,
                caption=str(record.get("table_caption") or "").strip() or None,
            )
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=sample_id,
                    question=question,
                    reference_answer=_encode_answer_list(answers),
                    prompt_context=prompt_context,
                    metadata={
                        "raw_index": index,
                        "paper_dataset_name": "WikiTQ",
                        "table_id": str(record.get("table_id") or sample_id),
                        "table_caption": record.get("table_caption"),
                        "table_text": table_rows,
                        "answers": answers,
                        "question_type": _infer_table_question_type(question),
                        "source_record": record,
                    },
                )
            )
    return samples


def _load_tabfact(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 Table-Critic 官方仓提供的 TabFact 测试 JSONL。"""

    path = resolve_dataset_source_path(config.source_path)
    raw2clean_path = path.with_name("raw2clean.jsonl")
    cleaned_index = _load_tabfact_cleaned_index(raw2clean_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            statement = str(record.get("statement") or "").strip()
            table_rows = record.get("table_text") or []
            table_caption = str(record.get("table_caption") or "").strip()
            table_id = str(record.get("table_id") or f"{config.sample_id_prefix}-table-{index:05d}")
            sample_id = f"{table_id}::{index:05d}"
            cleaned_statement = cleaned_index.get((table_id, statement))
            prompt_context = _render_table_context(table_rows, caption=table_caption or None)
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=sample_id,
                    question=statement,
                    reference_answer="entailed" if int(record.get("label") or 0) == 1 else "refuted",
                    prompt_context=prompt_context,
                    metadata={
                        "raw_index": index,
                        "paper_dataset_name": "TabFact",
                        "table_id": table_id,
                        "table_caption": table_caption,
                        "table_text": table_rows,
                        "label": int(record.get("label") or 0),
                        "cleaned_statement": cleaned_statement,
                        "question_type": "verification",
                        "source_record": record,
                    },
                )
            )
    return samples


def _load_webquestions(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 WebQuestions，并尽量拼接可用的 Freebase 路径注释为静态候选子图。"""
    path = resolve_dataset_source_path(config.source_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    relation_paths = _optional_webquestions_annotation(path.parent / "relation_paths_test.json")
    branched_relation_paths = _optional_webquestions_annotation(path.parent / "branched_relation_paths_test.json")
    freebase_keys = _optional_webquestions_annotation(path.parent / "freebase_key_test.json")
    freebase_mids = _optional_webquestions_annotation(path.parent / "freebase_mids_test.json")
    question_entities = _optional_webquestions_annotation(path.parent / "entities_test.json")
    question_dumps = _optional_webquestions_annotation(path.parent / "question_dump_test.json")

    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        sample_id = str(record.get("qId") or f"{config.sample_id_prefix}-{index:05d}")
        answers = [str(item).strip() for item in record.get("answers", []) if str(item).strip()]
        graph = _build_webquestions_candidate_graph(
            url=str(record.get("url") or "").strip(),
            freebase_key=freebase_keys.get(sample_id, {}).get("freebaseKey"),
            freebase_mids=freebase_mids.get(sample_id, {}).get("freebaseMids", []),
            question_entities=question_entities.get(sample_id, {}).get("entities", []),
            question_dump=question_dumps.get(sample_id, {}),
            relation_paths=branched_relation_paths.get(sample_id, {}).get("relPaths")
            or relation_paths.get(sample_id, {}).get("relPaths", []),
        )
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("qText") or record.get("question") or "").strip(),
                reference_answer=_encode_answer_list(answers),
                prompt_context=_render_candidate_graph(graph),
                metadata={
                    "raw_index": index,
                    "url": record.get("url"),
                    "answer_aliases": answers,
                    "freebase_key": freebase_keys.get(sample_id, {}).get("freebaseKey"),
                    "freebase_mids": freebase_mids.get(sample_id, {}).get("freebaseMids", []),
                    "question_entities": question_entities.get(sample_id, {}).get("entities", []),
                    "question_dump": question_dumps.get(sample_id, {}),
                    "relation_paths": relation_paths.get(sample_id, {}).get("relPaths", []),
                    "branched_relation_paths": branched_relation_paths.get(sample_id, {}).get("relPaths", []),
                    "candidate_subgraph": graph,
                    "graph_source": "webquestions_freebase_paths",
                },
            )
        )
    return samples


def _load_grailqa(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 GrailQA Parquet，并把 `graph_query` 转成可直接提示的静态候选子图。"""
    table = pq.read_table(resolve_dataset_source_path(config.source_path))
    payload = table.to_pylist()
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        answers = _extract_grailqa_answers(record.get("answer"))
        graph = _build_grailqa_candidate_graph(record)
        sample_id = str(record.get("qid") or f"{config.sample_id_prefix}-{index:05d}")
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("question") or "").strip(),
                reference_answer=_encode_answer_list(answers),
                prompt_context=_render_candidate_graph(graph),
                metadata={
                    "raw_index": index,
                    "answer_aliases": answers,
                    "function": record.get("function"),
                    "num_node": record.get("num_node"),
                    "num_edge": record.get("num_edge"),
                    "graph_query": record.get("graph_query"),
                    "sparql_query": record.get("sparql_query"),
                    "domains": list(record.get("domains") or []),
                    "level": record.get("level"),
                    "s_expression": record.get("s_expression"),
                    "candidate_subgraph": graph,
                    "graph_source": "grailqa_graph_query",
                },
            )
        )
    return samples


def _load_dog_webquestions(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载官方 DoG 仓的 WebQuestions JSON。"""

    payload = json.loads(resolve_dataset_source_path(config.source_path).read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        sample_id = str(record.get("qId") or record.get("QuestionId") or f"{config.sample_id_prefix}-{index:05d}")
        answers = [str(item).strip() for item in record.get("answers", []) if str(item).strip()]
        topic_entity_id, topic_entity_name = _extract_first_topic_entity(record.get("topic_entity"))
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("question") or "").strip(),
                reference_answer=_encode_answer_list(answers),
                prompt_context="",
                metadata={
                    "raw_index": index,
                    "paper_dataset_name": "WebQuestions",
                    "dog_task_family": "freebase",
                    "topic_entity": dict(record.get("topic_entity") or {}),
                    "topic_entity_id": topic_entity_id,
                    "topic_entity_name": topic_entity_name,
                    "answers": answers,
                    "qid_topic_entity": dict(record.get("qid_topic_entity") or {}),
                    "source_record": record,
                },
            )
        )
    return samples


def _load_dog_grailqa(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载官方 DoG 仓的 GrailQA JSON。"""

    payload = json.loads(resolve_dataset_source_path(config.source_path).read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        sample_id = str(record.get("qid") or f"{config.sample_id_prefix}-{index:05d}")
        answers = [
            str(item.get("entity_name") or item.get("answer_argument") or "").strip()
            for item in record.get("answer", [])
            if isinstance(item, dict)
        ]
        answers = [item for item in answers if item]
        topic_entity_id, topic_entity_name = _extract_first_topic_entity(record.get("topic_entity"))
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("question") or "").strip(),
                reference_answer=_encode_answer_list(answers),
                prompt_context="",
                metadata={
                    "raw_index": index,
                    "paper_dataset_name": "GrailQA",
                    "dog_task_family": "freebase",
                    "topic_entity": dict(record.get("topic_entity") or {}),
                    "topic_entity_id": topic_entity_id,
                    "topic_entity_name": topic_entity_name,
                    "answers": answers,
                    "graph_query": record.get("graph_query"),
                    "sparql_query": record.get("sparql_query"),
                    "domains": list(record.get("domains") or []),
                    "level": record.get("level"),
                    "s_expression": record.get("s_expression"),
                    "source_record": record,
                },
            )
        )
    return samples


def _load_dog_webqsp(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载官方 DoG 仓的 WebQSP JSON。"""

    payload = json.loads(resolve_dataset_source_path(config.source_path).read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        sample_id = str(record.get("QuestionId") or f"{config.sample_id_prefix}-{index:05d}")
        answers: list[str] = []
        for parse in record.get("Parses", []):
            if not isinstance(parse, dict):
                continue
            for item in parse.get("Answers", []):
                if not isinstance(item, dict):
                    continue
                alias = str(item.get("EntityName") or item.get("AnswerArgument") or "").strip()
                if alias:
                    answers.append(alias)
        answers = list(dict.fromkeys(answers))
        topic_entity_id, topic_entity_name = _extract_first_topic_entity(record.get("topic_entity"))
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("RawQuestion") or record.get("ProcessedQuestion") or "").strip(),
                reference_answer=_encode_answer_list(answers),
                prompt_context="",
                metadata={
                    "raw_index": index,
                    "paper_dataset_name": "WebQSP",
                    "dog_task_family": "freebase",
                    "topic_entity": dict(record.get("topic_entity") or {}),
                    "topic_entity_id": topic_entity_id,
                    "topic_entity_name": topic_entity_name,
                    "answers": answers,
                    "processed_question": record.get("ProcessedQuestion"),
                    "parses": record.get("Parses"),
                    "source_record": record,
                },
            )
        )
    return samples


def _load_dog_cwq(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载官方 DoG 仓的 CWQ JSON。"""

    payload = json.loads(resolve_dataset_source_path(config.source_path).read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        sample_id = str(record.get("ID") or f"{config.sample_id_prefix}-{index:05d}")
        answer_payload = record.get("answer")
        if isinstance(answer_payload, list):
            answers = [str(item).strip() for item in answer_payload if str(item).strip()]
        else:
            answer_value = str(answer_payload or "").strip()
            answers = [answer_value] if answer_value else []
        topic_entity_id, topic_entity_name = _extract_first_topic_entity(record.get("topic_entity"))
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("question") or "").strip(),
                reference_answer=_encode_answer_list(answers),
                prompt_context="",
                metadata={
                    "raw_index": index,
                    "paper_dataset_name": "CWQ",
                    "dog_task_family": "freebase",
                    "topic_entity": dict(record.get("topic_entity") or {}),
                    "topic_entity_id": topic_entity_id,
                    "topic_entity_name": topic_entity_name,
                    "answers": answers,
                    "webqsp_id": record.get("webqsp_ID"),
                    "source_record": record,
                },
            )
        )
    return samples


def _load_dog_metaqa(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载官方 DoG 仓的 MetaQA 测试文件。"""

    path = resolve_dataset_source_path(config.source_path)
    hop_count = _extract_metaqa_hop_count(path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            question_text, _, answer_text = line.partition("\t")
            answers = [item.strip() for item in answer_text.split("|") if item.strip()]
            topic_entity_name = _extract_metaqa_topic_entity(question_text)
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=f"{config.sample_id_prefix}-{index:05d}",
                    question=question_text.strip(),
                    reference_answer=_encode_answer_list(answers),
                    prompt_context="",
                    metadata={
                        "raw_index": index,
                        "paper_dataset_name": f"MetaQA {hop_count}-hop",
                        "dog_task_family": "metaqa",
                        "topic_entity": {topic_entity_name: topic_entity_name} if topic_entity_name else {},
                        "topic_entity_id": topic_entity_name,
                        "topic_entity_name": topic_entity_name,
                        "answers": answers,
                        "hop_count": hop_count,
                        "source_record": {"line": line},
                    },
                )
            )
    return samples


def _load_mmlu_pro(config: BenchmarkConfig) -> list[DatasetSample]:
    table = pq.read_table(resolve_dataset_source_path(config.source_path))
    payload = table.to_pylist()
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        options = [str(item).strip() for item in record.get("options", [])]
        answer_letter = _normalize_option_letter(record.get("answer"))
        answer_index = record.get("answer_index")
        if answer_letter is None and answer_index is not None:
            answer_letter = _choice_letter(int(answer_index))
        if answer_letter is None:
            raise ValueError(f"MMLU-Pro row {index} is missing a usable answer label.")
        option_index = ord(answer_letter) - ord("A")
        option_text = options[option_index] if 0 <= option_index < len(options) else ""
        question_id = record.get("question_id")
        sample_id = f"{config.sample_id_prefix}-{question_id}" if question_id is not None else f"{config.sample_id_prefix}-{index:05d}"
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record["question"]).strip(),
                reference_answer=f"{answer_letter}|||{option_text}",
                prompt_context=_render_multiple_choice_options(options),
                metadata={
                    "raw_index": index,
                    "question_id": question_id,
                    "options": options,
                    "answer_letter": answer_letter,
                    "answer_index": answer_index,
                    "answer_text": option_text,
                    "category": record.get("category"),
                    "src": record.get("src"),
                },
            )
        )
    return samples


def _load_gpqa_zip_csv(config: BenchmarkConfig) -> list[DatasetSample]:
    archive_member = config.archive_member or "dataset/gpqa_diamond.csv"
    archive_password = (config.archive_password or "").encode("utf-8") if config.archive_password else None
    samples: list[DatasetSample] = []
    with zipfile.ZipFile(resolve_dataset_source_path(config.source_path)) as archive:
        with archive.open(archive_member, pwd=archive_password) as handle:
            reader = csv.DictReader((line.decode("utf-8") for line in handle))
            for index, record in enumerate(reader):
                question = str(record.get("Question") or "").strip()
                correct = str(record.get("Correct Answer") or "").strip()
                incorrects = [
                    str(record.get("Incorrect Answer 1") or "").strip(),
                    str(record.get("Incorrect Answer 2") or "").strip(),
                    str(record.get("Incorrect Answer 3") or "").strip(),
                ]
                choices = [("correct", correct), *[(f"incorrect_{offset}", value) for offset, value in enumerate(incorrects, start=1)]]
                shuffled_choices = choices[:]
                random.Random(f"{config.random_seed}:{record.get('Record ID') or index}").shuffle(shuffled_choices)
                options = [value for _, value in shuffled_choices if value]
                answer_position = next(
                    position
                    for position, (label, value) in enumerate(shuffled_choices)
                    if label == "correct" and value
                )
                answer_letter = _choice_letter(answer_position)
                sample_id = str(record.get("Record ID") or f"{config.sample_id_prefix}-{index:05d}")
                samples.append(
                    DatasetSample(
                        dataset=config.slug,
                        sample_id=sample_id,
                        question=question,
                        reference_answer=f"{answer_letter}|||{correct}",
                        prompt_context=_render_multiple_choice_options(options),
                        metadata={
                            "raw_index": index,
                            "record_id": sample_id,
                            "options": options,
                            "answer_letter": answer_letter,
                            "answer_text": correct,
                            "high_level_domain": record.get("High-level domain"),
                            "subdomain": record.get("Subdomain"),
                        },
                    )
                )
    return samples


def _load_gsm_symbolic(config: BenchmarkConfig) -> list[DatasetSample]:
    path = resolve_dataset_source_path(config.source_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=f"{config.sample_id_prefix}-{int(record.get('id', index)):05d}",
                    question=str(record["question"]).strip(),
                    reference_answer=_extract_gsm8k_gold(str(record["answer"])),
                    prompt_context="",
                    metadata={
                        "raw_index": index,
                        "instance": record.get("instance"),
                        "original_id": record.get("original_id"),
                    },
                )
            )
    return samples


def _optional_webquestions_annotation(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("qId") or "").strip()
        if sample_id:
            rows[sample_id] = item
    return rows


def _load_tabfact_cleaned_index(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            statement = str(record.get("statement") or "").strip()
            table_id = str(record.get("table_id") or "").strip()
            cleaned_statement = str(record.get("cleaned_statement") or "").strip()
            if statement and table_id and cleaned_statement:
                rows[(table_id, statement)] = cleaned_statement
    return rows


def _render_table_context(table_rows: list[list[Any]], *, caption: str | None = None) -> str:
    if not table_rows:
        return "Table:\n<empty>"
    header = [str(cell).strip() for cell in table_rows[0]]
    body = [
        [str(cell).strip() for cell in row]
        for row in table_rows[1:]
    ]
    lines: list[str] = []
    if caption:
        lines.append(f"Caption: {caption}")
    lines.append("Table:")
    lines.append("/*")
    lines.append("col : " + " | ".join(header))
    for index, row in enumerate(body, start=1):
        lines.append(f"row {index} : " + " | ".join(row))
    lines.append("*/")
    return "\n".join(lines)


def _infer_table_question_type(question: str) -> str:
    normalized = question.lower()
    if normalized.startswith("how many") or "number of" in normalized or "count" in normalized:
        return "count"
    if normalized.startswith("which") or "what is the name" in normalized:
        return "lookup"
    if "average" in normalized or "sum" in normalized or "total" in normalized:
        return "aggregation"
    if "most" in normalized or "least" in normalized or "highest" in normalized or "lowest" in normalized:
        return "superlative"
    if normalized.startswith("is ") or normalized.startswith("are ") or normalized.startswith("did "):
        return "boolean"
    return "table_qa"


def _build_webquestions_candidate_graph(
    *,
    url: str,
    freebase_key: Any,
    freebase_mids: list[dict[str, Any]],
    question_entities: list[list[str]],
    question_dump: dict[str, Any],
    relation_paths: list[list[Any]],
) -> dict[str, Any]:
    topic_seed = _best_webquestions_topic_seed(question_dump, freebase_key, url)
    topic_id = f"topic:{topic_seed}"
    nodes = [
        {"id": topic_id, "label": topic_seed, "type": "topic_seed"},
        {"id": "node:?answer", "label": "?answer", "type": "answer_slot"},
    ]
    edges: list[dict[str, Any]] = []

    clue_rows = list(question_dump.get("Clue") or [])
    concept_rows = list(question_dump.get("Concept") or [])
    for offset, concept in enumerate(concept_rows[:4]):
        label = str(concept.get("fullLabel") or concept.get("cookedLabel") or "").strip()
        if not label:
            continue
        node_id = f"concept:{offset}"
        nodes.append({"id": node_id, "label": label, "type": "concept_candidate"})
        edges.append(
            {
                "source": topic_id,
                "relation": "concept_candidate",
                "target": node_id,
                "friendly_name": "concept candidate",
                "support": round(float(concept.get("score") or 0.0), 6) if concept.get("score") is not None else 0.0,
            }
        )

    for offset, clue in enumerate(clue_rows[:5]):
        label = str(clue.get("label") or "").strip()
        clue_type = str(clue.get("type") or "clue").strip()
        if not label:
            continue
        node_id = f"clue:{offset}"
        nodes.append({"id": node_id, "label": label, "type": "question_clue", "clue_type": clue_type})
        edges.append(
            {
                "source": topic_id,
                "relation": clue_type,
                "target": node_id,
                "friendly_name": clue_type.replace("Clue", "clue "),
                "support": round(float(clue.get("weight") or 0.0), 6) if clue.get("weight") is not None else 0.0,
            }
        )

    include_linked_entities = bool(relation_paths)
    for offset, entity in enumerate(freebase_mids[:6] if include_linked_entities else []):
        concept = str(entity.get("concept") or f"linked_entity_{offset + 1}").strip()
        mid = str(entity.get("mid") or f"mid_{offset + 1}").strip()
        node_id = f"mid:{mid}"
        nodes.append({"id": node_id, "label": concept or mid, "type": "linked_entity", "mid": mid})
        edges.append(
            {
                "source": topic_id,
                "relation": "linked_entity",
                "target": node_id,
                "friendly_name": "linked_entity",
                "support": 1,
            }
        )

    for offset, entity in enumerate(question_entities):
        mention = str(entity[0]).strip() if entity else f"mention_{offset + 1}"
        tag = str(entity[1]).strip() if len(entity) > 1 else "question_mention"
        node_id = f"mention:{offset}"
        nodes.append({"id": node_id, "label": mention, "type": "question_mention", "tag": tag})
        edges.append(
            {
                "source": f"question:{offset}",
                "relation": tag or "question_mention",
                "target": node_id,
                "friendly_name": tag or "question_mention",
                "support": 1,
            }
        )

    for path_spec in relation_paths[:8]:
        relation_chain = [str(item).strip() for item in (path_spec[0] if path_spec else []) if str(item).strip()]
        support = int(path_spec[1]) if len(path_spec) > 1 else 1
        if not relation_chain:
            continue
        relation_text = " -> ".join(_humanize_webquestions_relation_name(item) for item in relation_chain)
        edges.append(
            {
                "source": topic_id,
                "relation": relation_text,
                "target": "node:?answer",
                "friendly_name": relation_text,
                "support": support,
            }
        )

    if not relation_paths:
        edges.append(
            {
                "source": topic_id,
                "relation": "candidate_answer_relation",
                "target": "node:?answer",
                "friendly_name": "candidate_answer_relation",
                "support": 1,
            }
        )

    return {
        "graph_kind": "webquestions_static_paths",
        "topic_seed": topic_seed,
        "nodes": nodes,
        "edges": edges,
        "question_clues": [
            {
                "label": str(item.get("label") or "").strip(),
                "type": str(item.get("type") or "").strip(),
            }
            for item in clue_rows
            if str(item.get("label") or "").strip()
        ],
        "concept_candidates": [
            str(item.get("fullLabel") or item.get("cookedLabel") or "").strip()
            for item in concept_rows
            if str(item.get("fullLabel") or item.get("cookedLabel") or "").strip()
        ],
    }


def _build_grailqa_candidate_graph(record: dict[str, Any]) -> dict[str, Any]:
    graph_query = record.get("graph_query") or {}
    nodes_payload = graph_query.get("nodes") or {}
    edges_payload = graph_query.get("edges") or {}
    node_rows: list[dict[str, Any]] = []
    node_id_to_label: dict[int, str] = {}

    node_count = min(
        len(nodes_payload.get("nid") or []),
        len(nodes_payload.get("id") or []),
    )
    for index in range(node_count):
        node_numeric_id = int((nodes_payload.get("nid") or [index])[index])
        label = str((nodes_payload.get("friendly_name") or [""])[index] or (nodes_payload.get("id") or [""])[index]).strip()
        node_rows.append(
            {
                "id": f"node:{node_numeric_id}",
                "label": label or str((nodes_payload.get("id") or [""])[index]),
                "type": str((nodes_payload.get("node_type") or ["unknown"])[index]),
                "kb_id": str((nodes_payload.get("id") or [""])[index]),
                "class": str((nodes_payload.get("class") or [""])[index]),
                "question_node": bool((nodes_payload.get("question_node") or [0])[index]),
                "function": str((nodes_payload.get("function") or ["none"])[index]),
            }
        )
        node_id_to_label[node_numeric_id] = label or str((nodes_payload.get("id") or [""])[index])

    edge_rows: list[dict[str, Any]] = []
    edge_count = min(
        len(edges_payload.get("start") or []),
        len(edges_payload.get("end") or []),
        len(edges_payload.get("relation") or []),
    )
    for index in range(edge_count):
        start_id = int((edges_payload.get("start") or [0])[index])
        end_id = int((edges_payload.get("end") or [0])[index])
        relation = str((edges_payload.get("relation") or [""])[index]).strip()
        friendly = str((edges_payload.get("friendly_name") or [""])[index]).strip() or relation
        edge_rows.append(
            {
                "source": f"node:{start_id}",
                "relation": relation,
                "target": f"node:{end_id}",
                "friendly_name": friendly,
                "source_label": node_id_to_label.get(start_id, f"node:{start_id}"),
                "target_label": node_id_to_label.get(end_id, f"node:{end_id}"),
                "support": 1,
            }
        )

    return {
        "graph_kind": "grailqa_graph_query",
        "topic_seed": str(record.get("qid") or ""),
        "nodes": node_rows,
        "edges": edge_rows,
        "domains": list(record.get("domains") or []),
        "level": str(record.get("level") or ""),
        "s_expression": str(record.get("s_expression") or ""),
    }


def _extract_first_topic_entity(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict) or not payload:
        return "", ""
    entity_id, entity_name = next(iter(payload.items()))
    return str(entity_id).strip(), str(entity_name).strip()


def _extract_metaqa_topic_entity(question_text: str) -> str:
    match = re.search(r"\[(?P<entity>.+?)\]", question_text)
    if not match:
        return ""
    return str(match.group("entity")).strip()


def _extract_metaqa_hop_count(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"(?P<hops>\d+)-hop", part)
        if match:
            return int(match.group("hops"))
    return 1


def _extract_grailqa_answers(answer_payload: Any) -> list[str]:
    if not isinstance(answer_payload, dict):
        return []
    aliases = [str(item).strip() for item in answer_payload.get("entity_name", []) if str(item).strip()]
    if aliases:
        return aliases
    return [str(item).strip() for item in answer_payload.get("answer_argument", []) if str(item).strip()]


def _encode_answer_list(answers: list[str]) -> str:
    unique_answers = list(dict.fromkeys(answer for answer in answers if answer))
    return json.dumps(unique_answers, ensure_ascii=False)


def _render_candidate_graph(graph: dict[str, Any]) -> str:
    """把统一候选子图渲染成 prompt 可直接使用的图证据块。"""
    lines = ["Candidate graph:", ""]
    topic_seed = str(graph.get("topic_seed") or "").strip()
    if topic_seed:
        lines.append(f"- topic_seed: {topic_seed}")
    domains = [str(item).strip() for item in graph.get("domains", []) if str(item).strip()]
    if domains:
        lines.append(f"- domains: {', '.join(domains)}")
    level = str(graph.get("level") or "").strip()
    if level:
        lines.append(f"- level: {level}")
    question_clues = graph.get("question_clues") or []
    if question_clues:
        rendered_clues = ", ".join(
            f"{item.get('label')} [{item.get('type')}]"
            for item in question_clues[:5]
            if str(item.get("label") or "").strip()
        )
        if rendered_clues:
            lines.append(f"- question_clues: {rendered_clues}")
    concept_candidates = [str(item).strip() for item in graph.get("concept_candidates", []) if str(item).strip()]
    if concept_candidates:
        lines.append(f"- concept_candidates: {', '.join(concept_candidates[:5])}")

    lines.extend(["", "Nodes:"])
    for node in graph.get("nodes", []):
        lines.append(f"- {node.get('id')}: {node.get('label')} [{node.get('type')}]")

    lines.extend(["", "Triples / path fragments:"])
    for edge in graph.get("edges", []):
        source_label = edge.get("source_label") or edge.get("source")
        target_label = edge.get("target_label") or edge.get("target")
        relation = edge.get("friendly_name") or edge.get("relation")
        support = edge.get("support")
        support_suffix = f" (support={support})" if support not in {None, ""} else ""
        lines.append(f"- ({source_label}, {relation}, {target_label}){support_suffix}")

    s_expression = str(graph.get("s_expression") or "").strip()
    if s_expression:
        lines.extend(["", "Query sketch:", f"- {s_expression}"])
    return "\n".join(lines).strip()


def _best_webquestions_topic_seed(question_dump: dict[str, Any], freebase_key: Any, url: str) -> str:
    for clue in question_dump.get("Clue") or []:
        clue_type = str(clue.get("type") or "").strip()
        if clue_type == "ClueSubjectPhrase":
            label = str(clue.get("label") or "").strip()
            if label:
                return label
    for clue in question_dump.get("Clue") or []:
        label = str(clue.get("label") or "").strip()
        if label:
            return label
    for concept in question_dump.get("Concept") or []:
        label = str(concept.get("fullLabel") or concept.get("cookedLabel") or "").strip()
        if label:
            return label
    fallback = str(freebase_key or url.rsplit("/", 1)[-1] or "topic").strip() or "topic"
    return fallback.replace("_", " ")


def _humanize_webquestions_relation_name(value: str) -> str:
    parts = [part for part in str(value).split("/") if part]
    if not parts:
        return str(value)
    tail = parts[-1].replace("_", " ")
    if " -> " in value:
        chain = [segment for segment in str(value).split(" -> ") if segment]
        return " -> ".join(_humanize_webquestions_relation_name(segment) for segment in chain)
    return tail


def _render_hotpot_context(context: dict[str, Any]) -> str:
    """把 HotpotQA 的标题与句子数组渲染成 prompt 可直接使用的上下文文本。"""
    titles = context["title"]
    paragraphs = context["sentences"]
    rendered: list[str] = []
    for title, sentences in zip(titles, paragraphs, strict=False):
        joined = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
        rendered.append(f"[{title}] {joined}")
    return "\n".join(rendered)


def _extract_gsm8k_gold(answer: str) -> str:
    """从 GSM8K 标注答案中抽取 `####` 之后的标准数字。"""
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    return answer.strip()


def _render_multiple_choice_options(options: list[str]) -> str:
    rendered = [f"{_choice_letter(index)}. {option}" for index, option in enumerate(options)]
    return "Options:\n" + "\n".join(rendered)


def _choice_letter(index: int) -> str:
    return chr(ord("A") + index)


def _normalize_option_letter(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    candidate = normalized[0]
    if "A" <= candidate <= "J":
        return candidate
    return None


