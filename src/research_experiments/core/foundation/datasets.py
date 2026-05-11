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

from research_experiments.core.foundation.config import BenchmarkConfig
from research_experiments.core.foundation.workspace import workspace_layout


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
        "gsm8k_jsonl": _load_gsm8k,
        "math500_jsonl": _load_math500,
        "strategyqa_json": _load_strategyqa,
        "hotpotqa_parquet": _load_hotpotqa,
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

        for split_name, sample_ids in split_specs:
            path = resolve_split_manifest_path(
                config.slug,
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
        benchmark.slug,
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
    return Path(splits_root) / split_dir_name / f"{dataset_slug}-seed{seed}.json"


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

