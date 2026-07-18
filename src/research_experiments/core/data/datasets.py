"""数据集加载与冻结 split 工具。

本模块把不同 benchmark 的原始样本统一抽象为 `DatasetSample`，并负责：
1. 从底层文件格式中加载全量样本；
2. 生成固定随机种子的冻结 split 清单；
3. 根据冻结 split 选出某轮实验真正要跑的样本。

这样各实验包只需要关心“本轮有哪些题”，不需要重复理解 JSONL、JSON 或 Parquet 的细节。
"""

from __future__ import annotations

import csv
import json
import random
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        "competition_math_zip": _load_competition_math,
        "humaneval_parquet": _load_humaneval,
        "gsm8k_jsonl": _load_gsm8k,
        "math500_jsonl": _load_math500,
        "omni_math_jsonl": _load_omni_math,
        "omni_math_2_filtered_jsonl": _load_omni_math_2_filtered,
        "bbeh_json_bundle": _load_bbeh,
        "mmlu_parquet": _load_mmlu,
        "strategyqa_json": _load_strategyqa,
        "hotpotqa_parquet": _load_hotpotqa,
        "webquestions_json": _load_webquestions,
        "mmlu_pro_parquet": _load_mmlu_pro,
        "gpqa_zip_csv": _load_gpqa_zip_csv,
        "realmistake_error_detection_zip": _load_realmistake_error_detection,
    }
    return _apply_record_filters(loader_map[config.loader](config), config.record_filters)


def _apply_record_filters(samples: list[DatasetSample], record_filters: list[dict[str, Any]]) -> list[DatasetSample]:
    if not record_filters:
        return samples
    filtered = samples
    for spec in record_filters:
        field_name = str(spec.get("field") or "").strip()
        operator = str(spec.get("operator") or "eq").strip()
        if not field_name:
            raise ValueError("record_filters entries require a non-empty field.")
        if operator == "eq":
            target = spec.get("value")
            filtered = [sample for sample in filtered if sample.metadata.get(field_name) == target]
            continue
        if operator == "in":
            values = spec.get("values")
            if not isinstance(values, list):
                raise ValueError("record_filters with operator 'in' require a list-valued 'values' field.")
            value_set = set(values)
            filtered = [sample for sample in filtered if sample.metadata.get(field_name) in value_set]
            continue
        raise ValueError(f"Unsupported record_filters operator: {operator}")
    return filtered


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
        return _canonical_prefix_split_specs(config, samples)

    indexed_ids = [sample.sample_id for sample in samples]
    shuffled = indexed_ids[:]
    random.Random(42).shuffle(shuffled)
    split_specs = [
        ("count20_seed42", shuffled[: min(config.smoke_size, len(shuffled))]),
        ("count100_seed42", shuffled[: min(100, len(shuffled))]),
    ]
    if len(indexed_ids) > 100 and len(indexed_ids) > 300:
        split_specs.append(("count300_seed42", shuffled[: min(config.main_size, len(shuffled))]))
    if len(indexed_ids) > 500:
        split_specs.append(("count500_seed42", shuffled[:500]))
    split_specs.append((f"full{len(indexed_ids)}_seed42", indexed_ids[:]))
    return _dedupe_split_specs(
        (
            _canonical_split_name(
                split_name,
                sample_ids,
                total_count=len(indexed_ids),
                fallback_seed=42,
            ),
            sample_ids,
        )
        for split_name, sample_ids in split_specs
    )


def _canonical_prefix_split_specs(
    config: BenchmarkConfig,
    samples: list[DatasetSample],
) -> list[tuple[str, list[str]]]:
    """Build every configured split from one deterministic seed-42 prefix.

    ``countN_seed42`` always means the first N records of the same shuffled
    order, and ``fullN_seed42`` means the complete source order.  This is the
    repository-wide pairing contract: count20 is therefore always contained in
    count100, count300, and every larger canonical count split.
    """
    indexed_ids = [sample.sample_id for sample in samples]
    shuffled = indexed_ids[:]
    random.Random(42).shuffle(shuffled)
    resolved: list[tuple[str, list[str]]] = []
    for preset in config.split_presets:
        raw_name = str(preset["name"])
        match = re.fullmatch(r"(?P<kind>count|full)(?P<size>\d+)_seed42", raw_name)
        if match is None:
            raise ValueError(
                f"Benchmark {config.slug!r} uses non-canonical split {raw_name!r}. "
                "Use countN_seed42 or fullN_seed42."
            )
        if match.group("kind") == "full":
            resolved.append((f"full{len(indexed_ids)}_seed42", indexed_ids[:]))
            continue
        requested_size = int(match.group("size"))
        if requested_size >= len(indexed_ids):
            resolved.append((f"full{len(indexed_ids)}_seed42", indexed_ids[:]))
            continue
        resolved.append((f"count{requested_size}_seed42", shuffled[: min(requested_size, len(shuffled))]))
    return _dedupe_split_specs(resolved)


def _dedupe_split_specs(split_specs: Iterable[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    deduped: dict[str, list[str]] = {}
    for split_name, sample_ids in split_specs:
        deduped.setdefault(split_name, sample_ids)
    return list(deduped.items())


def _canonical_split_name(
    split_name: str,
    sample_ids: list[str],
    *,
    total_count: int,
    fallback_seed: int,
) -> str:
    raw_name, seed = _split_name_and_seed(split_name, fallback_seed)
    if re.fullmatch(r"full\d*", raw_name):
        return f"full{total_count}_seed{seed}"
    count_target = _count_split_target(raw_name)
    if count_target is not None and len(sample_ids) >= total_count and total_count <= count_target:
        return f"full{total_count}_seed{seed}"
    return split_name


def _split_name_and_seed(split_name: str, fallback_seed: int) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<name>.+?)_seed(?P<seed>\d+)", split_name)
    if match:
        return match.group("name"), int(match.group("seed"))
    return split_name, int(fallback_seed)


def _count_split_target(split_name_without_seed: str) -> int | None:
    match = re.fullmatch(r"count(?P<count>\d+)(?:_.+)?", split_name_without_seed)
    if not match:
        return None
    return int(match.group("count"))


def load_split_ids(
    dataset_slug: str,
    split_name: str,
    splits_root: str | Path = "configs/core/shared/benchmarks/splits",
    random_seed: int = 42,
) -> list[str]:
    """读取某个冻结 split 中的样本 ID 列表。"""
    manifest_path = resolve_split_manifest_path(
        dataset_slug,
        split_name,
        splits_root=splits_root,
        random_seed=random_seed,
    )
    if not manifest_path.exists():
        fallback_path = _resolve_full_fallback_manifest_path(
            dataset_slug,
            split_name,
            splits_root=splits_root,
            random_seed=random_seed,
        )
        if fallback_path is None:
            raise FileNotFoundError(
                f"Split manifest not found for dataset={dataset_slug!r}, split={split_name!r}. "
                f"Expected path: {manifest_path}"
            )
        manifest_path = fallback_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list):
        raise ValueError(
            f"Split manifest for dataset={dataset_slug!r}, split={split_name!r} is missing a valid sample_ids list: "
            f"{manifest_path}"
        )
    if not all(isinstance(item, str) for item in sample_ids):
        raise ValueError(
            f"Split manifest for dataset={dataset_slug!r}, split={split_name!r} contains non-string sample_ids: "
            f"{manifest_path}"
        )
    return sample_ids


def _resolve_full_fallback_manifest_path(
    dataset_slug: str,
    split_name: str,
    *,
    splits_root: str | Path,
    random_seed: int,
) -> Path | None:
    raw_name, seed = _split_name_and_seed(split_name, random_seed)
    count_target = _count_split_target(raw_name)
    if count_target is None:
        return None
    full_path = resolve_split_manifest_path(
        dataset_slug,
        f"full0_seed{seed}",
        splits_root=splits_root,
        random_seed=random_seed,
    )
    if not full_path.exists():
        return None
    payload = json.loads(full_path.read_text(encoding="utf-8"))
    sample_count = int(payload.get("sample_count") or len(payload.get("sample_ids") or []))
    if sample_count <= count_target:
        return full_path
    return None


def select_samples(
    benchmark: BenchmarkConfig,
    split_name: str,
    splits_root: str | Path = "configs/core/shared/benchmarks/splits",
) -> list[DatasetSample]:
    """按冻结 split 从全量 benchmark 中选出本轮样本。"""
    manifest_path = resolve_split_manifest_path(
        benchmark.cache_namespace or benchmark.slug,
        split_name,
        splits_root=splits_root,
        random_seed=benchmark.random_seed,
    )
    if not manifest_path.exists():
        generate_split_manifests([benchmark], splits_root)
    split_ids = load_split_ids(
        benchmark.cache_namespace or benchmark.slug,
        split_name,
        splits_root=splits_root,
        random_seed=benchmark.random_seed,
    )
    sample_map = {sample.sample_id: sample for sample in load_samples(benchmark)}
    missing_ids = [sample_id for sample_id in split_ids if sample_id not in sample_map]
    if missing_ids:
        preview = ", ".join(repr(sample_id) for sample_id in missing_ids[:5])
        if len(missing_ids) > 5:
            preview += ", ..."
        raise KeyError(
            f"Split selection for benchmark={benchmark.slug!r}, split={split_name!r} references "
            f"{len(missing_ids)} missing sample_id(s): {preview}"
        )
    return [sample_map[sample_id] for sample_id in split_ids]


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


def _load_omni_math(config: BenchmarkConfig) -> list[DatasetSample]:
    path = resolve_dataset_source_path(config.source_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("id") or f"{config.sample_id_prefix}-{index:05d}")
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=sample_id,
                    question=str(record.get("problem") or "").strip(),
                    reference_answer=str(record.get("answer") or "").strip(),
                    prompt_context="",
                    metadata={
                        "raw_index": index,
                        "domain": list(record.get("domain") or []),
                        "primary_domain": str(next(iter(record.get("domain") or []), "unknown")),
                        "difficulty": record.get("difficulty"),
                        "stratum": _omni_math_stratum(record),
                        "source": record.get("source"),
                        "solution": record.get("solution"),
                    },
                )
            )
    return samples


def _load_omni_math_2_filtered(config: BenchmarkConfig) -> list[DatasetSample]:
    """Load the exact-answer, untagged Omni-MATH-2 subset.

    Omni-MATH-2 exposes tags for items needing proof, visual interpretation, or
    estimation.  The BRD confirmation set excludes every tagged record rather
    than quietly applying a task-specific judge to them.
    """

    path = resolve_dataset_source_path(config.source_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            tags = _normalized_omni_math_2_tags(record.get("tags"))
            if tags:
                continue
            sample_id = str(record.get("id") or f"{config.sample_id_prefix}-{index:05d}")
            domain = record.get("domain")
            domains = domain if isinstance(domain, list) else [domain] if domain else []
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=sample_id,
                    question=str(record.get("problem") or "").strip(),
                    reference_answer=str(record.get("answer") or "").strip(),
                    prompt_context="",
                    metadata={
                        "raw_index": index,
                        "domain": domains,
                        "primary_domain": str(domains[0]) if domains else "unknown",
                        "difficulty": record.get("difficulty"),
                        "stratum": _omni_math_2_stratum(record),
                        "source": record.get("source"),
                        "solution": record.get("solution"),
                        "tags": tags,
                    },
                )
            )
    return samples


def _normalized_omni_math_2_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip() and item.strip().lower() not in {"none", "null", "[]"}]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip() and str(item).strip().lower() not in {"none", "null"}]
    return [str(value).strip()] if str(value).strip() else []


def _omni_math_2_stratum(record: dict[str, Any]) -> str:
    domain = record.get("domain")
    primary_domain = str(domain[0] if isinstance(domain, list) and domain else domain or "unknown")
    try:
        difficulty_band = int(float(record.get("difficulty")) // 2 * 2)
    except (TypeError, ValueError):
        difficulty_band = -1
    return f"{primary_domain}|difficulty_{difficulty_band}"


def _omni_math_stratum(record: dict[str, Any]) -> str:
    primary_domain = str(next(iter(record.get("domain") or []), "unknown"))
    try:
        difficulty_band = int(float(record.get("difficulty")) // 2 * 2)
    except (TypeError, ValueError):
        difficulty_band = -1
    return f"{primary_domain}|difficulty_{difficulty_band}"


def _load_bbeh(config: BenchmarkConfig) -> list[DatasetSample]:
    path = resolve_dataset_source_path(config.source_path)
    task_payloads: list[tuple[str, dict[str, Any]]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if "/benchmark_tasks/bbeh_" in name and name.endswith("/task.json")
            ]
            for member in sorted(members):
                task_name = Path(member).parent.name.removeprefix("bbeh_")
                with archive.open(member) as handle:
                    task_payloads.append((task_name, json.loads(handle.read().decode("utf-8"))))
    else:
        for task_path in sorted(path.rglob("benchmark_tasks/bbeh_*/task.json")):
            task_name = task_path.parent.name.removeprefix("bbeh_")
            task_payloads.append((task_name, json.loads(task_path.read_text(encoding="utf-8"))))

    samples: list[DatasetSample] = []
    raw_index = 0
    for task_name, payload in task_payloads:
        for task_index, record in enumerate(payload.get("examples") or []):
            question = str(record.get("input") or "").strip()
            answer_contract = _parse_bbeh_answer_contract(question)
            options = list(answer_contract["options"])
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=f"{config.sample_id_prefix}-{task_name}-{task_index:04d}",
                    question=question,
                    reference_answer=str(record.get("target") or "").strip(),
                    prompt_context="",
                    metadata={
                        "raw_index": raw_index,
                        "task": task_name,
                        "task_index": task_index,
                        "options": options,
                        "answer_contract": answer_contract,
                        "options_block_start": answer_contract["block_start"],
                        "options_block_end": answer_contract["block_end"],
                        "options_trailing_note": answer_contract["trailing_note"],
                        "option_selection_mode": answer_contract["selection_mode"],
                    },
                )
            )
            raw_index += 1
    return samples


def _parse_bbeh_options(question: str) -> list[dict[str, str]]:
    """Extract the sample's structured BBEH option table."""

    return list(_parse_bbeh_answer_contract(question)["options"])


def _parse_bbeh_answer_contract(question: str) -> dict[str, Any]:
    """Return the exact answer contract embedded in a BBEH question.

    BBEH uses both titled terminal option blocks and four task families whose
    choices are rendered directly as consecutive ``(A) ...`` lines.  The
    latter were previously treated as free text, which split a label and its
    option text into artificial answer classes.
    """

    source = str(question or "")
    titled = _parse_bbeh_option_block(source)
    if titled is not None:
        return {
            "kind": "single_choice",
            "options": list(titled["options"]),
            "block_start": int(titled["start"]),
            "block_end": int(titled["end"]),
            "source_style": "titled_terminal",
            "selection_mode": "single",
            "trailing_note": str(titled["trailing_note"]),
        }
    # A malformed titled table must not be reinterpreted as an inline table.
    if re.search(r"(?:^|\n)Options:\s*\n", source):
        return _free_text_bbeh_contract()

    inline = _parse_bbeh_inline_option_block(source)
    if inline is None:
        return _free_text_bbeh_contract()
    concatenated = bool(
        re.search(
            r"concatenation\s+of\s+all\s+the\s+correct\s+choices",
            source,
            flags=re.IGNORECASE,
        )
    )
    return {
        "kind": "multi_choice" if concatenated else "single_choice",
        "options": list(inline["options"]),
        "block_start": int(inline["start"]),
        "block_end": int(inline["end"]),
        "source_style": "inline",
        "selection_mode": "concatenated" if concatenated else "single",
        "trailing_note": "",
    }


def _free_text_bbeh_contract() -> dict[str, Any]:
    return {
        "kind": "free_text",
        "options": [],
        "block_start": None,
        "block_end": None,
        "source_style": "none",
        "selection_mode": "none",
        "trailing_note": "",
    }


def _parse_bbeh_inline_option_block(question: str) -> dict[str, Any] | None:
    """Find one unique longest, consecutive, A-led inline option table."""

    source = str(question or "")
    matches = list(
        re.finditer(r"(?m)^\s*\(([A-Za-z])\)\s+(.+?)\s*$", source)
    )
    sequences: list[list[re.Match[str]]] = []
    for start_index, match in enumerate(matches):
        if match.group(1).upper() != "A":
            continue
        expected = ord("A")
        sequence: list[re.Match[str]] = []
        for candidate in matches[start_index:]:
            if ord(candidate.group(1).upper()) != expected:
                break
            sequence.append(candidate)
            expected += 1
        if len(sequence) >= 2:
            sequences.append(sequence)
    if not sequences:
        return None
    longest = max(len(sequence) for sequence in sequences)
    winners = [sequence for sequence in sequences if len(sequence) == longest]
    if len(winners) != 1:
        return None
    winner = winners[0]
    return {
        "options": [
            {"label": match.group(1).upper(), "text": match.group(2).strip()}
            for match in winner
        ],
        "start": winner[0].start(),
        "end": winner[-1].end(),
    }


def _parse_bbeh_option_block(question: str) -> dict[str, Any] | None:
    """Return the final BBEH option table and its source boundary.

    ``geometric_shapes`` appends a fixed coordinate-rounding note after its
    choices.  That corpus-authored note is part of the terminal option region,
    not an additional choice or an arbitrary explanation.
    """

    source = str(question or "")
    markers = list(re.finditer(r"(?:^|\n)(?P<label>Options:)\s*\n", source))
    marker = markers[-1] if markers else None
    if marker is None:
        return None
    options: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    option_lines: list[str] = []
    trailing_lines: list[str] = []
    in_trailing_note = False
    for line in source[marker.end() :].splitlines():
        matched = re.fullmatch(r"\s*\(([A-Za-z])\)\s+(.+?)\s*", line)
        if matched is None:
            in_trailing_note = True
            trailing_lines.append(line)
            continue
        if in_trailing_note:
            return None
        label = matched.group(1).upper()
        text = matched.group(2).strip()
        if label in seen_labels or not text:
            return None
        seen_labels.add(label)
        options.append({"label": label, "text": text})
        option_lines.append(line)
    if not options:
        return None
    trailing_note = "\n".join(trailing_lines).strip()
    if trailing_note and re.fullmatch(
        r"Coordinates have been rounded to \d+ decimal places so ignore slight differences\.",
        trailing_note,
        flags=re.IGNORECASE,
    ) is None:
        return None
    option_start = marker.start("label")
    option_text = "\n".join(option_lines)
    option_end = marker.end() + len(option_text)
    return {
        "options": options,
        "start": option_start,
        "end": option_end,
        "trailing_note": trailing_note,
    }


def question_without_answer_contract(sample: DatasetSample) -> str:
    """Remove a structured answer region before blinded measurement."""

    if sample.dataset != "bbeh":
        return sample.question
    contract = sample.metadata.get("answer_contract")
    start = contract.get("block_start") if isinstance(contract, dict) else None
    if not isinstance(start, int):
        start = sample.metadata.get("options_block_start")
    if isinstance(start, int) and 0 <= start <= len(sample.question):
        return sample.question[:start].rstrip()
    parsed = _parse_bbeh_answer_contract(sample.question)
    start = parsed.get("block_start")
    if not isinstance(start, int):
        return sample.question
    return sample.question[:start].rstrip()


def question_without_bbeh_options(sample: DatasetSample) -> str:
    """Backward-compatible alias for answer-contract removal."""

    return question_without_answer_contract(sample)


def _load_competition_math(config: BenchmarkConfig) -> list[DatasetSample]:
    path = resolve_dataset_source_path(config.source_path)
    records: list[tuple[str, dict[str, Any]]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            member_names = [
                name
                for name in archive.namelist()
                if name.startswith("MATH/test/") and name.endswith(".json")
            ]
            for member_name in sorted(member_names):
                with archive.open(member_name) as handle:
                    records.append((member_name, json.loads(handle.read().decode("utf-8"))))
    else:
        test_root = path / "MATH" / "test" if path.is_dir() and (path / "MATH" / "test").exists() else path / "test"
        for member_path in sorted(test_root.rglob("*.json")):
            records.append((member_path.relative_to(test_root.parent).as_posix(), json.loads(member_path.read_text(encoding="utf-8"))))

    samples: list[DatasetSample] = []
    for index, (member_name, record) in enumerate(records):
        member_parts = Path(member_name).parts
        subject = str(record.get("type") or (member_parts[2] if len(member_parts) >= 3 else "")).strip()
        sample_id = str(record.get("unique_id") or Path(member_name).with_suffix("").as_posix())
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("problem") or "").strip(),
                reference_answer=str(record.get("answer") or "").strip(),
                prompt_context="",
                metadata={
                    "raw_index": index,
                    "subject": subject,
                    "level": record.get("level"),
                    "unique_id": sample_id,
                    "solution": record.get("solution"),
                    "source_member": member_name,
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


def _load_realmistake_error_detection(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 ReaLMistake 官方压缩包中的某个错误检测任务。"""

    path = resolve_dataset_source_path(config.source_path)
    archive_member_prefix = (config.archive_member or "").strip().strip("/\\")
    archive_password = (config.archive_password or "").encode("utf-8") if config.archive_password else None
    if not archive_member_prefix:
        raise ValueError(f"ReaLMistake benchmark {config.slug} requires a non-empty archive_member prefix.")

    records = _load_realmistake_records(path, archive_member_prefix=archive_member_prefix, archive_password=archive_password)
    samples: list[DatasetSample] = []
    for index, record in enumerate(records):
        metadata = dict(record.get("metadata") or {})
        candidate_response = str(record.get("llm_response") or "").strip()
        verdict = str(record.get("error_label") or record.get("label") or "").strip()
        sample_id = str(metadata.get("id") or f"{config.sample_id_prefix}-{index:05d}")
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record.get("input") or "").strip(),
                reference_answer=verdict,
                prompt_context="",
                metadata={
                    "raw_index": index,
                    "task_name": str(metadata.get("task_name") or config.source_split or config.slug),
                    "task_source": metadata.get("task_source"),
                    "difficulty": metadata.get("difficulty"),
                    "candidate_response": candidate_response,
                    "candidate_response_model": str(metadata.get("llm_response_model") or "unknown"),
                    "error_categories": list(record.get("error_categories") or []),
                    "human_explanation": str(record.get("human_explanation") or "").strip(),
                    "metadata": metadata,
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
    with zipfile.ZipFile(resolve_dataset_source_path(config.source_path)) as archive, archive.open(archive_member, pwd=archive_password) as handle:
        reader = csv.DictReader(line.decode("utf-8") for line in handle)
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


def _load_realmistake_records(
    path: Path,
    *,
    archive_member_prefix: str,
    archive_password: bytes | None,
) -> list[dict[str, Any]]:
    """从 ReaLMistake 官方压缩包或已解压目录读取指定任务的全部记录。"""

    records: list[dict[str, Any]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            member_names = [
                name
                for name in archive.namelist()
                if name.startswith(f"{archive_member_prefix}/") and name.endswith(".jsonl")
            ]
            for member_name in sorted(member_names):
                with archive.open(member_name, pwd=archive_password) as handle:
                    for raw_line in handle:
                        line = raw_line.decode("utf-8").strip()
                        if line:
                            records.append(json.loads(line))
        return records

    task_dir = path
    if path.is_dir():
        candidate = path / Path(archive_member_prefix).name
        if candidate.exists():
            task_dir = candidate
    if task_dir.is_file():
        task_dir = task_dir.parent
    for member_path in sorted(task_dir.glob("*.jsonl")):
        with member_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


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
