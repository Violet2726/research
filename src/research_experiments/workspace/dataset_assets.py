"""项目级数据集资产下载、盘点与 split 重建工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import tomllib
import urllib.request

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from research_experiments.core.config import BenchmarkConfig, load_benchmark_config
from research_experiments.core.data.datasets import (
    generate_split_manifests,
    load_samples,
    resolve_dataset_source_path,
)
from research_experiments.workspace.layout import default_datasets_root
DATASETS_DOCS_ROOT = Path("datasets")
SPLITS_ROOT = Path("configs/core/shared/benchmarks/splits")
CONFIGS_ROOT = Path("configs")


@dataclass(frozen=True)
class DatasetAssetSpec:
    """单个数据集资产文件的下载说明。"""

    slug: str
    dataset_name: str
    asset_id: str
    purpose: str
    relative_path: Path
    source_kind: str
    source_label: str
    source_url: str
    source_split: str
    repo_id: str | None = None
    repo_type: str | None = None
    filename: str | None = None
    revision: str = "main"
    notes: str = ""
    runtime_required: bool = False


@dataclass(frozen=True)
class DatasetDownloadResult:
    """单个资产文件下载动作的结果。"""

    slug: str
    asset_id: str
    purpose: str
    local_path: Path
    source_label: str
    source_url: str
    status: str
    size_bytes: int


def discover_used_benchmark_config_paths(configs_root: str | Path = CONFIGS_ROOT) -> list[Path]:
    """扫描项目配置，找出实际用到的 benchmark 配置文件。"""
    root = Path(configs_root)
    discovered: set[Path] = set()

    shared_benchmarks_root = root / "core" / "shared" / "benchmarks"
    if shared_benchmarks_root.exists():
        for benchmark_path in shared_benchmarks_root.rglob("*.toml"):
            discovered.add(benchmark_path.resolve())

    for experiment_path in root.glob("families/*/experiments/*.toml"):
        payload = _load_toml(experiment_path)
        for raw_path in payload.get("benchmark_configs", []):
            resolved = _resolve_config_reference(Path(str(raw_path)), root)
            if resolved is not None:
                discovered.add(resolved)

    return sorted(discovered)


def load_used_benchmark_configs(configs_root: str | Path = CONFIGS_ROOT) -> list[BenchmarkConfig]:
    """加载项目当前实际用到的全部 benchmark 配置。"""
    return [load_benchmark_config(path) for path in discover_used_benchmark_config_paths(configs_root)]


def build_primary_dataset_specs(benchmarks: list[BenchmarkConfig]) -> list[DatasetAssetSpec]:
    """构建主评测源文件的下载清单。"""
    source_map = {
        "gpqa_diamond": DatasetAssetSpec(
            slug="gpqa_diamond",
            dataset_name="GPQA Diamond",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("gpqa/dataset.zip"),
            source_kind="http_file",
            source_label="GitHub raw",
            source_url="https://github.com/idavidrein/gpqa/raw/main/dataset.zip",
            source_split="diamond",
            notes="官方 zip 同时内嵌 gpqa_main、gpqa_experts 与 gpqa_extended；请遵循上游许可与使用说明。",
        ),
        "gsm8k": DatasetAssetSpec(
            slug="gsm8k",
            dataset_name="GSM8K",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("gsm8k/test.jsonl"),
            source_kind="http_file",
            source_label="GitHub raw",
            source_url="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
            source_split="test",
        ),
        "gsm_symbolic": DatasetAssetSpec(
            slug="gsm_symbolic",
            dataset_name="GSM-Symbolic",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("gsm-symbolic/GSM_symbolic.jsonl"),
            source_kind="hf_file",
            source_label="Hugging Face dataset",
            source_url="https://huggingface.co/datasets/apple/GSM-Symbolic/blob/main/main/test.jsonl",
            source_split="test",
            repo_id="apple/GSM-Symbolic",
            repo_type="dataset",
            filename="main/test.jsonl",
            notes="公开版本只提供生成后的 test 集。",
        ),
        "hotpotqa": DatasetAssetSpec(
            slug="hotpotqa",
            dataset_name="HotpotQA",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("hotpotqa/validation_distractor.parquet"),
            source_kind="hf_file",
            source_label="Hugging Face dataset",
            source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            source_split="validation_distractor",
            repo_id="hotpotqa/hotpot_qa",
            repo_type="dataset",
            filename="distractor/validation-00000-of-00001.parquet",
        ),
        "grailqa": DatasetAssetSpec(
            slug="grailqa",
            dataset_name="GrailQA",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("grailqa/validation.parquet"),
            source_kind="hf_file",
            source_label="Hugging Face dataset mirror",
            source_url="https://huggingface.co/datasets/Hieuman/grail_qa",
            source_split="validation",
            repo_id="Hieuman/grail_qa",
            repo_type="dataset",
            filename="data/validation-00000-of-00001.parquet",
            notes="上游官方主页提供下载入口；这里使用 Hugging Face parquet 镜像，便于单文件恢复与本地 split 重建。",
        ),
        "dog_grailqa": DatasetAssetSpec(
            slug="dog_grailqa",
            dataset_name="DoG GrailQA",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("dog-freebase/grailqa.json"),
            source_kind="http_file",
            source_label="DoG official GitHub",
            source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/grailqa.json",
            source_split="test",
            notes="DoG 官方仓提供的 GrailQA 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。",
        ),
        "math500": DatasetAssetSpec(
            slug="math500",
            dataset_name="MATH500",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("math500/test.jsonl"),
            source_kind="hf_file",
            source_label="Hugging Face dataset",
            source_url="https://huggingface.co/datasets/math-ai/Math-500",
            source_split="test",
            repo_id="math-ai/Math-500",
            repo_type="dataset",
            filename="test.jsonl",
            notes="官方公开数据仅提供 test 集。",
        ),
        "mmlu_pro": DatasetAssetSpec(
            slug="mmlu_pro",
            dataset_name="MMLU-Pro",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("mmlu-pro/test.parquet"),
            source_kind="hf_file",
            source_label="Hugging Face dataset",
            source_url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
            source_split="test",
            repo_id="TIGER-Lab/MMLU-Pro",
            repo_type="dataset",
            filename="data/test-00000-of-00001.parquet",
        ),
        "strategyqa": DatasetAssetSpec(
            slug="strategyqa",
            dataset_name="StrategyQA",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("strategyqa/dev.json"),
            source_kind="http_file",
            source_label="GitHub raw",
            source_url="https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/dev.json",
            source_split="dev",
        ),
        "webquestions": DatasetAssetSpec(
            slug="webquestions",
            dataset_name="WebQuestions",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("webquestions/test.json"),
            source_kind="http_file",
            source_label="GitHub raw",
            source_url="https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/main/test.json",
            source_split="test",
            notes="主文件只包含问题与答案。若要恢复更完整的图注释，请再下载 supplementary 里的 Freebase 路径与实体链接文件。",
        ),
        "dog_webquestions": DatasetAssetSpec(
            slug="dog_webquestions",
            dataset_name="DoG WebQuestions",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("dog-freebase/WebQuestions.json"),
            source_kind="http_file",
            source_label="DoG official GitHub",
            source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/WebQuestions.json",
            source_split="test",
            notes="DoG 官方仓提供的 WebQuestions 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。",
        ),
        "dog_webqsp": DatasetAssetSpec(
            slug="dog_webqsp",
            dataset_name="DoG WebQSP",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("dog-freebase/WebQSP.json"),
            source_kind="http_file",
            source_label="DoG official GitHub",
            source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/WebQSP.json",
            source_split="test",
            notes="DoG 官方仓提供的 WebQSP 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。",
        ),
        "dog_cwq": DatasetAssetSpec(
            slug="dog_cwq",
            dataset_name="DoG CWQ",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("dog-freebase/cwq.json"),
            source_kind="http_file",
            source_label="DoG official GitHub",
            source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/cwq.json",
            source_split="test",
            notes="DoG 官方仓提供的 CWQ 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。",
        ),
        "dog_metaqa_1hop": DatasetAssetSpec(
            slug="dog_metaqa_1hop",
            dataset_name="DoG MetaQA 1-hop",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("dog-metaqa/1-hop/qa_test.txt"),
            source_kind="http_file",
            source_label="DoG official GitHub",
            source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/1-hop/qa_test.txt",
            source_split="test",
            notes="DoG 官方仓提供的 MetaQA 1-hop 测试集；运行时还需要共享的 `dog-metaqa/kb.txt` 图后端。",
        ),
        "dog_metaqa_2hop": DatasetAssetSpec(
            slug="dog_metaqa_2hop",
            dataset_name="DoG MetaQA 2-hop",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("dog-metaqa/2-hop/qa_test.txt"),
            source_kind="http_file",
            source_label="DoG official GitHub",
            source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/2-hop/qa_test.txt",
            source_split="test",
            notes="DoG 官方仓提供的 MetaQA 2-hop 测试集；运行时还需要共享的 `dog-metaqa/kb.txt` 图后端。",
        ),
        "dog_metaqa_3hop": DatasetAssetSpec(
            slug="dog_metaqa_3hop",
            dataset_name="DoG MetaQA 3-hop",
            asset_id="evaluation",
            purpose="evaluation",
            relative_path=Path("dog-metaqa/3-hop/qa_test.txt"),
            source_kind="http_file",
            source_label="DoG official GitHub",
            source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/3-hop/qa_test.txt",
            source_split="test",
            notes="DoG 官方仓提供的 MetaQA 3-hop 测试集；运行时还需要共享的 `dog-metaqa/kb.txt` 图后端。",
        ),
    }
    specs: dict[str, DatasetAssetSpec] = {}
    for benchmark in benchmarks:
        spec = source_map.get(benchmark.slug)
        if spec is None:
            raise ValueError(f"Unsupported benchmark slug for primary download: {benchmark.slug}")
        specs[benchmark.slug] = spec
    return [specs[slug] for slug in sorted(specs)]


def build_supplementary_dataset_specs(benchmarks: list[BenchmarkConfig]) -> list[DatasetAssetSpec]:
    """构建训练集及补充上游 split 的下载清单。"""
    source_map = {
        "gsm8k": [
            DatasetAssetSpec(
                slug="gsm8k",
                dataset_name="GSM8K",
                asset_id="train",
                purpose="train",
                relative_path=Path("gsm8k/train.jsonl"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
                source_split="train",
            )
        ],
        "strategyqa": [
            DatasetAssetSpec(
                slug="strategyqa",
                dataset_name="StrategyQA",
                asset_id="train",
                purpose="train",
                relative_path=Path("strategyqa/train.json"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json",
                source_split="train",
            )
        ],
        "hotpotqa": [
            DatasetAssetSpec(
                slug="hotpotqa",
                dataset_name="HotpotQA",
                asset_id="train_shard_0",
                purpose="train",
                relative_path=Path("hotpotqa/distractor/train-00000-of-00002.parquet"),
                source_kind="hf_file",
                source_label="Hugging Face dataset",
                source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa",
                source_split="distractor_train",
                repo_id="hotpotqa/hotpot_qa",
                repo_type="dataset",
                filename="distractor/train-00000-of-00002.parquet",
            ),
            DatasetAssetSpec(
                slug="hotpotqa",
                dataset_name="HotpotQA",
                asset_id="train_shard_1",
                purpose="train",
                relative_path=Path("hotpotqa/distractor/train-00001-of-00002.parquet"),
                source_kind="hf_file",
                source_label="Hugging Face dataset",
                source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa",
                source_split="distractor_train",
                repo_id="hotpotqa/hotpot_qa",
                repo_type="dataset",
                filename="distractor/train-00001-of-00002.parquet",
            ),
        ],
        "mmlu_pro": [
            DatasetAssetSpec(
                slug="mmlu_pro",
                dataset_name="MMLU-Pro",
                asset_id="validation",
                purpose="validation",
                relative_path=Path("mmlu-pro/validation.parquet"),
                source_kind="hf_file",
                source_label="Hugging Face dataset",
                source_url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
                source_split="validation",
                repo_id="TIGER-Lab/MMLU-Pro",
                repo_type="dataset",
                filename="data/validation-00000-of-00001.parquet",
                notes="官方未公开 train split；这里只保留 validation 作为唯一非测试补充源。",
            )
        ],
        "grailqa": [
            DatasetAssetSpec(
                slug="grailqa",
                dataset_name="GrailQA",
                asset_id="train",
                purpose="train",
                relative_path=Path("grailqa/train.parquet"),
                source_kind="hf_file",
                source_label="Hugging Face dataset mirror",
                source_url="https://huggingface.co/datasets/Hieuman/grail_qa",
                source_split="train",
                repo_id="Hieuman/grail_qa",
                repo_type="dataset",
                filename="data/train-00000-of-00001.parquet",
            ),
            DatasetAssetSpec(
                slug="grailqa",
                dataset_name="GrailQA",
                asset_id="test",
                purpose="test_public",
                relative_path=Path("grailqa/test.parquet"),
                source_kind="hf_file",
                source_label="Hugging Face dataset mirror",
                source_url="https://huggingface.co/datasets/Hieuman/grail_qa",
                source_split="test",
                repo_id="Hieuman/grail_qa",
                repo_type="dataset",
                filename="data/test-00000-of-00001.parquet",
                notes="用于额外泛化检查；正式 family v1 默认仍以 validation split 进入 count20/count100/count300。",
            ),
        ],
        "webquestions": [
            DatasetAssetSpec(
                slug="webquestions",
                dataset_name="WebQuestions",
                asset_id="question_dump_test",
                purpose="annotation",
                relative_path=Path("webquestions/question_dump_test.json"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-dump/test.json",
                source_split="test",
                notes="YodaQA 生成的问题概念、clue 与词汇注释，可用于更稳的 topic seed 与角色提示。",
                runtime_required=True,
            ),
            DatasetAssetSpec(
                slug="webquestions",
                dataset_name="WebQuestions",
                asset_id="freebase_key_test",
                purpose="annotation",
                relative_path=Path("webquestions/freebase_key_test.json"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase/test.json",
                source_split="test",
                notes="官方提供的单实体 Freebase key 注释，可作为 topic seed。",
                runtime_required=True,
            ),
            DatasetAssetSpec(
                slug="webquestions",
                dataset_name="WebQuestions",
                asset_id="freebase_mids_test",
                purpose="annotation",
                relative_path=Path("webquestions/freebase_mids_test.json"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-mids/test.json",
                source_split="test",
                notes="问题概念到 Freebase MID 的链接结果。",
                runtime_required=True,
            ),
            DatasetAssetSpec(
                slug="webquestions",
                dataset_name="WebQuestions",
                asset_id="relation_paths_test",
                purpose="annotation",
                relative_path=Path("webquestions/relation_paths_test.json"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-rp/test.json",
                source_split="test",
                notes="官方发布的关系路径注释，是 v1 WebQuestions 图视角的主要证据源。",
                runtime_required=True,
            ),
            DatasetAssetSpec(
                slug="webquestions",
                dataset_name="WebQuestions",
                asset_id="branched_relation_paths_test",
                purpose="annotation",
                relative_path=Path("webquestions/branched_relation_paths_test.json"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-brp/test.json",
                source_split="test",
                notes="Branched relation paths 是 relation-path 视角的更丰富补充；若存在，loader 会优先使用它。",
                runtime_required=True,
            ),
            DatasetAssetSpec(
                slug="webquestions",
                dataset_name="WebQuestions",
                asset_id="entities_test",
                purpose="annotation",
                relative_path=Path("webquestions/entities_test.json"),
                source_kind="http_file",
                source_label="GitHub raw",
                source_url="https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-entities/test.json",
                source_split="test",
                notes="问题实体识别结果，用于 neighborhood 视角构造。",
                runtime_required=True,
            ),
        ],
        "dog_metaqa_1hop": [
            DatasetAssetSpec(
                slug="dog_metaqa_1hop",
                dataset_name="DoG MetaQA",
                asset_id="kb",
                purpose="backend",
                relative_path=Path("dog-metaqa/kb.txt"),
                source_kind="http_file",
                source_label="DoG official GitHub",
                source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/kb.txt",
                source_split="shared",
                notes="MetaQA 论文复现共享知识图谱后端，供 1/2/3-hop 三个 benchmark 共用。",
                runtime_required=True,
            )
        ],
        "dog_metaqa_2hop": [
            DatasetAssetSpec(
                slug="dog_metaqa_2hop",
                dataset_name="DoG MetaQA",
                asset_id="kb",
                purpose="backend",
                relative_path=Path("dog-metaqa/kb.txt"),
                source_kind="http_file",
                source_label="DoG official GitHub",
                source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/kb.txt",
                source_split="shared",
                notes="MetaQA 论文复现共享知识图谱后端，供 1/2/3-hop 三个 benchmark 共用。",
                runtime_required=True,
            )
        ],
        "dog_metaqa_3hop": [
            DatasetAssetSpec(
                slug="dog_metaqa_3hop",
                dataset_name="DoG MetaQA",
                asset_id="kb",
                purpose="backend",
                relative_path=Path("dog-metaqa/kb.txt"),
                source_kind="http_file",
                source_label="DoG official GitHub",
                source_url="https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/kb.txt",
                source_split="shared",
                notes="MetaQA 论文复现共享知识图谱后端，供 1/2/3-hop 三个 benchmark 共用。",
                runtime_required=True,
            )
        ],
    }
    specs: list[DatasetAssetSpec] = []
    for benchmark in sorted(benchmarks, key=lambda item: item.slug):
        specs.extend(source_map.get(benchmark.slug, []))
    return _deduplicate_dataset_specs(specs)


def build_runtime_support_dataset_specs(benchmarks: list[BenchmarkConfig]) -> list[DatasetAssetSpec]:
    """构建公开可下载且属于运行必需品的补充资产清单。"""
    return [spec for spec in build_supplementary_dataset_specs(benchmarks) if spec.runtime_required]


def download_primary_dataset_sources(
    benchmarks: list[BenchmarkConfig],
    *,
    force: bool = False,
) -> list[DatasetDownloadResult]:
    """下载主评测源文件。"""
    return _download_specs(build_primary_dataset_specs(benchmarks), force=force)


def download_supplementary_dataset_sources(
    benchmarks: list[BenchmarkConfig],
    *,
    force: bool = False,
) -> list[DatasetDownloadResult]:
    """下载训练集与可用的非测试补充源。"""
    return _download_specs(build_supplementary_dataset_specs(benchmarks), force=force)


def download_runtime_support_dataset_sources(
    benchmarks: list[BenchmarkConfig],
    *,
    force: bool = False,
) -> list[DatasetDownloadResult]:
    """下载公开可得且运行实验所必需的补充资产。"""
    return _download_specs(build_runtime_support_dataset_specs(benchmarks), force=force)


def regenerate_used_dataset_splits(
    benchmarks: list[BenchmarkConfig],
    *,
    output_dir: str | Path = SPLITS_ROOT,
) -> list[Path]:
    """按当前 benchmark 配置重建冻结 split。"""
    return generate_split_manifests(benchmarks, output_dir)


def collect_dataset_inventory(
    benchmarks: list[BenchmarkConfig],
    *,
    splits_root: str | Path = SPLITS_ROOT,
) -> dict[str, object]:
    """收集主评测源与补充源的盘点信息。"""
    primary_specs = build_primary_dataset_specs(benchmarks)
    supplementary_specs = build_supplementary_dataset_specs(benchmarks)
    primary_spec_by_slug = {spec.slug: spec for spec in primary_specs}

    primary_assets: list[dict[str, object]] = []
    splits_root_path = Path(splits_root)
    for benchmark in sorted(benchmarks, key=lambda item: item.slug):
        spec = primary_spec_by_slug[benchmark.slug]
        source_path = resolve_dataset_source_path(benchmark.source_path)
        samples = load_samples(benchmark)
        primary_assets.append(
            {
                "slug": benchmark.slug,
                "name": benchmark.name,
                "benchmark_config_path": benchmark.config_path,
                "config_source_path": benchmark.source_path,
                "local_path": source_path.as_posix(),
                "source_split": benchmark.source_split,
                "source_label": spec.source_label,
                "source_url": spec.source_url,
                "sample_count": len(samples),
                "size_bytes": source_path.stat().st_size if source_path.exists() else 0,
                "split_files": sorted(
                    path.relative_to(splits_root_path).as_posix()
                    for path in splits_root_path.rglob(f"{Path(str(benchmark.cache_namespace or benchmark.slug)).name}-*.json")
                    if path.parent.as_posix().endswith(Path(str(benchmark.cache_namespace or benchmark.slug)).parent.as_posix())
                    or Path(str(benchmark.cache_namespace or benchmark.slug)).parent.as_posix() == "."
                ),
                "notes": spec.notes,
            }
        )

    supplementary_assets = [_describe_asset(spec) for spec in supplementary_specs]
    unavailable_assets = _describe_unavailable_supplementary_assets(benchmarks)

    return {
        "datasets_root": default_datasets_root(),
        "dataset_count": len(primary_assets),
        "primary_assets": primary_assets,
        "supplementary_assets": supplementary_assets,
        "unavailable_supplementary_assets": unavailable_assets,
    }


def write_dataset_inventory_files(
    benchmarks: list[BenchmarkConfig],
    *,
    datasets_root: str | Path | None = None,
    docs_root: str | Path = DATASETS_DOCS_ROOT,
    splits_root: str | Path = SPLITS_ROOT,
) -> dict[str, Path]:
    """刷新本地数据集清单与仓库内说明文档。"""
    datasets_root_path = Path(datasets_root) if datasets_root is not None else _current_datasets_root()
    docs_root_path = Path(docs_root)
    datasets_root_path.mkdir(parents=True, exist_ok=True)
    docs_root_path.mkdir(parents=True, exist_ok=True)
    inventory = collect_dataset_inventory(benchmarks, splits_root=splits_root)

    manifest_path = datasets_root_path / "manifest.json"
    manifest_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_lines = [
        "# datasets",
        "",
        "这个目录不再承载正式数据文件。",
        "",
        "项目的数据集资产现在统一放在本地工作区 `local/datasets/`，并通过 `research_cli tools dataset-assets` 一键恢复公开可下载的数据文件与冻结 split。",
        "",
        "这组命令只负责数据资产、split 与盘点文档，不负责外部后端服务、数据库、模型访问凭证或 family 运行时依赖的安装配置。",
        "",
        "这样做的目标是：",
        "",
        "- 不把上游原始数据作为 Git 大文件长期提交",
        "- 保持 benchmark split 可复现",
        "- 把合规说明、恢复命令和本地资产边界写清楚",
        "",
        "## 层级约束",
        "",
        "- `configs/core/shared/benchmarks/` 下的 benchmark 配置必须镜像 `local/datasets/` 的相对路径层级，并使用“去掉数据文件扩展名后的路径”作为配置路径。",
        "- `local/cache/providers/<provider>/<model>/...` 下的数据集缓存分片必须使用同一套层级键，避免把方法名或实验线名写成 dataset shard 名。",
        "- 示例：`local/datasets/dog-freebase/cwq.json` 对应 `configs/core/shared/benchmarks/dog-freebase/cwq.toml` 与 `local/cache/providers/<provider>/<model>/dog-freebase/cwq/requests.sqlite`。",
        "",
        "## 当前本地资产根目录",
        "",
        f"- 默认路径：`{inventory['datasets_root']}`",
        "- 环境变量覆盖：`RESEARCH_DATASETS_ROOT`",
        "",
        "## 一键恢复",
        "",
        "恢复主评测源、公开可下载的运行必需补充资产，并重建 split：",
        "",
        "```powershell",
        "uv run research_cli tools dataset-assets prepare-used",
        "```",
        "",
        "同时恢复训练集与可公开下载的验证补充源：",
        "",
        "```powershell",
        "uv run research_cli tools dataset-assets prepare-all-sources",
        "```",
        "",
        "强制覆盖已有本地文件：",
        "",
        "```powershell",
        "uv run research_cli tools dataset-assets prepare-all-sources --force",
        "```",
        "",
        "说明：",
        "",
        "- `prepare-used` 会下载当前项目实际用到的主评测数据资产，以及公开可下载的运行必需补充资产，重建 frozen split，并刷新 `datasets/README.md` 与 `local/datasets/manifest.json`。",
        "- `prepare-all-sources` 会额外下载公开可得的训练集、验证集与注释补充源，但仍只处理数据资产本身。",
        "- 这些命令不会自动安装或启动外部后端，例如 Freebase/Virtuoso、SPARQL 服务或其他 family 专属运行时依赖。",
        "",
        "## 主评测源文件",
        "",
    ]

    for item in inventory["primary_assets"]:
        readme_lines.extend(
            [
                f"### {item['name']} (`{item['slug']}`)",
                "",
                f"- benchmark 配置：`{item['benchmark_config_path']}`",
                f"- 数据相对路径：`{item['config_source_path']}`",
                f"- 本地资产：`{item['local_path']}`",
                f"- 上游来源：{item['source_label']}，`{item['source_url']}`",
                f"- 上游 split：`{item['source_split']}`",
                f"- 样本数：`{item['sample_count']}`",
                f"- 文件大小：`{_format_bytes(int(item['size_bytes']))}`",
                f"- 冻结 split：{', '.join(f'`{name}`' for name in item['split_files'])}",
            ]
        )
        if item["notes"]:
            readme_lines.append(f"- 说明：{item['notes']}")
        readme_lines.append("")

    readme_lines.extend(["## 训练集与补充上游 split", ""])
    supplementary_assets = inventory["supplementary_assets"]
    if supplementary_assets:
        for item in supplementary_assets:
            readme_lines.extend(
                [
                    f"### {item['dataset_name']} / {item['asset_id']}",
                    "",
                    f"- 本地资产：`{item['local_path']}`",
                    f"- 用途：`{item['purpose']}`",
                    f"- 上游来源：{item['source_label']}，`{item['source_url']}`",
                    f"- 上游 split：`{item['source_split']}`",
                    f"- 文件大小：`{_format_bytes(int(item['size_bytes']))}`" if item["size_bytes"] else "- 文件大小：尚未下载",
                ]
            )
            if item["sample_count"] is not None:
                readme_lines.append(f"- 样本数：`{item['sample_count']}`")
            if item["notes"]:
                readme_lines.append(f"- 说明：{item['notes']}")
            readme_lines.append("")
    else:
        readme_lines.extend(["当前没有额外可下载的训练或验证补充源。", ""])

    unavailable_assets = inventory["unavailable_supplementary_assets"]
    if unavailable_assets:
        readme_lines.extend(["## 未公开或不建议镜像的补充源", ""])
        for item in unavailable_assets:
            readme_lines.append(f"- `{item['slug']}`：{item['reason']}")
        readme_lines.append("")

    readme_lines.extend(
        [
            "## 合规与治理说明",
            "",
            "- 本仓库只保留下载逻辑、冻结 split 和说明文档，不默认镜像上游原始数据。",
            "- 具体使用时请遵循各数据集上游仓库、自带 license 文件以及发布页面的约束。",
            "- `gpqa_diamond` 这种带官方压缩包与密码的资产，尤其应以原始发布方说明为准，不建议再公开二次分发。",
            "- 本地盘点清单位于 `local/datasets/manifest.json`，用于脚本读取，不进入 Git 主线。",
        ]
    )

    readme_path = docs_root_path / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return {"readme": readme_path, "manifest": manifest_path}


def prepare_used_datasets(
    *,
    configs_root: str | Path = CONFIGS_ROOT,
    splits_root: str | Path = SPLITS_ROOT,
    force: bool = False,
) -> dict[str, object]:
    """下载主评测数据资产与公开运行依赖、重建 split，并刷新数据集清单文档。"""
    benchmarks = load_used_benchmark_configs(configs_root)
    primary_downloads = download_primary_dataset_sources(benchmarks, force=force)
    runtime_support_downloads = download_runtime_support_dataset_sources(benchmarks, force=force)
    downloads = [*primary_downloads, *runtime_support_downloads]
    created_splits = regenerate_used_dataset_splits(benchmarks, output_dir=splits_root)
    inventory_paths = write_dataset_inventory_files(benchmarks, datasets_root=_current_datasets_root(), docs_root=DATASETS_DOCS_ROOT, splits_root=splits_root)
    inventory = collect_dataset_inventory(benchmarks, splits_root=splits_root)
    return {
        "benchmark_count": len(benchmarks),
        "primary_downloads": [_serialize_download_result(item) for item in primary_downloads],
        "runtime_support_downloads": [_serialize_download_result(item) for item in runtime_support_downloads],
        "downloads": [_serialize_download_result(item) for item in downloads],
        "splits_created": [path.as_posix() for path in created_splits],
        "inventory": inventory,
        "readme_path": inventory_paths["readme"].as_posix(),
        "manifest_path": inventory_paths["manifest"].as_posix(),
    }


def prepare_all_dataset_sources(
    *,
    configs_root: str | Path = CONFIGS_ROOT,
    splits_root: str | Path = SPLITS_ROOT,
    force: bool = False,
) -> dict[str, object]:
    """下载公开可得的数据资产、重建 split，并刷新数据集清单文档。"""
    benchmarks = load_used_benchmark_configs(configs_root)
    primary_downloads = download_primary_dataset_sources(benchmarks, force=force)
    supplementary_downloads = download_supplementary_dataset_sources(benchmarks, force=force)
    created_splits = regenerate_used_dataset_splits(benchmarks, output_dir=splits_root)
    inventory_paths = write_dataset_inventory_files(benchmarks, datasets_root=_current_datasets_root(), docs_root=DATASETS_DOCS_ROOT, splits_root=splits_root)
    inventory = collect_dataset_inventory(benchmarks, splits_root=splits_root)
    return {
        "benchmark_count": len(benchmarks),
        "primary_downloads": [_serialize_download_result(item) for item in primary_downloads],
        "supplementary_downloads": [_serialize_download_result(item) for item in supplementary_downloads],
        "splits_created": [path.as_posix() for path in created_splits],
        "inventory": inventory,
        "readme_path": inventory_paths["readme"].as_posix(),
        "manifest_path": inventory_paths["manifest"].as_posix(),
    }


def _download_specs(specs: list[DatasetAssetSpec], *, force: bool) -> list[DatasetDownloadResult]:
    results: list[DatasetDownloadResult] = []
    for spec in specs:
        local_path = _resolve_asset_path(spec.relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and not force:
            results.append(
                DatasetDownloadResult(
                    slug=spec.slug,
                    asset_id=spec.asset_id,
                    purpose=spec.purpose,
                    local_path=local_path,
                    source_label=spec.source_label,
                    source_url=spec.source_url,
                    status="skipped",
                    size_bytes=local_path.stat().st_size,
                )
            )
            continue
        downloaded = _download_source_file(spec, local_path)
        shutil.copyfile(downloaded, local_path)
        if downloaded != local_path and downloaded.name.startswith(".") and downloaded.suffix == ".download":
            downloaded.unlink(missing_ok=True)
        results.append(
            DatasetDownloadResult(
                slug=spec.slug,
                asset_id=spec.asset_id,
                purpose=spec.purpose,
                local_path=local_path,
                source_label=spec.source_label,
                source_url=spec.source_url,
                status="downloaded",
                size_bytes=local_path.stat().st_size,
            )
        )
    return results


def _download_source_file(spec: DatasetAssetSpec, local_path: Path) -> Path:
    if spec.source_kind == "hf_file":
        if spec.repo_id is None or spec.repo_type is None or spec.filename is None:
            raise ValueError(f"Incomplete Hugging Face download spec for {spec.slug}:{spec.asset_id}.")
        return Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type=spec.repo_type,
                filename=spec.filename,
                revision=spec.revision,
            )
        )
    if spec.source_kind == "http_file":
        temporary_path = local_path.parent / f".{local_path.name}.download"
        with urllib.request.urlopen(spec.source_url, timeout=300) as response:
            temporary_path.write_bytes(response.read())
        return temporary_path
    raise ValueError(f"Unsupported source kind: {spec.source_kind}")


def _describe_asset(spec: DatasetAssetSpec) -> dict[str, object]:
    local_path = _resolve_asset_path(spec.relative_path)
    size_bytes = local_path.stat().st_size if local_path.exists() else 0
    sample_count = _count_samples_from_file(local_path) if local_path.exists() else None
    return {
        "slug": spec.slug,
        "dataset_name": spec.dataset_name,
        "asset_id": spec.asset_id,
        "purpose": spec.purpose,
        "local_path": local_path.as_posix(),
        "source_label": spec.source_label,
        "source_url": spec.source_url,
        "source_split": spec.source_split,
        "size_bytes": size_bytes,
        "sample_count": sample_count,
        "notes": spec.notes,
    }


def _describe_unavailable_supplementary_assets(benchmarks: list[BenchmarkConfig]) -> list[dict[str, str]]:
    known_unavailable = {
        "gpqa_diamond": "官方未提供独立 train split；补充题型已内嵌在 dataset.zip 中，不建议额外镜像分发。",
        "gsm_symbolic": "官方公开版本只提供生成后的 test 集。",
        "math500": "官方公开版本只提供 test 集。",
        "dog_webquestions": "需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。",
        "dog_grailqa": "需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。",
        "dog_webqsp": "需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。",
        "dog_cwq": "需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。",
    }
    items: list[dict[str, str]] = []
    for benchmark in sorted(benchmarks, key=lambda item: item.slug):
        reason = known_unavailable.get(benchmark.slug)
        if reason is not None:
            items.append({"slug": benchmark.slug, "reason": reason})
    return items


def _deduplicate_dataset_specs(specs: list[DatasetAssetSpec]) -> list[DatasetAssetSpec]:
    unique: dict[tuple[str, str, str, str], DatasetAssetSpec] = {}
    for spec in specs:
        key = (
            spec.relative_path.as_posix(),
            spec.asset_id,
            spec.purpose,
            spec.source_url,
        )
        unique.setdefault(key, spec)
    return list(unique.values())


def _count_samples_from_file(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return len(payload)
        return None
    if suffix == ".parquet":
        return pq.read_table(path).num_rows
    return None


def _resolve_asset_path(relative_path: Path) -> Path:
    return resolve_dataset_source_path(relative_path)


def _current_datasets_root() -> Path:
    return Path(default_datasets_root())


def _resolve_config_reference(path: Path, configs_root: Path) -> Path | None:
    candidate = (configs_root.parent / path).resolve()
    if candidate.exists():
        return candidate
    candidate = (configs_root / path).resolve()
    if candidate.exists():
        return candidate
    if path.exists():
        return path.resolve()
    return None


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _serialize_download_result(result: DatasetDownloadResult) -> dict[str, object]:
    return {
        "slug": result.slug,
        "asset_id": result.asset_id,
        "purpose": result.purpose,
        "local_path": result.local_path.as_posix(),
        "source_label": result.source_label,
        "source_url": result.source_url,
        "status": result.status,
        "size_bytes": result.size_bytes,
    }


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MiB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GiB"

