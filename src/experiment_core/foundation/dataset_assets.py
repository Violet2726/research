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

from experiment_core.foundation.config import BenchmarkConfig, load_benchmark_config
from experiment_core.foundation.datasets import (
    generate_split_manifests,
    load_samples,
    resolve_dataset_source_path,
)
from experiment_core.foundation.workspace import default_datasets_root
DATASETS_DOCS_ROOT = Path("datasets")
SPLITS_ROOT = Path("configs/shared/benchmarks/splits")
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

    shared_benchmarks_root = root / "shared" / "benchmarks"
    if shared_benchmarks_root.exists():
        for benchmark_path in shared_benchmarks_root.glob("*.toml"):
            discovered.add(benchmark_path.resolve())

    for experiment_path in root.glob("*/experiments/*.toml"):
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
    }
    specs: list[DatasetAssetSpec] = []
    for benchmark in sorted(benchmarks, key=lambda item: item.slug):
        specs.extend(source_map.get(benchmark.slug, []))
    return specs


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
                "config_source_path": benchmark.source_path,
                "local_path": source_path.as_posix(),
                "source_split": benchmark.source_split,
                "source_label": spec.source_label,
                "source_url": spec.source_url,
                "sample_count": len(samples),
                "size_bytes": source_path.stat().st_size if source_path.exists() else 0,
                "split_files": sorted(
                    path.relative_to(splits_root_path).as_posix()
                    for path in splits_root_path.rglob(f"{benchmark.slug}-*.json")
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
        "项目的数据集资产现在统一放在本地工作区 `local/datasets/`，并通过 `dataset_assets_cli` 一键恢复。",
        "",
        "这样做的目标是：",
        "",
        "- 不把上游原始数据作为 Git 大文件长期提交",
        "- 保持 benchmark split 可复现",
        "- 把合规说明、恢复命令和本地资产边界写清楚",
        "",
        "## 当前本地资产根目录",
        "",
        f"- 默认路径：`{inventory['datasets_root']}`",
        "- 环境变量覆盖：`RESEARCH_DATASETS_ROOT`",
        "",
        "## 一键恢复",
        "",
        "只恢复主评测源并重建 split：",
        "",
        "```powershell",
        "uv run dataset_assets_cli prepare-used",
        "```",
        "",
        "同时恢复训练集与可公开下载的验证补充源：",
        "",
        "```powershell",
        "uv run dataset_assets_cli prepare-all-sources",
        "```",
        "",
        "强制覆盖已有本地文件：",
        "",
        "```powershell",
        "uv run dataset_assets_cli prepare-all-sources --force",
        "```",
        "",
        "## 主评测源文件",
        "",
    ]

    for item in inventory["primary_assets"]:
        readme_lines.extend(
            [
                f"### {item['name']} (`{item['slug']}`)",
                "",
                f"- 配置路径：`{item['config_source_path']}`",
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
    """下载主评测源并重建 split。"""
    benchmarks = load_used_benchmark_configs(configs_root)
    downloads = download_primary_dataset_sources(benchmarks, force=force)
    created_splits = regenerate_used_dataset_splits(benchmarks, output_dir=splits_root)
    inventory_paths = write_dataset_inventory_files(benchmarks, datasets_root=_current_datasets_root(), docs_root=DATASETS_DOCS_ROOT, splits_root=splits_root)
    inventory = collect_dataset_inventory(benchmarks, splits_root=splits_root)
    return {
        "benchmark_count": len(benchmarks),
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
    """下载主评测源、训练/验证补充源，并重建 split。"""
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
    }
    items: list[dict[str, str]] = []
    for benchmark in sorted(benchmarks, key=lambda item: item.slug):
        reason = known_unavailable.get(benchmark.slug)
        if reason is not None:
            items.append({"slug": benchmark.slug, "reason": reason})
    return items


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
