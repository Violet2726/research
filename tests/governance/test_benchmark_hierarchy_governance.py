"""benchmark 配置层级与数据资产层级的一致性治理测试。"""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import resolve_split_manifest_path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = ROOT / "configs" / "core" / "shared" / "benchmarks"


def _expected_benchmark_config_relative_path(benchmark) -> str:
    """派生子集 benchmark 按自身命名空间对齐，其余 benchmark 继续按源数据路径对齐。"""
    if benchmark.record_filters or benchmark.cache_namespace_override:
        benchmark_key = str(benchmark.cache_namespace or benchmark.slug).replace("\\", "/")
        return Path(benchmark_key).with_suffix(".toml").as_posix()
    return Path(str(benchmark.source_path).replace("\\", "/")).with_suffix(".toml").as_posix()


def test_shared_benchmark_configs_mirror_dataset_asset_hierarchy() -> None:
    for path in BENCHMARKS_ROOT.rglob("*.toml"):
        benchmark = load_benchmark_config(path)
        expected = _expected_benchmark_config_relative_path(benchmark)
        actual = path.relative_to(BENCHMARKS_ROOT).as_posix()
        assert actual == expected, f"{path} should mirror source_path hierarchy {expected}"


def test_split_manifest_path_mirrors_benchmark_dataset_hierarchy() -> None:
    for path in BENCHMARKS_ROOT.rglob("*.toml"):
        benchmark = load_benchmark_config(path)
        resolved = resolve_split_manifest_path(benchmark.cache_namespace or benchmark.slug, "count20_seed42")
        relative = resolved.as_posix().split("configs/core/shared/benchmarks/splits/count20/", 1)[-1]
        expected = Path(str(benchmark.source_path).replace("\\", "/")).with_suffix("").parent.as_posix()
        if expected == ".":
            assert "/" not in relative
            continue
        assert relative.startswith(expected + "/"), f"{resolved} should mirror split hierarchy {expected}"


def test_benchmark_paths_and_cache_namespaces_use_official_dataset_names() -> None:
    for path in BENCHMARKS_ROOT.rglob("*.toml"):
        benchmark = load_benchmark_config(path)
        config_relative = path.relative_to(BENCHMARKS_ROOT).as_posix().lower()
        cache_namespace = str(benchmark.cache_namespace or benchmark.slug).replace("\\", "/").lower()
        assert "dog-" not in config_relative
        assert not any(part.startswith("dog_") for part in Path(config_relative).parts)
        assert "dog-" not in cache_namespace
        assert not any(part.startswith("dog_") for part in Path(cache_namespace).parts)
