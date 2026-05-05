"""单智能体实验配置加载。

本模块负责解析单智能体基线实验的顶层配置，并把实验级、phase 级的模型约束
与 benchmark 约束整理成统一接口，供 runner 在真正发请求之前做筛选与 fail-fast 校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


SINGLE_AGENT_CONFIG_ROOT = Path("configs/single_agent")


@dataclass(frozen=True)
class ExperimentConfig:
    """单智能体实验的顶层配置。"""

    name: str
    description: str
    method_catalog: Path
    benchmark_configs: list[Path]
    required_model_tags: list[str]
    benchmark_required_tags: dict[str, list[str]]
    global_seed: int
    reruns_per_method: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """加载单智能体实验配置文件。"""
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return ExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        method_catalog=Path(payload["method_catalog"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        required_model_tags=[str(item) for item in payload.get("required_model_tags", [])],
        benchmark_required_tags={
            str(benchmark): [str(tag) for tag in tags]
            for benchmark, tags in payload.get("benchmark_required_tags", {}).items()
        },
        global_seed=int(payload["global_seed"]),
        reruns_per_method=int(payload["reruns_per_method"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=_optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=_optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


def phase_metadata(experiment: ExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回某个 phase 的原始配置副本。"""
    return dict(experiment.raw["phases"][phase_name])


def required_model_tags(experiment: ExperimentConfig, phase_name: str) -> list[str]:
    """合并实验级与 phase 级的模型标签要求。"""
    phase = phase_metadata(experiment, phase_name)
    combined = list(experiment.required_model_tags)
    combined.extend(str(item) for item in phase.get("required_model_tags", []))
    return _dedupe_preserving_order(combined)


def required_benchmark_tags(
    experiment: ExperimentConfig,
    phase_name: str,
    benchmark_slug: str,
) -> list[str]:
    """合并实验级与 phase 级的 benchmark 标签要求。"""
    phase = phase_metadata(experiment, phase_name)
    combined = list(experiment.benchmark_required_tags.get(benchmark_slug, []))
    combined.extend(str(item) for item in phase.get("benchmark_required_tags", {}).get(benchmark_slug, []))
    return _dedupe_preserving_order(combined)


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """按出现顺序去重，便于输出稳定且可读。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    """读取可选整数字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return int(value)
