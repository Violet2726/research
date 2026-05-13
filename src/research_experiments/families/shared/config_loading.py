"""family 配置加载器的共享辅助函数。

本模块只承接配置解析阶段反复出现的低层工具，
避免各个 family 在读取 TOML、抽取 phase 字段和解析模型引用时重复写样板代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import tomllib

from research_experiments.core.config import (
    BenchmarkConfig,
    ResolvedModelConfig,
    load_benchmark_config,
    resolve_model_ref,
)


class SupportsRawPhases(Protocol):
    """约束拥有原始 `phases` 载荷的实验配置对象。"""

    raw: dict[str, Any]


class SupportsBenchmarkConfigs(Protocol):
    """约束显式列出 benchmark 配置路径的实验配置对象。"""

    benchmark_configs: list[Path]


def load_toml(path: str | Path) -> dict[str, Any]:
    """从磁盘读取一个 TOML 载荷。"""

    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def optional_int(payload: dict[str, Any], key: str) -> int | None:
    """读取一个可选整数字段。"""

    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def optional_float(payload: dict[str, Any], key: str) -> float | None:
    """读取一个可选浮点数字段。"""

    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def optional_str(payload: dict[str, Any], key: str) -> str | None:
    """读取一个可选非空字符串字段。"""

    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def first_str(payload: dict[str, Any], *keys: str) -> str | None:
    """从候选字段列表里返回第一个有值的字符串。"""

    for key in keys:
        value = optional_str(payload, key)
        if value is not None:
            return value
    return None


def phase_metadata(experiment: SupportsRawPhases, phase_name: str) -> dict[str, Any]:
    """返回指定 phase 配置的防御性拷贝。"""

    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: SupportsBenchmarkConfigs) -> list[BenchmarkConfig]:
    """解析实验配置里引用的全部 benchmark 配置文件。"""

    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_model(model_ref: str) -> ResolvedModelConfig:
    """把共享模型引用解析成可直接运行的模型配置。"""

    return resolve_model_ref(model_ref)

