"""family 配置加载器的共享辅助函数。

本模块只承接配置解析阶段反复出现的低层工具，
避免各个 family 在读取 TOML、抽取 phase 字段和解析模型引用时重复写样板代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from research_experiments.core.config import (
    BenchmarkConfig,
    ResolvedModelConfig,
    load_benchmark_config,
    resolve_model_ref,
)
from research_experiments.core.execution.rate_limits import standard_runtime_limits
from research_experiments.core.io import read_toml


class SupportsRawPhases(Protocol):
    """约束拥有原始 `phases` 载荷的实验配置对象。"""

    raw: dict[str, Any]


class SupportsBenchmarkConfigs(Protocol):
    """约束显式列出 benchmark 配置路径的实验配置对象。"""

    benchmark_configs: list[Path]


class RuntimeConfigPayload(Protocol):
    """约束携带统一运行时限流字段的配置对象。"""

    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None


def load_toml(path: str | Path) -> dict[str, Any]:
    """从磁盘读取一个 TOML 载荷。"""

    return read_toml(path)


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


def apply_runtime_defaults(payload: dict[str, Any]) -> dict[str, int]:
    """返回补齐项目标准默认值后的运行时限流配置。"""

    defaults = standard_runtime_limits()
    return {
        "max_concurrent_requests": int(payload.get("max_concurrent_requests", defaults["max_concurrent_requests"])),
        "requests_per_minute_limit": int(
            payload.get("requests_per_minute_limit", defaults["requests_per_minute_limit"])
        ),
        "tokens_per_minute_limit": int(payload.get("tokens_per_minute_limit", defaults["tokens_per_minute_limit"])),
    }


def phase_metadata(experiment: SupportsRawPhases, phase_name: str) -> dict[str, Any]:
    """返回指定 phase 配置的防御性拷贝。"""

    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: SupportsBenchmarkConfigs) -> list[BenchmarkConfig]:
    """解析实验配置里引用的全部 benchmark 配置文件。"""

    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_model(model_ref: str) -> ResolvedModelConfig:
    """把共享模型引用解析成可直接运行的模型配置。"""

    return resolve_model_ref(model_ref)

