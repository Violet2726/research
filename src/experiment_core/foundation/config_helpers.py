"""Shared helpers for family config loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import tomllib

from experiment_core.foundation.config import (
    BenchmarkConfig,
    ResolvedModelConfig,
    load_benchmark_config,
    resolve_model_ref,
)


class SupportsRawPhases(Protocol):
    raw: dict[str, Any]


class SupportsBenchmarkConfigs(Protocol):
    benchmark_configs: list[Path]


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load one TOML payload from disk."""

    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def optional_int(payload: dict[str, Any], key: str) -> int | None:
    """Read one optional integer field."""

    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def optional_float(payload: dict[str, Any], key: str) -> float | None:
    """Read one optional float field."""

    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def optional_str(payload: dict[str, Any], key: str) -> str | None:
    """Read one optional non-empty string field."""

    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def first_str(payload: dict[str, Any], *keys: str) -> str | None:
    """Return the first populated string field from a candidate list."""

    for key in keys:
        value = optional_str(payload, key)
        if value is not None:
            return value
    return None


def phase_metadata(experiment: SupportsRawPhases, phase_name: str) -> dict[str, Any]:
    """Return a defensive copy of one phase payload."""

    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: SupportsBenchmarkConfigs) -> list[BenchmarkConfig]:
    """Resolve benchmark config files referenced by one experiment."""

    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_model(model_ref: str) -> ResolvedModelConfig:
    """Resolve one shared model ref into a runnable config."""

    return resolve_model_ref(model_ref)
