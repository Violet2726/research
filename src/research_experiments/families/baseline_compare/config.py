"""独立基准对比 family 的配置加载逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import (
    apply_runtime_defaults,
    load_toml,
)
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog


@dataclass(frozen=True)
class ProtocolConfig:
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class RosterConfig:
    agent_count: int


@dataclass(frozen=True)
class ExperimentSetup:
    name: str
    protocol: Path
    roster: Path


@dataclass(frozen=True)
class BaselineCompareExperimentConfig:
    name: str
    description: str
    benchmark_configs: list[Path]
    control_catalog: Path
    control_methods: list[str]
    method_order: list[str]
    setups: list[ExperimentSetup]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    payload = load_toml(path)
    return ProtocolConfig(
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
    )


def load_roster_config(path: str | Path) -> RosterConfig:
    payload = load_toml(path)
    return RosterConfig(agent_count=int(payload["agent_count"]))


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> BaselineCompareExperimentConfig:
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    setups = [
        ExperimentSetup(
            name=str(item["name"]),
            protocol=Path(item["protocol"]),
            roster=Path(item["roster"]),
        )
        for item in payload.get("setups", [])
    ]
    control_methods = [str(name) for name in payload.get("control_methods", [])]
    method_order = [str(name) for name in payload.get("method_order", [])]
    _validate_method_inventory(control_methods, method_order, setups)
    return BaselineCompareExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        control_catalog=Path(payload["control_catalog"]),
        control_methods=control_methods,
        method_order=method_order,
        setups=setups,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        tokens_per_minute_limit=runtime["tokens_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


def _validate_method_inventory(
    control_methods: list[str],
    method_order: list[str],
    setups: list[ExperimentSetup],
) -> None:
    ordered_unique = list(dict.fromkeys(method_order))
    if len(ordered_unique) != len(method_order):
        raise ValueError("baseline_compare method_order must not contain duplicates.")
    setup_names = [setup.name for setup in setups]
    declared_methods = set(control_methods) | set(setup_names)
    if set(method_order) != declared_methods:
        raise ValueError("baseline_compare method_order must exactly cover control_methods and setup names.")
