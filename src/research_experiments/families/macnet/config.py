"""MacNet family 的配置加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_experiments.families.shared.config_loading import (
    load_benchmarks,
    load_toml,
    optional_int,
)


@dataclass(frozen=True)
class ProtocolConfig:
    """MacNet 协议配置。"""

    protocol_kind: str
    actor_temperature: float
    critic_temperature: float
    top_p: float
    max_output_tokens: int
    parent_artifact_token_cap: int
    inbound_instruction_token_cap: int
    memory_control_max_parents: int
    profile_asset_path: Path
    default_direction_mode: str
    terminal_fuse_temperature: float


@dataclass(frozen=True)
class MacnetMethodSpec:
    """MacNet 单个方法声明。"""

    name: str
    mode: str
    note: str = ""
    topology_type: str | None = None


@dataclass(frozen=True)
class MacnetExperimentConfig:
    """MacNet 顶层实验配置。"""

    name: str
    description: str
    experiment_kind: str
    benchmark_configs: list[Path]
    protocol: Path
    methods: list[MacnetMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载 MacNet 协议配置。"""

    payload = load_toml(path)
    return ProtocolConfig(
        protocol_kind=str(payload["protocol_kind"]),
        actor_temperature=float(payload["actor_temperature"]),
        critic_temperature=float(payload["critic_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        parent_artifact_token_cap=int(payload["parent_artifact_token_cap"]),
        inbound_instruction_token_cap=int(payload["inbound_instruction_token_cap"]),
        memory_control_max_parents=int(payload["memory_control_max_parents"]),
        profile_asset_path=Path(str(payload["profile_asset_path"])),
        default_direction_mode=str(payload["default_direction_mode"]),
        terminal_fuse_temperature=float(payload["terminal_fuse_temperature"]),
    )


def load_experiment_config(path: str | Path) -> MacnetExperimentConfig:
    """加载 MacNet 顶层实验配置。"""

    payload = load_toml(path)
    methods = [
        MacnetMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            note=str(item.get("note") or "").strip(),
            topology_type=str(item["topology_type"]) if item.get("topology_type") else None,
        )
        for item in payload.get("methods", [])
    ]
    return MacnetExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        experiment_kind=str(payload["experiment_kind"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(str(payload["protocol"])),
        methods=methods,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )
