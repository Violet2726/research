"""comm_necessary 实验配置加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from experiment_core.config import BenchmarkConfig, ResolvedModelConfig, load_benchmark_config, resolve_model_ref


@dataclass(frozen=True)
class CommNecessaryProtocolConfig:
    """HotpotQA 通信必要性实验协议。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    update_temperature: float
    top_p: float
    max_output_tokens: int
    answer_only_token_cap: int
    evidence_token_cap: int
    full_packet_token_cap: int


@dataclass(frozen=True)
class CommNecessaryExperimentConfig:
    """comm_necessary 顶层实验配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    methods: list[str]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_backbone: str
    raw: dict[str, Any]


def _load_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 文件。"""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_config(path: str | Path) -> CommNecessaryProtocolConfig:
    """加载协议配置。"""
    payload = _load_toml(path)
    return CommNecessaryProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        update_temperature=float(payload["update_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        answer_only_token_cap=int(payload["answer_only_token_cap"]),
        evidence_token_cap=int(payload["evidence_token_cap"]),
        full_packet_token_cap=int(payload["full_packet_token_cap"]),
    )


def load_experiment_config(path: str | Path) -> CommNecessaryExperimentConfig:
    """加载顶层实验配置。"""
    payload = _load_toml(path)
    return CommNecessaryExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        methods=[str(item) for item in payload["methods"]],
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=_optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=_optional_int(payload, "tokens_per_minute_limit"),
        primary_backbone=str(payload["primary_backbone"]),
        raw=payload,
    )


def phase_metadata(experiment: CommNecessaryExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回 phase 原始配置。"""
    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: CommNecessaryExperimentConfig) -> list[BenchmarkConfig]:
    """加载实验声明的 benchmark。"""
    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_backbone(model_ref: str) -> ResolvedModelConfig:
    """解析 backbone 模型。"""
    return resolve_model_ref(model_ref)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)

