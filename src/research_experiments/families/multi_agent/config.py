"""多智能体实验配置加载。

本模块负责解析 Vanilla MAD 风格多智能体实验所需的配置，
包括 debate 协议、agent roster、实验 setup，以及与之配套的等预算控制方法。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import (
    apply_runtime_defaults,
    load_toml,
)
from research_experiments.core.controls.control_prompts import CONSISTENT_JSON_V2_PROMPT_VERSION as CONTROL_PROMPT_V2
from research_experiments.family_runtime.answer_contracts import (
    JSON_ANSWER_ANCHOR_V2_CONTRACT,
    MULTI_AGENT_CONSISTENT_JSON_V2_PROMPT,
    validate_answer_contract,
    validate_prompt_answer_contract,
)
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog


@dataclass(frozen=True)
class ProtocolConfig:
    """Vanilla MAD 协议配置。"""

    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float


@dataclass(frozen=True)
class RosterConfig:
    """多智能体 roster 配置。"""

    agent_count: int


@dataclass(frozen=True)
class ExperimentSetup:
    """单个多智能体 setup 的声明。"""

    name: str
    protocol: Path
    roster: Path
    matched_controls: list[str]


@dataclass(frozen=True)
class MultiAgentExperimentConfig:
    """多智能体实验的顶层配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    control_catalog: Path
    setups: list[ExperimentSetup]
    global_seed: int
    control_prompt_version: str
    control_answer_contract: str
    mad_prompt_version: str
    mad_answer_contract: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载多智能体协议信息。"""
    payload = load_toml(path)
    return ProtocolConfig(
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
    )


def load_roster_config(path: str | Path) -> RosterConfig:
    """加载 agent roster 配置。"""
    payload = load_toml(path)
    return RosterConfig(agent_count=int(payload["agent_count"]))


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载与多智能体 setup 配套的控制方法目录。"""
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> MultiAgentExperimentConfig:
    """加载多智能体实验配置。"""
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    setups = [
        ExperimentSetup(
            name=str(item["name"]),
            protocol=Path(item["protocol"]),
            roster=Path(item["roster"]),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
        )
        for item in payload.get("setups", [])
    ]
    control_prompt_version = str(payload["control_prompt_version"])
    if control_prompt_version != CONTROL_PROMPT_V2:
        raise ValueError(f"Unsupported multi_agent control_prompt_version={control_prompt_version!r}.")
    control_answer_contract = validate_answer_contract(str(payload["control_answer_contract"]))
    mad_prompt_version = str(payload["mad_prompt_version"])
    mad_answer_contract = validate_prompt_answer_contract(
        mad_prompt_version,
        str(payload["mad_answer_contract"]),
    )
    if control_answer_contract != JSON_ANSWER_ANCHOR_V2_CONTRACT:
        raise ValueError("multi_agent control_answer_contract must be json_answer_anchor_v2.")
    if mad_prompt_version != MULTI_AGENT_CONSISTENT_JSON_V2_PROMPT:
        raise ValueError("multi_agent mad_prompt_version must be multi_agent_consistent_json_v2.")
    return MultiAgentExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        control_catalog=Path(payload["control_catalog"]),
        setups=setups,
        global_seed=int(payload["global_seed"]),
        control_prompt_version=control_prompt_version,
        control_answer_contract=control_answer_contract,
        mad_prompt_version=mad_prompt_version,
        mad_answer_contract=mad_answer_contract,
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )




