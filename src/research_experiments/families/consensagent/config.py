"""CONSENSAGENT 实验配置加载。

本模块负责解析 CONSENSAGENT 框架所需的配置，包括：
- 辩论协议（轮数、温度、触发阈值等）
- 智能体 roster 配置
- 实验 setup 和基线方法
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_experiments.core.families.config_loading import (
    load_toml,
    optional_int,
)
from research_experiments.core.families.config_loading import (
    phase_metadata as _phase_metadata,
)
from research_experiments.core.families.method_catalog import MethodConfig, load_method_catalog


def phase_metadata(experiment: ConsensagentExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回指定 phase 配置的防御性拷贝。"""
    return _phase_metadata(experiment, phase_name)


@dataclass(frozen=True)
class TriggerConfig:
    """触发机制配置。"""

    # 停滞触发阈值：当智能体连续未回应对方解释的轮数
    stagnation_threshold: int = 2
    # 谄媚触发阈值：一致性得分超过此值时判定为谄媚
    sycophancy_consistency_threshold: float = 0.8
    # 是否在达成共识时仍检查谄媚
    check_sycophancy_on_consensus: bool = True


@dataclass(frozen=True)
class Phase3Config:
    """Phase 3 提示优化配置（in-context learning 替代微调）。"""

    enabled: bool = True
    optimizer_temperature: float = 0.3
    post_optimization_rounds: int = 1
    max_optimizer_output_tokens: int = 512


@dataclass(frozen=True)
class ProtocolConfig:
    """CONSENSAGENT 辩论协议配置。"""

    agent_count: int
    max_debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int
    trigger: TriggerConfig
    phase3: Phase3Config = field(default_factory=Phase3Config)
    method_type: str = "consensagent"


@dataclass(frozen=True)
class AgentProfile:
    """单个智能体的配置。"""

    agent_id: int
    persona_name: str
    persona_instruction: str
    # 可选 per-agent 温度覆盖，None 表示使用协议默认温度
    temperature_override: float | None = None


@dataclass(frozen=True)
class RosterConfig:
    """CONSENSAGENT roster 配置。"""

    agents: list[AgentProfile]

    @property
    def agent_count(self) -> int:
        return len(self.agents)


@dataclass(frozen=True)
class ExperimentSetup:
    """单个 CONSENSAGENT setup 的声明。"""

    name: str
    protocol: Path
    roster: Path
    matched_controls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConsensagentExperimentConfig:
    """CONSENSAGENT 实验的顶层配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    control_catalog: Path | None
    setups: list[ExperimentSetup]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_trigger_config(payload: dict[str, Any]) -> TriggerConfig:
    """加载触发机制配置。"""
    trigger_payload = payload.get("trigger", {})
    return TriggerConfig(
        stagnation_threshold=int(trigger_payload.get("stagnation_threshold", 2)),
        sycophancy_consistency_threshold=float(trigger_payload.get("sycophancy_consistency_threshold", 0.8)),
        check_sycophancy_on_consensus=bool(trigger_payload.get("check_sycophancy_on_consensus", True)),
    )


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载 CONSENSAGENT 协议配置。"""
    payload = load_toml(path)
    return ProtocolConfig(
        agent_count=int(payload["agent_count"]),
        max_debate_rounds=int(payload["max_debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        trigger=load_trigger_config(payload),
        phase3=load_phase3_config(payload),
        method_type=str(payload.get("method_type", "consensagent")),
    )


def load_phase3_config(payload: dict[str, Any]) -> Phase3Config:
    """加载 Phase 3 提示优化配置。"""
    p3 = payload.get("phase3_optimizer", {})
    return Phase3Config(
        enabled=bool(p3.get("enabled", True)),
        optimizer_temperature=float(p3.get("optimizer_temperature", 0.3)),
        post_optimization_rounds=int(p3.get("post_optimization_rounds", 1)),
        max_optimizer_output_tokens=int(p3.get("max_optimizer_output_tokens", 512)),
    )


def load_roster_config(path: str | Path) -> RosterConfig:
    """加载 CONSENSAGENT roster 配置。"""
    payload = load_toml(path)
    agents = [
        AgentProfile(
            agent_id=int(item["agent_id"]),
            persona_name=str(item["persona_name"]),
            persona_instruction=str(item["persona_instruction"]),
            temperature_override=float(item["temperature_override"]) if "temperature_override" in item else None,
        )
        for item in payload.get("agents", [])
    ]
    return RosterConfig(agents=agents)


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载控制方法目录。"""
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> ConsensagentExperimentConfig:
    """加载 CONSENSAGENT 实验配置。"""
    payload = load_toml(path)
    setups = [
        ExperimentSetup(
            name=str(item["name"]),
            protocol=Path(item["protocol"]),
            roster=Path(item["roster"]),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
        )
        for item in payload.get("setups", [])
    ]
    return ConsensagentExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        control_catalog=Path(payload["control_catalog"]) if payload.get("control_catalog") else None,
        setups=setups,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )

