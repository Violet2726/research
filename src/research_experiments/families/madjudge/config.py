"""MADJudge 实验配置加载。

基于论文 "Multi-Agent Debate for LLM Judges with Adaptive Stability Detection" 的配置结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import (
    apply_runtime_defaults,
    load_toml,
)
from research_experiments.family_runtime.config_helpers import (
    phase_metadata as _phase_metadata,
)
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog


def phase_metadata(experiment: MadJudgeExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回指定 phase 配置的防御性拷贝。"""
    return _phase_metadata(experiment, phase_name)


@dataclass(frozen=True)
class StabilityConfig:
    """稳定性检测配置。"""

    # KS 统计量阈值（论文使用 0.05）
    ks_threshold: float = 0.05
    # 需要连续稳定的轮数（论文使用 2）
    consecutive_stable_required: int = 2


@dataclass(frozen=True)
class ProtocolConfig:
    """MADJudge 辩论协议配置。"""

    agent_count: int
    max_debate_rounds: int
    temperature: float
    top_p: float
    max_output_tokens: int
    stability: StabilityConfig
    method_type: str = "madjudge"


@dataclass(frozen=True)
class AgentProfile:
    """单个智能体的配置。"""

    agent_id: int
    persona_name: str
    persona_instruction: str
    temperature_override: float | None = None


@dataclass(frozen=True)
class RosterConfig:
    """MADJudge roster 配置。"""

    agents: list[AgentProfile]

    @property
    def agent_count(self) -> int:
        return len(self.agents)


@dataclass(frozen=True)
class ExperimentSetup:
    """单个 MADJudge setup 的声明。"""

    name: str
    protocol: Path
    roster: Path
    matched_controls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MadJudgeExperimentConfig:
    """MADJudge 实验的顶层配置。"""

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


def load_stability_config(payload: dict[str, Any]) -> StabilityConfig:
    """加载稳定性检测配置。"""
    stability_payload = payload.get("stability", {})
    return StabilityConfig(
        ks_threshold=float(stability_payload.get("ks_threshold", 0.05)),
        consecutive_stable_required=int(stability_payload.get("consecutive_stable_required", 2)),
    )


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载 MADJudge 协议配置。"""
    payload = load_toml(path)
    return ProtocolConfig(
        agent_count=int(payload["agent_count"]),
        max_debate_rounds=int(payload["max_debate_rounds"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        stability=load_stability_config(payload),
        method_type=str(payload.get("method_type", "madjudge")),
    )


def load_roster_config(path: str | Path) -> RosterConfig:
    """加载 MADJudge roster 配置。"""
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


def load_experiment_config(path: str | Path) -> MadJudgeExperimentConfig:
    """加载 MADJudge 实验配置。"""
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
    return MadJudgeExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        control_catalog=Path(payload["control_catalog"]) if payload.get("control_catalog") else None,
        setups=setups,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        tokens_per_minute_limit=runtime["tokens_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )

