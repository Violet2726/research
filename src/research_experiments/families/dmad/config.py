"""DMAD family 配置加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import (
    apply_runtime_defaults,
    load_toml,
)
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog


@dataclass(frozen=True)
class ProtocolConfig:
    """DMAD 辩论协议参数。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    reflection_temperature: float
    top_p: float


@dataclass(frozen=True)
class AgentProfile:
    """单个智能体的人设与推理策略配置。"""

    agent_id: int
    persona_name: str
    persona_instruction: str
    strategy_name: str
    strategy_instruction: str = ""


@dataclass(frozen=True)
class RosterConfig:
    """DMAD 智能体阵容定义。"""

    diversity_mode: str
    agents: list[AgentProfile]

    @property
    def agent_count(self) -> int:
        return len(self.agents)


@dataclass(frozen=True)
class DmadMethodSpec:
    """实验配置中的单个 DMAD 方法规格。"""

    name: str
    mode: str
    roster: Path | None = None
    debate_call_style: str = "split_process_answer"
    note: str = ""
    matched_controls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DmadExperimentConfig:
    """DMAD 顶层实验配置。"""

    name: str
    description: str
    evaluation_scope: str
    paper_alignment_version: str
    benchmark_configs: list[Path]
    protocol: Path
    control_catalog: Path | None
    methods: list[DmadMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """从 TOML 加载 DMAD 辩论协议。"""

    payload = load_toml(path)
    return ProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        reflection_temperature=float(payload["reflection_temperature"]),
        top_p=float(payload["top_p"]),
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载 DMAD 使用的共享基线与对照方法目录。"""

    return load_method_catalog(path)


def load_roster_config(path: str | Path) -> RosterConfig:
    """从 TOML 加载 DMAD 智能体阵容。"""

    payload = load_toml(path)
    agents = [
        AgentProfile(
            agent_id=int(item["agent_id"]),
            persona_name=str(item["persona_name"]),
            persona_instruction=str(item["persona_instruction"]),
            strategy_name=str(item["strategy_name"]),
            strategy_instruction=str(item.get("strategy_instruction") or "").strip(),
        )
        for item in payload.get("agents", [])
    ]
    return RosterConfig(
        diversity_mode=str(payload["diversity_mode"]),
        agents=agents,
    )


def load_experiment_config(path: str | Path) -> DmadExperimentConfig:
    """从 TOML 加载 DMAD 实验配置。"""

    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    methods = [
        DmadMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            roster=Path(item["roster"]) if item.get("roster") else None,
            debate_call_style=str(item.get("debate_call_style") or "split_process_answer"),
            note=str(item.get("note") or "").strip(),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
        )
        for item in payload.get("methods", [])
    ]
    return DmadExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        evaluation_scope=str(payload.get("evaluation_scope") or "paper_main"),
        paper_alignment_version=str(payload.get("paper_alignment_version") or "dmad_iclr2025_llm_text_v1"),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        control_catalog=Path(payload["control_catalog"]) if payload.get("control_catalog") else None,
        methods=methods,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )

