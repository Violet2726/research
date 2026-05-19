"""DMAD family configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_experiments.families.shared.method_catalog import MethodConfig, load_method_catalog
from research_experiments.families.shared.config_loading import (
    load_benchmarks,
    load_toml,
    optional_int,
    phase_metadata,
    resolve_model,
)


@dataclass(frozen=True)
class ProtocolConfig:
    """DMAD debate protocol."""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    reflection_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class AgentProfile:
    """One agent's persona/strategy profile."""

    agent_id: int
    persona_name: str
    persona_instruction: str
    strategy_name: str
    strategy_instruction: str = ""


@dataclass(frozen=True)
class RosterConfig:
    """A DMAD roster definition."""

    diversity_mode: str
    agents: list[AgentProfile]

    @property
    def agent_count(self) -> int:
        return len(self.agents)


@dataclass(frozen=True)
class DmadMethodSpec:
    """A configured DMAD method."""

    name: str
    mode: str
    roster: Path | None = None
    note: str = ""
    matched_controls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DmadExperimentConfig:
    """Top-level DMAD experiment configuration."""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    control_catalog: Path | None
    methods: list[DmadMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """Load a DMAD protocol file."""

    payload = load_toml(path)
    return ProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        reflection_temperature=float(payload["reflection_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """Load the shared baseline/control catalog used by DMAD."""

    return load_method_catalog(path)


def load_roster_config(path: str | Path) -> RosterConfig:
    """Load a DMAD roster definition."""

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
    """Load a DMAD experiment configuration."""

    payload = load_toml(path)
    methods = [
        DmadMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            roster=Path(item["roster"]) if item.get("roster") else None,
            note=str(item.get("note") or "").strip(),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
        )
        for item in payload.get("methods", [])
    ]
    return DmadExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        control_catalog=Path(payload["control_catalog"]) if payload.get("control_catalog") else None,
        methods=methods,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )
