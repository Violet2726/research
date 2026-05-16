"""ECON family 的配置加载逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_experiments.families.shared.config_loading import (
    load_benchmarks,
    load_toml,
    optional_float,
    optional_int,
    phase_metadata,
    resolve_model,
)


@dataclass(frozen=True)
class ProtocolConfig:
    """ECON 低通信协调协议。"""

    agent_count: int
    initial_temperature: float
    belief_temperature: float
    top_p: float
    max_output_tokens: int
    peer_packet_token_cap: int
    disagreement_weight: float
    confidence_dispersion_weight: float
    rationale_conflict_weight: float
    expected_gain_weight: float
    communication_cost_weight: float
    vote_bonus: float
    query_best_peer_bonus: float
    query_two_peers_bonus: float
    communication_cost_divisor: float


@dataclass(frozen=True)
class EconMethodSpec:
    """ECON 实验中的单个方法声明。"""

    name: str
    mode: str
    note: str = ""
    matched_controls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EconExperimentConfig:
    """ECON 顶层实验配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    methods: list[EconMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载 ECON 协议配置。"""

    payload = load_toml(path)
    return ProtocolConfig(
        agent_count=int(payload["agent_count"]),
        initial_temperature=float(payload["initial_temperature"]),
        belief_temperature=float(payload["belief_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        peer_packet_token_cap=int(payload["peer_packet_token_cap"]),
        disagreement_weight=float(payload["disagreement_weight"]),
        confidence_dispersion_weight=float(payload["confidence_dispersion_weight"]),
        rationale_conflict_weight=float(payload["rationale_conflict_weight"]),
        expected_gain_weight=float(payload["expected_gain_weight"]),
        communication_cost_weight=float(payload["communication_cost_weight"]),
        vote_bonus=float(payload["vote_bonus"]),
        query_best_peer_bonus=float(payload["query_best_peer_bonus"]),
        query_two_peers_bonus=float(payload["query_two_peers_bonus"]),
        communication_cost_divisor=float(payload["communication_cost_divisor"]),
    )


def load_experiment_config(path: str | Path) -> EconExperimentConfig:
    """加载 ECON 实验配置。"""

    payload = load_toml(path)
    methods = [
        EconMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            note=str(item.get("note") or "").strip(),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
        )
        for item in payload.get("methods", [])
    ]
    return EconExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        methods=methods,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )
