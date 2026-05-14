"""DoG family 的配置加载逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from research_experiments.families.shared.config_loading import (
    load_benchmarks,
    load_toml,
    optional_int,
    phase_metadata,
    resolve_model,
)


ExperimentKind = Literal["paper", "static"]
ProtocolKind = Literal["paper", "static"]


@dataclass(frozen=True)
class StaticProtocolConfig:
    """静态子图消融线使用的共享协议参数。"""

    protocol_kind: Literal["static"]
    agent_count: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int
    max_evidence_triples: int
    max_answer_path_hops: int


@dataclass(frozen=True)
class PaperProtocolConfig:
    """高保真论文主线使用的协议参数。"""

    protocol_kind: Literal["paper"]
    max_hops: int
    max_selected_relations: int
    selector_temperature: float
    enough_answer_temperature: float
    simplifier_temperature: float
    fallback_temperature: float
    top_p: float
    max_output_tokens: int
    direct_fallback_enabled: bool
    freebase_sparql_url: str
    freebase_backend_mode: str
    role_names: list[str] = field(default_factory=list)


DogGraphProtocolConfig = StaticProtocolConfig | PaperProtocolConfig
# 为当前静态消融链路保留旧名字，避免 legacy 模块导入失败。
ProtocolConfig = StaticProtocolConfig


@dataclass(frozen=True)
class GraphMethodSpec:
    """DoG 实验中的单个方法声明。"""

    name: str
    mode: str
    agent_count: int | None = None
    round_limit: int | None = None
    view_mode: str | None = None
    matched_controls: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class DogGraphExperimentConfig:
    """DoG 实验的顶层配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    methods: list[GraphMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    experiment_kind: ExperimentKind
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> DogGraphProtocolConfig:
    """加载 DoG 协议配置。"""

    payload = load_toml(path)
    protocol_kind = str(payload.get("protocol_kind") or "static").strip().lower()
    if protocol_kind == "paper":
        return PaperProtocolConfig(
            protocol_kind="paper",
            max_hops=int(payload["max_hops"]),
            max_selected_relations=int(payload["max_selected_relations"]),
            selector_temperature=float(payload["selector_temperature"]),
            enough_answer_temperature=float(payload["enough_answer_temperature"]),
            simplifier_temperature=float(payload["simplifier_temperature"]),
            fallback_temperature=float(payload["fallback_temperature"]),
            top_p=float(payload["top_p"]),
            max_output_tokens=int(payload["max_output_tokens"]),
            direct_fallback_enabled=bool(payload.get("direct_fallback_enabled", True)),
            freebase_sparql_url=str(payload.get("freebase_sparql_url") or "http://localhost:8890/sparql"),
            freebase_backend_mode=str(payload.get("freebase_backend_mode") or "local_reduced"),
            role_names=[str(item) for item in payload.get("role_names", [])],
        )
    return StaticProtocolConfig(
        protocol_kind="static",
        agent_count=int(payload["agent_count"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        max_evidence_triples=int(payload["max_evidence_triples"]),
        max_answer_path_hops=int(payload["max_answer_path_hops"]),
    )


def load_experiment_config(path: str | Path) -> DogGraphExperimentConfig:
    """加载 DoG 实验配置。"""

    payload = load_toml(path)
    methods = [
        GraphMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            agent_count=_optional_method_int(item, "agent_count"),
            round_limit=_optional_method_int(item, "round_limit"),
            view_mode=_optional_method_str(item, "view_mode"),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
            note=str(item.get("note") or "").strip(),
        )
        for item in payload.get("methods", [])
    ]
    experiment_kind = str(payload.get("experiment_kind") or "static").strip().lower()
    if experiment_kind not in {"paper", "static"}:
        raise ValueError(f"Unsupported DoG experiment_kind: {experiment_kind!r}")
    return DogGraphExperimentConfig(
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
        experiment_kind=experiment_kind,
        raw=payload,
    )


def _optional_method_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _optional_method_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
