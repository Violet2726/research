"""Table-Critic family 的配置加载逻辑。"""

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


ExperimentKind = Literal["paper"]


@dataclass(frozen=True)
class ProtocolConfig:
    """Table-Critic 原论文主线协议。"""

    protocol_kind: Literal["paper"]
    max_refine_rounds: int
    initial_temperature: float
    judge_temperature: float
    critic_temperature: float
    refiner_temperature: float
    curator_temperature: float
    top_p: float
    max_output_tokens: int
    max_template_examples: int
    template_tree_max_summaries_per_node: int
    use_curator: bool
    seed_template_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TableCriticMethodSpec:
    """Table-Critic 实验中的单个方法声明。"""

    name: str
    mode: str
    matched_controls: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class TableCriticExperimentConfig:
    """Table-Critic 实验的顶层配置。"""

    name: str
    description: str
    experiment_kind: ExperimentKind
    benchmark_configs: list[Path]
    protocol: Path
    methods: list[TableCriticMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载 Table-Critic 协议配置。"""

    payload = load_toml(path)
    protocol_kind = str(payload.get("protocol_kind") or "paper").strip().lower()
    if protocol_kind != "paper":
        raise ValueError(f"Unsupported Table-Critic protocol_kind: {protocol_kind!r}")
    return ProtocolConfig(
        protocol_kind="paper",
        max_refine_rounds=int(payload["max_refine_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        judge_temperature=float(payload["judge_temperature"]),
        critic_temperature=float(payload["critic_temperature"]),
        refiner_temperature=float(payload["refiner_temperature"]),
        curator_temperature=float(payload["curator_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        max_template_examples=int(payload["max_template_examples"]),
        template_tree_max_summaries_per_node=int(payload["template_tree_max_summaries_per_node"]),
        use_curator=bool(payload.get("use_curator", True)),
        seed_template_paths=[str(item) for item in payload.get("seed_template_paths", [])],
    )


def load_experiment_config(path: str | Path) -> TableCriticExperimentConfig:
    """加载 Table-Critic 实验配置。"""

    payload = load_toml(path)
    experiment_kind = str(payload.get("experiment_kind") or "paper").strip().lower()
    if experiment_kind != "paper":
        raise ValueError(f"Unsupported Table-Critic experiment_kind: {experiment_kind!r}")
    methods = [
        TableCriticMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
            note=str(item.get("note") or "").strip(),
        )
        for item in payload.get("methods", [])
    ]
    return TableCriticExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        experiment_kind="paper",
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

