"""ColMAD family 的配置加载逻辑。"""

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
    """ColMAD 论文主线协议。"""

    protocol_kind: Literal["paper"]
    opening_temperature: float
    reply_temperature: float
    judge_temperature: float
    top_p: float
    max_output_tokens: int
    max_evidence_points: int
    max_failure_modes: int
    max_debate_rounds: int


@dataclass(frozen=True)
class ColmadMethodSpec:
    """ColMAD 实验中的单个方法声明。"""

    name: str
    mode: str
    matched_controls: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class ColmadExperimentConfig:
    """ColMAD 实验的顶层配置。"""

    name: str
    description: str
    experiment_kind: ExperimentKind
    benchmark_configs: list[Path]
    protocol: Path
    methods: list[ColmadMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载 ColMAD 协议配置。"""

    payload = load_toml(path)
    protocol_kind = str(payload.get("protocol_kind") or "paper").strip().lower()
    if protocol_kind != "paper":
        raise ValueError(f"Unsupported ColMAD protocol_kind: {protocol_kind!r}")
    return ProtocolConfig(
        protocol_kind="paper",
        opening_temperature=float(payload["opening_temperature"]),
        reply_temperature=float(payload["reply_temperature"]),
        judge_temperature=float(payload["judge_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        max_evidence_points=int(payload["max_evidence_points"]),
        max_failure_modes=int(payload["max_failure_modes"]),
        max_debate_rounds=int(payload["max_debate_rounds"]),
    )


def load_experiment_config(path: str | Path) -> ColmadExperimentConfig:
    """加载 ColMAD 实验配置。"""

    payload = load_toml(path)
    experiment_kind = str(payload.get("experiment_kind") or "paper").strip().lower()
    if experiment_kind != "paper":
        raise ValueError(f"Unsupported ColMAD experiment_kind: {experiment_kind!r}")
    methods = [
        ColmadMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
            note=str(item.get("note") or "").strip(),
        )
        for item in payload.get("methods", [])
    ]
    return ColmadExperimentConfig(
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

