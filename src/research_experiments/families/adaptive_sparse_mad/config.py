"""A-SMAD 配置加载与实验入口解析。

本模块把 TOML 配置转换为运行时 dataclass，并集中校验聚合方法与提示词版本的兼容性。
路径字段在这里保持为 `Path`，由运行层再解析到具体工作区位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION
from research_experiments.families.adaptive_sparse_mad.prompts import (
    FREE_TEXT_DEBATE_PROMPT_VERSION,
    STAGE_A_V4_PROMPT_VERSION,
)
from research_experiments.family_runtime.config_helpers import (
    apply_runtime_defaults,
    load_benchmarks,
    load_toml,
)
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog
from research_experiments.family_runtime.output_protocols import validate_output_protocol

ADAPTIVE_SPARSE_DEBATE_METHOD = "adaptive_sparse_debate_v1"
STRUCTURED_STAGE_A_PROMPT_VERSIONS = frozenset({STAGE_A_V4_PROMPT_VERSION, FREE_TEXT_DEBATE_PROMPT_VERSION})
RESPONSE_FORMAT_MODES = frozenset({"free_text", "json_object"})

ACTIVE_AGGREGATE_METHODS = frozenset(
    {
        "hetero_vote_3",
        "ega_only_v4",
        "adaptive_gate_v4",
        "adaptive_dual_open_v5",
        "adaptive_counterfactual_v1",
        ADAPTIVE_SPARSE_DEBATE_METHOD,
    }
)
STRUCTURED_STAGE_A_METHODS = frozenset(
    {
        "ega_only_v4",
        "adaptive_gate_v4",
        "adaptive_dual_open_v5",
        "adaptive_counterfactual_v1",
        ADAPTIVE_SPARSE_DEBATE_METHOD,
    }
)
ADAPTIVE_POLICY_METHODS = frozenset(
    {
        "adaptive_gate_v4",
        "adaptive_dual_open_v5",
        "adaptive_counterfactual_v1",
        ADAPTIVE_SPARSE_DEBATE_METHOD,
    }
)


@dataclass(frozen=True)
class AdaptiveSparseMadProtocolConfig:
    """A-SMAD 协议级参数，控制 Stage A 调用和基础聚合阈值。"""

    agent_count: int
    top_p: float
    stage_a_temperature: float
    consensus_confidence_threshold: float
    majority_confidence_threshold: float
    majority_margin_threshold: float
    debate_rounds: int = 1
    debate_temperature: float | None = None
    debate_trigger_mode: str = "adaptive_gate"


@dataclass(frozen=True)
class AdaptiveSparseMadExperimentConfig:
    """单个 A-SMAD 实验配置，承载数据集、方法、模型与运行时限流参数。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    control_catalog: Path
    aggregate_methods: tuple[str, ...]
    max_adaptive_addon_calls: int
    global_seed: int
    control_prompt_version: str
    control_output_protocol: str
    prompt_version: str
    stage_a_prompt_version: str
    adaptive_prompt_version: str
    stage_a_response_format_mode: str
    adaptive_response_format_mode: str
    legacy_json_mode: bool
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> AdaptiveSparseMadProtocolConfig:
    """读取协议 TOML，并返回强类型的协议配置。"""
    payload = load_toml(path)
    return AdaptiveSparseMadProtocolConfig(
        agent_count=int(payload["agent_count"]),
        top_p=float(payload["top_p"]),
        stage_a_temperature=float(payload["stage_a_temperature"]),
        consensus_confidence_threshold=float(payload["consensus_confidence_threshold"]),
        majority_confidence_threshold=float(payload["majority_confidence_threshold"]),
        majority_margin_threshold=float(payload["majority_margin_threshold"]),
        debate_rounds=int(payload.get("debate_rounds", 1)),
        debate_temperature=float(payload.get("debate_temperature", payload["stage_a_temperature"])),
        debate_trigger_mode=str(payload.get("debate_trigger_mode", "adaptive_gate")),
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载 no-comm 对照方法目录。"""
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> AdaptiveSparseMadExperimentConfig:
    """读取实验 TOML，校验 A-SMAD 方法组合并补齐运行时默认值。"""
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    aggregate_methods = tuple(str(item) for item in payload.get("aggregate_methods", ["hetero_vote_3"]))
    unsupported_methods = sorted(set(aggregate_methods) - ACTIVE_AGGREGATE_METHODS)
    if unsupported_methods:
        raise ValueError(
            "Unsupported adaptive_sparse_mad aggregate_methods: "
            + ", ".join(unsupported_methods)
        )
    max_adaptive_addon_calls = int(
        payload.get(
            "max_adaptive_addon_calls",
            1 if any(method_name in ADAPTIVE_POLICY_METHODS for method_name in aggregate_methods) else 0,
        )
    )
    prompt_version = str(payload["prompt_version"])
    stage_a_prompt_version = str(payload.get("stage_a_prompt_version", prompt_version))
    adaptive_prompt_version = str(payload.get("adaptive_prompt_version", prompt_version))
    control_prompt_version = str(payload.get("control_prompt_version", FREE_TEXT_V1_PROMPT_VERSION))
    if control_prompt_version != FREE_TEXT_V1_PROMPT_VERSION:
        raise ValueError("adaptive_sparse_mad control_prompt_version must be single_agent_free_text_v1.")
    control_output_protocol = validate_output_protocol(
        str(payload.get("control_output_protocol", FREE_TEXT_ANSWER_PROTOCOL_V1))
    )
    if control_output_protocol != FREE_TEXT_ANSWER_PROTOCOL_V1:
        raise ValueError("adaptive_sparse_mad control_output_protocol must be free_text_answer_v1.")
    default_response_format_mode = (
        "free_text" if stage_a_prompt_version == FREE_TEXT_DEBATE_PROMPT_VERSION else "json_object"
    )
    stage_a_response_format_mode = str(payload.get("stage_a_response_format_mode", default_response_format_mode))
    adaptive_default_response_format_mode = (
        "free_text" if adaptive_prompt_version == FREE_TEXT_DEBATE_PROMPT_VERSION else "json_object"
    )
    adaptive_response_format_mode = str(
        payload.get("adaptive_response_format_mode", adaptive_default_response_format_mode)
    )
    if stage_a_response_format_mode not in RESPONSE_FORMAT_MODES:
        raise ValueError(
            "adaptive_sparse_mad stage_a_response_format_mode must be one of: "
            + ", ".join(sorted(RESPONSE_FORMAT_MODES))
        )
    if adaptive_response_format_mode not in RESPONSE_FORMAT_MODES:
        raise ValueError(
            "adaptive_sparse_mad adaptive_response_format_mode must be one of: "
            + ", ".join(sorted(RESPONSE_FORMAT_MODES))
        )
    if (
        stage_a_response_format_mode == "free_text"
        and stage_a_prompt_version != FREE_TEXT_DEBATE_PROMPT_VERSION
    ):
        raise ValueError(
            "adaptive_sparse_mad free-text Stage A requires "
            f"stage_a_prompt_version={FREE_TEXT_DEBATE_PROMPT_VERSION}."
        )
    if (
        adaptive_response_format_mode == "free_text"
        and adaptive_prompt_version != FREE_TEXT_DEBATE_PROMPT_VERSION
    ):
        raise ValueError(
            "adaptive_sparse_mad free-text adaptive turns require "
            f"adaptive_prompt_version={FREE_TEXT_DEBATE_PROMPT_VERSION}."
        )
    legacy_json_mode = bool(
        payload.get(
            "legacy_json_mode",
            stage_a_response_format_mode == "json_object" or adaptive_response_format_mode == "json_object",
        )
    )
    if ADAPTIVE_SPARSE_DEBATE_METHOD in aggregate_methods:
        required_versions = {
            "prompt_version": prompt_version,
            "stage_a_prompt_version": stage_a_prompt_version,
            "adaptive_prompt_version": adaptive_prompt_version,
        }
        wrong_versions = [
            f"{name}={value}"
            for name, value in required_versions.items()
            if value != FREE_TEXT_DEBATE_PROMPT_VERSION
        ]
        if wrong_versions:
            raise ValueError(
                "adaptive_sparse_debate_v1 requires "
                f"{FREE_TEXT_DEBATE_PROMPT_VERSION}: "
                + ", ".join(wrong_versions)
            )
        if not legacy_json_mode:
            if stage_a_response_format_mode != "free_text":
                raise ValueError("adaptive_sparse_debate_v1 requires stage_a_response_format_mode=free_text.")
            if adaptive_response_format_mode != "free_text":
                raise ValueError("adaptive_sparse_debate_v1 requires adaptive_response_format_mode=free_text.")
    if any(method_name in STRUCTURED_STAGE_A_METHODS for method_name in aggregate_methods):
        if stage_a_prompt_version not in STRUCTURED_STAGE_A_PROMPT_VERSIONS:
            raise ValueError(
                "adaptive_sparse_mad structured aggregate methods require "
                "stage_a_prompt_version in "
                + ", ".join(sorted(STRUCTURED_STAGE_A_PROMPT_VERSIONS))
            )
        if prompt_version not in STRUCTURED_STAGE_A_PROMPT_VERSIONS:
            raise ValueError(
                "adaptive_sparse_mad structured aggregate methods require "
                "prompt_version in "
                + ", ".join(sorted(STRUCTURED_STAGE_A_PROMPT_VERSIONS))
            )
    if (
        any(method_name in ADAPTIVE_POLICY_METHODS for method_name in aggregate_methods)
        and adaptive_prompt_version not in STRUCTURED_STAGE_A_PROMPT_VERSIONS
    ):
        raise ValueError(
            "adaptive_sparse_mad adaptive policy methods require "
            "adaptive_prompt_version in "
            + ", ".join(sorted(STRUCTURED_STAGE_A_PROMPT_VERSIONS))
        )
    return AdaptiveSparseMadExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        control_catalog=Path(payload["control_catalog"]),
        aggregate_methods=aggregate_methods,
        max_adaptive_addon_calls=max_adaptive_addon_calls,
        global_seed=int(payload["global_seed"]),
        control_prompt_version=control_prompt_version,
        control_output_protocol=control_output_protocol,
        prompt_version=prompt_version,
        stage_a_prompt_version=stage_a_prompt_version,
        adaptive_prompt_version=adaptive_prompt_version,
        stage_a_response_format_mode=stage_a_response_format_mode,
        adaptive_response_format_mode=adaptive_response_format_mode,
        legacy_json_mode=legacy_json_mode,
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


def inspect_methods(experiment: AdaptiveSparseMadExperimentConfig) -> list[str]:
    """返回 CLI inspect 视图展示的全部方法名。"""
    controls = load_control_catalog(experiment.control_catalog)
    return list(controls) + list(experiment.aggregate_methods)


def inspect_benchmarks(experiment: AdaptiveSparseMadExperimentConfig) -> list[str]:
    """返回实验实际引用的 benchmark slug 列表。"""
    return [benchmark.slug for benchmark in load_benchmarks(experiment)]
