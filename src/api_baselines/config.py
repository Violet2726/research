from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key_env: str
    chat_path: str
    default_temperature: float
    default_top_p: float
    default_max_output_tokens: int
    supports_response_format: bool
    response_format: str | None
    timeout_seconds: int
    max_retries: int


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    slug: str
    loader: str
    source_path: str
    source_split: str
    sample_id_prefix: str
    question_field: str
    answer_field: str
    smoke_size: int
    pilot_size: int
    main_size: int
    random_seed: int
    notes: str


@dataclass(frozen=True)
class MethodConfig:
    name: str
    family: str
    budget_calls: int
    temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    model_configs: list[Path]
    benchmark_configs: list[Path]
    global_seed: int
    reruns_per_method: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    raw: dict[str, Any]


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_model_config(path: str | Path) -> ModelConfig:
    payload = _load_toml(Path(path))
    return ModelConfig(**payload)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    payload = _load_toml(Path(path))
    return BenchmarkConfig(**payload)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path)
    payload = _load_toml(resolved)
    return ExperimentConfig(
        name=payload["name"],
        description=payload["description"],
        model_configs=[Path(item) for item in payload["model_configs"]],
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        global_seed=payload["global_seed"],
        reruns_per_method=payload["reruns_per_method"],
        prompt_version=payload["prompt_version"],
        max_concurrent_requests=payload["max_concurrent_requests"],
        requests_per_minute_limit=payload.get("requests_per_minute_limit"),
        tokens_per_minute_limit=payload.get("tokens_per_minute_limit"),
        raw=payload,
    )


def phase_metadata(experiment: ExperimentConfig, phase_name: str) -> dict[str, Any]:
    return dict(experiment.raw["phases"][phase_name])
