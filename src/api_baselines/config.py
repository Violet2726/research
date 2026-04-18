"""配置解析层。

当前项目的配置链为：
``CLI --model provider/model -> experiment -> method catalog -> model catalog -> provider``。

本模块负责把分散在 TOML 中的静态配置解析成结构化对象，并在运行前
合并 provider 默认值与可选的模型级覆盖项。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


DEFAULT_MODEL_CATALOG_PATH = Path("configs/model_catalog.toml")


@dataclass(frozen=True)
class ProviderConfig:
    """供应商级传输配置。

    该层只描述连接参数与默认请求参数，不维护模型目录。
    """

    name: str
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
class ModelCatalogEntry:
    """已登记模型的元信息与可选覆盖项。"""

    model_ref: str
    tags: list[str]
    supports_response_format: bool | None
    response_format: str | None
    default_max_output_tokens: int | None
    timeout_seconds: int | None
    max_retries: int | None


@dataclass(frozen=True)
class ResolvedModelConfig:
    """最终可运行的模型配置。

    它是 provider 默认值与 model catalog 覆盖项合并后的结果，
    也是 runner 与 provider 客户端真正消费的结构。
    """

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
    tags: list[str]


@dataclass(frozen=True)
class BenchmarkConfig:
    """单个 benchmark 的数据源与切分规则。"""

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
    """方法目录中的单个方法定义。"""

    name: str
    family: str
    budget_calls: int
    temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class ExperimentConfig:
    """实验规格。

    experiment 不再绑定默认模型，只声明方法目录、benchmark、phase 与标签约束。
    """

    name: str
    description: str
    method_catalog: Path
    benchmark_configs: list[Path]
    required_model_tags: list[str]
    benchmark_required_tags: dict[str, list[str]]
    global_seed: int
    reruns_per_method: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    raw: dict[str, Any]


def _load_toml(path: Path) -> dict[str, Any]:
    """以二进制模式读取 TOML，交给 ``tomllib`` 解析。"""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _provider_config_path(provider_name: str) -> Path:
    """根据 provider 名称定位配置文件路径。"""
    return Path("configs/providers") / f"{provider_name}.toml"


def load_provider_config(path: str | Path) -> ProviderConfig:
    """加载供应商配置。"""
    payload = _load_toml(Path(path))
    return ProviderConfig(
        name=payload["name"],
        base_url=payload["base_url"],
        api_key_env=payload["api_key_env"],
        chat_path=payload["chat_path"],
        default_temperature=payload["default_temperature"],
        default_top_p=payload["default_top_p"],
        default_max_output_tokens=payload["default_max_output_tokens"],
        supports_response_format=payload["supports_response_format"],
        response_format=payload.get("response_format"),
        timeout_seconds=payload["timeout_seconds"],
        max_retries=payload["max_retries"],
    )


def load_model_catalog(path: str | Path = DEFAULT_MODEL_CATALOG_PATH) -> dict[str, ModelCatalogEntry]:
    """加载模型元信息目录。

    目录不存在时返回空字典，这意味着运行时仍可直接使用
    ``provider/model``，只是没有 tags 和模型级覆盖项。
    """
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {}
    payload = _load_toml(catalog_path)
    models = payload.get("models", {})
    entries: dict[str, ModelCatalogEntry] = {}
    for model_ref, item in models.items():
        entries[str(model_ref)] = ModelCatalogEntry(
            model_ref=str(model_ref),
            tags=[str(tag) for tag in item.get("tags", [])],
            supports_response_format=_optional_bool(item, "supports_response_format"),
            response_format=item.get("response_format"),
            default_max_output_tokens=_optional_int(item, "default_max_output_tokens"),
            timeout_seconds=_optional_int(item, "timeout_seconds"),
            max_retries=_optional_int(item, "max_retries"),
        )
    return entries


def load_method_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载共享方法目录。"""
    payload = _load_toml(Path(path))
    methods = payload.get("methods", {})
    return {
        str(name): MethodConfig(name=str(name), **config)
        for name, config in methods.items()
    }


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """加载单个 benchmark 配置。"""
    payload = _load_toml(Path(path))
    return BenchmarkConfig(**payload)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """加载实验规格。"""
    payload = _load_toml(Path(path))
    return ExperimentConfig(
        name=payload["name"],
        description=payload["description"],
        method_catalog=Path(payload["method_catalog"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        required_model_tags=[str(item) for item in payload.get("required_model_tags", [])],
        benchmark_required_tags={
            str(benchmark): [str(tag) for tag in tags]
            for benchmark, tags in payload.get("benchmark_required_tags", {}).items()
        },
        global_seed=payload["global_seed"],
        reruns_per_method=payload["reruns_per_method"],
        prompt_version=payload["prompt_version"],
        max_concurrent_requests=payload["max_concurrent_requests"],
        requests_per_minute_limit=payload.get("requests_per_minute_limit"),
        tokens_per_minute_limit=payload.get("tokens_per_minute_limit"),
        raw=payload,
    )


def phase_metadata(experiment: ExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回某个 phase 的原始配置字典。"""
    return dict(experiment.raw["phases"][phase_name])


def resolve_model_ref(
    model_ref: str,
    model_catalog_path: str | Path = DEFAULT_MODEL_CATALOG_PATH,
) -> ResolvedModelConfig:
    """解析 ``provider/model``，并合并 provider 默认值与 catalog 覆盖项。"""
    provider_name, model_name = parse_model_ref(model_ref)
    provider = load_provider_config(_provider_config_path(provider_name))
    catalog_entry = load_model_catalog(model_catalog_path).get(model_ref)
    return ResolvedModelConfig(
        name=model_ref,
        provider=provider_name,
        model_id=model_name,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        chat_path=provider.chat_path,
        default_temperature=provider.default_temperature,
        default_top_p=provider.default_top_p,
        default_max_output_tokens=(
            catalog_entry.default_max_output_tokens
            if catalog_entry and catalog_entry.default_max_output_tokens is not None
            else provider.default_max_output_tokens
        ),
        supports_response_format=(
            catalog_entry.supports_response_format
            if catalog_entry and catalog_entry.supports_response_format is not None
            else provider.supports_response_format
        ),
        response_format=(
            catalog_entry.response_format
            if catalog_entry and catalog_entry.response_format is not None
            else provider.response_format
        ),
        timeout_seconds=(
            catalog_entry.timeout_seconds
            if catalog_entry and catalog_entry.timeout_seconds is not None
            else provider.timeout_seconds
        ),
        max_retries=(
            catalog_entry.max_retries
            if catalog_entry and catalog_entry.max_retries is not None
            else provider.max_retries
        ),
        tags=list(catalog_entry.tags) if catalog_entry is not None else [],
    )


def parse_model_ref(model_ref: str) -> tuple[str, str]:
    """拆解 ``provider/model`` 形式的运行时模型引用。"""
    if "/" not in model_ref:
        raise RuntimeError(
            f"Invalid model ref '{model_ref}'. Use the format 'provider/model_name', for example 'dashscope/qwen2.5-7b-instruct'."
        )
    provider_name, model_name = model_ref.split("/", 1)
    if not provider_name or not model_name:
        raise RuntimeError(
            f"Invalid model ref '{model_ref}'. Use the format 'provider/model_name'."
        )
    return provider_name, model_name


def required_model_tags(experiment: ExperimentConfig, phase_name: str) -> list[str]:
    """汇总 experiment 级与 phase 级模型标签要求。"""
    phase = phase_metadata(experiment, phase_name)
    combined = list(experiment.required_model_tags)
    combined.extend(str(item) for item in phase.get("required_model_tags", []))
    return _dedupe_preserving_order(combined)


def required_benchmark_tags(
    experiment: ExperimentConfig,
    phase_name: str,
    benchmark_slug: str,
) -> list[str]:
    """汇总某个 benchmark 在当前 phase 下需要满足的额外标签。"""
    phase = phase_metadata(experiment, phase_name)
    combined = list(experiment.benchmark_required_tags.get(benchmark_slug, []))
    combined.extend(str(item) for item in phase.get("benchmark_required_tags", {}).get(benchmark_slug, []))
    return _dedupe_preserving_order(combined)


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """去重但保持原顺序，方便在报错信息里还原配置语义。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    """安全读取可选布尔字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return bool(value)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    """安全读取可选整数字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return int(value)
