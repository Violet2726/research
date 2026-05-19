"""共享配置加载。

本模块负责把 `configs/core/shared/` 与各实验包引用的 TOML 配置解析成具名数据结构。
它的职责不是做实验逻辑决策，而是把“原始配置文本”转换成“可执行、可校验、可追踪”的运行配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any
import tomllib


SHARED_CONFIG_ROOT = Path("configs/core/shared")
DEFAULT_MODEL_CATALOG_PATH = SHARED_CONFIG_ROOT / "model_catalog.toml"
DEFAULT_PROVIDERS_DIR = SHARED_CONFIG_ROOT / "providers"
REASONING_EFFORT_VALUES = {"high", "medium", "low", "none"}


@dataclass(frozen=True)
class ProviderConfig:
    """Provider 的原始默认配置。"""

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
    """模型目录中的单条模型覆盖项。"""

    model_ref: str
    tags: list[str]
    reasoning_effort: str | None
    supports_response_format: bool | None
    response_format: str | None
    default_max_output_tokens: int | None
    timeout_seconds: int | None
    max_retries: int | None


@dataclass(frozen=True)
class ResolvedModelConfig:
    """合并 provider 默认值与模型级覆盖项后的可运行模型配置。"""

    name: str
    provider: str
    model_id: str
    base_url: str
    api_key_env: str
    chat_path: str
    default_temperature: float
    default_top_p: float
    default_max_output_tokens: int
    reasoning_effort: str | None
    supports_response_format: bool
    response_format: str | None
    timeout_seconds: int
    max_retries: int
    tags: list[str]


@dataclass(frozen=True)
class BenchmarkConfig:
    """单个 benchmark 的数据集配置。"""

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
    options_field: str | None = None
    answer_index_field: str | None = None
    archive_member: str | None = None
    archive_password: str | None = None
    split_presets: list[dict[str, Any]] = field(default_factory=list)
    record_filters: list[dict[str, Any]] = field(default_factory=list)
    config_path: str | None = None
    cache_namespace_override: str | None = None
    cache_namespace: str | None = None


def _load_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 文件并返回原始字典。"""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _provider_config_path(provider_name: str) -> Path:
    """根据 provider 名称定位其共享配置文件。"""
    return DEFAULT_PROVIDERS_DIR / f"{provider_name}.toml"


def load_provider_config(path: str | Path) -> ProviderConfig:
    """加载单个 provider 配置文件。"""
    payload = _load_toml(path)
    return ProviderConfig(
        name=str(payload["name"]),
        base_url=str(payload["base_url"]),
        api_key_env=str(payload["api_key_env"]),
        chat_path=str(payload["chat_path"]),
        default_temperature=float(payload["default_temperature"]),
        default_top_p=float(payload["default_top_p"]),
        default_max_output_tokens=int(payload["default_max_output_tokens"]),
        supports_response_format=bool(payload["supports_response_format"]),
        response_format=payload.get("response_format"),
        timeout_seconds=int(payload["timeout_seconds"]),
        max_retries=int(payload["max_retries"]),
    )


def load_model_catalog(path: str | Path = DEFAULT_MODEL_CATALOG_PATH) -> dict[str, ModelCatalogEntry]:
    """加载模型目录。

    如果目录文件不存在，则返回空字典，方便最小化测试环境直接沿用 provider 默认值。
    """
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {}
    payload = _load_toml(catalog_path)
    models = payload.get("models", {})
    return {
        str(model_ref): ModelCatalogEntry(
            model_ref=str(model_ref),
            tags=[str(tag) for tag in item.get("tags", [])],
            reasoning_effort=_optional_reasoning_effort(item, "reasoning_effort"),
            supports_response_format=_optional_bool(item, "supports_response_format"),
            response_format=item.get("response_format"),
            default_max_output_tokens=_optional_int(item, "default_max_output_tokens"),
            timeout_seconds=_optional_int(item, "timeout_seconds"),
            max_retries=_optional_int(item, "max_retries"),
        )
        for model_ref, item in models.items()
    }


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """加载单个 benchmark 配置。"""
    payload = _load_toml(path)
    config_path = Path(path)
    try:
        config_path_text = Path(os.path.relpath(config_path, Path.cwd())).as_posix()
    except ValueError:
        config_path_text = config_path.as_posix()
    payload_copy = dict(payload)
    cache_namespace_override = _optional_str(payload_copy, "cache_namespace_override")
    payload_copy.pop("cache_namespace_override", None)
    return BenchmarkConfig(
        **payload_copy,
        config_path=config_path_text,
        cache_namespace_override=cache_namespace_override,
        cache_namespace=cache_namespace_override or benchmark_cache_namespace(str(payload["source_path"])),
    )


def resolve_model_ref(
    model_ref: str,
    model_catalog_path: str | Path = DEFAULT_MODEL_CATALOG_PATH,
) -> ResolvedModelConfig:
    """把 `provider/model_name` 解析成最终可执行的模型配置。

    解析顺序是先读取 provider 默认配置，再叠加模型目录中的可选覆盖项，
    最后得到 runner 可以直接消费的 `ResolvedModelConfig`。
    """
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
        reasoning_effort=(
            catalog_entry.reasoning_effort
            if catalog_entry and catalog_entry.reasoning_effort is not None
            else None
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
    """解析形如 `provider/model_name` 的模型引用。"""
    if "/" not in model_ref:
        raise RuntimeError(
            f"Invalid model ref '{model_ref}'. Use the format 'provider/model_name'."
        )
    provider_name, model_name = model_ref.split("/", 1)
    if not provider_name or not model_name:
        raise RuntimeError(
            f"Invalid model ref '{model_ref}'. Use the format 'provider/model_name'."
        )
    return provider_name, model_name


def benchmark_cache_namespace(source_path: str | Path) -> str:
    """把数据集源路径转换成 cache / config 共用的层级键。"""

    normalized = str(source_path).replace("\\", "/").strip()
    path = Path(normalized)
    if not path.is_absolute():
        return path.with_suffix("").as_posix()

    parts = list(path.parts)
    lowered = [str(part).lower() for part in parts]
    if "datasets" in lowered:
        anchor = lowered.index("datasets") + 1
        return Path(*parts[anchor:]).with_suffix("").as_posix()

    return path.with_suffix("").name


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    """读取可选布尔字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return bool(value)


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    """读取可选非空字符串字段。"""

    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    """读取可选整数字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _optional_reasoning_effort(payload: dict[str, Any], key: str) -> str | None:
    """读取并校验可选的 `reasoning_effort` 字段。"""
    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value)
    if normalized not in REASONING_EFFORT_VALUES:
        raise RuntimeError(
            f"Invalid reasoning_effort {normalized!r}. Expected one of {sorted(REASONING_EFFORT_VALUES)}."
        )
    return normalized
