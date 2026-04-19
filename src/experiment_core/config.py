"""共享配置加载。

本模块负责 provider、模型目录和 benchmark 配置的解析，是三条实验线
统一依赖的配置入口。这里返回的 dataclass 尽量保持“已解析、可直接运行”
的形态，避免上层 runner 再拼装底层细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


SHARED_CONFIG_ROOT = Path("configs/shared")
DEFAULT_MODEL_CATALOG_PATH = SHARED_CONFIG_ROOT / "model_catalog.toml"
DEFAULT_PROVIDERS_DIR = SHARED_CONFIG_ROOT / "providers"


@dataclass(frozen=True)
class ProviderConfig:
    """provider 原始配置。"""

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
    """模型目录中的单条覆盖项。"""

    model_ref: str
    tags: list[str]
    supports_response_format: bool | None
    response_format: str | None
    default_max_output_tokens: int | None
    timeout_seconds: int | None
    max_retries: int | None


@dataclass(frozen=True)
class ResolvedModelConfig:
    """合并 provider 默认值与模型覆盖项后的可运行模型配置。"""

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
    """基准数据集配置。"""

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


def _load_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 文件并返回原始字典。"""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _provider_config_path(provider_name: str) -> Path:
    """根据 provider 名称推导其配置文件路径。"""
    return DEFAULT_PROVIDERS_DIR / f"{provider_name}.toml"


def load_provider_config(path: str | Path) -> ProviderConfig:
    """加载单个 provider 配置。"""
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
    """加载模型目录；目录不存在时返回空字典。"""
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {}
    payload = _load_toml(catalog_path)
    models = payload.get("models", {})
    return {
        str(model_ref): ModelCatalogEntry(
            model_ref=str(model_ref),
            tags=[str(tag) for tag in item.get("tags", [])],
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
    return BenchmarkConfig(**_load_toml(path))


def resolve_model_ref(
    model_ref: str,
    model_catalog_path: str | Path = DEFAULT_MODEL_CATALOG_PATH,
) -> ResolvedModelConfig:
    """解析 ``provider/model`` 形式的模型引用并合并默认值。"""
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
    """拆解 ``provider/model_name`` 形式的模型引用。"""
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


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    """读取可选布尔字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return bool(value)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    """读取可选整数值。"""
    value = payload.get(key)
    if value is None:
        return None
    return int(value)
