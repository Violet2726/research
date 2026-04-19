"""共享方法目录加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class MethodConfig:
    """统一的方法配置结构。"""

    name: str
    family: str
    budget_calls: int
    temperature: float
    top_p: float
    max_output_tokens: int


def load_method_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """从 TOML 方法目录加载方法定义。"""
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    methods = payload.get("methods", {})
    return {
        str(name): MethodConfig(name=str(name), **config)
        for name, config in methods.items()
    }
