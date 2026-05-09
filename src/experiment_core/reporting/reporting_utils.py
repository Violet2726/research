"""跨实验报告生成时复用的轻量工具函数。"""

from __future__ import annotations

from typing import Any


def resolve_manifest_model_name(manifest: dict[str, Any]) -> str:
    """从 manifest 的多种字段中提取最终展示用模型名。"""
    resolved = manifest.get("resolved_model")
    if isinstance(resolved, dict) and resolved.get("name"):
        return str(resolved["name"])
    backbone = manifest.get("backbone")
    if isinstance(backbone, dict) and backbone.get("name"):
        return str(backbone["name"])
    primary = manifest.get("primary_model_ref")
    if primary:
        return str(primary)
    return "unknown-model"
