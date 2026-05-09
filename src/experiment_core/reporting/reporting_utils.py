"""跨实验报告生成时复用的轻量工具函数。"""

from __future__ import annotations

import re
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


def slugify_report_fragment(value: Any) -> str:
    """把报告文件名片段标准化成稳定 slug。"""

    text = str(value or "").strip().lower()
    normalized = re.sub(r"[^0-9a-zA-Z._-]+", "-", text.replace("/", "-"))
    collapsed = re.sub(r"-{2,}", "-", normalized).strip("-")
    return collapsed or "unknown"


def build_published_report_name(manifest: dict[str, Any], *, stem: str = "report") -> str:
    """统一生成发布到 `reports/<family>/` 的 Markdown 文件名。"""

    created_at = str(manifest.get("created_at") or "")
    created_date = created_at.split("T", 1)[0] if "T" in created_at else "unknown-date"
    experiment = slugify_report_fragment(manifest.get("experiment") or manifest.get("name"))
    phase = slugify_report_fragment(manifest.get("phase") or manifest.get("phase_name"))
    backbone = slugify_report_fragment(resolve_manifest_model_name(manifest))
    suffix = slugify_report_fragment(stem)
    return f"{created_date}-{experiment}-{phase}-{backbone}-{suffix}.md"
