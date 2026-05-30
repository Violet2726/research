"""family 运行 manifest 的统一规范化入口。"""

from __future__ import annotations

from typing import Any

from research_experiments.core.contracts import ARTIFACT_SCHEMA_VERSION
from research_experiments.families.registry import get_family_registration


def finalize_family_manifest(
    manifest: dict[str, Any],
    *,
    family_name: str,
    matrix_membership: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """补齐正式 run manifest 必需的稳定合同字段。"""

    registration = get_family_registration(family_name)
    payload = dict(manifest)
    payload["family_name"] = family_name
    payload["prototype"] = registration.prototype
    payload["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    payload["artifact_schema"] = registration.artifact_schema.to_manifest_payload()
    payload["matrix_membership"] = list(matrix_membership or payload.get("matrix_membership") or [])
    return payload
