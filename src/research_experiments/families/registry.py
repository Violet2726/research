"""实验家族平台注册中心。"""

from __future__ import annotations

from functools import cache
from importlib import import_module
from pathlib import Path

from research_experiments.core.contracts import FamilyManifest

FAMILIES_ROOT = Path(__file__).resolve().parent


@cache
def _family_manifest_map() -> dict[str, FamilyManifest]:
    manifests: dict[str, FamilyManifest] = {}
    for path in sorted(FAMILIES_ROOT.iterdir()):
        if not path.is_dir() or path.name == "shared":
            continue
        if not (path / "__init__.py").exists():
            continue
        manifest_path = path / "family_manifest.py"
        if not manifest_path.exists():
            continue
        module = import_module(f"research_experiments.families.{path.name}.family_manifest")
        manifest = getattr(module, "MANIFEST", None)
        if not isinstance(manifest, FamilyManifest):
            raise TypeError(f"{manifest_path} 必须导出 FamilyManifest 类型的 MANIFEST。")
        if manifest.family_name != path.name:
            raise ValueError(f"{manifest_path} 的 family_name 必须与目录名一致。")
        manifests[path.name] = manifest
    return manifests


def get_family_manifest(family_name: str) -> FamilyManifest:
    """按 family 名返回平台注册合同。"""

    return _family_manifest_map()[family_name]


def get_family_spec(family_name: str) -> FamilyManifest:
    """兼容旧调用点，返回 family manifest。"""

    return get_family_manifest(family_name)


def registered_family_manifests() -> tuple[FamilyManifest, ...]:
    """返回稳定排序后的 family manifest 列表。"""

    return tuple(_family_manifest_map()[name] for name in registered_family_names())


def registered_family_names() -> tuple[str, ...]:
    """返回稳定排序后的 family 名列表。"""

    return tuple(sorted(_family_manifest_map()))


def validator_map() -> dict[str, object]:
    """返回按 family 名索引的校验函数映射。"""

    return {
        family_name: manifest.validator
        for family_name, manifest in _family_manifest_map().items()
    }
