"""实验家族注册中心。"""

from __future__ import annotations

from functools import cache
from importlib import import_module
from pathlib import Path

from research_experiments.core.contracts import FamilyRegistration

FAMILIES_ROOT = Path(__file__).resolve().parent


@cache
def _family_registration_map() -> dict[str, FamilyRegistration]:
    registrations: dict[str, FamilyRegistration] = {}
    for path in sorted(FAMILIES_ROOT.iterdir()):
        if not path.is_dir() or path.name == "shared":
            continue
        if not (path / "__init__.py").exists():
            continue
        registration_path = path / "registration.py"
        if not registration_path.exists():
            continue
        module = import_module(f"research_experiments.families.{path.name}.registration")
        registration = getattr(module, "REGISTRATION", None)
        if not isinstance(registration, FamilyRegistration):
            raise TypeError(f"{registration_path} 必须导出 FamilyRegistration 类型的 REGISTRATION。")
        if registration.family_name != path.name:
            raise ValueError(f"{registration_path} 的 family_name 必须与目录名一致。")
        registrations[path.name] = registration
    return registrations


def get_family_registration(family_name: str) -> FamilyRegistration:
    """按 family 名返回注册对象。"""

    return _family_registration_map()[family_name]


def registered_family_registrations() -> tuple[FamilyRegistration, ...]:
    """返回稳定排序后的 family 注册列表。"""

    return tuple(_family_registration_map()[name] for name in registered_family_names())


def registered_family_names() -> tuple[str, ...]:
    """返回稳定排序后的 family 名列表。"""

    return tuple(sorted(_family_registration_map()))


def validator_map() -> dict[str, object]:
    """返回按 family 名索引的校验函数映射。"""

    return {
        family_name: registration.validate_run
        for family_name, registration in _family_registration_map().items()
    }
