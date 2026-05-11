"""约束包间导入边界，防止实验层与核心层职责串扰。"""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

from experiment_core.orchestration.registry import registered_family_names


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENT_PACKAGES = set(registered_family_names())


def test_no_cross_experiment_imports() -> None:
    violations: list[str] = []
    for package in EXPERIMENT_PACKAGES:
        for path in (SRC / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    root = name.split(".", 1)[0]
                    if root in EXPERIMENT_PACKAGES and root != package:
                        violations.append(f"{path}: {name}")
    assert not violations, "\n".join(violations)


def test_no_legacy_package_imports() -> None:
    legacy_markers = (
        "api_baselines",
        "experiment_common",
        "single_agent_baselines",
        "multi_agent_baselines",
        "smoke20_orchestrator",
        "smoke20_matrix",
        "smoke20_matrix_cli",
    )
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in legacy_markers):
            violations.append(str(path))
    assert not violations, "\n".join(violations)


def test_no_legacy_matrix_cli_entrypoint() -> None:
    pyproject_path = ROOT / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]

    assert "faithful_matrix_cli" in scripts
    assert "smoke20_matrix_cli" not in scripts
