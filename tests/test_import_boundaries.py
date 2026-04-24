from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENT_PACKAGES = {
    "single_agent_baselines",
    "multi_agent_baselines",
    "selective_comm",
    "sparc",
    "budget_comm",
    "sid_lite",
    "free_mad_lite",
    "comm_necessary",
}


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
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "api_baselines" in text or "experiment_common" in text:
            violations.append(str(path))
    assert not violations, "\n".join(violations)
