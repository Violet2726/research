"""统一维护 faithful 矩阵的实验规格。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib


TRACK_SAME_CONTEXT = "same_context"
TRACK_SPLIT_CONTEXT = "split_context"
EVIDENCE_HEADLINE = "headline"
EVIDENCE_SUPPORTING = "supporting"
EVIDENCE_DIAGNOSTIC = "diagnostic"
EVIDENCE_REFERENCE = "reference"


@dataclass(frozen=True)
class ExperimentMatrixSpec:
    """单个实验入口在 faithful 矩阵中的比较规格。"""

    evaluation_track: str
    evidence_tier: str
    primary_method_name: str
    best_no_comm_candidates: tuple[str, ...]
    full_comm_reference: str | None = None
    full_context_reference: str | None = None
    token_gate_basis: str = "none"


MATRIX_SPEC_PATH = Path("configs/core/matrix/faithful_matrix.toml")


@lru_cache(maxsize=1)
def _load_matrix_spec_rows() -> tuple[tuple[str, ExperimentMatrixSpec], ...]:
    with MATRIX_SPEC_PATH.open("rb") as handle:
        payload = tomllib.load(handle)
    rows: list[tuple[str, ExperimentMatrixSpec]] = []
    for entry in payload.get("entries", []):
        config_path = str(entry["config_path"])
        rows.append(
            (
                config_path,
                ExperimentMatrixSpec(
                    evaluation_track=str(entry["evaluation_track"]),
                    evidence_tier=str(entry["evidence_tier"]),
                    primary_method_name=str(entry["primary_method_name"]),
                    best_no_comm_candidates=tuple(str(item) for item in entry.get("best_no_comm_candidates", [])),
                    full_comm_reference=_optional_str(entry, "full_comm_reference"),
                    full_context_reference=_optional_str(entry, "full_context_reference"),
                    token_gate_basis=str(entry.get("token_gate_basis", "none")),
                ),
            )
        )
    return tuple(rows)


def ordered_matrix_config_paths() -> tuple[str, ...]:
    """返回 faithful matrix 的正式执行顺序。"""

    return tuple(config_path for config_path, _ in _load_matrix_spec_rows())


def _optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def get_experiment_matrix_spec(config_path: str) -> ExperimentMatrixSpec:
    """按实验配置路径返回 faithful 矩阵规格。"""
    for registered_path, spec in _load_matrix_spec_rows():
        if registered_path == config_path:
            return spec
    raise KeyError(f"Missing matrix spec for config {config_path}")
